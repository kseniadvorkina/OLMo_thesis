import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

import sys
import os
stderr_path = os.path.join(os.getcwd(), f"stderr_rank{os.environ.get('RANK', '0')}.log")
sys.stderr = open(stderr_path, 'w')


# === Load Pre-trained Expert Models ===


expert_dirs = {
    "[1710-1780]": "./fine_tuning_E_7B_1710_1780/checkpoint-90",
    "[1780-1850]": "./fine_tuning_E_7B_1780_1850/checkpoint-100",
    "[1850-1920]": "./fine_tuning_E_7B_1850_1920/checkpoint-112"
}
expert_models = []
for path in expert_dirs.values():
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    model.requires_grad_(False)
    expert_models.append(model)

tokenizer = AutoTokenizer.from_pretrained(expert_dirs["[1710-1780]"], trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expert_models = [m.to(device) for m in expert_models]


# In[1]:


class MoEModel(nn.Module):
    def __init__(self, expert_models):
        super(MoEModel, self).__init__()
        self.expert_models = nn.ModuleList(expert_models)
        self.num_experts = len(expert_models)

        # Freeze all expert parameters
        for expert in self.expert_models:
            for param in expert.parameters():
                param.requires_grad = False

        # Shared token embedding layer (from first expert)
        self.token_embedding_layer = self.expert_models[0].get_input_embeddings()
        self.embedding_dim = self.token_embedding_layer.embedding_dim

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_experts)
        )

    def forward(self, input_ids):
        # Input embeddings: [B, T, E]
        gate_inputs = self.token_embedding_layer(input_ids)

        # Mean pooling across tokens: [B, E]
        pooled_inputs = gate_inputs.mean(dim=1)

        # Compute gating logits and softmax: [B, num_experts]
        gate_logits = self.gate(pooled_inputs)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        # Combine expert outputs
        weighted_logits = 0
        for i, model in enumerate(self.expert_models):
            logits = model(input_ids).logits  # [B, T, V]
            weight = gate_probs[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            weighted_logits += logits * weight

        return {
            "lm_logits": weighted_logits,  # final output
            "gate_logits": gate_logits     # for training the gate
        }


moe_model = MoEModel(expert_models).to(device)

# === Dataset and Preprocessing ===
def prepare_chunks(text, year, max_context_size=2048):
    time_prompt = f"This text was written in the year {year}.\n\n"
    time_prompt_ids = tokenizer(time_prompt, return_tensors="pt", truncation=False)["input_ids"][0]
    text_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    # Reserve space for time prompt and eos_token
    max_chunk = max_context_size - len(time_prompt_ids) - 1

    # Chunk the text only
    chunks = [text_ids[i:i + max_chunk] for i in range(0, len(text_ids), max_chunk)]

    # Prepend time prompt and append eos_token to each chunk
    return [torch.cat([time_prompt_ids, chunk, torch.tensor([tokenizer.eos_token_id])]) for chunk in chunks]


def year_to_index(year_range):
    return {
        "[1710-1780]": 0,
        "[1780-1850]": 1,
        "[1850-1920]": 2
    }[year_range]

# In[1]:

class GatingDataset(Dataset):
    def __init__(self, df):
        self.samples = []
        for _, row in df.iterrows():
            chunks = prepare_chunks(row["cleaned_text"], row["printedDate"])
            label = year_to_index(row["yearRange"])  # still used for supervision
            for chunk in chunks:
                self.samples.append((chunk, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, label = self.samples[idx]
        return {"input_ids": input_ids, "label": label}



def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    padded_input_ids = padded["input_ids"].to(torch.long)
    return {"input_ids": padded_input_ids, "labels": labels}



# === Data Loaders ===
train_df = pd.read_csv("./data_OLMo/train_df.csv")
#train_df = train_df.iloc[[1]]

train_data = GatingDataset(train_df)


train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

# In[1]:

# === Setup ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(moe_model.gate.parameters(), lr=1e-5)
epochs = 5

for epoch in range(epochs):
    moe_model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)  # must be integers: 0, 1, or 2

        outputs = moe_model(input_ids)
        gate_logits = outputs["gate_logits"]  # [B, num_experts]

        loss = criterion(gate_logits, labels)  # labels: [B]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} — Train Loss: {avg_train_loss:.4f}")

    avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss))
    print(f"Average Train Perplexity: {avg_train_perplexity:.4f}")

    # Save model checkpoint (gating network only)
    torch.save(moe_model.gate.state_dict(), f"./moe_gate_7B_E_epoch{epoch+1}.pt")
    print(f"Saved gating model for epoch {epoch+1}\n")

