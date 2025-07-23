import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

# === Load Pre-trained Expert Models ===

expert_dirs = {
    "[1710-1780]": "./fine_tuned_model_full_E_1710_1780",
    "[1780-1850]": "./fine_tuned_model_full_E_1780_1850",
    "[1850-1920]": "./fine_tuned_model_full_E_1850_1920"
}

expert_models = []
for path in expert_dirs.values():
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    expert_models.append(model)

tokenizer = AutoTokenizer.from_pretrained(expert_dirs["[1710-1780]"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expert_models = [m.to(device) for m in expert_models]

# === MoE Model Definition ===

class MoEModel(nn.Module):
    def __init__(self, expert_models):
        super(MoEModel, self).__init__()
        self.expert_models = nn.ModuleList(expert_models)
        self.num_experts = len(expert_models)

        # Shared token embedding layer from expert 0
        self.token_embedding_layer = self.expert_models[0].get_input_embeddings()
        self.embedding_dim = self.token_embedding_layer.embedding_dim

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_experts)
        )

    def forward(self, input_ids):
        gate_inputs = self.token_embedding_layer(input_ids)
        pooled_inputs = gate_inputs.mean(dim=1)

        gate_logits = self.gate(pooled_inputs)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        expert_logits = [model(input_ids).logits for model in self.expert_models]
        expert_logits = torch.stack(expert_logits, dim=0)  # [num_experts, B, T, V]

        gate_probs = gate_probs.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [num_experts, B, 1, 1]
        weighted_logits = torch.sum(gate_probs * expert_logits, dim=0)  # [B, T, V]

        return {
            "lm_logits": weighted_logits,
            "gate_logits": gate_logits
        }

moe_model = MoEModel(expert_models).to(device)

# === Load Trained Gating Network ===

moe_model.gate.load_state_dict(torch.load("./moe_gate_3epochs_sentence_epoch3.pt", map_location=device))
moe_model.eval()



# === Dataset and Collate ===

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

# Inverse map for printing
index_to_year = {
    0: "[1710-1780]",
    1: "[1780-1850]",
    2: "[1850-1920]"
}


class GatingDataset(Dataset):
    def __init__(self, df):
        self.samples = []
        for idx, row in df.iterrows():
            chunks = prepare_chunks(row["cleaned_text"], row["printedDate"])
            label = year_to_index(row["yearRange"])
            for chunk in chunks:
                self.samples.append((chunk, label, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, label, original_idx = self.samples[idx]
        return {"input_ids": input_ids, "label": label, "original_idx": original_idx}


def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    original_idx = [x["original_idx"] for x in batch]
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")

    return {
        "input_ids": padded["input_ids"].to(torch.long),
        "labels": labels,
        "original_idx": original_idx
    }

# === Load Test Data ===

test_df = pd.read_csv("./data_OLMo/test_df.csv")
test_dataset = GatingDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === Run Inference on Test Set ===

test_results = []
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Validating"):
        torch.cuda.empty_cache()
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"]            # integer tensor [B]
        labels_ll = input_ids.clone()

        # Compute gate logits and softmax
        gate_inputs = moe_model.token_embedding_layer(input_ids)
        pooled_inputs = gate_inputs.mean(dim=1)
        gate_logits = moe_model.gate(pooled_inputs)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        # Manually get each expert's logits
        expert_logits = [model(input_ids).logits for model in moe_model.expert_models]
        expert_logits = torch.stack(expert_logits, dim=0)  # [E, B, T, V]

        # Hard gating: pick the expert with highest gate probability
        selected_expert_idx = torch.argmax(gate_probs, dim=-1)  # [B]

        # Gather selected expert logits per batch item
        batch_idx = torch.arange(expert_logits.size(1), device=device)  # [B]
        selected_logits = expert_logits[
            selected_expert_idx,
            batch_idx,
            :, :
        ]  # [B, T, V]

        # Compute causal language modeling loss
        shift_logits = selected_logits[..., :-1, :].contiguous()
        shift_labels = labels_ll[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        test_results.append(loss.item())
        original_indices = batch["original_idx"]
        print(f"Batch loss: {loss.item():.4f}")

avg_loss = sum(test_results) / len(test_results)
avg_perplexity = torch.exp(torch.tensor(avg_loss))

print(f"\nAverage Test Loss: {avg_loss:.4f}")
print(f"Average Test Perplexity: {avg_perplexity:.4f}")
