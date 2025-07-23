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

    def forward(self, input_ids, expert_index):
        expert_model = self.expert_models[expert_index]
        return expert_model(input_ids=input_ids)

moe_model = MoEModel(expert_models).to(device)

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
        input_ids, label = self.samples[idx]
        return {"input_ids": input_ids, "label": label}

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    return {"input_ids": padded["input_ids"].to(torch.long), "labels": labels}

# === Load Test Data ===

test_df = pd.read_csv("./data_OLMo/test_df.csv")
test_dataset = GatingDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === Run Inference on Test Set ===

test_results = []
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        torch.cuda.empty_cache()
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"]
        labels_ll = input_ids.clone()

        expert_index = labels.item()  # batch_size = 1
        outputs = moe_model(input_ids=input_ids, expert_index=expert_index)
        lm_logits = outputs.logits

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels_ll[..., 1:].contiguous()

        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        test_results.append({
            "loss": loss.item(),
            "perplexity": torch.exp(torch.tensor(loss.item())),
            "true_year_range": index_to_year[expert_index]
        })

# === Save to CSV ===

results_df = pd.DataFrame(test_results)
results_df.to_csv("moe_E_test_predictions_rule_based.csv", index=False)

# === Print Summary ===

avg_loss = results_df["loss"].mean()
avg_perplexity = torch.exp(torch.tensor(avg_loss))

print(f"\nAverage Test Loss: {avg_loss:.4f}")
print(f"Average Test Perplexity: {avg_perplexity:.4f}")
