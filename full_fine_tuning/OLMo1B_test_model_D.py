# In[1]:
from sympy import false
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, PrinterCallback, TrainerCallback
import os
import re
from hf_olmo import OLMoForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import math
import evaluate
import logging
import pandas as pd

import platform
from tqdm import tqdm
import torch.multiprocessing as mp
import bitsandbytes
import sys
import gc



device = torch.device("cuda")
pin_memory_setting = True

model_dir = "./fine_tuned_model_D"

# Load the model and tokenizer in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = torch.nn.DataParallel(model)
model.to(device)


test_df  = pd.read_csv("./data_OLMo/test_df.csv")
seed = 42
test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)


# In[9]:

def prepare_chunks(text, yearRange, max_context_size=2048):
    # Create a natural-language temporal prompt
    yearRange = yearRange.strip("[]")
    time_prompt = f"This is English text written between {yearRange}.\n\n"

    # Tokenize the time prompt and the full text
    time_prompt_ids = tokenizer(time_prompt, return_tensors="pt", truncation=False)["input_ids"][0]
    text_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    # Reserve space for eos_token and the prompt in each chunk
    max_chunk_size = max_context_size - len(time_prompt_ids) - 1

    # Split text into chunks
    text_chunks = [
        text_ids[i:i + max_chunk_size]
        for i in range(0, len(text_ids), max_chunk_size)
    ]

    # Add prompt and eos token to each chunk
    processed_chunks = [
        torch.cat([
            time_prompt_ids,
            chunk,
            torch.tensor([tokenizer.eos_token_id])
        ])
        for chunk in text_chunks
    ]

    return processed_chunks


# Prepare encodings dynamically from a dataframe
def prepare_encodings(df_subset):
    encodings = []
    for _, row in df_subset.iterrows():
        yearRange = row['yearRange']
        text = row['cleaned_text']
        encodings.extend(prepare_chunks(text, yearRange))
    return encodings


# Dataset class
class FineTuneDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {"input_ids": self.encodings[idx], "labels": self.encodings[idx]}


# Prepare training, validation, and test datasets
# Convert processed encodings into datasets

test_dataset = FineTuneDataset(prepare_encodings(test_df))


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract input_ids and labels
        input_ids = [feature["input_ids"] for feature in features]

        # Dynamically pad sequences to the max length in the batch
        padded_batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,  # Pads to the max length in the batch
            return_tensors="pt"
        )

        # Set the "labels" field to be the same as input_ids
        padded_batch["labels"] = padded_batch["input_ids"].clone()

        return padded_batch


# In[35]:


data_collator = CustomDataCollator(tokenizer)


# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[36]:
# for the fine-tuned model eval
model.eval()


test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=CustomDataCollator(tokenizer))

# Run inference without Trainer
model.eval()
test_results = []
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in test_loader:
        torch.cuda.empty_cache()
        input_ids = batch["input_ids"].to(device)
        labels = input_ids
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        test_results.append(outputs.loss.item())
        print(f"Batch loss: {loss.item()}")

   
avg_loss = sum(test_results) / len(test_results) if test_results else 0
print(f"Average test loss: {avg_loss}, Average test perplexity: {torch.exp(torch.tensor(avg_loss))}")

