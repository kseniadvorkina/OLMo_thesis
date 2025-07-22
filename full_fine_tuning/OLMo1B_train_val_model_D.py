# In[1]:
from sympy import false
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, PrinterCallback, TrainerCallback
import os
import re
from hf_olmo import OLMoForCausalLM
from torch.utils.data import Dataset, DataLoader
#import torch.nn as nn
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



#device = torch.device("cuda")
pin_memory_setting = True
#print("Running on GPU:", torch.cuda.get_device_name(device))


# Load the model and tokenizer in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B-hf",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf",)

#model.to(device)


# data
train_df = pd.read_csv("./data_OLMo/train_df.csv")
val_df   = pd.read_csv("./data_OLMo/val_df.csv")


seed = 42  #seeds = [42, 123, 456, 789, 101112]
train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

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


def prepare_encodings(df_subset):
    encodings = []
    for _, row in df_subset.iterrows():
        text = row['cleaned_text']
        yearRange = row['yearRange']
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
train_dataset = FineTuneDataset(prepare_encodings(train_df))
val_dataset = FineTuneDataset(prepare_encodings(val_df))


# In[29]:
from transformers import DataCollatorForLanguageModeling

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


# Set training arguments
training_args = TrainingArguments(
    output_dir="./",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    #save_steps=500,
    save_total_limit=3,
    save_strategy="no",
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    logging_dir="./logs",
    logging_steps=1,
    logging_first_step=True,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=0.5,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=16,
    warmup_ratio=0.1,
    #warmup_steps=100,
    save_safetensors=False,
    fp16=False,
    bf16=True,
    #dataloader_pin_memory=pin_memory_setting,  # Enables pin_memory on GPU
    #dataloader_num_workers=50,
    weight_decay=0.005
)

data_collator = CustomDataCollator(tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[PrinterCallback()]
)


# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# In[36]:

# Train the model
train_result = trainer.train()

# Save training metrics
trainer.save_metrics("train", train_result.metrics)

# Evaluate on the validation set
eval_result = trainer.evaluate()

# Save evaluation metrics
trainer.save_metrics("eval", eval_result)

# Print training and evaluation metrics
print("Training Metrics:", train_result.metrics)
print("Evaluation Metrics:", eval_result)


# In[36]:
# for the fine-tuned model eval
model.eval()

model.save_pretrained("./fine_tuned_model_D")
tokenizer.save_pretrained("./fine_tuned_model_D")
