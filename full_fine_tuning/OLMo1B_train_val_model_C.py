# In[1]:
from sympy import false
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, PrinterCallback, TrainerCallback
import os
import re
from hf_olmo import OLMoForCausalLM
from torch.utils.data import Dataset, DataLoader
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


# Load the model and tokenizer in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B-hf",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf",)


# add time range tokens, resize the tokenizer
special_tokens = {
    "additional_special_tokens": ["[1710-1780]", "[1780-1850]", "[1850-1920]"]
}

tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)


# data
train_df = pd.read_csv("./data_OLMo/train_df.csv")
val_df   = pd.read_csv("./data_OLMo/val_df.csv")


seed = 42  #seeds = [42, 123, 456, 789, 101112]
train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# In[9]:

def prepare_chunks(text, time_token, max_context_size=2048):
    # Reserve space for the time token and eos_token
    max_chunk_size = max_context_size - 2

    # Tokenize the full text
    input_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    # Split into chunks of max_chunk_size
    chunks = [
        input_ids[i:i + max_chunk_size]
        for i in range(0, len(input_ids), max_chunk_size)
    ]

    # Add the time token and eos_token to each chunk
    processed_chunks = [
        torch.cat([
            torch.tensor([tokenizer.convert_tokens_to_ids(time_token)]),
            chunk,
            torch.tensor([tokenizer.eos_token_id])
        ])
        for chunk in chunks
    ]

    return processed_chunks


# Prepare encodings dynamically from a dataframe
def prepare_encodings(df_subset):
    encodings = []
    for _, row in df_subset.iterrows():
        text = row['cleaned_text']
        time_token = row['yearRange']  # Use yearRange as the special time token
        encodings.extend(prepare_chunks(text, time_token))
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


training_args = TrainingArguments(
    output_dir="./",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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
    save_safetensors=False,
    fp16=False,
    bf16=True,
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

model.save_pretrained("./fine_tuned_model_C")
tokenizer.save_pretrained("./fine_tuned_model_C")
