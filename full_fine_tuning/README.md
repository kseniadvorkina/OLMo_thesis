## Full Fine-tuning `full_fine_tuning/` Folder

This folder contains training and evaluation scripts for models B–F used in the full fine-tuning experiments.

Each model implements a different strategy for temporal adaptation, applied to the CLMET dataset using the OLMo 1B architecture. All scripts are named according to the corresponding model variant.

**Included scripts:**

- `OLMo1B_train_val_model_B.py` — Train and evaluate model B (no temporal adaptation)  
- `OLMo1B_train_val_model_C.py` — Train and evaluate model C (special token year ranges)  
- `OLMo1B_train_val_model_D.py` — Train and evaluate model D (year range in natural language)  
- `OLMo1B_train_val_model_E.py` — Train and evaluate model E (exact year in natural language)  
- `OLMo1B_train_val_model_F.py` — Train and evaluate model F (year as a single word)  

- `OLMo1B_test_model_B.py` to `OLMo1B_test_model_F.py` — Corresponding testing scripts
- `OLMo1B_test_FAKE_model_C.py` to `OLMo1B_test_FAKE_model_F.py` — Corresponding testing scripts with incorrect temporal embeddigns (see *Evaluation with Incorrect Temporal Embeddings* below)
- `full_fine_tuning_job_sample.sh` — SLURM script template for submitting full fine-tuning jobs (train, val, test)

---

### Model Overview

| Model | Temporal Embedding Strategy | Example Input |
|-------|------------------------------|----------------|
| **B** | No temporal context | `Preface Of The Author...` |
| **C** | Special token with year range | `[1710–1780] Preface Of The Author...` |
| **D** | Natural language year range | `This is English text written between 1710 and 1780.\n\nPreface Of The Author...` |
| **E** | Natural language exact year | `This text was written in the year 1776.\n\nPreface Of The Author...` |
| **F** | Year as a word | `1776.\n\nPreface Of The Author...` |

### Key Implementation Details

- **Model B**: No temporal embeddings are added to the input.
- **Model C**: Uses special year range tokens (e.g., `[1710–1780]`) prepended to the input. The tokenizer is extended and resized to handle the additional tokens.
- **Models D–F**: Differ in how natural language prompts are constructed in the `prepare_chunks()` function.
  - The `time_prompt` string is generated differently for each (year range, full sentence with year, or year as a word).
  - Additionally, models E and F rely on retrieving the `year` from the dataset via `prepare_encodings()` and passing it into `prepare_chunks()`. Model D retrieves and passes `yearRange`.


---

### Evaluation with Incorrect Temporal Embeddings

To assess the effectiveness of temporal conditioning strategies, we performed additional evaluations using **deliberately incorrect temporal information**.

This setup tests whether the models rely meaningfully on temporal cues or simply ignore them.

**Key details:**

- **Model B** was excluded, as it does not use any temporal embeddings.
- **Models C and D** (trained with year ranges):
  - The correct time range token (C) or natural language year range (D) was replaced with a **random incorrect time period** from outside the original context.
- **Models E and F** (trained with exact years):
  - The correct year in the prompt was replaced with a **random year from a different time period**.

**Fake evaluation scripts:**

- `OLMo1B_test_FAKE_model_C.py`  — Evaluate model C (special tokens year ranges)  with incorrect temporal time tokens
- `OLMo1B_test_FAKE_model_D.py`  — Evaluate model D (year range in natural language)  with incorrect year ranges
- `OLMo1B_test_FAKE_model_E.py`  — Evaluate model E (exact year in natural language)  with incorrect years
- `OLMo1B_test_FAKE_model_F.py`  — Evaluate model F (year as a single word)  with incorrect years


These scripts are included in this folder and mirror the structure of the original testing scripts, with modifications applied during input preparation.

Use these evaluations to compare model sensitivity to temporal misalignment and report performance degradation (if any) in your results.

---


To run training or evaluation for any model:

1. Create your virtual environment using `requirements.txt`
2. Pre-process data using `data_cleaning_CLMET.ipynd` and place resulting test_df, val_df and train_df in the correct folder, adjust paths
3. Adjust `full_fine_tuning_job_sample.sh` to match the desired model script and path of your virtual environment
4. Submit via SLURM:  
   ```bash
   sbatch full_fine_tuning_job_sample.sh
5. Retrieve log file once job is finished
