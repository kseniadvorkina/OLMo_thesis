## Mixture of Experts — `MoE/` Folder

This folder contains scripts for training and evaluating **Mixture of Experts (MoE)** models for temporal adaptation on the CLMET dataset.  
The approach builds on fine-tuned OLMo 1B experts trained on specific historical periods and combines them using soft or rule-based gating mechanisms.

---

### Data Preparation

To adapt CLMET for Mixture of Experts:

1. **Split** the original `train_df`, `val_df`, and `test_df` into **three temporal periods** using the `yearRange` column  
   → This results in **nine** datasets: 3 splits × 3 partitions (train/val/test)
2. **Save** each subset into a designated data directory

Then, retain the original full `train_df`, `val_df`, and `test_df` files for **training and evaluating the gating mechanism**.

> **Note:** Validation and test for the MoE gate are conducted using identical script structures — the only difference lies in the input data passed.

---

### Expert Models

To create expert models:

- Use the scripts from `full_fine_tuning/`, applied to the **temporal subsets** of the data
- For replication of the thesis results, use the following configurations:
  - **Model B** — No temporal conditioning
  - **Model C** — Year range as special token
  - **Model E** — Exact year in natural language prompt

Train one expert per temporal period per configuration. These experts will then be passed to the MoE gate.

---

### Gating Mechanism

After training the experts, use the scripts in this folder to train and evaluate the **Mixture of Experts gate**.

Three evaluation strategies are supported:

- **Soft gating** — Weighted combination of expert outputs based on gate probabilities
- **Hard gating (argmax)** — Choose the single most probable expert
- **Oracle (rule-based)** — Selects expert based on gold-standard temporal metadata (for upper-bound comparison)

---

### Scaling Up: MoE with OLMo 7B

To assess whether diachronic adaptation benefits scale with model size, we fine-tuned the larger **OLMo 7B** model using the best-performing configuration (**Model E**) and trained a corresponding MoE system.

> Note: This step requires significant compute — at least one H100 GPU per training job.

---

### Script List

Scripts are grouped by expert configuration (**B**, **C**, **E**) and function:

#### Model B
- `OLMo1B_MoE_B_train.py` — Train gate using Model B experts
- `OLMo1B_MoE_B_test_soft_gate.py` — Evaluate using soft gating
- `OLMo1B_MoE_B_test_argmax.py` — Evaluate using hard (argmax) gating
- `OLMo1B_MoE_B_test_rule_based.py` — Evaluate using oracle gating

#### Model C
- `OLMo1B_MoE_C_train.py`
- `OLMo1B_MoE_C_test_soft_gate.py`
- `OLMo1B_MoE_C_test_argmax.py`
- `OLMo1B_MoE_C_test_rule_based.py`

#### Model E
- `OLMo1B_MoE_E_train.py`
- `OLMo1B_MoE_E_test_soft_gate.py`
- `OLMo1B_MoE_E_test_argmax.py`
- `OLMo1B_MoE_E_test_rule_based.py`

---

### Usage

1. **Train experts** using temporally split datasets (see `full_fine_tuning/`)
2. **Train MoE gate** using the corresponding `*_train.py` script
3. **Evaluate** with one or more of the `*_test_*.py` scripts
4. (Optional) Log gate weights for downstream tasks (e.g., year prediction)

---

Let me know if you would like to include:
- SLURM job templates here
- A table summarising which scripts use which gating strategy
- Results plots or evaluation metrics as output screenshots
