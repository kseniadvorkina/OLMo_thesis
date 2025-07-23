# Mixture of Experts — `MoE/` Folder

This folder contains scripts for training and evaluating **Mixture of Experts (MoE)** models for temporal adaptation on the CLMET dataset.  
The approach builds on fine-tuned OLMo 1B experts trained on specific historical periods and combines them using soft or rule-based gating mechanisms.  
We also investigate whether diachronic adaptation benefits scale with model size by adapting the MoE approach to the larger OLMo 7B model.

---

## Data Preparation

To adapt CLMET for Mixture of Experts:

1. **Split** the original `train_df`, `val_df`, and `test_df` into **three temporal periods** using the `yearRange` column  
   → This results in **nine** datasets: 3 splits × 3 partitions (train/val/test)
2. **Save** each subset into a designated data directory

Then, retain the original full `train_df`, `val_df`, and `test_df` files for **training and evaluating the gating mechanism**.

> **Note:** Validation and test for the MoE gate are conducted using identical script structures — the only difference lies in the input data passed.

---

## Expert Models

To create expert models:

- Use the scripts from `full_fine_tuning/`, applied to the **temporal subsets** of the data
- For replication of the thesis results, use the following configurations:
  - **Model B** — No temporal conditioning
  - **Model C** — Year range as special token
  - **Model E** — Exact year in natural language prompt

Train one expert per temporal period per configuration. These experts will then be passed to the MoE gate.

---

## Gating Mechanism

After training the experts, use the scripts in this folder to train and evaluate the **Mixture of Experts gate**.

Three evaluation strategies are supported:

- **Soft gating** — Weighted combination of expert outputs based on gate probabilities
- **Hard gating (argmax)** — Choose the single most probable expert
- **Oracle (rule-based)** — Selects expert based on gold-standard temporal metadata (for upper-bound comparison)

---

## Scaling Up: MoE with OLMo 7B

To assess whether diachronic adaptation benefits scale with model size, we fine-tuned the larger **OLMo 7B** model using the best-performing configuration (**Model E**).  
We trained a corresponding OLMo 7B MoE (E) gate using the script below. The evaluation scripts used for OLMo 1B MoE (E) can be **reused** without modification — just ensure that:

- Expert paths point to the 7B fine-tuned models
- Gate weights are loaded using:

  ```python
  moe_model.gate.load_state_dict(torch.load("moe_gate_7B_E_epoch3.pt", map_location=device))

> Note: This step requires significant compute — at least one H100 GPU per training job.

---

### Script List

Scripts are grouped by expert configuration (**B**, **C**, **E**) and function:

#### MoE Model B (OLMo 1B)
- `OLMo1B_MoE_B_train.py` — Train gate using Model B experts
- `OLMo1B_MoE_B_test_soft_gate.py` — Evaluate using soft gating
- `OLMo1B_MoE_B_test_argmax.py` — Evaluate using hard (argmax) gating
- `OLMo1B_MoE_B_test_rule_based.py` — Evaluate using oracle gating

#### MoE Model C (OLMo 1B)
- `OLMo1B_MoE_C_train.py`
- `OLMo1B_MoE_C_test_soft_gate.py`
- `OLMo1B_MoE_C_test_argmax.py`
- `OLMo1B_MoE_C_test_rule_based.py`

#### MoE Model E (OLMo 1B)
- `OLMo1B_MoE_E_train.py`
- `OLMo1B_MoE_E_test_soft_gate.py`
- `OLMo1B_MoE_E_test_argmax.py`
- `OLMo1B_MoE_E_test_rule_based.py`

#### MoE Model E (OLMo 7B)
- `OLMo7B_5epochs_train_MoE_E.py` — Train gate using 7B experts (E configuration)
- Reuse the OLMo 1B MoE (E) evaluation scripts for testing, as described above
  
---

### Usage

1. **Train experts** using temporally split datasets (see `full_fine_tuning/`)
2. **Train MoE gate** using the corresponding `_train.py` script
3. **Evaluate** with one or more of the `_test_.py` scripts
4. (Optional) Save gate weights for downstream tasks, this fucntionality is availible at soft gating test (e.g., year prediction)

