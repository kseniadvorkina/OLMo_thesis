# Master Thesis: Temporal Adaptation Techniques in Diachronic Language Modelling

This repository contains the codebase for the master’s thesis *Temporal Adaptation Techniques in Diachronic Language Modelling*.  
It implements several fine-tuning and adaptation strategies for historical language modelling, including a Mixture of Experts (MoE) architecture trained on temporally split English corpora.  
Key components include temporal conditioning, soft-gated expert selection, and a continuous year prediction heuristic based on gating probabilities.  
Experiments are conducted on the CLMET corpus, using OLMo models and the Hugging Face ecosystem.

---

## Repository Structure

This repository is structured as follows:

- `full_fine_tuning/`: Scripts for full fine-tuning with various temporal adaptation strategies (models B–F)
- `MoE/`: Scripts for training and evaluating Mixture of Experts
- `data_cleaning_CLMET.ipynb`: Preprocessing and cleaning of the CLMET dataset
- `year_prediction.ipynb`: Predicts publication year from MoE gate weights
- `requirements.txt`: Python dependencies
- Sample `.sh` scripts: SLURM job definitions for training and evaluation

---

## Computational Resources

All experiments were conducted on the **Galadriel** node of the SLURM cluster at the University of Vienna.  
Due to the scale of the models and data, training and evaluation cannot be replicated on a standard PC. Access to substantial compute is required.

**Galadriel node specifications:**

- 4× NVIDIA H100 Tensor Core GPUs  
- 192 logical CPU cores  
- ~2 TB of RAM  
- `x86_64` architecture  
- Miniforge-based Python environment

A dedicated virtual environment was used for all experiments.  
Dependencies are listed in `requirements.txt`.

Jobs were submitted using SLURM with `.sh` shell scripts.  
These scripts define resource requests and log outputs.  
Models and MoE gate weights were saved after each run.  
Sample job scripts are available in the respective folders.

---

## Data Preparation

The thesis uses **CLMET 3.1**, a diachronic English corpus.  
Download it from: [https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html](https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html)

To prepare the data:

1. Open and run `data_cleaning_CLMET.ipynb`
2. Follow the notebook steps to clean, split, and save:
   - `train_df.pkl`
   - `val_df.pkl`
   - `test_df.pkl`

Place the resulting files in your designated data directory.

---

## Full Fine-Tuning

We test several temporal adaptation strategies using full fine-tuning.  
To illustrate the input transformations, we use a sample text (`CLMET3_1_1_67`, year: 1776, opening: *"Preface Of The Author."*).

| Model | Description | Temporal Embedding | Example Input |
|-------|-------------|---------------------|----------------|
| **B** | OLMo 1B (no temporal context) | None | `Preface Of The Author...` |
| **C** | OLMo 1B + year range tokens | Special token | `[1710–1780] Preface Of The Author...` |
| **D** | OLMo 1B + year range prompt | Natural language | `This is English text written between 1710 and 1780.\n\nPreface Of The Author...` |
| **E** | OLMo 1B + exact year prompt | Natural language | `This text was written in the year 1776.\n\nPreface Of The Author...` |
| **F** | OLMo 1B + exact year as word | Natural language | `1776.\n\nPreface Of The Author...` |

Each model (B–F) has corresponding training and evaluation scripts located in the `full_fine_tuning/` folder.  
Fake-temporal baselines (with irrelevant year prompts) are also included for models C–F.

Key implementation details:

- Models **D–F** differ in the natural language prompts generated in `prepare_chunks`
- Model **C** requires modifying the tokenizer to add special tokens
- Model **B** has no temporal input modifications

---

## Mixture of Experts (MoE)

To adapt CLMET for Mixture of Experts:

1. **Stratify** `train_df`, `val_df`, and `test_df` by the `yearRange` column  
2. **Split** each into 3 temporal periods, resulting in 9 datasets total  
3. **Save** them into a designated folder

Next:

- Fine-tune models using configurations **B**, **C**, and **E** on each subset (see `full_fine_tuning/`)
- Train the soft gating model using expert outputs
- Evaluate using:
  - **Soft gating**
  - **Hard gating**
  - **Oracle-based** gating (rule-based)

All relevant training and evaluation scripts are provided in the `MoE/` folder.

---

## Year Prediction

To extend MoE outputs beyond generation, we developed a **year prediction heuristic**.  
The notebook `year_prediction.ipynb` takes MoE gate weights as input and predicts the publication year.

Refer to the thesis manuscript for methodology details and the definition of the linear mapping function.

---

## Citation

If you use this code or build upon this work, please cite the associated thesis:

> *Ksenia Dvorkina, “Temporal Adaptation Techniques in Diachronic Language Modelling,” Master’s thesis, University of Vienna, 2025.*

---
