# Master Thesis: Temporal Adaptation Techniques in Diachronic Language Modelling

This repository contains the codebase for the master’s thesis *Temporal Adaptation Techniques in Diachronic Language Modelling*.  Full text available online at https://utheses.univie.ac.at/detail/76983/.

It implements several fine-tuning and adaptation strategies for historical language modelling, including a Mixture of Experts (MoE) architecture trained on temporally split English corpora.  
Key components include temporal conditioning, soft-gated expert selection, and a year prediction heuristic based on gating probabilities.  
Experiments are conducted on the CLMET corpus, using OLMo models and the Hugging Face ecosystem.

---

## Repository Structure

This repository is structured as follows:

- `full_fine_tuning/`: Scripts for full fine-tuning with various temporal adaptation strategies (models B–F)
- `MoE/`: Scripts for training and evaluating Mixture of Experts
- `base_models_test/`: Scripts for testing the original OLMo base models
- `data_cleaning_CLMET.ipynb`: Preprocessing and cleaning of the CLMET dataset
- `year_prediction.ipynb`: Predicts publication year from MoE gate weights
- `requirements.txt`: Python dependencies

---

## Computational Resources

All experiments were conducted on the **Galadriel** node of the SLURM cluster at the University of Vienna, Faculty of Computer Science.  
Due to the scale of the models and data, training and evaluation cannot be replicated on a standard PC. Access to substantial compute is required.

**Galadriel node specifications:**

- 4× NVIDIA H100 Tensor Core GPUs  
- 192 logical CPU cores  
- ~2 TB of RAM  
- `x86_64` architecture  
- Miniforge-based Python environment

A dedicated virtual environment was used for all experiments.  Dependencies are listed in `requirements.txt`.

Jobs were submitted using SLURM with `.sh` shell scripts via `sbatch`. These scripts define resource requests and log outputs. Models and MoE gate weights were saved after each run. Sample job scripts are available in the respective folders.

---

## Data Preparation

The thesis uses **CLMET 3.1**, a diachronic English corpus.  
Download it from: [https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html](https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html)

To prepare the data:

1. Open and run `data_cleaning_CLMET.ipynb`
2. Follow the notebook steps to clean, split, and save:
   - `train_df.csv`
   - `val_df.csv`
   - `test_df.csv`

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
Fake-temporal baselines (with wrong year prompts) are also included for models C–F.

Key implementation details:

- Models **D–F** only differ in the natural language prompts generated in `prepare_chunks`
- Model **C** requires modifying the tokenizer to add special tokens
- Model **B** has no explicit temporal input modifications

---

## Mixture of Experts (MoE)

To adapt CLMET for Mixture of Experts:

1. **Split** `train_df`, `val_df`, and `test_df` into 3 temporal periods by the `yearRange` column, resulting in 9 datasets total 
2. **Save** them into a designated folder

Next:

- Fine-tune expert models using desired configurations from `full_fine_tuning/` on each subset; if you wish to replicate results of this thesis, choose  **B**, **C**, and **E**
- Train the soft gating model using expert outputs using a script from `MoE/`
- Evaluate using:
  - **Soft gating**
  - **Hard gating**
  - **Oracle-based** gating (rule-based)

In addition to the 1B experiments, we applied the Mixture of Experts approach to the OLMo 7B model using the same temporal splits and prompt strategy as Model E. Expert models were fine-tuned individually on each temporal range, and the MoE gate was trained and evaluated in the same manner as for 1B.

All relevant training and evaluation scripts are provided in the `MoE/` folder.

---

## Year Prediction

To extend MoE outputs beyond generation, we developed a **year prediction heuristic**.  
The notebook `year_prediction.ipynb` takes MoE gate weights as input and predicts the publication year.

Refer to the thesis manuscript for methodology details and the definition of the linear mapping function.

---

## Base Models Testing

To compare the performance of temporally adapted models with non-adapted baselines, use the scripts provided in the `base_models_test/` folder.
These scripts evaluate pre-trained base models (e.g. unmodified OLMo) on the CLMET dataset without any temporal adaptation.

Shell scripts for SLURM job submission are also included in this folder.

---

## Citation

If you use this code or build upon this work, please cite the associated thesis:

> *Ksenia Dvorkina, “Temporal Adaptation Techniques in Diachronic Language Modelling,” Master’s thesis, University of Vienna, 2025.*

---
