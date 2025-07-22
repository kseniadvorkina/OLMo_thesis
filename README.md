# Master Thesis: Temporal Adaptation Techniques in Diachronic Language Modelling
This repository contains the codebase for the master’s thesis Temporal Adaptation Techniques in Diachronic Language Modelling.
It implements several fine-tuning and adaptation strategies for historical language modelling, including a Mixture of Experts (MoE) architecture trained on temporally split English corpora. Key components include temporal conditioning, soft-gated expert selection, and a continuous year prediction heuristic based on gating probabilities. Experiments are conducted on the CLMET corpus, with model training based on OLMo and Hugging Face libraries.

This repository is structuree as follows...

# Computational resorces

All experiments in this thesis were conducted using the Galadriel node on SLURM cluster the University of Vienna. To replicated model training and evaluation, standart PC is not sufficient, and bigger computational capacotoes are required. 
Node used ofr this thesis is equipped with four NVIDIA H100 Tensor Core GPUs, 192 logical CPU cores, and approximately 2 TB of RAM. The system runs on an x86_64 architecture and uses the Miniforge distribution for environment management. At least one such GPU is reired to replicate results of this thesis.
A dedicated Python virtual environment was used for all jobs. Exact list of needed libraries is availible in requirements.txt.

# Data Preparation

This thesis employs CLMET 3.1 as dataset. This corpues is freely availible to download at https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html.

To pre-process the data, clean and aplit it, jupited notebook data_cleaning_CLMET.ipynb was used. To reproduce this research, please download CLMET and follow steps in this notebook. Then, save train_df, val_df, test_df to your designated folder.

# Full fine-tuning


# Mixture of Experts

To adapt existing data split for Mixture of Experts, pleqase simply stratify the exiting train_df, val_df, test_df by 

# Year Prediction

Year Prediction 
