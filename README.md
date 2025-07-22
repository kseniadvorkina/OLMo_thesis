# Master Thesis: Temporal Adaptation Techniques in Diachronic Language Modelling
This repository contains the codebase for the master’s thesis Temporal Adaptation Techniques in Diachronic Language Modelling.
It implements several fine-tuning and adaptation strategies for historical language modelling, including a Mixture of Experts (MoE) architecture trained on temporally split English corpora. Key components include temporal conditioning, soft-gated expert selection, and a continuous year prediction heuristic based on gating probabilities. Experiments are conducted on the CLMET corpus, with model training based on OLMo and Hugging Face libraries.

This repository is structuree as follows...

## Computational resorces

All experiments in this thesis were conducted using the Galadriel node on SLURM cluster the University of Vienna. To replicated model training and evaluation, standart PC is not sufficient, and bigger computational capacotoes are required. 
Node used ofr this thesis is equipped with four NVIDIA H100 Tensor Core GPUs, 192 logical CPU cores, and approximately 2 TB of RAM. The system runs on an x86_64 architecture and uses the Miniforge distribution for environment management. At least one such GPU is reired to replicate results of this thesis.
A dedicated Python virtual environment was used for all jobs. Exact list of needed libraries is availible in requirements.txt.

To run jobs on the SLURM cluster, Terminal scripts .sh are ctreated. They outline required computational resources for every job and define logging. Jobs were then run via sbatch, and the outpit was later taken from the log files. In addition, we save model after fine-tuning and save gating weights for MoE. We provide sample shell fine for full finetuning and MoE in respectfull folders.

## Data Preparation

This thesis employs CLMET 3.1 as dataset. This corpues is freely availible to download at https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html.

To pre-process the data, clean and aplit it, jupited notebook data_cleaning_CLMET.ipynb was used. To reproduce this research, please download CLMET and follow steps in this notebook. Then, save train_df, val_df, test_df to your designated folder.

## Full fine-tuning
During the full fine-tuning, we use varuios temporal adaptation strategies. To illustrate the various temporal embedding strategies, we use a sample text CLMET3_1_1_67. This sample, beginning with the phrase “Preface Of The Author.” and dated to the year 1776, serves as a running example to show how each temporal embedding is incorporated during the encoding preparation phase.  (see table):

Model	Model Description	Embedding	Example Input (for CLMET3_1_1_67 text)
B	Fine-tuned OLMo 1B	No temporal conditioning	Preface Of The Author...
C	Fine-tuned OLMo 1B + year ranges as special tokens	Special year range tokens like [1710–1780], [1780–1850] and [1850–1920] prepended	[1710–1780] Preface Of The Author...
D	Fine-tuned OLMo 1B + year range in natural language prompts	Year range in natural language	This is English text written between 1710 and 1780.\n\nPreface Of The Author...
E	Fine-tuned OLMo 1B + exact year in natural language prompts	Exact year in natural language	This text was written in the year 1776.\n\nPreface Of The Author...
F	Fine-tuned OLMo 1B + exact year (word) in natural language	Exact year as a single word in natural language	1776.\n\nPreface Of The Author...

For each adaptation strategy, we provide a separate pythin script for training and eval, and for testing. All scripts are named after model configuration (B-F) are are made availible in the subfolder full_fine_tuning. Scripts for evaluation strategies with fake temporal embeddings for models C-F are also availible in this folder.

Scripts of models D-F only differ in natural language prompt from the function prepare_chunks, while for the model C we also add tokens and resize the tokeniser, and for the model B we do not use any explicit temporal embeddings.

## Mixture of Experts

To adapt existing data split for Mixture of Experts, pleqase simply stratify the exiting train_df, val_df, test_df by yearRange column and save nine resulting datasets in a designated folder.

Then, we use strategies B, C and E from the fine-tuning section to create subsets of experts. To achieve this, simply take the desired script from the privious section and change dataframe to the dataframe strarified by time periods. Then run for every period.

Then, we train the soft gating mechanism and evaluate it via soft gating and hard gating. We also evaluate sets of experts via rule-basd gate (oracle).

All scripts are made availible in the folder MoE.

## Year Prediction

To applu results of Mixture of Experts beyond text generation, we developed a liner function used to predict the year said text was published it. This function is availible in year_prediction.ipynb. It takes gate weights as an input and outputspredicted year. Please refer to the thesis monuscript for the methodological details.
