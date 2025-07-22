#!/bin/bash
#SBATCH --nodelist=galadriel
#SBATCH --partition=p_csunivie_gres
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=full_fine_tuning_model_C.out  # Redirects output to a file

# Activate the virtual environment
VENV_PATH="/srv/environments/a12103237cs/venvs/olmo/bin/python3"

# Run multiple scripts sequentially. This job is for Model C
$VENV_PATH OLMo1B_train_val_model_C.py
$VENV_PATH OLMo1B_test_model_C.py
