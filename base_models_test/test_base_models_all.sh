#!/bin/bash
#SBATCH --nodelist=galadriel
#SBATCH --partition=p_csunivie_gres
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256gb
#SBATCH --time=02:30:00
#SBATCH --output=all_base_models_test.out  # Redirects output to a file

# Activate the virtual environment
VENV_PATH="/srv/environments/a12103237cs/venvs/olmo/bin/python3"

# Run multiple scripts sequentially
$VENV_PATH OLMo1B_base_test.py
$VENV_PATH OLMo7B_base_test.py
$VENV_PATH OLMo13B_base_test.py