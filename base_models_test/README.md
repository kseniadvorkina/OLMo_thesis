## Base Models Test `base_models_test/` Folder

This folder contains scripts for evaluating **pre-trained OLMo models** on the CLMET dataset without any temporal adaptation. These results serve as a baseline for comparison with fine-tuned and temporally adapted models.

**Included scripts:**

- `OLMo1B_base_test.py` — Evaluates OLMo 1B model  
- `OLMo7B_base_test.py` — Evaluates OLMo 7B model  
- `OLMo13B_base_test.py` — Evaluates OLMo 13B model  

**Batch job script:**

- `test_base_models_all.sh` — Runs all three tests sequentially in a single SLURM job

To run only a subset of the models, simply comment out or remove `$VENV_PATH` with the corresponding Python scripts from `test_base_models_all.sh`.  
Make sure to adjust the requested computational resources in the script to match the model size(s) you are running.
