## Mixture of Experts `MoE/` Folder.

To adapt CLMET for Mixture of Experts:

1. **Split** `train_df`, `val_df`, and `test_df` into 3 temporal periods by the `yearRange` column, resulting in 9 datasets total 
2. **Save** them into a designated folder

Then again use the original `train_df`, `val_df`, and `test_df` for MoE gate training.

Due to how the models are implemented, validation and test in this configurationg are perfomed on identical scripts, with the only difference being data.

Next:

- Fine-tune expert models using desired configurations from `full_fine_tuning/` on each subset; if you wish to replicate results of this thesis, choose  **B**, **C**, and **E**
- Train the soft gating model using expert outputs using a script from `MoE/`
- Evaluate using:
  - **Soft gating**
  - **Hard gating**
  - **Oracle-based** gating (rule-based)

Scaling Up: Diachronic Mixture of Experts with OLMo 7B

To investigate whether diachronic adaptation benefits scale with model size, we also fine-tuned a larger OLMo 7B language model using the best-performing configuration identified in master thesis (MoE (E)). This adaptation requires bigger computational resorces.
