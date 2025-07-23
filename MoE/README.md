### Mixture of Experts `MoE/` Folder.

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

