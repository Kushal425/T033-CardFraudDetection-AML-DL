# Shared-File Change Requests — Vani Rudra (P2_03)

To: Kushal Sarkar (Integration Lead)

Per the P2_03 file-ownership rules, I did not edit any shared files directly. Please review and apply the following minimal updates during Phase 2 integration (`P2_05`):

1. **`requirements.txt`**
   - Uncomment or enable `torch>=2.0` (tested locally with `torch==2.15.0` on CPU).

2. **`tests/test_score_contract.py`**
   - Add `"ae"` and `"ae_dae"` to `REAL_MODELS` once all Phase 2 score files are merged, so the shared score-contract test suite validates `results/scores/ae_{val,test}.npy` and `results/scores/ae_dae_{val,test}.npy` automatically.

3. **`reports/ieee_report/refs.bib`**
   - Merge entries from `reports/refs_vani.bib` into the canonical bibliography (and verify/remove `[VERIFY]` tags during the final citation audit).

