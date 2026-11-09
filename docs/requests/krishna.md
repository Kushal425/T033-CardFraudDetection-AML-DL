# Change Requests & Integration Notes — Krishna Pramod Dubale (Task P2_01)

**Author:** Krishna Pramod Dubale (Enrollment: 2401010239)  
**Branch:** `feat/krishna-p2`  
**Date:** 2026-11-10  
**Target:** Kushal Sarkar (Integration Lead / P2_05) & Team T033  

---

## 1. Shared File Requests for Lead Integration (P2_05)

### Request 1: LaTeX Section Integration in `reports/ieee_report/main_phase2.tex`
Please include the three Phase 2 deliverable sections authored for P2_01 in the final IEEE manuscript:
```latex
\input{sections/p2_introduction.tex}
\input{sections/p2_dl_literature.tex}
\input{sections/p2_dl_data_pipeline.tex}
```

### Request 2: Bibliography Merge into `reports/ieee_report/refs.bib`
All verified academic citations for Deep Learning foundations, denoising score matching, anomaly detection, optimization, and dataset shift are consolidated in:
`reports/refs_krishna_p2.bib`
Please merge these BibTeX entries into `reports/ieee_report/refs.bib` during the Step 1 bibliography consolidation audit.

---

## 2. Shared Data Pipeline Modules for Teammates

### For Vani (P2_03 - Autoencoder Architecture & Training):
You can import the verified PyTorch tensors and DataLoader builder directly from `src.eda_dl`:
```python
from src.data import get_features
from src.eda_dl import create_dl_tensors, create_dl_dataloaders

splits = get_features(feature_set="base")
tensors = create_dl_tensors(splits)
loaders = create_dl_dataloaders(tensors, batch_size=512)

train_loader = loaders["train_loader"]        # pure legitimate, (x, x) pairs, shuffled
val_legit_loader = loaders["val_legit_loader"] # for early stopping, (x, x) pairs, no shuffle
val_loader = loaders["val_loader"]            # (x, y) pairs
test_loader = loaders["test_loader"]          # (x, y) pairs
```
- Tensors are guaranteed `torch.float32`, strictly finite, and have $0$ fraud in `X_train_legit`.
- Early-stopping loader contains only the $56,904$ legitimate validation transactions, preventing label leakage.

### For Apoorva (P2_02 - Latent GMM) & Prachee (P2_04 - Hybrids):
- Feature variance dominance analysis and summary statistics are saved to:
  - `results/tables/dl_data_feature_stats.csv`
  - `results/tables/dl_data_variance_dominance.csv`
- All unit tests pass via `pytest tests/test_dl_data.py -v`.
