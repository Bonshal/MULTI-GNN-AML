# Multi-GNN for Anti-Money Laundering (Optimized Hybrid TGNN)

This repository contains state-of-the-art Graph Neural Network (GNN) architectures designed explicitly for Anti-Money Laundering (AML) in financial transaction networks. 

This project builds upon the foundational Multi-GNN research by integrating **Explainable Hybrid Temporal Graph Attention Networks (GAT)**, optimized specifically for extreme class imbalances and operational investigator workload.

## 🚀 Key Architectural Upgrades (Hybrid TGNN)

This repository has been heavily optimized for production-grade AML environments:

1. **Explainability (GAT vs GIN):** Switched the core architecture from a "black-box" Graph Isomorphism Network (GIN) to an Explainable Graph Attention Network (GAT). The model natively outputs mathematical Attention Weights for every transaction, allowing investigators to see exactly *why* a fraud ring was flagged.
2. **Temporal Awareness (Time-Deltas & Ports):** Instead of treating graphs as static snapshots, the model injects chronological time-gaps (`--tds`) and sequence orderings (`--ports`) directly into the edge features. This allows the Attention Heads to catch rapid "smurfing" money flow patterns.
3. **PR-AUC Early Stopping:** Standard ROC-AUC is misleading in AML due to millions of True Negatives. The training loop (`training.py`) now dynamically evaluates and saves models based strictly on **PR-AUC**, mathematically optimizing the ratio between High Recall and Investigator False Positives.
4. **Focal Loss Scaling:** Configured `model_settings.json` with a mathematically balanced Focal Loss penalty (`w_ce2 = 20`) to aggressively prioritize Recall without collapsing Precision.

## 🛠 Setup & Data Engineering

To use the repository, install the conda environment:
```bash
conda env create -f env.yml
```

### Automated Dataset Preprocessing
The transaction data is sourced from the IBM AML Kaggle Datasets (e.g., `Small_LI`, `Small_HI`, `Medium_HI`). We have provided a streamlined `setup_dataset.py` script to automate the extraction and formatting of this data:
```bash
python setup_dataset.py --dataset Small_HI
```
This automatically downloads the Kaggle data and runs the PyTorch graph conversion. *Note: Ensure your `data_config.json` paths point to your respective dataset directories.*

## ⚡ Usage

To launch the ultimate **Hybrid Temporal GAT**, execute the following command:

```bash
python main.py --data Small_HI --model gat --tds --ports --batch_size 81920 --n_epochs 50 --save_model --unique_name tgnn_gat_production
```

### Architecture Arguments
You can customize the network by toggling the following flags:
| Argument       | Description                  |
| -------------- | ---------------------------- |
| `--model`      | The core GNN: `[gat, gin, rgcn, pna]` |
| `--tds`        | Injects Time-Deltas into edges (Highly Recommended) |
| `--ports`      | Injects chronological port sequences into edges |
| `--emlps`      | Edge updates via MLPs        |
| `--reverse_mp` | Reverse Message Passing (Creates Heterogeneous Graph) |

## 📊 Post-Training (Threshold Sweeping)
Because PR-AUC is threshold-independent, we have included `threshold_search.py`. Once your model is trained, run this script to sweep probability thresholds and extract the exact operational point that matches your organization's required Precision-to-Recall ratio.

## Licence
Apache License
Version 2.0, January 2004