# Anti-Money Laundering (AML) Multi-GNN Context

This document provides complete context on the state of the AML Graph Neural Network pipeline. It is designed to be fed into an LLM to evaluate advanced strategies for drastically improving Precision while maintaining a high Recall (80%+).

## 1. Project Overview & Data
* **Objective:** Detect illicit money laundering flows in massive financial transaction networks.
* **Dataset:** The IBM Synthetic AML Dataset (`HI-Medium_Trans.csv`).
* **Scale:** ~3.5 million transactions (edges) mapped as a heterogeneous graph.
* **Class Imbalance:** Extreme. Only ~0.05% of the network edges represent actual illicit fraud (Label `1`), while 99.95% are legitimate (Label `0`).

## 2. Model Architecture
* **Framework:** PyTorch & PyTorch Geometric (PyG).
* **Model Type:** Hybrid Graph Isomorphism Network (GINe) and Graph Attention Network (GATe), specifically modified to support **Edge Updates** (incorporating transaction features like amount and time dynamically into the hidden states).
* **Hyperparameters:**
  * 2 GNN Layers
  * 66 Hidden Dimensions
  * Dropout = 0.01
  * Final Dropout (MLP Head) = 0.10
* **Dataloader:** Uses PyG's `LinkNeighborLoader` (batch size 81,920) for CPU-multithreaded localized subgraph sampling, feeding an H100 GPU.

## 3. Training State & Loss Function
* **Training Hardware:** Lightning AI H100 Instance (80GB VRAM).
* **Loss Function:** **Focal Loss** (`gamma = 2.0`, `alpha = 0.25`) combined with heavily weighted Cross-Entropy.
  * Legitimate Class Weight (`w_ce1`): `1.0`
  * Fraud Class Weight (`w_ce2`): `6.27`
* **Current Status:** The model was trained for 37 epochs and achieved early-stopping convergence at Epoch 34. The weights are locked in `checkpoint_medium_hi_run1.tar`.

## 4. Current Metrics (The Baseline)
The model achieved an outstanding separation metric:
* **Validation ROC-AUC:** `0.9831`

However, because the dataset is so imbalanced, a high ROC-AUC does not directly map to perfect Precision. We ran a Precision-Recall Curve threshold search on the final weights:
* **At 80% Recall** (Business Target): Threshold = `0.25988` | **Precision = 11.37%**
* **At 95% Recall** (Aggressive Target): Threshold = `0.14339` | **Precision = 1.28%**

## 5. The Core Problem
An 11.37% Precision means that out of every 100 flagged transactions, roughly 11 are actual money laundering and 89 are false positives. While this vastly outperforms traditional bank rule-based systems (<1% precision), the engineering goal is to drastically push Precision higher (e.g., 30%+) without dropping Recall below the 80% compliance threshold. 

## 6. Attempted Solutions (Failed)
* **The "Feature Factory" Approach:** We bypassed the GNN's MLP head, extracted the 198-dimensional relational embeddings from the PyTorch model, and fed them into XGBoost (`tree_method='hist'`) using a `scale_pos_weight` of 6.27. 
* **Result:** XGBoost underperformed the end-to-end PyTorch GNN, dropping Precision to **8.19%** at 80% Recall. The PyTorch MLP had co-adapted perfectly with its own embeddings during Focal Loss training.

## 7. Remaining Options to Evaluate
Please evaluate the following theoretical options to aggressively increase Precision during retraining. Which of these will provide the highest ROI for solving the False Positive problem without severely breaking the 80% Recall baseline?

1. **Reduce the Fraud Class Weight (`w_ce2`):** Currently set to `6.27`, the network is heavily penalized for missing criminals, forcing it to over-predict fraud. Slashing this to `2.0` or `1.0` could drop false positives organically, relying on post-training threshold tuning to maintain Recall.
2. **Online Hard Negative Mining (OHEM):** Modify the PyTorch training loop to identify the top 5% of legitimate transactions that the GNN misclassifies with high fraud probability (the hardest false positives) and heavily multiply their loss penalty (e.g., 10x) to force the model to learn the microscopic differences in legitimate high-volume flow.
3. **Differentiable F-Beta Loss:** Delete Focal Loss/Cross-Entropy entirely and build a custom PyTorch loss function that explicitly optimizes for an F-Beta score where Beta is < 1 (e.g., `Beta = 0.5`), forcing gradient descent to directly optimize for Precision.
4. **Upgrade to Heterogeneous Graph Transformer (HGT):** Replace the GIN/GAT backbone with HGT to allow the network to apply dynamic attention weights based strictly on edge-types or temporal features, filtering out noisy neighbor connections during the `LinkNeighborLoader` sub-sampling.
5. **Neighborhood Sampling Modifications:** Modify `LinkNeighborLoader` to use Importance-Based Sampling or Random Walk with Restart instead of uniform sampling, preventing massive legitimate corporate nodes from injecting noise into the local subgraphs.
