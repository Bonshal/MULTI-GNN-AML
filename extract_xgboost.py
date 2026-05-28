import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import json

from train_util import extract_param, add_arange_ids, get_loaders
from training import get_model
from data_loading import get_data
from util import create_parser, set_seed

def run_feature_factory():
    print("=> Loading data and configurations...")
    parser = create_parser()
    args = parser.parse_args(['--data', 'Medium_HI', '--model', 'gin', '--testing', '--batch_size', '81920', '--tqdm'])
    
    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)
        
    set_seed(1)
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    add_arange_ids([tr_data, val_data, te_data])
    
    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, None, args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=> Loading optimized weights...")
    sample_batch = next(iter(tr_loader))
    
    class DummyConfig:
        n_gnn_layers = 2
        n_hidden = 66
        dropout = 0.01
        final_dropout = 0.10
        w_ce1 = 1.0
        w_ce2 = 6.27
    
    model = get_model(sample_batch, DummyConfig(), args)
    checkpoint = torch.load('models/checkpoint_medium_hi_run1.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Hook to extract embeddings right before the MLP
    global current_embedding
    current_embedding = None
    def hook_fn(module, args):
        global current_embedding
        current_embedding = args[0]
    
    model.mlp.register_forward_pre_hook(hook_fn)
    
    def extract_features(loader, original_inds):
        X_list = []
        y_list = []
        for batch in tqdm(loader, desc="Extracting"):
            inds = original_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            batch.edge_attr = batch.edge_attr[:, 1:]
            
            with torch.no_grad():
                batch.to(device)
                _ = model(batch.x, batch.edge_index, batch.edge_attr)
                
                # Retrieve embedding from hook
                emb = current_embedding[mask]
                labels = batch.y[mask]
                
                X_list.append(emb.cpu().numpy())
                y_list.append(labels.cpu().numpy())
                
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    print("=> Extracting GNN embeddings for Training Set...")
    X_train, y_train = extract_features(tr_loader, tr_inds)
    
    print("=> Extracting GNN embeddings for Test Set...")
    X_test, y_test = extract_features(te_loader, te_inds)
    
    print(f"=> Training XGBoost on {X_train.shape[0]} edges with {X_train.shape[1]} GNN features...")
    clf = xgb.XGBClassifier(
        tree_method='hist',
        device='cpu',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=6.27
    )
    clf.fit(X_train, y_train)
    
    print("=> Predicting on Test Set...")
    pred_probs = clf.predict_proba(X_test)[:, 1]
    
    print("=> Calculating Precision-Recall Curve (Feature Factory)...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_probs)
    
    target_recall_80 = 0.80
    valid_indices_80 = np.where(recalls >= target_recall_80)[0]
    best_idx_80 = valid_indices_80[np.argmax(precisions[valid_indices_80])] if len(valid_indices_80) > 0 else 0
    
    target_recall_95 = 0.95
    valid_indices_95 = np.where(recalls >= target_recall_95)[0]
    best_idx_95 = valid_indices_95[np.argmax(precisions[valid_indices_95])] if len(valid_indices_95) > 0 else 0
    
    print(f"\n--- XGBOOST (FEATURE FACTORY) RESULTS ---")
    print(f"--- BUSINESS TARGET: 80% RECALL ---")
    print(f"REQUIRED THRESHOLD: {thresholds[best_idx_80]:.5f}")
    print(f"Resulting Precision: {precisions[best_idx_80]:.4f}")
    
    print(f"\n--- BUSINESS TARGET: 95% RECALL ---")
    print(f"REQUIRED THRESHOLD: {thresholds[best_idx_95]:.5f}")
    print(f"Resulting Precision: {precisions[best_idx_95]:.4f}")

if __name__ == "__main__":
    run_feature_factory()
