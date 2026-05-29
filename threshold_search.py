import torch
import numpy as np
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from data_loading import get_data
from training import get_model
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders
import json
from util import create_parser, set_seed

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    sample_batch = next(iter(tr_loader))
    
    class Config:
        lr = extract_param("lr", args)
        n_hidden = extract_param("n_hidden", args)
        n_gnn_layers = extract_param("n_gnn_layers", args)
        dropout = extract_param("dropout", args)
        final_dropout = extract_param("final_dropout", args)
        w_ce1 = extract_param("w_ce1", args)
        w_ce2 = extract_param("w_ce2", args)
        n_heads = extract_param("n_heads", args) if args.model == 'gat' else None
    
    config = Config()
    model = get_model(sample_batch, config, args)

    checkpoint_name = f'checkpoint_{args.unique_name}.tar' if args.unique_name else 'checkpoint_tgnn_gat_v2.tar'
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/{checkpoint_name}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Evaluating on Test Set to extract raw probabilities...")
    preds = []
    ground_truths = []
    pred_probs = []
    
    loader = te_loader
    inds = te_inds
    data = te_data

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        inds_cpu = inds.detach().cpu()
        batch_edge_inds = inds_cpu[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())
        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()
        
            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)
            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        batch.edge_attr = batch.edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            probs = torch.nn.functional.softmax(out, dim=-1)[:, 1]
            pred_probs.append(probs)
            ground_truths.append(batch.y[mask])

    pred_probs = torch.cat(pred_probs, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    print("\n" + "="*60)
    print("THRESHOLD SWEEP RESULTS (TEST SET)")
    print("="*60)
    print(f"{'Threshold':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'FP':<8} | {'TP':<8}")
    print("-" * 75)

    best_f1 = 0
    best_thresh = 0

    for thresh in np.arange(0.01, 0.51, 0.02):
        pred = (pred_probs >= thresh).astype(int)
        f1 = f1_score(ground_truth, pred, zero_division=0)
        prec = precision_score(ground_truth, pred, zero_division=0)
        rec = recall_score(ground_truth, pred, zero_division=0)
        cm = confusion_matrix(ground_truth, pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"{thresh:<12.2f} | {f1:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {fp:<8} | {tp:<8}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("="*60)
    print(f"Optimal Threshold (Highest F1): {best_thresh:.2f} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    main()
