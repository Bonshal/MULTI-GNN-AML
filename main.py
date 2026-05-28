import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data
from training import train_gnn
from inference import infer_gnn
import torch
import json

def main():
    # Enable TF32 for up to 3x faster matrix multiplications on A100/H100 Tensor Cores
    torch.set_float32_matmul_precision('high')
    parser = create_parser()
    args = parser.parse_args()

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        #Training
        logging.info(f"Running Training")
        train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
    else:
        #Inference
        logging.info(f"Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

if __name__ == "__main__":
    main()
