#!/usr/bin/env bash
set -e

echo "=== Deleting old virtual environment ==="
rm -rf /home/zeus/.venv

echo "=== Recreating virtual environment ==="
python3 -m venv /home/zeus/.venv
source /home/zeus/.venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing PyTorch 2.3.0+cu121 ==="
pip install torch==2.3.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing PyG Extension Binaries ==="
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

echo "=== Installing remaining requirements ==="
pip install torch-geometric pandas scikit-learn tqdm wandb jinja2

echo "=== Installation complete ==="
