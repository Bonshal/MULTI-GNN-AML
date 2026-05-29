import os
import json
import subprocess

creds = {
    "username": "bonshallangthasa",
    "key": "KGAT_2ad5574a4312e6aa43400378f16af386"
}

kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')

with open(kaggle_file, 'w') as f:
    json.dump(creds, f)

os.chmod(kaggle_file, 0o600)

dataset_dir = '/teamspace/studios/this_studio/Multi-GNN/AML_dataset'
os.makedirs(dataset_dir, exist_ok=True)
os.chdir(dataset_dir)

print("Downloading Kaggle dataset...")
subprocess.run(['/home/zeus/.venv/bin/kaggle', 'datasets', 'download', '-d', 'ealtman2019/ibm-transactions-for-anti-money-laundering-aml', '-f', 'HI-Small_Trans.csv'], check=True)

print("Unzipping dataset...")
subprocess.run(['unzip', '-o', 'HI-Small_Trans.csv.zip'], check=True)

print("Formatting dataset...")
subprocess.run(['/home/zeus/.venv/bin/python', '../format_kaggle_files.py', 'HI-Small_Trans.csv'], check=True)

print("Done!")
