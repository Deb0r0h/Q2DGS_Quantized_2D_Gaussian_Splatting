import subprocess
import os
from tqdm import tqdm

dataset_path = 'output/date'

scans = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106",
    "scan110", "scan114", "scan118", "scan122"
]

test = ["scan69", "scan83", "scan97", "scan105", "scan106"]

for scan in tqdm(scans, desc="Metrics on DTU dataset"):
    scan_path = os.path.join(dataset_path, scan)

    command = [
        "python", "metrics_test.py",
        "--model_paths", scan_path]

    print(f"\nComputing metrics {scan}...")
    subprocess.run(command)

print("Metrics calculation completed")