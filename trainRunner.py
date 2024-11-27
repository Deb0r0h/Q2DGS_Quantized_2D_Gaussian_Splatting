import subprocess
import os
from tqdm import tqdm

dataset_path = 'dataset/DTU'
output_path = 'output/date'
depth_ratio = 0
r_value = 2

scans = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106",
    "scan110", "scan114", "scan118", "scan122"
]

scans_test = ["scan83"]

for scan in tqdm(scans_test, desc="Training DTU dataset"):
    scan_path = os.path.join(dataset_path, scan)
    output_folder = os.path.join(output_path, scan)

    command = [
        "python", "train.py",
        "-s", scan_path,
        "-m", output_folder,
        "-r", str(r_value),
        "--depth_ratio", str(depth_ratio)
    ]

    print(f"\nTraining {scan}...")
    subprocess.run(command)

print("Training completed")

#tesss