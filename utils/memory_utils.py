import os
import csv
from tqdm import tqdm

data_path = 'output/date'
csv_path = 'memory.csv'

scans = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106",
    "scan110", "scan114", "scan118", "scan122"
]

test = ["scan24", "scan37", "scan40"]

for scan in tqdm(scans, desc="Metrics on DTU dataset"):
    print(f"\nComputing memory usage {scan}")
    scan_path = os.path.join(data_path, scan)
    ply_file = os.path.join(scan_path, "train", "ours_30000", "fuse_post.ply")

    size_bytes = os.path.getsize(ply_file)
    size_mb = size_bytes / (1024 * 1024)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([scan, size_mb])