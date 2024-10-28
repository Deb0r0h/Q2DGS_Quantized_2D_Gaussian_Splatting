import subprocess
import os
from tqdm import tqdm


output_path = 'output/date'
depth_ratio = 0
r_value = 2

scans = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106",
    "scan110", "scan14", "scan118", "scan122"
]

for scan in tqdm(scans, desc="Rendering DTU dataset"):
    scan_path = os.path.join(dataset_path, scan)
    output_folder = os.path.join(output_path, scan)

    command = [
        "python", "render.py",
        "-r", str(r_value),
        "--depth_ratio", str(depth_ratio),
        "--skip_test",
        "--skin_train",
        "--model_path",
        output_folder
    ]

    print(f"\nRendering {scan}...")
    subprocess.run(command)

print("Rendering of all scenes completed")