import subprocess
import os
from tqdm import tqdm


output_path = 'output/date'
gt_mesh_path = 'dataset/DTU_GT/mesh'
depth_ratio = 0
r_value = 2

scans = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106",
    "scan110", "scan14", "scan118", "scan122"
]

scans_test = ["scan83"]

for scan in tqdm(scans_test, desc="Rendering DTU dataset"):
    output_folder = os.path.join(output_path, scan)
    gt_mesh = os.path.join(gt_mesh_path, f"{scan}.ply")

    command = [
        "python", "render.py",
        "-r", str(r_value),
        "--depth_ratio", str(depth_ratio),
        "--skip_test",
        "--skip_train",
        "--model_path", output_folder,
        "--gt_mesh", gt_mesh
    ]

    print(f"\nRendering {scan}...")
    subprocess.run(command)

print("Rendering completed")