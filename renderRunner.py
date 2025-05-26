import subprocess
import os
from tqdm import tqdm


output_path = 'output/date'
gt_mesh_path = 'dataset/DTU_GT/mesh'
depth_ratio = 1
r_value = 2

scans = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106",
    "scan110", "scan114", "scan118", "scan122"
]
test = ["scan63"]

for scan in tqdm(test, desc="Rendering DTU dataset"):
    output_folder = os.path.join(output_path, scan)
    gt_mesh = os.path.join(gt_mesh_path, f"{scan}.ply")

    # command = [
    #     "python", "render.py",
    #     "-r", str(r_value),
    #     "--depth_ratio", str(depth_ratio),
    #     "--skip_test",
    #     "--skip_train",
    #     "--model_path", output_folder,
    #     "--gt_mesh", gt_mesh
    # ]

    command = [
        "python", "render.py",
        "--model_path", output_folder,
        "-r", str(r_value),
        "--depth_ratio", str(depth_ratio),
        "--load_quant"
    ]

    print(f"\nRendering {scan}...")
    subprocess.run(command)

print("Rendering completed")