import os
import pandas as pd
import yaml
import csv
import shutil


def compute_ply_size(file_path, file):
    try:
        file_size = os.path.getsize(file_path + file)
        file_mb = file_size / (1024 * 1024)
        return file_mb
    except FileNotFoundError:
        return ("File not found")

def compute(file_path_base, file_path_light, file_list_base, file_list_light, output_file, type):
    try:
        if type == "txt":
            with open(output_file, "w") as f:
                f.write("BASE VERSION\tLIGHT VERSION\n")
                for base, light in zip(file_list_base, file_list_light):
                    base_size = compute_ply_size(file_path_base, base)
                    light_size = compute_ply_size(file_path_light, light)
                    base_size_str = f"{base} {base_size:.2f} MB" if base_size else "Not found"
                    light_size_str = f"{light} {light_size:.2f} MB" if light_size else "Not found"
                    f.write(f"{base_size_str}\t{light_size_str}\n")

        elif type == "csv":
            with open(output_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["BASE VERSION", "MB", "LIGHT VERSION", "MB"])
                for base, light in zip(file_list_base, file_list_light):
                    base_size = compute_ply_size(file_path_base, base)
                    light_size = compute_ply_size(file_path_light, light)
                    base_size_str = f"{base_size:.2f}" if base_size else "Not found"
                    light_size_str = f"{light_size:.2f}" if light_size else "Not found"
                    writer.writerow([base, base_size_str, light, light_size_str])

    except FileNotFoundError:
        return ("File not found")

def temp():
    file_path = "../output/date"
    data = []
    for scan_folder in os.listdir(file_path):
        for file in os.listdir(os.path.join(file_path, scan_folder,"train","ours_30000")):
            name = "fuse_post.ply"
            if name in file:
                size = os.path.getsize(os.path.join(file_path, scan_folder,"train","ours_30000",file))
                size = size/(1024*1024)
                data.append(size)
    print(data)

def bo():
    file_path = "../output/date"
    output_folder = "modified/"
    for scan_folder in os.listdir(file_path):
        source_file = os.path.join(file_path, scan_folder,"train","ours_30000","fuse_post.ply")
        destination_file = os.path.join(output_folder, f"{scan_folder}.ply")
        shutil.copy2(source_file, destination_file)


file_path = "base/"
file_list = ["24.ply", "37.ply", "40.ply", "55.ply", "63.ply", "65.ply", "69.ply", "83.ply", "97.ply", "105.ply","106.ply", "110.ply", "114.ply", "118.ply", "122.ply"]

