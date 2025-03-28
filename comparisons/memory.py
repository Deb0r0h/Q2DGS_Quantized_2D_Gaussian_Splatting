import os
import csv
import shutil


def get_ply_size(file_path, file):
    try:
        file_size = os.path.getsize(file_path + file)
        file_mb = file_size / (1024 * 1024)
        return file_mb
    except FileNotFoundError:
        return ("File not found")

def compute_comparison_file(file_path_base, file_path_light, file_list_base, file_list_light, output_file, type):
    try:
        if type == "txt":
            with open(output_file, "w") as f:
                f.write("BASE VERSION\tLIGHT VERSION\n")
                for base, light in zip(file_list_base, file_list_light):
                    base_size = get_ply_size(file_path_base, base)
                    light_size = get_ply_size(file_path_light, light)
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

def get_ply_files_from_output_folder():
    file_path = "../output/date"
    output_folder = "modified/"
    for scan_folder in os.listdir(file_path):
        scan_number = scan_folder.replace('scan','',1)
        source_file = os.path.join(file_path, scan_folder,"train","ours_30000","fuse_post.ply")
        destination_file = os.path.join(output_folder, f"{scan_number}.ply")
        shutil.copy2(source_file, destination_file)

def get_list_items(file_path):
    list = []
    for file in os.listdir(file_path):
        list.append(file)
    list = sorted(list, key=lambda x : int(x.split(".")[0]))
    return list


file_path_base = "base/"
file_path_light = "modified/"
file_list_base = get_list_items(file_path_base)
file_list_light = get_list_items(file_path_light)


compute_comparison_file(file_path_base, file_path_light, file_list_base, file_list_light, "size_comparison.txt", "txt")
