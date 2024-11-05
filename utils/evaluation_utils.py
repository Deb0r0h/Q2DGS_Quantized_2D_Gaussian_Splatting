import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
import trimesh
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh, find_visible_points, chamfer_distance_and_f1_score
from utils.render_utils import generate_path, create_videos
import numpy as np
import math
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
from scipy.spatial import KDTree
import open3d as o3d



def compare_with_GT(gt_mesh, post_mesh,scene):
    vert = np.asarray(post_mesh.vertices).tolist()
    trian = np.asarray(post_mesh.triangles).tolist()

    rendered_mesh = trimesh.Trimesh(vertices=vert, faces=trian)
    original_mesh = trimesh.load_mesh(gt_mesh)

    gt_points = find_visible_points(scene.getTrainCameras(),original_mesh)
    post_points = find_visible_points(scene.getTrainCameras(), rendered_mesh)

    pcl = trimesh.points.PointCloud(gt_points)

    chamfer_dist, f1_score = chamfer_distance_and_f1_score(gt_points,post_points,f_threshold=0.5)

    return chamfer_dist,f1_score,pcl