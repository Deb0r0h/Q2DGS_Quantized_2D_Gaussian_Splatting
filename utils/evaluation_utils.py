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
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
import numpy as np
import math
from utils.render_utils import save_img_f32, save_img_u8

# TODO manca gen_rays_at
#from utils.graphics_utils import gen_rays_at
from functools import partial
from scipy.spatial import KDTree
import trimesh
import open3d as o3d


# Function to compute chamfer,f1,rmse,hausdorff metrics
def compute_metrics(original_points, rendered_points, threshold = 0.5):
    distance_original_rendered, _ = KDTree(original_points).query(rendered_points, k=1, p=2)
    distance_rendered_original, _ = KDTree(rendered_points).query(original_points, k=1, p=2)

    # CHAMFER
    chamfer = (np.mean(distance_rendered_original) + np.mean(distance_original_rendered)) / 2

    # F1
    precision = np.mean(distance_original_rendered < threshold)
    recall = np.mean(distance_rendered_original < threshold)
    f1_score = 2 * ((precision*recall)/(precision+recall))

    # RMSE
    rmse = np.sqrt(np.mean(np.sum((original_points-rendered_points)**2, axis=1)))

    # HAUSDORFF
    haus = max(np.max(distance_original_rendered), np.max(distance_rendered_original))


    return chamfer,f1_score,rmse,haus

# Function that returns the visible points from a collection of scene and a mesh
def find_visible_points(viewpoints, mesh):
    points = []
    for viewpoint in viewpoints:
        rays_origin, rays_direction = gen_rays_at(viewpoint)
        rays_origin, rays_direction = rays_origin.cpu().detach().numpy(), rays_direction.cpu().detach().numpy()
        rays_origin = rays_origin[None, ...].repeat(rays_direction.shape[0], axis=0)
        rays_direction = rays_direction /np.linalg.norm(rays_direction, axis=1, keepdims=True)
        locations, index_ray, index_triangle = mesh.ray.intersects_location(ray_origins = rays_origin, ray_directions = rays_direction, multiple_hits = False)
        points.append(locations)

        return np.concatenate(points, axis = 0)

# Function to compare the GT with the rendered one
def compareWithGT(gt_mesh,mesh_post,scene):
    vertices = np.asarray(mesh_post.vertices).tolist()
    triangles = np.asarray(mesh_post.triangles).tolist()

    original_mesh = trimesh.load_mesh(gt_mesh)
    #Matteo : rendered_mesh = trimesh.Trimesh(vertices=vertices, triangles=triangles)
    rendered_mesh = trimesh.load_mesh(mesh_post)

    original_points = find_visible_points(scene.getTrainCameras(),original_mesh)
    rendered_points = find_visible_points(scene.getTrainCameras(),rendered_mesh)

    chamfer,f1,rmse,haus = compute_metrics(original_points, rendered_points, threshold = 0.5)

    return chamfer,f1,rmse,haus