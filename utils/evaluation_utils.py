from scene import Scene
import trimesh
from utils.mesh_utils import find_visible_points, compute_metrics
import numpy as np


def compare_with_GT(gt_mesh, post_mesh,scene):
    print("Computing metrics...")
    vert = np.asarray(post_mesh.vertices).tolist()
    trian = np.asarray(post_mesh.triangles).tolist()

    rendered_mesh = trimesh.Trimesh(vertices=vert, faces=trian)
    original_mesh = trimesh.load_mesh(gt_mesh)

    gt_points = find_visible_points(scene.getTrainCameras(),original_mesh)
    post_points = find_visible_points(scene.getTrainCameras(), rendered_mesh)

    pcl = trimesh.points.PointCloud(gt_points)

    chamfer_dist, f1_score = compute_metrics(gt_points,post_points,f_threshold=0.5)

    return chamfer_dist,f1_score,pcl

