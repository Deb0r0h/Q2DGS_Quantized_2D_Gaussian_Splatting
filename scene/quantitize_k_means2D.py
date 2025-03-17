# Part of the code comes from a re-adaptation of the wonderful
# work done in Compact3D, source https://github.com/UCDvision/compact3d/tree/main

import numpy as np
import os
import pdb
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn


class Quantize_kMeans():

    def __init__(self, num_clusters = 100, num_iters = 10):
        self.num_clusters = num_clusters
        self.num_k_iterations = num_iters
        self.nn_index = torch.empty(0)
        self.centers = torch.empty(0)
        self.vec_dim = 0
        self.clusters_ids= torch.empty(0)
        self.cls_ids = torch.empty(0)
        self.excl_clusters = []
        self.excl_cluster_ids = []
        self.cluster_len = torch.empty(0)
        self.max_cnt = 0
        self.n_excl_cls = 0

    # Compute the distance between all the vectors in x and y
    # x = (m,dim); y = (n,dim); dist = (m,n)
    # Chunk mode divides x into blocks to avoid memory problems when x is large: calculates distances and combines all blocks into a single matrix
    def get_distance(self, x, y, mode='euclidean'):
        if mode == 'euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0] # unsqueeze is necessary to use cdist, requieres batch size 1 at beg, [0] removes it
        elif mode == 'euclidean_chunk':
            dist = []
            max_block = 65536
            if x.shape[0] < max_block:
                max_block = x.shape[0] # y will be lower than x so chunk is not useful (y is the centroid)
            for i in range(np.ceil(x.shape[0] / max_block).astype(int)):
                dist.append(torch.cdist(x[(i * max_block): (i + 1) * max_block, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_distance':
            dist = torch.cdist(x,y,p=2)
        return dist

    # Update the clusters centroids
    def update_centers(self, features):
        features = features.detach().reshape(-1, self.vec_dim)
        features = torch.cat([features, torch.zeros_like(features[:1]).cuda()], 0) # to avoid indexing errors with .cluster_ids (below)
        self.centers = torch.sum(features[self.cluster_ids, :].reshape(self.num_clusters, self.max_cnt, -1), dim=1)
        # Some clusters can be excluded during the update
        if len(self.excl_cluster_ids) > 0:
            for i, cls in enumerate(self.excl_clusters):
                self.centers[cls] += torch.sum(features[self.excl_cluster_ids[i], :], dim=0) # Only the extra elements in the bigger clusters are added here.v
        self.centers /= (self.cluster_len + 1e-6) # Division by num_points in cluster is done during the one-shot averaging of all

    # Update the clusters centroids using mask matrix multiplication from the distance matrix
    # (ho tolto nn_index=None dai parametri perchè non usato)
    def update_centers_with_mask(self, features, cluster_mask=None, nn_index = None ,avg=False):
        features = features.detach().reshape(-1, self.vec_dim)
        centers = (cluster_mask.T @ features)
        if avg:
            self.centers /= counts.unsqueeze(-1)
        return centers

    # Equalize the cluster size adding some not relevant elements based on the size of the max cluster
    def equalize_cluster_size(self):
        unq, n_unq = torch.unique(self.nn_index, return_counts=True) # list of the index + counter of how a index is present
        topk = 100
        if len(n_unq) < topk:
            topk = len(n_unq)
        max_cnt_topk, topk_idx = torch.topk(n_unq, topk) # descendent order
        self.max_cnt = max_cnt_topk[0]

        # Exlude cluster over a threshold
        idx = 0
        self.excl_clusters = []
        self.excl_cluster_ids = []
        threshold = 5000
        while (self.max_cnt > threshold):
            self.excl_clusters.append(unq[topk_idx[idx]])
            idx += 1
            if idx < topk:
                self.max_cnt = max_cnt_topk[idx]
            else:
                break
        self.n_excl_cls = len(self.excl_clusters)
        self.excl_clusters = sorted(self.excl_clusters)

        # Creation of the cluster index list
        all_ids = []
        cls_len = []
        for i in range(self.num_clusters):
            cur_cluster_ids = torch.where(self.nn_index == i)[0]
            cls_len.append(torch.Tensor([len(cur_cluster_ids)]))
            # For excluded clusters, use only the first max_cnt elements for averaging along with other clusters
            if i in self.excl_clusters:
                self.excl_cluster_ids.append(cur_cluster_ids[self.max_cnt:])
                cur_cluster_ids = cur_cluster_ids[:self.max_cnt]
            # Equalize the size
            all_ids.append(torch.cat([cur_cluster_ids, -1 * torch.ones((self.max_cnt - len(cur_cluster_ids)), dtype=torch.long).cuda()]))
        all_ids = torch.cat(all_ids).type(torch.long)
        cls_len = torch.cat(cls_len).type(torch.long)
        self.cluster_ids = all_ids
        self.cluster_len = cls_len.unsqueeze(1).cuda()
        self.cls_ids = self.nn_index

    # Assign each element to a cluster, they will have all the same size thanks to the previous function
    # Basically performs K-means
    def cluster_assign(self, feat, feat_scaled=None):
        feat = feat.detach()
        feat = feat.reshape(-1, self.vec_dim)
        if feat_scaled is None:
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8)
        if len(self.centers) == 0:
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
        centers = torch.zeros_like(self.centers)

        # chunk is used for memory issues
        chunk = True
        for iteration in range(self.num_k_iterations):
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                while True:
                    dist = self.get_distance(feat[i * chunk:(i + 1) * chunk, :], self.centers)
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32) # Assign a single cluster when distance to multiple clusters is same
                    curr_centers = self.update_centers_with_mask(feat[i * chunk:(i + 1) * chunk, :], dist, curr_nn_index, avg=False)
                    counts += dist.detach().sum(0) + 1e-6
                    centers += curr_centers
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i * chunk > feat.shape[0]:
                        break

            self.centers = centers / counts.unsqueeze(-1)
            centers[centers != 0] = 0.
            counts[counts > 0.1] = 0.

        if chunk:
            self.nn_index = None
            i = 0
            while True:
                dist = self.get_distance(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                curr_nn_index = torch.argmin(dist, dim=-1)
                if self.nn_index == None:
                    self.nn_index = curr_nn_index
                else:
                    self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                i += 1
                if i * chunk > feat.shape[0]:
                    break
        self.equalize_cluster_size()

    # Rescale the features in [-1,+1]
    def rescale(self, feat, scale=None):
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)



    # Forward pass for each parameter of the gaussians: position, scale. rotation. translation. sh, dc ...  + combinations (experiments)


    def forward_position(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._xyz.shape[1] # OCCHIO A QUESTO VALORE [1] era 3dgs
        if assign:
            self.cluster_assign(gaussian._xyz)
        else:
            self.update_centers(gaussian._xyz)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._xyz_q = gaussian._xyz - gaussian._xyz.detach() + sampled_centers

    def forward_scale(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._scaling.shape[1]
        if assign:
            self.cluster_assign(gaussian._scaling)
        else:
            self.update_centers(gaussian._scaling)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._scaling_q = gaussian._scaling - gaussian._scaling.detach() + sampled_centers

    def forward_rotation(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1]
        if assign:
            self.cluster_assign(gaussian._rotation)
        else:
            self.update_centers(gaussian._rotation)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._rotation_q = gaussian._rotation - gaussian._rotation.detach() + sampled_centers

    def forward_dc(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2]
        if assign:
            self.cluster_assign(gaussian._features_dc)
        else:
            self.update_centers(gaussian._features_dc)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_dc_q = gaussian._features_dc - gaussian._features_dc.detach() + sampled_centers.reshape(-1,1,3) # OCCHIO era (-1,1,3)

    def forward_features_rest(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2]
        if assign:
            self.cluster_assign(gaussian._features_rest)
        else:
            self.update_centers(gaussian._features_rest)
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest_q = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)

    def forward_scale_rotation(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1] + gaussian._scaling.shape[1]
        feat_scaled = torch.cat([self.rescale(gaussian._scaling), self.rescale(gaussian._rotation)], 1)
        feat = torch.cat([gaussian._scaling, gaussian._rotation], 1)
        if assign:
            self.cluster_assign(feat, feat_scaled)
        else:
            self.update_centers(feat)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._scaling_q = gaussian._scaling - gaussian._scaling.detach() + sampled_centers[:, :3] # prima era 3
        gaussian._rotation_q = gaussian._rotation - gaussian._rotation.detach() + sampled_centers[:, 3:] # prima era 3

    def forward_dc_features_rest(self, gaussian, assign = False):
        if self.vec_dim == 0:
            self.vec_dim = (gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2] + gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2])
        if assign:
            self.cluster_assign(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        else:
            self.update_centers(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_dc_q = gaussian._features_dc - gaussian._features_dc.detach() + sampled_centers[:,:3].reshape(-1, 1, 3)
        gaussian._features_rest_q = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers[:,3:].reshape(-1, deg, 3)

    def replace_with_centers(self, gaussian):
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)



