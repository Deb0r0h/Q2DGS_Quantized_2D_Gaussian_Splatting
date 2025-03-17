#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from os.path import join
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import torch.nn.functional as F
from bitarray import bitarray #QUANTIZATION
from collections import OrderedDict

class GaussianModel:

    #Build the covariance using the matrix R and S as in the 3D Gaussian (𝚺 = R S S^⊤ R^⊤) 
    #The matrix contains rotation and scaling in the superior part of the matrix (3x3) and the traslation (center) in the fourth row
    #Define functions and activation as instance attribute -> make the model configurabile and dynamics
    # TEST PER PUSH
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1) # mi mette 1 per farla homo e scambia gli assi con permute in modo da portarla come std
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    #Constructor, initializa all as empty or zero
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # QUANTIZATION
        self._xyz_q = torch.empty(0)
        self._features_dc_q = torch.empty(0)
        self._features_rest_q = torch.empty(0)
        self._scaling_q = torch.empty(0)
        self._rotation_q = torch.empty(0)

        ### LOD PART ###
        self.lod_thres = None
        self.lod_reduction_factors = None
        ### LOD PART ###


        ### GEOMETRY PART ###
        self.last_depth = None
        self.last_normal = None
        ### GEOMETRY PART ###

        self.setup_functions()

    #Snapshote of the model state: actual state of the model, useful to save the state of the model for comparison etc
    #Return a tupla with all the values defined in __init__
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,

            self.last_depth,
            self.last_normal,
        )
    
    #Restore the state of the model with the values model_args (variabili interne)
    #Call a training function
    #Restore gradient and normalization values
    #Load the state of the optmizer, which has weights and info from previous optimization
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    #Functions to get the scaling, rotation, xyz, features, opacity obtained from the activation variable defined in setup_functions()
    # added QUANTIZATION
    # -----------------------------------------------------------------
    @property
    def get_scaling(self):
        if len(self._scaling_q) == 0:
            scaling = self._scaling
        else:
            scaling = self._scaling_q
        return self.scaling_activation(scaling) #.clamp(max=1)

    @property
    def get_scaling_no_quantize(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if len(self._rotation_q) == 0:
            rotation = self._rotation
        else:
            rotation = self._rotation_q
        return self.rotation_activation(rotation)
    
    @property
    def get_xyz(self):
        if len(self._xyz_q) == 0:
            xyz = self._xyz
        else:
            xyz = self._xyz_q
        return xyz
    
    @property
    def get_features(self):
        if len(self._features_dc_q) == 0:
            features_dc = self._features_dc
        else:
            features_dc = self._features_dc_q
        if len(self._features_rest_q) == 0:
            features_rest = self._features_rest
        else:
            features_rest = self._features_rest_q
        return torch.cat((features_dc, features_rest), dim=1)

    # NO QUANTIZATION APPLIED
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    # -----------------------------------------------------------------

    # Compute the covariance (get it) calling the variable defined in setup_functions()
    # It pass also the scaling modifier (=1 as base)
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    # Function that increases the degree of SH by 1 if the degree is less than the maximum degree
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        # Il grado SH indica la complessità o il livello di dettaglio delle armoniche sferiche utilizzate.
        # Le armoniche sferiche sono una famiglia di funzioni che servono a rappresentare funzioni definite sulla superficie di una sfera

    # Initialize the model using a PointCloud from which obtain a vector of features, used as input data
    # The input data, the ones initialized in __init__ here are assigned, these parameters are tensor -> during backpropagation the model can change/update them
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # conversion to PyTorch tensor + pass to GPU. Contains x,y,z of each points
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # same but with the colors but representation throw SH
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # tensor for features with size (number of points, 3 [RGB], total number of SH coeff) NB: **2 = ^2
        # Insert the color of each point only in the first channel [0] & set the others as zeros
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) #distance from points in the cloud, set also a lower-bound
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2) #scale for each point, anisotropic scale
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # set the initial opacity as 0.1 for each point

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    # Configure the training process, assign the requiered parameters (ex: gradient,denominator,norm) and 
    # the setting for the optimization process + rules on how update the param (xyz,features,colors,opacity,R,S) during training
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # dim = number of points, used to accumulate gradients of spatial coordinates
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # idem but for the normalization

        # Parameters for the optimizer: each param has its own learning rate
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # Adam optimizer, to update the model parameters. The global learing rate is 0 but it is actually managed by the rate in the list l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        #Control how the learning rate change for the spatial coordinates xyz during training
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # Modify dinamically the learing rate during training based on the current iteration aka "learning rate scheduling"
    # Works only for the xyz parameters
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # Create a list of attributes that describes the model components: the xyz coordinates, the nx,ny,nz normals, features SH (at high and low frequency)
    # the opacity, scales, rotations
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # A DIFFERENZA DELL'ALTRA PASSA UNA LISTA
    # OCIO A "NORMALS"
    def construct_list_of_attributes_q(self, save_att=None):
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l = []
        if 'xyz' in save_att:
            l += ['x', 'y', 'z']
        if 'normals' in save_att:
            l += ['nx', 'ny', 'nz']
        # All channels except the 3 DC
        if 'f_dc' in save_att:
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
        if 'f_rest' in save_att:
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        if 'scale' in save_att:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
        if 'rotation' in save_att:
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
        return l

    # Save all the model data (the data in the list of previous method) into a PLY format, that will be used to represent the point cloud
    # TODO ho messo una flag per la quntizzazione
    def save_ply(self, path, save_q=[], save_attributes=None):
        mkdir_p(os.path.dirname(path))

        # detach() is used to separate a tensor from the computational graph by returning a new tensor that doesn’t require a gradient
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        quantization = True
        if quantization:
            if 'pos' in save_q:
                xyz = self._xyz_q.detach().cpu().numpy()
            if 'dc' in save_q or 'sh_dc' in save_q:
                f_dc = self._features_dc_q.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            if 'sh' in save_q or 'sh_dc' in save_q:
                f_rest = self._features_rest_q.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            if 'scale' in save_q or 'scale_rot' in save_q:
                scale = self._scaling_q.detach().cpu().numpy()
            if 'rot' in save_q or 'scale_rot' in save_q:
                rotation = self._rotation_q.detach().cpu().numpy()

            all_attributes = {'xyz': xyz, 'f_dc': f_dc, 'f_rest': f_rest, 'opacities': opacities,
                              'scale': scale, 'rotation': rotation}
            if save_attributes is None:
                save_attributes = list(all_attributes.keys())
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_q(save_attributes)]
            #print('non-quantized attributes: ', save_attributes)
            #print('quantized attributes: ', save_q)

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate(tuple([val for (key, val) in all_attributes.items() if key in save_attributes]),axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
        # ---------- fine quantization -------

        else:
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)

    # Reduce the opacity: set the opacity to 0.01 if is higher, otherwise nothing change
    # And update the new opacity to the optimizer
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # TODO al 95% da togliere
    def reset_opacity_geo(self):
        # Preserve opacity in geometrically complex regions
        if hasattr(self, 'last_depth'):
            depth_grad = torch.abs(torch.gradient(self.last_depth, dim=(1,2))[0]).mean(dim=0)
            preserve_mask = depth_grad > 0.1
            preserved_opacity = self.get_opacity[preserve_mask]
        
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        
        if hasattr(self, 'last_depth'):
            opacities_new[preserve_mask] = preserved_opacity
            
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # Convert binary b to decimal integer, code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    # Used in load_ply() (quant)
    def bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, torch.int64)
        return torch.sum(mask * b, -1)

    # Load the point cloud in PLY format and obtain the model parameters: xyz, SH (at high and law frequency), opacity, scales, rotations
    # After the loading move the parameter to tensor and make them optimizable throw to requires_grad_(True)
    def load_ply_original(self, path):
        plydata = PlyData.read(path)

        # COORDINATES
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # OPACITIES
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # SH LOW FREQUENCY
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # SH HIGH FREQUENCY
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # SCALES
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # ROTATIONS
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # CONVERSION FROM NUMPY TO PYTORCH TENSOR -> this tensors can be optimize during training
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # SET THE SH DEGREE MAX
        self.active_sh_degree = self.max_sh_degree

    # TODO questa in realtà è quella nuovo
    def load_ply(self, path, load_quant = False):
        plydata = PlyData.read(path)
        quant_params = []

        # Load indices and codebook for quantized params
        if load_quant:
            base_path = '/'.join(path.split('/')[:-1])
            inds_file = join(base_path, 'kmeans_inds.bin')
            codebook_file = join(base_path, 'kmeans_centers.pth')
            args_file = join(base_path, 'kmeans_args.npy')
            codebook = torch.load(codebook_file)
            args_dict = np.load(args_file, allow_pickle=True).item()
            quant_params = args_dict['params']
            loaded_bitarray = bitarray()
            with open(inds_file, 'rb') as file:
                loaded_bitarray.fromfile(file)
            # bitarray pads 0s if array is not divisible by 8. ignore extra 0s at end when loading
            total_len = args_dict['total_len']
            loaded_bitarray = loaded_bitarray[:total_len].tolist()
            indices = np.reshape(loaded_bitarray, (-1, args_dict['n_bits']))
            indices = self.bin2dec(torch.from_numpy(indices), args_dict['n_bits'])
            indices = np.reshape(indices.cpu().numpy(), (len(quant_params), -1))
            indices_dict = OrderedDict()
            for i, key in enumerate(args_dict['params']):
                indices_dict[key] = indices[i]

        if 'xyz' in quant_params:
            xyz = np.expand_dims(codebook['xyz'][indices_dict['xyz']].cpu().numpy(), -1)
        else:
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])), axis=1)
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if 'dc' in quant_params:
            features_dc = np.expand_dims(codebook['dc'][indices_dict['dc']].cpu().numpy(), -1)
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if 'sh' in quant_params:
            features_extra = codebook['sh'][indices_dict['sh']].cpu().numpy()
            features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))
            features_extra = features_extra.transpose((0, 2, 1))
        else:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        if 'scale' in quant_params:
            scales = codebook['scale'][indices_dict['scale']].cpu().numpy()
        else:
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if 'rot' in quant_params:
            rots = codebook['rot'][indices_dict['rot']].cpu().numpy()
        else:
            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # Replace a tensor, used when a tensor is update or modified and we want to train the model with the new tensor
    # but keeping the state of the optimizer for that parameter (the param_group are as usually xyz, SH, opacity, ...)
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # Recover the internal state of the parameter & (since it is replacing the tensor) re-initialize = 0 the state "exp_avg" "exp_avg_sq" for the new tensor
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # Delete of the old and replace with the newest
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # Reduce the parameters of the model and the optimizer state according to a given mask
    # The mask allows to mantain only the selected parameters, discarding those not necessary
    # Ex: remove not-used or uneffective nodes during training
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)

            # if the state "exist" -> prune
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # creation of new tensor that contains only the parameters selected by the mask
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            
            # if the state "does not exist" -> proceeds in the same way but without assigning states
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # Applies a mask to remove unnecessary points from the model and the optimization process after pruning, 
    # only “valid” points are retained and the optimizer and model tensors are updated accordingly
    # * Used in densify_and_split() & densify_and_prune()
    def prune_points(self, mask):
        valid_points_mask = ~mask # since  ~mask -> valid_points_mask is True when the point must be manteined, False if must be removed
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # call the previous function that remove the parameters from the model

        # Update the model parameters after the pruning: the original tensors (containing all points) are replaced with the reduced ones,
        # which contain only the valid points
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Pruning of "new" variable that depends from model's points as gradient and radius
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # Opposite operation to pruning: add new tensors to existing parameters in the model and properly update the optimizer, 
    # it is used to expand the model by adding new parameters to be optimized, without losing the information already accumulated by the optimizer for the existing parameters
    # * Used in densification_postfix()
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                
                # cat() = concatenation (first concatenate the state then the tensor, since i put the state into the tensor)
                # NB: Concatenation of tensors of zeros initializes new parameters without affecting existing parameters. 
                # New moment and variance values start from zero and will be updated as the model trains
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            
            # If there is no state associated with the parameter, it simply concatenates the original tensor with the new tensor and saves it
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # Add new point to the model (densification) and update the optimizer state to handle this new points
    # * Used in densify_and_split() & densify_and_clone()
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # previous function: add + concatenation
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Re-initialize to fit the new point density in the model
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # Generates new points from an existing set of points, selecting them based on a gradient threshold and then adding them to the current representation
    # Filters and removes points that do not meet certain conditions
    # * Used in densify_and_prune
    # TODO la parte di sampling è leggermente diversa (nel quant = 3DGS)
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # select the point with gradient >= threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling_no_quantize, dim=1).values > self.percent_dense*scene_extent) # selected points must have sufficient scale

        # N = number of sample that will be created for each selected point

        # Sampling of new points around the selected ones
        stds = self.get_scaling_no_quantize[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        
        # New points are created by applying rotation, position, scaling, SH, opacity on sampled points 
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling_no_quantize[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # Update the optimizer using the previous function
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        # Removal of items that do not meet the criteria, including those newly added if they do not meet the conditions
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # Select a subset of points based on a gradient and density criterion and then CLONE them
    # The function follows a logic similar to densify_and_split but without the sampling
    # * Used in densify_and_prune 
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling_no_quantize, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # Add new points by cloning existing points and/or by sampling new points & remove points with low opacity
    # keeping only "usefull points"
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):

        # Calculation of normalized gradients with respect to an accumulated quantity (denom = used to normalize)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Mask to identify the points that will be pruned based on the opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # If a specific screen size is given, the mask is updated to include points that are "too big"
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_no_quantize.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    # Accumulates gradient information for selected points through a filter. This let to know how “significant” the changes in gradients are for each point
    # It updates both the accumulated total of gradients and the number of updates so that gradients can be normalized later
    # Useful for monitoring the points undergoing the greatest changes in the optimization process, to decide which points to densify or prune
    # TODO anche qua differisce dal quant = 3DGS -> .grad[update_flter,:2]
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # immondizia
    def add_densification_stats_grad(self, viewspace_point_tensor, update_filter, depth_grad = None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)

        if depth_grad is not None:
            with torch.no_grad():
                gaussian_depth_grad = depth_grad.mean(dim=(-1,-2))
                self.depth_gradient_accum[update_filter] += gaussian_depth_grad.unsqueeze(-1)
        self.denom[update_filter] += 1

    # TODO non è presente nè in 3DGS che in 2DGS, serve per opacity_regulation
    # copy the prune elements in densify_and_prune
    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling_nq.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    # ---------- fine base + quant ------------








    ### LOD PART ###
    # per ora non va però ci si può lavorare
    def compute_screen_size(self,viewpoint_camera):
        h_pos = torch.cat([self._xyz, torch.ones_like(self._xyz[:,:1])], dim=1)
        view_pos = (h_pos @ viewpoint_camera.world_view_transform.T)
        depth = view_pos[:,2].clamp(min=0.01)
        focal_lenght = viewpoint_camera.image_width / (2 * math.tan(viewpoint_camera.FoVx / 2))
        scale_magnitude = torch.norm(self.get_scaling, dim=1)
        size = (scale_magnitude * focal_lenght) / depth
        return size

        """
        screen_pos = (h_pos @ projection_matrix.T)
        screen_pos = screen_pos[:,:3] / screen_pos[:,3:4]
        scaling = self.get_scaling
        return torch.norm(scaling,dim=1) / screen_pos[:,2]"""
    def apply_LOD(self,viewpoint_camera,thresholds, reduction_factors):
        screen_size = self.compute_screen_size(viewpoint_camera)
        mask = torch.ones_like(screen_size,dtype=torch.bool)
        for lod_threshold,lod_reduction_factor in zip(thresholds,reduction_factors):
            mask_lv = screen_size < lod_threshold
            subsample_mask = torch.rand_like(screen_size) < lod_reduction_factor
            mask &= ~(mask_lv & subsample_mask)
        self.prune_points(~mask)
    ### LOD PART ###


    #  NEW VERSION FOR GEOMETRY LOSS DI D&P
    def densify_and_prune_geometry(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if hasattr(self, 'last_depth') and hasattr(self, 'last_normal'):
            depth_grad = torch.abs(torch.gradient(self.last_depth, dim=(1, 2))[0]).mean(dim=0)
            normal_diff = 1 - (self.last_normal[:, 1:] * self.last_normal[:, :-1]).sum(dim=0)
            target_shape = normal_diff.shape
            depth_grad_resized = F.interpolate(depth_grad.unsqueeze(0).unsqueeze(0), target_shape, mode='bilinear', align_corners=False).squeeze()
            geo_complexity = (depth_grad_resized + normal_diff).clamp(0, 1)
            geo_weights = geo_complexity[self.visibility_filter.cpu().numpy()]
            weighted_grads = grads * (1 + 3 * torch.tensor(geo_weights, device="cuda"))
        else:
            weighted_grads = grads

        self.densify_and_clone(weighted_grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()