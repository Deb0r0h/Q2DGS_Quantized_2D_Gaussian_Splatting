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
# Part of the code comes from a re-adaptation of the wonderful
# work done in Compact3D, source https://github.com/UCDvision/compact3d/tree/main

import numpy as np
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.quantitize_k_means2D import Quantize_kMeans
from bitarray import bitarray
from os.path import join

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# Manage the training cycles of the rendering model basing on Gaussian, parameters, checkpoint and the rendering pipeline
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # if it has a checkpoint -> loads and update iter
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Initialize the background color (black or white)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Used to measure the execution time at each iteration
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    number_of_gaussian_per_iter = []
    # quantized_params = args.quant_params
    # n_cls = args.kmeans_ncls
    # n_cls_sh = args.kmeans_ncls_sh
    # n_cls_dc = args.kmeans_ncls_dc
    # n_it = args.kmeans_iters
    # kmeans_st_iter = args.kmeans_st_iter
    # freq_cls_assn = args.kmeans_freq

    quantized_params = ['rot','scale','sh','dc']
    n_cls = 1024
    n_cls_sh = 128
    n_cls_dc = 1024
    n_it = 30
    kmeans_st_iter = 15000
    freq_cls_assn = 50


    if 'pos' in quantized_params:
        kmeans_pos_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'dc' in quantized_params:
        kmeans_dc_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'sh' in quantized_params:
        kmeans_sh_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)
    if 'scale' in quantized_params:
        kmeans_sc_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'rot' in quantized_params:
        kmeans_rot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'scale_rot' in quantized_params:
        kmeans_scrot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'sh_dc' in quantized_params:
        kmeans_shdc_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)


    # TRAINING CYCLE
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # Update the learning rate based on the current iteration
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # (come from 3D Gaussian splatting: we start by optimizing only the zero-order component and then introduce one band of the SH after every 1000 iterations until all 4 bands of SH are represente)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        # Above 3100 iterations the assignment frequency is 100, while above a higher threshold it is 5000
        # TODO fittare meglio questi parametri
        if iteration > 3500:
            freq_cls_assn = 500
            if iteration > (opt.iterations - 5000):
                freq_cls_assn = 50

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))



        # Quant parameters
        if iteration > kmeans_st_iter:
            if iteration % freq_cls_assn == 1:
                assign = True
            else:
                assign = False

            if 'pos' in quantized_params:
                kmeans_pos_q.forward_position(gaussians, assign=assign)
            if 'dc' in quantized_params:
                kmeans_dc_q.forward_dc(gaussians, assign=assign)
            if 'sh' in quantized_params:
                kmeans_sh_q.forward_features_rest(gaussians, assign=assign)
            if 'scale' in quantized_params:
                kmeans_sc_q.forward_scale(gaussians, assign=assign)
            if 'rot' in quantized_params:
                kmeans_rot_q.forward_rotation(gaussians, assign=assign)
            if 'scale_rot' in quantized_params:
                kmeans_scrot_q.forward_scale_rotation(gaussians, assign=assign)
            if 'sh_dc' in quantized_params:
                kmeans_shdc_q.forward_dc_features_rest(gaussians, assign=assign)


        # Dynamic res changing
        if opt.dynamic_resolution:
            if iteration < 10000:
                viewpoint_cam.change_resolution(2)  # low
            elif iteration < 20000:
                viewpoint_cam.change_resolution(1)  # med
            else:
                viewpoint_cam.change_resolution(0)  # normal


        # Rendering based on camera and Gaussian
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # It returns: the rendered image, the points in the viewspace, a visibility filter, the rays (radii) of each point
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Ground Truth
        gt_image = viewpoint_cam.original_image.cuda()

        # L1 loss between render and gt + SSIM (structural similarity index to improve visual quality)
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        # TODO immondizia
        ### GEOMETRY PART ###
        # Store geometric information
        gaussians.last_depth = render_pkg['surf_depth'].detach()
        gaussians.last_normal = render_pkg['surf_normal'].detach()
        # New geometric regularization loss
        depth_geo = torch.exp(-10 * torch.abs(render_pkg['surf_depth'] - render_pkg['render_depth_expected']))
        normal_geo = (render_pkg['surf_normal'] * render_pkg['rend_normal']).sum(dim=0)
        geometry_loss = 0.1 * (1 - depth_geo.mean()) + 0.1 * (1 - normal_geo.mean())
        ### GEOMETRY PART ###



        # Regularization:
        # - Normal consistency -> L𝑛 = ∑︁𝜔𝑖(1−n𝑖^T N) based on the difference between the rendered normals and the the ones from the surfaces. Used after 7000 iter
        # - Depth distortion -> L𝑑 = ∑︁𝜔𝑖𝜔𝑗|𝑧𝑖−𝑧𝑗| based on the distance between the rendered points. Used after 3000 iter
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        depth_grad = render_pkg["depth_gradient_magnitude"]


        # TODO aggiungere opacity loss


        # FINAL LOSS
        total_loss = loss + dist_loss + normal_loss

        # Compute the gradient
        total_loss.backward()

        iter_end.record()

        # Temporarily disable automatic gradient calculation
        with torch.no_grad():

            # PROGRESS BAR
            # Ituses an exponential moving average (EMA) to get a smoothed version of the metrics
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            # Every 10 iteration the values in the progress bar are updated
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # Metrics are recorded to display graphically during training
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))


            if (iteration in saving_iterations):

                print(args.model_path)
                # print(f'PSNR Train: {psnr_train}, PSNR Test: {psnr_test}')
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Save only the non-quantized parameters in ply file.
                all_attributes = {'xyz': 'xyz', 'dc': 'f_dc', 'sh': 'f_rest', 'opacities': 'opacities',
                                  'scale': 'scale', 'rot': 'rotation', 'pos': 'position'}
                save_attributes = [val for (key, val) in all_attributes.items() if key not in quantized_params]
                if iteration > kmeans_st_iter:
                    scene.save(iteration, save_q=quantized_params, save_attributes=save_attributes)

                    # Save indices and codebook for quantized parameters
                    kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'sh': kmeans_sh_q, 'dc': kmeans_dc_q}
                    #kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'scale_rot': kmeans_scrot_q, 'sh_dc': kmeans_shdc_q}
                    kmeans_list = []
                    for param in quantized_params:
                        kmeans_list.append(kmeans_dict[param])
                    out_dir = join(scene.model_path, 'point_cloud/iteration_%d' % iteration) # Model PATH output/date/scan40 + pointcloud/iterationXXXXX
                    save_kmeans(kmeans_list, quantized_params, out_dir)
                else:
                    scene.save(iteration, save_q=[])

                # prima c'erano solo loro
                #print("\n[ITER {}] Saving Gaussians".format(iteration))
                #scene.save(iteration)

            # DENSIFICATION: new Gaussian points are added to improve the representation of the model
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                new_policy = False
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and new_policy == False:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,size_threshold)  # add and remove points

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and new_policy == True:
                    gaussians.densify(opt.densify_grad_threshold, scene.cameras_extent)
                    # prune by contribution
                    xyz = gaussians.get_xyz
                    # load all cameras
                    top_k = 3
                    viewpoint_stack = scene.getTrainCameras().copy()
                    wc = torch.zeros((xyz.shape[0], top_k), dtype=xyz.dtype, device=xyz.device)
                    prev = torch.clone(wc)
                    for idx, viewpoint_cam in enumerate(viewpoint_stack):
                        with torch.no_grad():
                            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                            max_weight = render_pkg['max_weight']
                            wc[..., 0] = torch.maximum(wc[..., 0], max_weight)
                            for i in range(1, min(idx + 1, top_k)):
                                temp = torch.minimum(prev[..., i - 1], max_weight)
                                wc[..., i] = torch.maximum(wc[..., i], temp)
                            prev[...] = wc[...]
                    wc = wc[..., -1]
                    mask = wc < opt.weight_cull
                    gaussians.prune_points(mask)

                ### LOD ### immondizia
                if pipe.apply_LOD:
                    gaussians.apply_LOD(viewpoint_cam, pipe.lod_thres, pipe.lod_reduction_factors)
                ### LOD ###

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # TODO azione della opacity regolation

            # OPTIMIZATION STEP
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save the checkpoint if in checkpoint iterations
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # NEW
        number_of_gaussian_per_iter.append(gaussians.get_xyz.shape[0])
        if iteration == 29999:
            print("NUMBER OF GAUSSIAN AT THE FINAL ITERATION: ", gaussians._xyz.shape[0])
        np.save(f'{scene.model_path}/num_g_per_iters.npy', np.array(number_of_gaussian_per_iter))

        # TODO passaggio un po' diverso, è messo nella parte iniziale della funzione
        with torch.no_grad():

            # Establishes a connection with a network graphical user interface (GUI) to monitor training in real time
            # -> showing real-time rendering images
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                   0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None


# Convert decimal integer x to binary, code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
def dec2binary(x, n_bits=None):
    if n_bits is None:
        n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
    mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

# Save the codebook and indices of KMeans
def save_kmeans(kmeans_list, quantized_params, out_dir):
    # Convert to bitarray object to save compressed version saving as npy or pth will use 8bits per digit (or boolean) for the indices
    # Convert to binary, concat the indices for all params and save.
    bitarray_all = bitarray([])
    for kmeans in kmeans_list:
        n_bits = int(np.ceil(np.log2(len(kmeans.cls_ids))))
        assignments = dec2binary(kmeans.cls_ids, n_bits)
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        # bitarr = bitarray(list(map(bool, assignments.cpu().numpy().flatten())))
        bitarray_all.extend(bitarr)
    with open(join(out_dir, 'kmeans_inds.bin'), 'wb') as file:
        bitarray_all.tofile(file)

    # Save details needed for loading
    args_dict = {}
    args_dict['params'] = quantized_params
    args_dict['n_bits'] = n_bits
    args_dict['total_len'] = len(bitarray_all)
    np.save(join(out_dir, 'kmeans_args.npy'), args_dict)
    centers_dict = {param: kmeans.centers for (kmeans, param) in zip(kmeans_list, quantized_params)}

    # Save codebook
    torch.save(centers_dict, join(out_dir, 'kmeans_centers.pth'))

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# TODO è un tot diversa, anche da 3DGS
@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                                 rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                                 surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                                 rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                                 rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()






if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")