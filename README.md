# Light 2D Gaussian Splatting (L-2DGS)



![Teaser image](assets/teaser.jpg)

This repo contains the implementation for the paper 2DGS ([source code](https://github.com/hbb1/2d-gaussian-splatting)) with a vector quantization, that with the contribution of a new regularization term redece the number of gaussians and the memory storage, taking inspiration from the work presents in [Compact3D](https://github.com/UCDvision/compact3d). This work is part of my master thesis


## Installation

```bash
# download
git clone https://github.com/Deb0r0h/Light_2DGS.git --recursive
conda env create --file environment.yml
conda activate surfel_splatting
```
## Training
To train a scene, simply use
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
# or
python trainRunner.py
```
Commandline arguments for regularizations
```bash
--lambda_normal  # hyperparameter for normal consistency
--lambda_distortion # hyperparameter for depth distortion
--depth_ratio # 0 for mean depth and 1 for median depth, 0 works for most cases
--opacity_regularization # hyperparameter for opacity regularization
```
**Tips for adjusting the parameters on your own dataset:**
- For unbounded/large scenes, we suggest using mean depth, i.e., ``depth_ratio=0``,  for less "disk-aliasing" artifacts.

## Rendering
### Bounded Mesh Extraction
To export a mesh within a bounded volume, simply use
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
# or
python renderRunner.py
```
Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion, use
```bash
--depth_ratio # 0 for mean depth and 1 for median depth
--voxel_size # voxel size
--depth_trunc # depth truncation
--load_quantization # to apply quantization on render
```
If these arguments are not specified, the script will automatically estimate them using the camera information.
### Unbounded Mesh Extraction
To export a mesh with an arbitrary size, we devised an unbounded TSDF fusion with space contraction and adaptive truncation.
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> --mesh_res 1024
```

### Quick Examples
Assuming you have downloaded [MipNeRF360](https://jonbarron.info/mipnerf360/), simply use
```bash
python train.py -s <path to m360>/<garden> -m output/m360/garden
# use our unbounded mesh extraction!!
python render.py -s <path to m360>/<garden> -m output/m360/garden --unbounded --skip_test --skip_train --mesh_res 1024
# or use the bounded mesh extraction if you focus on foreground
python render.py -s <path to m360>/<garden> -m output/m360/garden --skip_test --skip_train --mesh_res 1024
```
If you have downloaded the [DTU dataset](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), you can use
```bash
python train.py -s <path to dtu>/<scan105> -m output/date/scan105 -r 2 --depth_ratio 1
python render.py -r 2 --depth_ratio 1 --skip_test --skip_train
```
**Custom Dataset**: We use the same COLMAP loader as 3DGS, you can prepare your data following [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes). 


## Note
:D

