# Installation

## Requirements

- Linux (tested on Ubuntu 14.04/16.04)
- Python 3.6+
- PyTorch 1.1
- CUDA 10.0
- CMake 3.13.2
- [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv)

## Environment

```shell
conda create --name 3dal_pytorch python=3.6
conda activate 3dal_pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
pip install waymo-open-dataset-tf-1-15-0==1.2.0

# Add 3DAL_PyTorch to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_THIS_REPO"

# Frustum PointNets
git clone https://github.com/simon3dv/frustum_pointnets_pytorch
export PYTHONPATH="${PYTHONPATH}:PATH_TO_FRUSTUM_POINTNETS_PYTORCH"

# Set the cuda path (change the path to your own cuda location)
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-10.0
export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

# Rotated NMS
cd 3DAL_PyTorch/det3d/ops/iou3d_nms
python setup.py build_ext --inplace

# Deformable Convolution (Optional and only works with old torch versions e.g. 1.1)
cd 3DAL_PyTorch/det3d/ops/dcn
python setup.py build_ext --inplace

# spconv
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *

# APEX (Optional)
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```