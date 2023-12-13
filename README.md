# SnowflakeNet Transformers for Point Cloud Completion

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointr-diverse-point-cloud-completion-with/point-cloud-completion-on-shapenet)](https://paperswithcode.com/sota/point-cloud-completion-on-shapenet?p=pointr-diverse-point-cloud-completion-with)

Adaptation led by Tahmid Awal, with contributions from former students including Ben Wagner.

This repository, originally based on the __PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers__ model (ICCV 2021 Oral Presentation), has been adapted to focus on SnowflakeNet Transformers for improved point cloud completion. We have implemented various modifications and enhancements to better support our current research objectives.

SnowflakeNet Transformers provide a more efficient and robust approach for processing and completing point cloud data, especially for complex real-world scenarios. Our current focus lies in exploring and expanding the capabilities of these transformers for diverse applications.

## Installation and Setup for SnowflakeNet Transformers

### Requirements

- PyTorch >= 1.7.0
- Python >= 3.10
- CUDA >= 11.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX


```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required. My environment has PyTorch = 2.0.1 and Python = 3.10.11

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.

# Chamfer Distance
cd extensions/chamfer_dist
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Cubic Feature Sampling, Gridding & Gridding Reverse, Gridding Loss (For GRNet)
cd $HOME/extensions/[specific_module]
python setup.py install --user


### New Installation Instructions (As of October 2023)

- Ensure Python version is 3.10 and CUDA version is 11.0 or above.
- Install dependencies using `pip install -r requirements.txt`.
- Set CUDA path: `export CUDAPATH=/usr/local/cuda-11`.
- Install additional required extensions using the provided installation scripts.

These changes are part of our ongoing efforts to enhance the performance and applicability of the SnowflakeNet Transformers in processing point cloud data.



### Dataset


The details of the data used so far for this project can be found in the ALPhA #data Slack Channel. Simulated 22Mg data has been the primary training dataset.

### Inference

To inference sample(s) with pretrained model

```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \

[--experimental] \
[--save_img_path <dir>] \
[--n_imgs <number>]
```

For example, inference 10 samples from `MidCutSnowflake.yaml` and save the results under `ATTPCPoinTr/imgs/`
```
python tools/inference.py \
cfgs/Mg22_Ne20pp_models/MidCutSnowflake.yaml path/to/ckpt.pth \ 
--save_img_path ./imgs/  \
--n_imgs=10 \

```

### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name>
```

####  Some examples:
Test the PoinTr pretrained model on the PCN benchmark:
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_PCN.pth \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name example
```
Test the PoinTr pretrained model on ShapeNet55 benchmark (*easy* mode):
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_ShapeNet55.pth \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --mode easy \
    --exp_name example
```
Test the PoinTr pretrained model on the KITTI benchmark:
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_KITTI.pth \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name example
CUDA_VISIBLE_DEVICES=0 python KITTI_metric.py \
    --vis <visualization_path> 
```

### Training

To train a point cloud completion model from scratch, run:


```
# Use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
####  Some examples:
Resume a checkpoint:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name example --resume
```

Finetune a PoinTr on PCNCars
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name example \
    --start_ckpts ./weight.pth
```

Train a PoinTr model with a single GPU:
```
bash ./scripts/train.sh 0 \
    --config ./cfgs/KITTI_models/PoinTr.yaml \

    --exp_name example
```

We also provide the Pytorch implementation of several baseline models including GRNet, PCN, TopNet and FoldingNet. For example, to train a GRNet model on ShapeNet-55, run:
```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/ShapeNet55_models/GRNet.yaml \
    --exp_name example
```


## License
MIT License

## Acknowledgements


Our code is inspired by [GRNet](https://github.com/hzxie/GRNet) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet/tree/main).


## Citation
If you find our work useful in your research, please consider citing: 
```
@inproceedings{yu2021pointr,
  title={PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers},
  author={Yu, Xumin and Rao, Yongming and Wang, Ziyi and Liu, Zuyan and Lu, Jiwen and Zhou, Jie},
  booktitle={ICCV},
  year={2021}
}
```
