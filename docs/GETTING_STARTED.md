# Getting Started

## Data Preparation

- Download the data from [here](https://waymo.com/open/download/) and organize as follows.

```
WAYMO_DATASET_ROOT
  ├── tfrecord_training
  ├── tfrecord_validation
  └── tfrecord_testing
```

- Convert the tfrecord data to pickle files.

```shell
# Train set
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_training/*.tfrecord' --root_path 'WAYMO_DATASET_ROOT/train/'
# Validation set
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_validation/*.tfrecord' --root_path 'WAYMO_DATASET_ROOT/val/'
# Testing set
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_testing/*.tfrecord' --root_path 'WAYMO_DATASET_ROOT/test/'
```

- Create a symlink to the dataset root.

```shell
# Remember to change the WAYMO_DATASET_ROOT to the actual path in your system.
mkdir data && cd data
ln -s WAYMO_DATASET_ROOT Waymo
```

- Create info files.

```shell
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=2
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=2
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=2
```

- The data and info files should be organized as follows.

```
3DAL_PyTorch
  └── data
    └── Waymo
      ├── tfrecord_training
      ├── tfrecord_validation
      ├── tfrecord_testing
      ├── train <-- all training frames and annotations
      ├── val   <-- all validation frames and annotations
      ├── test  <-- all testing frames and annotations
      ├── infos_train_02sweeps_filter_zero_gt.pkl
      ├── infos_val_02sweeps_filter_zero_gt.pkl
      └── infos_test_02sweeps_filter_zero_gt.pkl
```

## Training & Testing

### Preparation

1. 3D Object Detection

> Download the pre-trained 3D object detector [CenterPoint](https://drive.google.com/file/d/1Pp-Df8R3Oh9WuPGk7HdS2-CqmSl8K4-S/view?usp=sharing) and place it into the directory ```work_dirs/{config_name}```.

```shell
python tools/dist_test.py configs/waymo/voxelnet/two_stage/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel.py --work_dir work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/{train or val} --checkpoint work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/CenterPoint.pth --speed_test
```

2. 3D Multi-Object Tracking

```shell
python tools/waymo_tracking/test.py --work_dir work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/{train or val} --checkpoint work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/{train or val}/prediction.pkl --info_path data/Waymo/infos_{train or val}_02sweeps_filter_zero_gt.pkl
```

3. Object Track Data Extraction

```shell
python tools/trackData.py --work_dir work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/{train or val}
```

4. Motion State Classification

```shell
python tools/motionState.py --track_train work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/track.pkl --track_val work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/track.pkl --trackGT_train work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/trackGT.pkl --trackGT_val work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackGT.pkl
```

### Training

- Static Object Auto-Labeling

```shell
python tools/static_train.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/trackStatic.pkl --infos data/Waymo/infos_train_02sweeps_filter_zero_gt.pkl --model_type one_box_est
```

- Dynamic Object Auto-Labeling

```shell
python tools/dynamic_train.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/trackDynamic.pkl --infos data/Waymo/infos_train_02sweeps_filter_zero_gt.pkl --model_type one_box_est
```

### Testing

- Static Object Auto-Labeling

```shell
python tools/static_eval.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackStatic.pkl --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --model_path work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/static/model/one_box_est/856392.pth --model_type one_box_est --det_annos work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/det_annos.pkl
```

- Dynamic Object Auto-Labeling

```shell
python tools/dynamic_eval.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackDynamic.pkl --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --model_path work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/dynamic/model/one_box_est/856392.pth --model_type one_box_est --det_annos work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/det_annos.pkl
```