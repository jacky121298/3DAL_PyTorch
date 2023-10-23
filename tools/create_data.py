import os
import fire
import copy
import pickle
from pathlib import Path

from det3d.datasets.waymo import waymo_common as waymo_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train':
        create_groundtruth_database(
            'WAYMO',
            root_path,
            Path(root_path) / 'infos_train_{:02d}sweeps_filter_zero_gt.pkl'.format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )

if __name__ == '__main__':
    fire.Fire()