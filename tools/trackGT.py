import pickle
import argparse
import numpy as np
from tqdm import tqdm

def get_obj(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def transform_box(box, pose):
    '''Transforms 3d upright boxes from one frame to another.
    Args:
    box: [1, 7] boxes.
    from_frame_pose: [4, 4] origin frame poses.
    to_frame_pose: [4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [1, 7] with the same type as box.
    '''
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0, 0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3], box[..., 0:3]) + np.expand_dims(transform[..., 0:3, 3], axis=-2)
    transformed = np.concatenate([center, box[..., 3:6], heading[..., np.newaxis]], axis=-1)
    return np.squeeze(transformed)

def main():
    # python3 tools/trackGT.py --infos data/Waymo/infos_train_02sweeps_filter_zero_gt.pkl --result work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/trackGT.pkl
    # python3 tools/trackGT.py --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --result work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackGT.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--infos', help='Path to infos file.')
    parser.add_argument('--result', help='Path to result file.')
    args = parser.parse_args()

    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)
    
    trackGT = {}
    for info in tqdm(infos):
        annos = get_obj(info['anno_path'])
        pose = np.reshape(annos['veh_to_global'], [4, 4])

        for obj in annos['objects']:
            name = obj['name']
            box = np.array(obj['box'])[[0, 1, 2, 3, 4, 5, -1]]
            box = transform_box(box[np.newaxis, ...], pose)
            vel = np.linalg.norm(np.array(obj['box'])[[6, 7]])
            num_points = obj['num_points']
            
            if name not in trackGT:
                trackGT[name] = {}
                trackGT[name]['box'] = [box]
                trackGT[name]['vel'] = [vel]
                trackGT[name]['pose'] = pose
                trackGT[name]['num_points'] = [num_points]
            else:
                trackGT[name]['box'].append(box)
                trackGT[name]['vel'].append(vel)
                trackGT[name]['num_points'].append(num_points)
    
    for name, obj in trackGT.items():
        bbox = np.array(obj['box'])
        dist = np.linalg.norm(bbox[0, :3] - bbox[-1, :3])
        vel = np.max(obj['vel'])
        if dist < 1 and vel < 1:
            trackGT[name]['static'] = 1
        else: trackGT[name]['static'] = 0

    with open(args.result, 'wb') as f:
        pickle.dump(trackGT, f)

if __name__ == '__main__':
    main()