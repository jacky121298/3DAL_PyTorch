import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

def reorganize_info(infos):
    new_info = {}
    for info in infos:
        token = info['token']
        new_info[token] = info
    return new_info

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
    return transformed

def main():
    # python3 tools/eval.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackStatic.pkl --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --static work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/static/static_labels.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', help='Path to track.pkl.')
    parser.add_argument('--infos', help='Path to infos file.')
    parser.add_argument('--static', help='Path to static_labels.pkl.')
    args = parser.parse_args()

    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Load track data')
    with open(args.track, 'rb') as f:
        track = pickle.load(f)

    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Load infos data')
    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)
    infos = reorganize_info(infos)

    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Load static data')
    with open(args.static, 'rb') as f:
        static = pickle.load(f)
    
    iou_track, iou_static = [], []
    for ID, obj in tqdm(static.items()):
        token = obj['token']
        static_bbox = obj['bbox']
        score = track[ID]['score']

        with open(infos[token]['anno_path'], 'rb') as f:
            annos = pickle.load(f)
        
        pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
        track_bbox = transform_box(track[ID]['bbox'][np.argmax(score)][np.newaxis, ...], pose)

        # Ground truth: (x, y, z, l, w, h, heading)
        bboxs = np.array([anno['box'] for anno in annos['objects']])
        bboxs = bboxs[:, [0, 1, 2, 3, 4, 5, -1]]
        bboxs_t = torch.tensor(bboxs).float().cuda()

        track_bbox = torch.tensor(track_bbox).float().cuda()
        track_iou = boxes_iou3d_gpu(track_bbox, bboxs_t)
        track_iou = track_iou.cpu().data.numpy().squeeze(axis=0)
        track_idx = np.argmax(track_iou)
        track_iou = np.max(track_iou)
        iou_track.append(track_iou)

        static_bbox = torch.tensor(static_bbox).float().cuda()
        static_iou = boxes_iou3d_gpu(static_bbox, bboxs_t)
        static_iou = static_iou.cpu().data.numpy().squeeze(axis=0)
        static_idx = np.argmax(static_iou)
        static_iou = np.max(static_iou)
        if static_iou > 1: continue
        iou_static.append(static_iou)
        
        # point = np.vstack(track[ID]['point'])
        # print(point.shape)
        # print(track_idx, static_idx)
        # print(bboxs_t[track_idx].cpu().data.numpy())
        # print(track_bbox.cpu().data.numpy())
        # print(static_bbox.cpu().data.numpy())
        # print(track_iou, static_iou)
        # print()

    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] mIOU of track: {np.mean(iou_track)}')
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] mIOU of static: {np.mean(iou_static)}')

if __name__ == '__main__':
    main()