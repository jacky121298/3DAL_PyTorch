import torch
import pickle
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from static_model import STATICTRACK
from utils import size2class, angle2class, compute_box3d_iou
from utils import reorganize_info

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3

MEAN_SIZE_ARR = np.array([
    [ 4.8, 1.8, 1.5],
    [10.0, 2.6, 3.2],
    [ 2.0, 1.0, 1.6],
])

def preprocessing(track, infos):
    del_keys = []
    for k, v in track.items():
        score = np.stack(v['score'])
        token = v['token'][np.argmax(score)]
        with open(infos[token]['anno_path'], 'rb') as f:
            annos = pickle.load(f)

        delete = True
        for obj in annos['objects']:
            if obj['name'] == v['match'][-1]:
                delete = False
                break    
        if delete:
            del_keys.append(k)

    for k in del_keys:
        del track[k]
    return track

def transform_box(box, pose):
    '''
    Transforms 3d upright boxes from one frame to another.
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

def calculate_init_iou(track, infos):
    init_iou2d = 0.0
    init_iou3d = 0.0
    init_iou3d_acc = 0.0

    n_samples = 0
    n_vehicle, n_cyclist = 0, 0
    for i, (key, value) in enumerate(tqdm(track.items())):
        bbox = np.vstack(value['bbox'])
        types = np.stack(value['type'])
        score = np.stack(value['score'])
        token = np.stack(value['token'])

        n_samples += bbox.shape[0]
        for j, t in enumerate(token):
            with open(infos[t]['anno_path'], 'rb') as f:
                annos = pickle.load(f)
            pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
            init_box = transform_box(bbox[[j], ...], pose)

            heading_scores = np.zeros((1, NUM_HEADING_BIN))
            heading_residuals = np.zeros((1, NUM_HEADING_BIN))
            size_scores = np.zeros((1, NUM_SIZE_CLUSTER))
            size_residuals = np.zeros((1, NUM_SIZE_CLUSTER, 3))

            heading_class, heading_residual = angle2class(0, NUM_HEADING_BIN)
            heading_scores[0, heading_class] = 1
            heading_residuals[0, heading_class] = heading_residual

            size_class, size_residual = size2class(init_box[0, 3:6])
            size_scores[0, size_class] = 1
            size_residuals[0, size_class, :] = size_residual

            bbox_gt = None
            for obj in annos['objects']:
                if obj['name'] == value['match'][-1]:
                    bbox_gt = obj['box']
            
            if bbox_gt is None:
                continue
            bbox_gt = bbox_gt[[0, 1, 2, 3, 4, 5, -1]]
            bbox_gt = bbox_gt[np.newaxis, ...]

            heading_class_label, heading_residual_label = angle2class(bbox_gt[0, -1] - init_box[0, -1], NUM_HEADING_BIN)
            size_class_label, size_residual_label = size2class(bbox_gt[0, 3:6])

            heading_class_label = np.array(heading_class_label)[np.newaxis, ...]
            heading_residual_label = np.array(heading_residual_label)[np.newaxis, ...]
            size_class_label = np.array(size_class_label)[np.newaxis, ...]
            size_residual_label = np.array(size_residual_label)[np.newaxis, ...]

            # Calculate iou2d, iou3d, iou3d_acc
            iou2ds, iou3ds = compute_box3d_iou(
                init_box[[0], :3],
                heading_scores,
                heading_residuals,
                size_scores,
                size_residuals,
                bbox_gt[[0], :3],
                heading_class_label,
                heading_residual_label,
                size_class_label,
                size_residual_label,
            )

            init_iou2d += np.sum(iou2ds)
            init_iou3d += np.sum(iou3ds)
            
            assert types[j] in [1, 4], 'Unknown type.'
            if types[j] == 1:
                init_iou3d_acc += np.sum(iou3ds >= 0.7)
                n_vehicle += 1
            elif types[j] == 4:
                init_iou3d_acc += np.sum(iou3ds >= 0.5)
                n_cyclist += 1

    init_iou2d /= n_samples
    init_iou3d /= n_samples
    init_iou3d_acc /= n_samples

    print(f'[Init] #Vehicle: {n_vehicle}, #Cyclist: {n_cyclist}')
    print(f'[Init] Box IoU (2D/3D): {init_iou2d:.4f}/{init_iou3d:.4f}')
    print(f'[Init] Box estimation accuracy: {init_iou3d_acc:.4f}')
    return init_iou2d, init_iou3d, init_iou3d_acc

def calculate_static_iou(track, infos, token2idx, det_annos, result_path):
    static_iou2d = 0.0
    static_iou3d = 0.0
    static_iou3d_acc = 0.0
    
    n_samples = 0
    n_vehicle, n_cyclist = 0, 0
    for i, (key, value) in enumerate(tqdm(track.items())):
        bbox = np.vstack(value['bbox'])
        types = np.stack(value['type'])
        score = np.stack(value['score'])
        token = np.stack(value['token'])
        best_box = bbox[[np.argmax(score)], ...]

        n_samples += bbox.shape[0]
        for j, t in enumerate(token):
            with open(infos[t]['anno_path'], 'rb') as f:
                annos = pickle.load(f)
            pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
            init_box = transform_box(best_box, pose)

            heading_scores = np.zeros((1, NUM_HEADING_BIN))
            heading_residuals = np.zeros((1, NUM_HEADING_BIN))
            size_scores = np.zeros((1, NUM_SIZE_CLUSTER))
            size_residuals = np.zeros((1, NUM_SIZE_CLUSTER, 3))

            heading_class, heading_residual = angle2class(0, NUM_HEADING_BIN)
            heading_scores[0, heading_class] = 1
            heading_residuals[0, heading_class] = heading_residual

            size_class, size_residual = size2class(init_box[0, 3:6])
            size_scores[0, size_class] = 1
            size_residuals[0, size_class, :] = size_residual

            bbox_gt = None
            for obj in annos['objects']:
                if obj['name'] == value['match'][-1]:
                    bbox_gt = obj['box']
            
            if bbox_gt is None:
                continue
            bbox_gt = bbox_gt[[0, 1, 2, 3, 4, 5, -1]]
            bbox_gt = bbox_gt[np.newaxis, ...]

            heading_class_label, heading_residual_label = angle2class(bbox_gt[0, -1] - init_box[0, -1], NUM_HEADING_BIN)
            size_class_label, size_residual_label = size2class(bbox_gt[0, 3:6])

            heading_class_label = np.array(heading_class_label)[np.newaxis, ...]
            heading_residual_label = np.array(heading_residual_label)[np.newaxis, ...]
            size_class_label = np.array(size_class_label)[np.newaxis, ...]
            size_residual_label = np.array(size_residual_label)[np.newaxis, ...]

            # Calculate iou2d, iou3d, iou3d_acc
            iou2ds, iou3ds = compute_box3d_iou(
                init_box[[0], :3],
                heading_scores,
                heading_residuals,
                size_scores,
                size_residuals,
                bbox_gt[[0], :3],
                heading_class_label,
                heading_residual_label,
                size_class_label,
                size_residual_label,
            )

            static_iou2d += np.sum(iou2ds)
            static_iou3d += np.sum(iou3ds)
            
            assert types[j] in [1, 4], 'Unknown type.'
            if types[j] == 1:
                static_iou3d_acc += np.sum(iou3ds >= 0.7)
                n_vehicle += 1
            elif types[j] == 4:
                static_iou3d_acc += np.sum(iou3ds >= 0.5)
                n_cyclist += 1

            # For det_annos
            bbox[j] = transform_box(bbox[[j], ...], pose).squeeze()
            exist = False
            for k, arr in enumerate(det_annos[token2idx[t]]['boxes_lidar']):
                if np.linalg.norm(arr[:3] - bbox[j, :3]) < 0.1:
                    det_annos[token2idx[t]]['boxes_lidar'][k, :] = init_box.squeeze()
                    det_annos[token2idx[t]]['score'][k] = np.max(score)
                    exist = True
                    break
            assert exist, 'Bounding box not in det_annos.'

    with open(result_path, 'wb') as f:
        pickle.dump(det_annos, f)
    
    static_iou2d /= n_samples
    static_iou3d /= n_samples
    static_iou3d_acc /= n_samples

    print(f'[Static] #Vehicle: {n_vehicle}, #Cyclist: {n_cyclist}')
    print(f'[Static] Box IoU (2D/3D): {static_iou2d:.4f}/{static_iou3d:.4f}')
    print(f'[Static] Box estimation accuracy: {static_iou3d_acc:.4f}')
    return static_iou2d, static_iou3d, static_iou3d_acc

def sort_detections(detections):
    indices = []
    for det in detections:
        indices.append(det['frame_id'])

    rank = list(np.argsort(np.array(indices)))
    detections = [detections[r] for r in rank]
    return detections

def main():
    # python3 tools/static_init.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackStatic.pkl --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --det_annos work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/det_annos.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', help='Path to trackStatic.pkl.')
    parser.add_argument('--infos', help='Path to infos file.')
    parser.add_argument('--det_annos', help='Path to detection annos.')
    args = parser.parse_args()

    with open(args.track, 'rb') as f:
        track = pickle.load(f)

    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)
    infos = reorganize_info(infos)

    with open(args.det_annos, 'rb') as f:
        det_annos = pickle.load(f)
    det_annos = sort_detections(det_annos)

    annos2idx, token2idx = {}, {}
    for i, annos in enumerate(det_annos):
        annos2idx[annos['frame_id']] = i
    for token in infos:
        with open(infos[token]['anno_path'], 'rb') as f:
            annos = pickle.load(f)
        scene_name = annos['scene_name']
        frame_id = annos['frame_id']
        token2idx[token] = annos2idx[f'segment-{scene_name}_with_camera_labels_{frame_id:03d}']
    
    track = preprocessing(track, infos)
    calculate_init_iou(track, infos)
    result_dir = pathlib.Path(args.track).parent / 'static'
    result_path = result_dir / 'static.pkl'
    calculate_static_iou(track, infos, token2idx, det_annos, result_path)

if __name__ == '__main__':
    main()