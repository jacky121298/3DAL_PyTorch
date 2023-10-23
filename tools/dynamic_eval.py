import torch
import pickle
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dynamic_model import DYNAMICTRACK
from dynamic_model import DynamicModel
from utils import class2angle, class2size, compute_box3d_iou
from utils import fixSeed, create_logger, reorganize_info
from utils import size2class, angle2class

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3
NUM_OBJECT_POINT = 512
NUM_POINT = 1024
NUM_FRAME = 5

MEAN_SIZE_ARR = np.array([
    [ 4.8, 1.8, 1.5],
    [10.0, 2.6, 3.2],
    [ 2.0, 1.0, 1.6],
])

def preprocessing(track):
    del_keys = []
    for k, v in track.items():
        if v['type'][0] == 2:
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

def postprocessing(track, infos, token2idx, final_bboxes, det_annos, result_path, logger):
    '''
        :param final_bboxes: (n, 7)
        :param det_annos: list of dict
    '''
    eval_iou2d = 0.0
    eval_iou3d = 0.0
    eval_iou3d_acc = 0.0

    n_samples = 0
    for i, (key, value) in enumerate(tqdm(track.items())):
        bbox = np.vstack(value['bbox'])
        types = np.stack(value['type'])
        score = np.stack(value['score'])
        token = np.stack(value['token'])
        best_box = bbox[[np.argmax(score)], ...]

        with open(infos[token[np.argmax(score)]]['anno_path'], 'rb') as f:
            _annos = pickle.load(f)
        _pose = np.reshape(_annos['veh_to_global'], [4, 4])

        n_samples += bbox.shape[0]
        for j, t in enumerate(token):
            with open(infos[t]['anno_path'], 'rb') as f:
                annos = pickle.load(f)
            pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
            # For det_annos
            bbox[j] = transform_box(bbox[[j], ...], pose).squeeze()
            final_bbox = transform_box(final_bboxes[[i], :], _pose)
            final_bbox = transform_box(final_bbox, pose)
            init_box = transform_box(best_box, pose)

            heading_scores = np.zeros((1, NUM_HEADING_BIN))
            heading_residuals = np.zeros((1, NUM_HEADING_BIN))
            size_scores = np.zeros((1, NUM_SIZE_CLUSTER))
            size_residuals = np.zeros((1, NUM_SIZE_CLUSTER, 3))

            heading_class, heading_residual = angle2class(final_bbox[0, -1] - init_box[0, -1], NUM_HEADING_BIN)
            heading_scores[0, heading_class] = 1
            heading_residuals[0, heading_class] = heading_residual

            size_class, size_residual = size2class(final_bbox[0, 3:6])
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
                final_bbox[[0], :3],
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

            eval_iou2d += np.sum(iou2ds)
            eval_iou3d += np.sum(iou3ds)
            
            assert types[j] in [1, 4], 'Unknown type.'
            if types[j] == 1:
                eval_iou3d_acc += np.sum(iou3ds >= 0.7)
            elif types[j] == 4:
                eval_iou3d_acc += np.sum(iou3ds >= 0.5)

            exist = False
            for k, arr in enumerate(det_annos[token2idx[t]]['boxes_lidar']):
                if np.linalg.norm(arr[:3] - bbox[j, :3]) < 0.1:
                    det_annos[token2idx[t]]['boxes_lidar'][k, :] = final_bbox.squeeze()
                    # det_annos[token2idx[t]]['score'][k] = np.max(score)
                    exist = True
                    break
            assert exist, 'Bounding box not in det_annos.'

    logger.info(f'Saving results to {result_path}')
    with open(result_path, 'wb') as f:
        pickle.dump(det_annos, f)
    
    eval_iou2d /= n_samples
    eval_iou3d /= n_samples
    eval_iou3d_acc /= n_samples

    logger.info(f'[Eval] Box IoU (2D/3D): {eval_iou2d:.4f}/{eval_iou3d:.4f}')
    logger.info(f'[Eval] Box estimation accuracy: {eval_iou3d_acc:.4f}')
    return eval_iou2d, eval_iou3d, eval_iou3d_acc, det_annos

def sort_detections(detections):
    indices = []
    for det in detections:
        indices.append(det['frame_id'])

    rank = list(np.argsort(np.array(indices)))
    detections = [detections[r] for r in rank]
    return detections

def eval_one_epoch(model, dataloader, criterion):
    eval_total_loss = 0.0
    eval_iou2d = 0.0
    eval_iou3d = 0.0
    eval_seg_acc = 0.0
    eval_iou3d_acc = 0.0

    n_samples = 0
    for i, data in enumerate(tqdm(dataloader)):
        model = model.eval()

        ID, bbox, bbox_gt, pts, token, mask_label, center_label, \
        heading_class_label, heading_residual_label, \
        size_class_label, size_residual_label = data
        
        if ID == None:
            continue
        
        n_samples += pts.shape[0]
        pts = pts.transpose(2, 1).float().cuda()
        bbox = bbox.transpose(2, 1).float().cuda()
        bbox_gt = bbox_gt.float().cuda()
        mask_label = mask_label.float().cuda()
        center_label = center_label.float().cuda()
        heading_class_label = heading_class_label.long().cuda()
        heading_residual_label = heading_residual_label.float().cuda()
        size_class_label = size_class_label.long().cuda()
        size_residual_label = size_residual_label.float().cuda()

        output = model(pts, bbox, bbox_gt)
        losses = criterion(output, mask_label, center_label, \
            heading_class_label, heading_residual_label, size_class_label, size_residual_label)
        
        total_loss = losses['total_loss']
        # Calculate loss, seg_acc, iou2d, iou3d, iou3d_acc
        eval_total_loss += total_loss.item()
        iou2ds, iou3ds = compute_box3d_iou(
            output['center'].cpu().detach().numpy(),
            output['heading_scores'].cpu().detach().numpy(),
            output['heading_residuals'].cpu().detach().numpy(),
            output['size_scores'].cpu().detach().numpy(),
            output['size_residuals'].cpu().detach().numpy(),
            center_label.cpu().detach().numpy(),
            heading_class_label.cpu().detach().numpy(),
            heading_residual_label.cpu().detach().numpy(),
            size_class_label.cpu().detach().numpy(),
            size_residual_label.cpu().detach().numpy(),
        )

        eval_iou2d += np.sum(iou2ds)
        eval_iou3d += np.sum(iou3ds)
        eval_iou3d_acc += np.sum(iou3ds >= 0.7)

        correct = torch.argmax(output['logits'], 2).eq(mask_label.long()).cpu().detach().numpy()
        eval_seg_acc += np.sum(correct)

    eval_total_loss /= n_samples
    eval_seg_acc /= n_samples * float(NUM_POINT * NUM_FRAME)
    eval_iou2d /= n_samples
    eval_iou3d /= n_samples
    eval_iou3d_acc /= n_samples

    return eval_total_loss, eval_seg_acc, eval_iou2d, eval_iou3d, eval_iou3d_acc

def test_one_epoch(model, dataloader, logger):
    final_bboxes = np.zeros((0, 7))
    for data in tqdm(dataloader):
        model = model.eval()

        ID, bbox, bbox_gt, pts, token, mask_label, center_label, \
        heading_class_label, heading_residual_label, \
        size_class_label, size_residual_label = data

        if ID == None:
            continue
        
        pts = pts.transpose(2, 1).float().cuda()
        bbox = bbox.transpose(2, 1).float().cuda()
        bbox_gt = bbox_gt.float().cuda()

        output = model(pts, bbox, bbox_gt)
        bs = output['center'].shape[0]
        heading_class = np.argmax(output['heading_scores'].cpu().detach().numpy(), 1) # (bs,)
        heading_residual = np.array([output['heading_residuals'].cpu().detach().numpy()[i, heading_class[i]] for i in range(bs)]) # (bs,)
        size_class = np.argmax(output['size_scores'].cpu().detach().numpy(), 1) # (bs,)
        size_residual = np.vstack([output['size_residuals'].cpu().detach().numpy()[i, size_class[i], :] for i in range(bs)]) # (bs, 3)
        
        box_size = np.zeros((bs, 3))
        heading_angle = np.zeros((bs, 1))
        for i in range(bs):
            box_size[i, :] = class2size(size_class[i], size_residual[i])
            heading_angle[i] = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
            heading_angle[i] += init_box[i, -1].cpu().detach().numpy()
        final_bbox = np.concatenate((output['center'].cpu().detach().numpy(), box_size, heading_angle), axis=1) # (bs, 7)
        final_bboxes = np.concatenate((final_bboxes, final_bbox), axis=0)

    return final_bboxes

def main():
    # CUDA_VISIBLE_DEVICES=0 python3 tools/dynamic_eval.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackDynamic.pkl --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --model_path work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/dynamic/model/one_box_est/acc0.856392_best.pth --model_type one_box_est --det_annos work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/det_annos.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', help='Path to trackStatic.pkl.')
    parser.add_argument('--infos', help='Path to infos file.')
    parser.add_argument('--model_path', help='Path to model.')
    parser.add_argument('--model_type', help='Type of model.')
    parser.add_argument('--det_annos', help='Path to detection annos.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64].')
    args = parser.parse_args()

    # Fix the random seed
    fixSeed(seed=10922081)

    assert args.model_type in ['one_box_est', 'two_box_est'], f'No model supports for model type \"{args.model_type}\".'
    result_dir = pathlib.Path(args.track).parent / 'static'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path('tools/log/static_eval')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{args.model_type}.txt'

    logger = create_logger(log_file=log_file)
    logger.info('Load track data')
    with open(args.track, 'rb') as f:
        track = pickle.load(f)

    logger.info('Load info data')
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
    test_set = STATICTRACK(track, infos)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    logger.info(f'Load model from {args.model_path}')
    model_dict = {
        'one_box_est': StaticModelOneBoxEst,
        'two_box_est': StaticModelTwoBoxEst,
    }
    model = model_dict[args.model_type](n_classes=3, n_channel=3).cuda()
    model_state = torch.load(args.model_path)
    model.load_state_dict(model_state['model_state_dict'])

    logger.info('Start testing')
    final_bboxes = test_one_epoch(model, test_loader, logger)
    logger.info('Start post processing')
    postprocessing(track, infos, token2idx, final_bboxes, det_annos, result_dir / f'{args.model_type}.pkl', logger)

if __name__ == '__main__':
    main()