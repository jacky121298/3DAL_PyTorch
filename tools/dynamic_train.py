import copy
import torch
import random
import pickle
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dynamic_eval import eval_one_epoch
from dynamic_model import DYNAMICTRACK
from dynamic_model import DynamicModel
from dynamic_model import DynamicModelLoss
from utils import fixSeed, create_logger, reorganize_info, compute_box3d_iou

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

def preprocessing(track, ratio=0.1):
    del_keys = []
    for k, v in track.items():
        if v['type'][0] == 2:
            del_keys.append(k)
    
    for k in del_keys:
        del track[k]
    
    track = list(track.items())
    random.shuffle(track)
    train_track = dict(track[int(ratio * len(track)):])
    val_track = dict(track[:int(ratio * len(track))])
    return train_track, val_track

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epoch, result_dir, logger):
    best_state = {}
    best_iou3d_acc = 0.0
    for epoch in range(n_epoch):
        # Record for one epoch
        train_total_loss = 0.0
        train_iou2d = 0.0
        train_iou3d = 0.0
        train_seg_acc = 0.0
        train_iou3d_acc = 0.0
        
        n_samples = 0
        for i, data in enumerate(tqdm(train_loader)):
            model.train()

            if len(data) == 1:
                continue

            ID, bbox, bbox_gt, pts, token, mask_label, center_label, \
            heading_class_label, heading_residual_label, \
            size_class_label, size_residual_label = data

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
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Calculate loss, seg_acc, iou2d, iou3d, iou3d_acc
            train_total_loss += total_loss.item()
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

            train_iou2d += np.sum(iou2ds)
            train_iou3d += np.sum(iou3ds)
            train_iou3d_acc += np.sum(iou3ds >= 0.7)

            correct = torch.argmax(output['logits'], 2).eq(mask_label.long()).cpu().detach().numpy()
            train_seg_acc += np.sum(correct)

        train_total_loss /= n_samples
        train_seg_acc /= n_samples * float(NUM_POINT)
        train_iou2d /= n_samples
        train_iou3d /= n_samples
        train_iou3d_acc /= n_samples

        logger.info(f'=== Epoch [{epoch + 1}/{n_epoch}] ===')
        logger.info(f'[Train] loss: {train_total_loss:.4f}, seg acc: {train_seg_acc:.4f}')
        logger.info(f'[Train] Box IoU (2D/3D): {train_iou2d:.4f}/{train_iou3d:.4f}')
        logger.info(f'[Train] Box estimation accuracy (IoU=0.7): {train_iou3d_acc:.4f}')

        scheduler.step()

        # Evaluate
        eval_total_loss, eval_seg_acc, eval_iou2d, eval_iou3d, eval_iou3d_acc = eval_one_epoch(model, val_loader, criterion)
        logger.info(f'[Eval] loss: {eval_total_loss:.4f}, seg acc: {eval_seg_acc:.4f}')
        logger.info(f'[Eval] Box IoU (2D/3D): {eval_iou2d:.4f}/{eval_iou3d:.4f}')
        logger.info(f'[Eval] Box estimation accuracy (IoU=0.7): {eval_iou3d_acc:.4f}')
        if eval_iou3d_acc >= best_iou3d_acc:
            best_iou3d_acc = eval_iou3d_acc
            if epoch > n_epoch / 5:
                savepath = result_dir / f'acc{eval_iou3d_acc:04f}_epoch{epoch + 1:03d}.pth'
                logger.info(f'Model save to {savepath}')
                state = {
                    'epoch': epoch + 1,
                    'train_iou3d_acc': train_iou3d_acc,
                    'eval_iou3d_acc': eval_iou3d_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                best_state = copy.deepcopy(state)

    savepath = result_dir / f'acc{best_iou3d_acc:04f}_best.pth'
    logger.info(f'Model save to {savepath}')
    torch.save(best_state, savepath)
    logger.info(f'Done.')

def main():
    # CUDA_VISIBLE_DEVICES=0 python3 tools/dynamic_train.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train/trackDynamic.pkl --infos data/Waymo/infos_train_02sweeps_filter_zero_gt.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', help='Path to trackDynamic.pkl.')
    parser.add_argument('--infos', help='Path to infos file.')
    parser.add_argument('--n_epoch', type=int, default=150, help='Epoch to run [default: 150].')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate [default: 0.001].')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64].')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight Decay of Adam [default: 1e-4].')
    args = parser.parse_args()

    # Fix the random seed
    fixSeed(seed=10922081)

    result_dir = pathlib.Path(args.track).parent / 'dynamic' / 'model'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path('tools/log/dynamic_train')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'model.txt'

    logger = create_logger(log_file=log_file)
    logger.info('Load track data')
    with open(args.track, 'rb') as f:
        track = pickle.load(f)

    logger.info('Load info data')
    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)
    infos = reorganize_info(infos)

    train_track, val_track = preprocessing(track)
    train_set = DYNAMICTRACK(train_track, infos)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_set = DYNAMICTRACK(val_track, infos)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = DynamicModel(n_classes=3, n_channel=4).cuda()
    criterion = DynamicModelLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_func(epoch, init_lr=args.lr, step_size=20, gamma=0.7, eta_min=0.00001):
        f = gamma ** (epoch // step_size)
        if init_lr * f > eta_min:
            return f
        else: return 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    
    # torch.autograd.set_detect_anomaly(True)
    logger.info('Start training')
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, args.n_epoch, result_dir, logger)

if __name__ == '__main__':
    main()