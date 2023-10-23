import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from det3d.core.bbox import box_np_ops
from utils import class2angle, class2size
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

def gather_object_pts(pts, mask, n_pts=NUM_FRAME * NUM_OBJECT_POINT):
    '''
        :param pts: (bs, 4, n)
        :param mask: (bs, n)
        :param n_pts: max number of points of an object
        :return:
            object_pts: (bs, 4, n_pts)
            indices: (bs, n_pts)
    '''
    bs = pts.shape[0]
    object_pts = torch.zeros((bs, pts.shape[1], n_pts))
    indices = torch.zeros((bs, n_pts), dtype=torch.int64)

    for i in range(bs):
        pos_indices = torch.nonzero(mask[i, :] > 0.5).squeeze(1)
        if len(pos_indices) > 0:
            if len(pos_indices) >= n_pts:
                choice = np.random.choice(len(pos_indices), n_pts, replace=False)
            else:
                choice = np.random.choice(len(pos_indices), n_pts - len(pos_indices), replace=True)
                choice = np.concatenate((np.arange(len(pos_indices)), choice))
            
            np.random.shuffle(choice)
            indices[i, :] = pos_indices[choice]
            object_pts[i, :, :] = pts[i, :, indices[i, :]]

    return object_pts, indices

def point_cloud_masking(pts, logits):
    '''
        :param pts: (bs, 4, n) in frustum
        :param logits: (bs, n, 2)
    '''
    bs = pts.shape[0]
    n_pts = pts.shape[2]
    # Binary Classification for each point
    mask = logits[:, :, 0] < logits[:, :, 1] # (bs, n)
    pts_xyz = pts[:, :4, :] # (bs, 4, n)
    object_pts, _ = gather_object_pts(pts_xyz, mask, NUM_FRAME * NUM_OBJECT_POINT)
    return object_pts.float(), mask

def parse_output_to_tensors(box_pred, logits, mask):
    '''
        :param box_pred: (bs, 59)
        :param logits: (bs, n, 2)
        :param mask: (bs, n)
        :return:
            center_boxnet: (bs, 3)
            heading_scores: (bs, 12)
            heading_residuals_normalized: (bs, 12), -1 to 1
            heading_residuals: (bs, 12)
            size_scores: (bs, 8)
            size_residuals_normalized: (bs, 8)
            size_residuals: (bs, 8)
    '''
    bs = box_pred.shape[0]
    # Center
    c = 3
    center_boxnet = box_pred[:, :c]

    # Heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residuals_normalized = box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residuals = heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # Size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]
    c += NUM_SIZE_CLUSTER
    size_residuals_normalized = box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    size_residuals_normalized = size_residuals_normalized.view(bs, NUM_SIZE_CLUSTER, 3)
    size_residuals = size_residuals_normalized * torch.from_numpy(MEAN_SIZE_ARR).unsqueeze(0).repeat(bs, 1, 1).float().cuda()
    return center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals

def rotz(angle: torch.float):
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotz = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

class DynamicModel(nn.Module):
    def __init__(self, n_classes=3, n_channel=4):
        super(DynamicModel, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.ins_seg = PointNetInstanceSeg(n_classes=n_classes, n_channel=n_channel)
        self.point_emb = PointEmbedding(n_classes=n_classes)
        self.box_emb = BoxEmbedding(n_classes=n_classes)
        self.box_est = PointNetEstimation(n_classes=n_classes)
    
    def forward(self, pts, box, bbox_gt):
        '''
            :param pts: (bs, 4, n)
            :param box: (bs, 8, 101)
            :param bbox_gt: (bs, 7)
        '''
        # 3D Instance Segmentation
        logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 4, NUM_FRAME * NUM_OBJECT_POINT)

        # Point Embedding
        point_e = self.point_emb(object_pts_xyz) # (bs, 256)

        # Box Embedding
        box_e = self.box_emb(box) # (bs, 128)
        embedding = torch.cat([point_e, box_e], dim=1) # (bs, 256 + 128)
        
        # 3D Box Estimation
        box_pred = self.box_est(embedding) # (bs, 59)
        center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = parse_output_to_tensors(box_pred, logits, mask)
        center = center_boxnet + box[:, :3, 50] # (bs, 3)
        
        output = {
            'logits': logits,
            'mask': mask,
            'center_boxnet': center_boxnet,
            'heading_scores': heading_scores,
            'heading_residuals_normalized': heading_residuals_normalized,
            'heading_residuals': heading_residuals,
            'size_scores': size_scores,
            'size_residuals_normalized': size_residuals_normalized,
            'size_residuals': size_residuals,
            'center': center,
        }
        return output

class PointNetInstanceSeg(nn.Module):
    def __init__(self, n_classes=3, n_channel=4):
        '''
            3D Instance Segmentation PointNet
            :param n_classes: 3
            :param n_channel: 4
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.dconv1 = nn.Conv1d(1088, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts):
        '''
            :param pts: [bs, 4, n]
            :return: logits: [bs, n, 2], scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))          # (bs, 64, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))         # (bs, 64, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))         # (bs, 64, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))         # (bs, 128, n)
        out5 = F.relu(self.bn5(self.conv5(out4)))         # (bs, 1024, n)
        global_feat = torch.max(out5, 2, keepdim=True)[0] # (bs, 1024, 1)

        global_feat_repeat = global_feat.view(bs, -1, 1).repeat(1, 1, n_pts) # (bs, 1024, n)
        concat_feat = torch.cat([out2, global_feat_repeat], 1)               # (bs, 1088, n)

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))   # (bs, 512, n)
        x = F.relu(self.dbn2(self.dconv2(x)))             # (bs, 256, n)
        x = F.relu(self.dbn3(self.dconv3(x)))             # (bs, 128, n)
        x = F.relu(self.dbn4(self.dconv4(x)))             # (bs, 128, n)
        x = self.dropout(x)
        x = self.dconv5(x)                                # (bs, 2, n)
        seg_pred = x.transpose(2, 1).contiguous()         # (bs, n, 2)
        return seg_pred

class PointEmbedding(nn.Module):
    def __init__(self, n_classes=3):
        '''
            :param n_classes: 3
        '''
        super(PointEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts):
        '''
            :param pts: [bs, 4, m]
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))             # (bs, 64, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))            # (bs, 128, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))            # (bs, 256, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))            # (bs, 512, n)
        global_feat = torch.max(out4, 2, keepdim=False)[0]   # (bs, 512)

        x = F.relu(self.fcbn1(self.fc1(global_feat)))        # (bs, 512)
        x = F.relu(self.fcbn2(self.fc2(x)))                  # (bs, 256)
        return x

class BoxEmbedding(nn.Module):
    def __init__(self, n_classes=3):
        '''
            :param n_classes: 3
        '''
        super(BoxEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, box):
        '''
            :param box: [bs, 8, 101]
        '''
        bs = box.size()[0]
        n_box = box.size()[2]

        out1 = F.relu(self.bn1(self.conv1(box)))             # (bs, 64, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))            # (bs, 64, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))            # (bs, 128, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))            # (bs, 512, n)
        global_feat = torch.max(out4, 2, keepdim=False)[0]   # (bs, 512)

        x = F.relu(self.fcbn1(self.fc1(global_feat)))        # (bs, 128)
        x = F.relu(self.fcbn2(self.fc2(x)))                  # (bs, 128)
        return x

class PointNetEstimation(nn.Module):
    def __init__(self, n_classes=3):
        '''
            :param n_classes: 3
        '''
        super(PointNetEstimation, self).__init__()
        self.fc1 = nn.Linear(256 + 128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_classes + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, embedding):
        '''
            :param embedding: [bs, 256 + 128]
            :return: box_pred: [bs, n_classes + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4]
                including box centers, heading bin class scores and residual,
                and size cluster scores and residual
        '''
        bs = embedding.size()[0]

        x = F.relu(self.fcbn1(self.fc1(embedding)))          # (bs, 128)
        x = F.relu(self.fcbn2(self.fc2(x)))                  # (bs, 128)
        box_pred = self.fc3(x)                               # (bs, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)
        return box_pred

def huber_loss(error, delta=1.0):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(losses)

class DynamicModelLoss(nn.Module):
    def __init__(self):
        super(DynamicModelLoss, self).__init__()

    def forward(self, output, mask_label, center_label, heading_class_label, \
                heading_residuals_label, size_class_label, size_residuals_label, w_box=1.0):
        '''
        1. 3D Instance Segmentation PointNet Loss
            logits: torch.Size([32, 1024, 2]) torch.float32
            mask_label: [32, 1024]
        2. Center Regression Loss
            center: torch.Size([32, 3]) torch.float32
            center_label: [32, 3]
        3. Heading Loss
            heading_scores: torch.Size([32, 12]) torch.float32
            heading_residuals_normalized: torch.Size([32, 12]) torch.float32
            heading_residuals: torch.Size([32, 12]) torch.float32
            heading_class_label: (32)
            heading_residuals_label: (32)
        4. Size Loss
            size_scores: torch.Size([32, 8]) torch.float32
            size_residuals_normalized: torch.Size([32, 8, 3]) torch.float32
            size_residuals: torch.Size([32, 8, 3]) torch.float32
            size_class_label: (32)
            size_residuals_label: (32, 3)
        5. Weighted sum of all losses
            w_box: float scalar
        '''
        logits = output['logits']
        center = output['center']
        heading_scores = output['heading_scores']
        heading_residuals_normalized = output['heading_residuals_normalized']
        heading_residuals = output['heading_residuals']
        size_scores = output['size_scores']
        size_residuals_normalized = output['size_residuals_normalized']
        size_residuals = output['size_residuals']

        bs = logits.shape[0]
        # 3D Instance Segmentation PointNet Loss
        logits = F.log_softmax(logits.view(-1, 2), dim=1) # torch.Size([32768, 2])
        mask_label = mask_label.view(-1).long() # torch.Size([32768])
        mask_loss = F.nll_loss(logits, mask_label)

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1) # (32,)
        center_loss = huber_loss(center_dist, delta=2.0)

        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores, dim=1), heading_class_label.long())
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long()].cuda() # (32, 12)
        heading_residuals_normalized_label = heading_residuals_label / (np.pi / NUM_HEADING_BIN) # (32,)
        heading_residuals_normalized_dist = torch.sum(heading_residuals_normalized * hcls_onehot.float(), dim=1) # (32,)
        heading_residuals_normalized_loss = huber_loss(heading_residuals_normalized_dist - heading_residuals_normalized_label, delta=1.0)
        
        # Size loss
        size_class_loss = F.nll_loss(F.log_softmax(size_scores, dim=1), size_class_label.long())
        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()].cuda() # (32, 3)
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3) # (32, 3, 3)
        predicted_size_residuals_normalized_dist = torch.sum(size_residuals_normalized * scls_onehot_repeat.cuda(), dim=1) # (32, 3)
        mean_size_arr_expand = torch.from_numpy(MEAN_SIZE_ARR).float().cuda().view(1, NUM_SIZE_CLUSTER, 3) # (1, 3, 3)
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1) # (32, 3)
        size_residuals_label_normalized = size_residuals_label / mean_size_label.cuda()
        size_normalized_dist = torch.norm(size_residuals_label_normalized - predicted_size_residuals_normalized_dist, dim=1) # (32,)
        size_residuals_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = mask_loss + w_box * (center_loss * 10 + heading_class_loss + size_class_loss + heading_residuals_normalized_loss * 20 + size_residuals_normalized_loss * 20)

        losses = {
            'total_loss': total_loss,
            'mask_loss': mask_loss,
            'center_loss': w_box * center_loss * 10,
            'heading_class_loss': w_box * heading_class_loss,
            'size_class_loss': w_box * size_class_loss,
            'heading_residuals_normalized_loss': w_box * heading_residuals_normalized_loss * 20,
            'size_residuals_normalized_loss': w_box * size_residuals_normalized_loss * 20,
        }
        return losses

class DYNAMICTRACK(Dataset):
    def __init__(self, track, infos, npoints=NUM_POINT):
        self.trackID = list(track.keys())
        self.track = list(track.values())
        self.npoints = npoints
        self.infos = infos

        self.len = 0
        self.heads = [0]
        for t in self.track:
            self.len += len(t['point'])
            self.heads.append(self.heads[-1] + len(t['point']))

        self.r = 2
        self.s = 50

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        for i, h in enumerate(self.heads):
            if index < h:
                track_idx = i - 1
                item_idx = index - self.heads[i - 1]
                break
            
        ID = self.trackID[track_idx]
        token = self.track[track_idx]['token'][item_idx]

        point = np.zeros((0, 4)) # (5 * 1024, 4)
        for j, i in enumerate(range(item_idx - self.r, item_idx + self.r + 1)):
            if i < 0 or i >= len(self.track[track_idx]['point']):
                point = np.vstack([point, np.hstack([np.zeros((self.npoints, 3)), np.full((self.npoints, 1), 0.1 * (j - self.r))])])
            else:
                if len(self.track[track_idx]['point'][i]) > 0:
                    choice = np.random.choice(len(self.track[track_idx]['point'][i]), self.npoints, replace=True)
                    point_cur = self.track[track_idx]['point'][i][choice]
                    point = np.vstack([point, np.hstack([point_cur, np.full((self.npoints, 1), 0.1 * (j - self.r))])])
                else:
                    point = np.vstack([point, np.hstack([np.zeros((self.npoints, 3)), np.full((self.npoints, 1), 0.1 * (j - self.r))])])

        bbox = np.zeros((0, 8)) # (101, 8)
        for j, i in enumerate(range(item_idx - self.s, item_idx + self.s + 1)):
            if i < 0 or i >= len(self.track[track_idx]['bbox']):
                bbox = np.vstack([bbox, np.hstack([np.zeros((1, 7)), np.full((1, 1), 0.1 * (j - self.s))])])
            else:
                bbox_cur = self.track[track_idx]['bbox'][i].reshape((1, 7))
                bbox = np.vstack([bbox, np.hstack([bbox_cur, np.full((1, 1), 0.1 * (j - self.s))])])
    
        with open(self.infos[token]['anno_path'], 'rb') as f:
            annos = pickle.load(f)
        pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
        bbox[:, :7] = self.transform_box(bbox[:, :7], pose)
        point[:, :3] = (pose @ np.concatenate([point[:, :3].T, np.ones((1, point.shape[0]))], axis=0)).T[:, :3]

        ########### Generate labels ###########
        # mask_label
        bbox_gt = []
        mask_label = np.zeros((0, self.npoints))
        for j, i in enumerate(range(item_idx - self.r, item_idx + self.r + 1)):
            if i < 0 or i >= len(self.track[track_idx]['bbox']):
                mask_label = np.vstack([mask_label, np.zeros((1, self.npoints))])
            else:
                t = self.track[track_idx]['token'][i]
                with open(self.infos[t]['anno_path'], 'rb') as f:
                    annos = pickle.load(f)
                _pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
                
                has_bbox_t = False
                for obj in annos['objects']:
                    if obj['name'] == self.track[track_idx]['match'][-1]:
                        bbox_t = obj['box']
                        bbox_t = bbox_t[[0, 1, 2, 3, 4, 5, -1]]
                        has_bbox_t = True
                        if i == item_idx:
                            bbox_gt = np.copy(bbox_t)
                        break
                
                if has_bbox_t:
                    p = np.copy(point[j * self.npoints: (j + 1) * self.npoints, :3]).T
                    p = _pose @ np.linalg.inv(pose) @ np.vstack([p, np.ones((1, p.shape[1]))])
                    mask_label = np.vstack([mask_label, box_np_ops.points_in_rbbox(p.T[:, :3], bbox_t[np.newaxis, ...]).reshape((1, self.npoints))])
                else:
                    mask_label = np.vstack([mask_label, np.zeros((1, self.npoints))])

        mask_label = mask_label.flatten().astype(np.float)
        if len(bbox_gt) == 0:
            return torch.tensor([0])
        
        # center_label
        center_label = bbox_gt[:3]

        # heading_class_label & heading_residual_label
        heading_class_label, heading_residual_label = angle2class(bbox_gt[-1] - bbox[self.s, -2], NUM_HEADING_BIN)

        # size_class_label & size_residual_label
        size_class_label, size_residual_label = size2class(bbox_gt[3:6])

        # Coordinate transform
        point[:, :3] = point[:, :3] - bbox[self.s, :3]
        point[:, :3] = (self.rotz(-bbox[self.s, -2]) @ point[:, :3].T).T

        bbox[:, :3] = bbox[:, :3] - bbox[self.s, :3]
        bbox[:, -2] = bbox[:, -2] - bbox[self.s, -2]

        return ID, torch.from_numpy(bbox), torch.from_numpy(bbox_gt), torch.from_numpy(point), token, mask_label, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label
    
    def transform_box(self, box, pose):
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

    def rotz(self, angle: np.float):
        c = np.cos(angle)
        s = np.sin(angle)
        rotz = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])
        return rotz