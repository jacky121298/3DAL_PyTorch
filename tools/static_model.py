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
NUM_POINT = 4096

MEAN_SIZE_ARR = np.array([
    [ 4.8, 1.8, 1.5],
    [10.0, 2.6, 3.2],
    [ 2.0, 1.0, 1.6],
])

def gather_object_pts(pts, mask, n_pts=NUM_OBJECT_POINT):
    '''
        :param pts: (bs, 3, n)
        :param mask: (bs, n)
        :param n_pts: max number of points of an object
        :return:
            object_pts: (bs, 3, n_pts)
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
        :param pts: (bs, 3, n) in frustum
        :param logits: (bs, n, 2)
    '''
    bs = pts.shape[0]
    n_pts = pts.shape[2]
    # Binary Classification for each point
    mask = logits[:, :, 0] < logits[:, :, 1] # (bs, n)
    pts_xyz = pts[:, :3, :] # (bs, 3, n)
    object_pts, _ = gather_object_pts(pts_xyz, mask, NUM_OBJECT_POINT)
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

class StaticModelOneBoxEst(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        super(StaticModelOneBoxEst, self).__init__()
        self.name = 'one_box_est'
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.ins_seg = PointNetInstanceSeg(n_classes=n_classes, n_channel=n_channel)
        self.box_est = PointNetEstimation(n_classes=n_classes)
    
    def forward(self, pts, init_box, bbox_gt):
        '''
            :param pts: (bs, 3, n)
            :param init_box: (bs, 7)
            :param bbox_gt: (bs, 7)
        '''
        # 3D Instance Segmentation
        logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 3, NUM_OBJECT_POINT)

        # 3D Box Estimation
        box_pred = self.box_est(object_pts_xyz) # (bs, 59)
        center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = parse_output_to_tensors(box_pred, logits, mask)
        center = center_boxnet + init_box[:, :3] # (bs, 3)
        
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

class StaticModelTwoBoxEst(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        super(StaticModelTwoBoxEst, self).__init__()
        self.name = 'two_box_est'
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.ins_seg = PointNetInstanceSeg(n_classes=n_classes, n_channel=n_channel)
        self.box_est_one = PointNetEstimation(n_classes=n_classes)
        self.box_est_two = PointNetEstimation(n_classes=n_classes)
    
    def forward(self, pts, init_box, bbox_gt):
        '''
            :param pts: (bs, 3, n)
            :param init_box: (bs, 7)
            :param bbox_gt: (bs, 7)
        '''
        # 3D Instance Segmentation
        logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 3, NUM_OBJECT_POINT)
        object_pts_xyz_two = torch.clone(object_pts_xyz) # (bs, 3, NUM_OBJECT_POINT)

        # 3D Box Estimation One
        box_pred_one = self.box_est_one(object_pts_xyz) # (bs, 59)
        center_one, heading_scores_one, heading_residuals_normalized_one, heading_residuals_one, \
        size_scores_one, size_residuals_normalized_one, size_residuals_one = parse_output_to_tensors(box_pred_one, logits, mask)
        center_one += init_box[:, :3] # (bs, 3)

        # Coordinate transform & Generate labels
        bs = center_one.shape[0]
        heading_class = np.argmax(heading_scores_one.cpu().detach().numpy(), 1) # (bs,)
        heading_residual = np.array([heading_residuals_one.cpu().detach().numpy()[i, heading_class[i]] for i in range(bs)]) # (bs,)
        size_class = np.argmax(size_scores_one.cpu().detach().numpy(), 1) # (bs,)
        size_residual = np.vstack([size_residuals_one.cpu().detach().numpy()[i, size_class[i], :] for i in range(bs)]) # (bs, 3)
        
        box_size = np.zeros((bs, 3))
        heading_angle = np.zeros((bs, 1))
        for i in range(bs):
            box_size[i, :] = class2size(size_class[i], size_residual[i])
            heading_angle[i] = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
            heading_angle[i] += init_box.cpu().detach().numpy()[i, -1]
        box_one = np.concatenate((center_one.cpu().detach().numpy(), box_size, heading_angle), axis=1) # (bs, 7)
        box_one = torch.from_numpy(box_one).float().cuda()

        heading_class_label_two = np.zeros((bs,))
        heading_residuals_label_two = np.zeros((bs,))

        for i in range(bs):
            object_pts_xyz_two[i] = rotz(init_box[i, -1]).float().cuda() @ object_pts_xyz_two[i]
            object_pts_xyz_two[i] = object_pts_xyz_two[i] + init_box[i, :3][..., None]

            object_pts_xyz_two[i] = object_pts_xyz_two[i] - box_one[i, :3][..., None]
            object_pts_xyz_two[i] = rotz(-box_one[i, -1]).float().cuda() @ object_pts_xyz_two[i]

            heading_class_label_two[i], heading_residuals_label_two[i] = angle2class(bbox_gt[i, -1] - box_one[i, -1], NUM_HEADING_BIN)

        heading_class_label_two = torch.from_numpy(heading_class_label_two).long().cuda()
        heading_residuals_label_two = torch.from_numpy(heading_residuals_label_two).float().cuda()

        # 3D Box Estimation Two
        box_pred_two = self.box_est_two(object_pts_xyz_two) # (bs, 59)
        center_two, heading_scores_two, heading_residuals_normalized_two, heading_residuals_two, \
        size_scores_two, size_residuals_normalized_two, size_residuals_two = parse_output_to_tensors(box_pred_two, logits, mask)
        center_two += center_one # (bs, 3)
        
        output = {
            'logits': logits,
            'mask': mask,
            'heading_scores_one': heading_scores_one,
            'heading_residuals_normalized_one': heading_residuals_normalized_one,
            'heading_residuals_one': heading_residuals_one,
            'size_scores_one': size_scores_one,
            'size_residuals_normalized_one': size_residuals_normalized_one,
            'size_residuals_one': size_residuals_one,
            'center_one': center_one,
            'box_one': box_one,
            'heading_scores_two': heading_scores_two,
            'heading_residuals_normalized_two': heading_residuals_normalized_two,
            'heading_residuals_two': heading_residuals_two,
            'size_scores_two': size_scores_two,
            'size_residuals_normalized_two': size_residuals_normalized_two,
            'size_residuals_two': size_residuals_two,
            'center_two': center_two,
            'heading_class_label_two': heading_class_label_two,
            'heading_residuals_label_two': heading_residuals_label_two,
            'center': center_two,
            'heading_scores': heading_scores_two,
            'heading_residuals': heading_residuals_two,
            'size_scores': size_scores_two,
            'size_residuals': size_residuals_two,
        }
        return output

class PointNetInstanceSeg(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        '''
            3D Instance Segmentation PointNet
            :param n_classes: 3
            :param n_channel: 3
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
            :param pts: [bs, 3, n]: x, y, z, intensity
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

class PointNetEstimation(nn.Module):
    def __init__(self, n_classes=3):
        '''
            Amodal 3D Box Estimation Pointnet
            :param n_classes: 3
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts):
        '''
            :param pts: [bs, 3, m]: x, y, z after InstanceSeg
            :return: box_pred: [bs, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4]
                including box centers, heading bin class scores and residual,
                and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))             # (bs, 128, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))            # (bs, 128, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))            # (bs, 256, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))            # (bs, 512, n)
        global_feat = torch.max(out4, 2, keepdim=False)[0]   # (bs, 512)

        x = F.relu(self.fcbn1(self.fc1(global_feat)))        # (bs, 512)
        x = F.relu(self.fcbn2(self.fc2(x)))                  # (bs, 256)
        box_pred = self.fc3(x)                               # (bs, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)
        return box_pred

def huber_loss(error, delta=1.0):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(losses)

class FrustumPointNetLossOneBoxEst(nn.Module):
    def __init__(self):
        super(FrustumPointNetLossOneBoxEst, self).__init__()

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

class FrustumPointNetLossTwoBoxEst(nn.Module):
    def __init__(self):
        super(FrustumPointNetLossTwoBoxEst, self).__init__()

    def forward(self, output, mask_label, center_label, heading_class_label, \
                heading_residuals_label, size_class_label, size_residuals_label, w_box=1.0):
        logits = output['logits']
        center_one = output['center_one']
        center_two = output['center_two']
        heading_scores_one = output['heading_scores_one']
        heading_scores_two = output['heading_scores_two']
        heading_residuals_normalized_one = output['heading_residuals_normalized_one']
        heading_residuals_normalized_two = output['heading_residuals_normalized_two']
        heading_residuals_one = output['heading_residuals_one']
        heading_residuals_two = output['heading_residuals_two']
        size_scores_one = output['size_scores_one']
        size_scores_two = output['size_scores_two']
        size_residuals_normalized_one = output['size_residuals_normalized_one']
        size_residuals_normalized_two = output['size_residuals_normalized_two']
        size_residuals_one = output['size_residuals_one']
        size_residuals_two = output['size_residuals_two']
        heading_class_label_two = output['heading_class_label_two']
        heading_residuals_label_two = output['heading_residuals_label_two']

        bs = logits.shape[0]
        # 3D Instance Segmentation PointNet Loss
        logits = F.log_softmax(logits.view(-1, 2), dim=1) # torch.Size([32768, 2])
        mask_label = mask_label.view(-1).long() # torch.Size([32768])
        mask_loss = F.nll_loss(logits, mask_label)

        # Center Regression Loss
        center_dist_one = torch.norm(center_one - center_label, dim=1) # (32,)
        center_loss_one = huber_loss(center_dist_one, delta=2.0)
        center_dist_two = torch.norm(center_two - center_label, dim=1) # (32,)
        center_loss_two = huber_loss(center_dist_two, delta=2.0)

        # Heading Loss
        heading_class_loss_one = F.nll_loss(F.log_softmax(heading_scores_one, dim=1), heading_class_label.long())
        hcls_onehot_one = torch.eye(NUM_HEADING_BIN)[heading_class_label.long()].cuda() # (32, 12)
        heading_residuals_normalized_label_one = heading_residuals_label / (np.pi / NUM_HEADING_BIN) # (32,)
        heading_residuals_normalized_dist_one = torch.sum(heading_residuals_normalized_one * hcls_onehot_one.float(), dim=1) # (32,)
        heading_residuals_normalized_loss_one = huber_loss(heading_residuals_normalized_dist_one - heading_residuals_normalized_label_one, delta=1.0)

        heading_class_loss_two = F.nll_loss(F.log_softmax(heading_scores_two, dim=1), heading_class_label_two.long())
        hcls_onehot_two = torch.eye(NUM_HEADING_BIN)[heading_class_label_two.long()].cuda() # (32, 12)
        heading_residuals_normalized_label_two = heading_residuals_label_two / (np.pi / NUM_HEADING_BIN) # (32,)
        heading_residuals_normalized_dist_two = torch.sum(heading_residuals_normalized_two * hcls_onehot_two.float(), dim=1) # (32,)
        heading_residuals_normalized_loss_two = huber_loss(heading_residuals_normalized_dist_two - heading_residuals_normalized_label_two, delta=1.0)
        
        # Size loss
        size_class_loss_one = F.nll_loss(F.log_softmax(size_scores_one, dim=1), size_class_label.long())
        scls_onehot_one = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()].cuda() # (32, 3)
        scls_onehot_repeat_one = scls_onehot_one.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3) # (32, 3, 3)
        predicted_size_residuals_normalized_dist_one = torch.sum(size_residuals_normalized_one * scls_onehot_repeat_one.cuda(), dim=1) # (32, 3)
        mean_size_arr_expand_one = torch.from_numpy(MEAN_SIZE_ARR).float().cuda().view(1, NUM_SIZE_CLUSTER, 3) # (1, 3, 3)
        mean_size_label_one = torch.sum(scls_onehot_repeat_one * mean_size_arr_expand_one, dim=1) # (32, 3)
        size_residuals_label_normalized_one = size_residuals_label / mean_size_label_one.cuda()
        size_normalized_dist_one = torch.norm(size_residuals_label_normalized_one - predicted_size_residuals_normalized_dist_one, dim=1) # (32,)
        size_residuals_normalized_loss_one = huber_loss(size_normalized_dist_one, delta=1.0)

        size_class_loss_two = F.nll_loss(F.log_softmax(size_scores_two, dim=1), size_class_label.long())
        scls_onehot_two = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()].cuda() # (32, 3)
        scls_onehot_repeat_two = scls_onehot_two.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3) # (32, 3, 3)
        predicted_size_residuals_normalized_dist_two = torch.sum(size_residuals_normalized_two * scls_onehot_repeat_two.cuda(), dim=1) # (32, 3)
        mean_size_arr_expand_two = torch.from_numpy(MEAN_SIZE_ARR).float().cuda().view(1, NUM_SIZE_CLUSTER, 3) # (1, 3, 3)
        mean_size_label_two = torch.sum(scls_onehot_repeat_two * mean_size_arr_expand_two, dim=1) # (32, 3)
        size_residuals_label_normalized_two = size_residuals_label / mean_size_label_two.cuda()
        size_normalized_dist_two = torch.norm(size_residuals_label_normalized_two - predicted_size_residuals_normalized_dist_two, dim=1) # (32,)
        size_residuals_normalized_loss_two = huber_loss(size_normalized_dist_two, delta=1.0)

        # Weighted sum of all losses
        total_loss = mask_loss + w_box * (
            center_loss_one * 10 + heading_class_loss_one + size_class_loss_one + heading_residuals_normalized_loss_one * 20 + size_residuals_normalized_loss_one * 20 + \
            center_loss_two * 10 + heading_class_loss_two + size_class_loss_two + heading_residuals_normalized_loss_two * 20 + size_residuals_normalized_loss_two * 20
        )

        losses = {
            'total_loss': total_loss,
            'mask_loss': mask_loss,
            'center_loss_one': w_box * center_loss_one * 10,
            'center_loss_two': w_box * center_loss_two * 10,
            'heading_class_loss_one': w_box * heading_class_loss_one,
            'heading_class_loss_two': w_box * heading_class_loss_two,
            'size_class_loss_one': w_box * size_class_loss_one,
            'size_class_loss_two': w_box * size_class_loss_two,
            'heading_residuals_normalized_loss_one': w_box * heading_residuals_normalized_loss_one * 20,
            'heading_residuals_normalized_loss_two': w_box * heading_residuals_normalized_loss_two * 20,
            'size_residuals_normalized_loss_one': w_box * size_residuals_normalized_loss_one * 20,
            'size_residuals_normalized_loss_two': w_box * size_residuals_normalized_loss_two * 20,
        }
        return losses

class STATICTRACK(Dataset):
    def __init__(self, track, infos, npoints=NUM_POINT):
        self.trackID = list(track.keys())
        self.track = list(track.values())
        self.npoints = npoints
        self.infos = infos

    def __len__(self):
        return len(self.track)
    
    def __getitem__(self, index):
        ID = self.trackID[index]
        bbox = np.vstack(self.track[index]['bbox'])
        point = np.vstack(self.track[index]['point'])
        score = np.stack(self.track[index]['score'])
        token = self.track[index]['token'][np.argmax(score)]
        
        with open(self.infos[token]['anno_path'], 'rb') as f:
            annos = pickle.load(f)
        pose = np.linalg.inv(np.reshape(annos['veh_to_global'], [4, 4]))
        bbox = self.transform_box(bbox[np.argmax(score)][np.newaxis, ...], pose)

        point = point.T
        point = pose @ np.concatenate([point, np.ones((1, point.shape[1]))], axis=0)
        point = point[:3, :].T

        # Resample point clouds
        choice = np.random.choice(point.shape[0], self.npoints, replace=True)
        point = point[choice, :] # (n, 3)

        ########### Generate labels ###########
        # mask_label
        for obj in annos['objects']:
            if obj['name'] == self.track[index]['match'][-1]:
                bbox_gt = obj['box']
        bbox_gt = bbox_gt[[0, 1, 2, 3, 4, 5, -1]]

        mask_label = box_np_ops.points_in_rbbox(point, bbox_gt[np.newaxis, ...])
        mask_label = mask_label.astype(np.float).squeeze()

        # center_label
        center_label = bbox_gt[:3]

        # heading_class_label & heading_residuals_label
        heading_class_label, heading_residuals_label = angle2class(bbox_gt[-1] - bbox[0, -1], NUM_HEADING_BIN)

        # size_class_label & size_residual_label
        size_class_label, size_residual_label = size2class(bbox_gt[3:6])

        # Coordinate transform
        point = point - bbox[:, :3]
        point = (self.rotz(-bbox[0, -1]) @ point.T).T
        
        return ID, torch.from_numpy(bbox), torch.from_numpy(bbox_gt), torch.from_numpy(point), token, mask_label, center_label, heading_class_label, heading_residuals_label, size_class_label, size_residual_label
    
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