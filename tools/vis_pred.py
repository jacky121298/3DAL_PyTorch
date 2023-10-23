import os
import pickle
import argparse
import numpy as np
import open3d as o3d
import numpy.matlib as matlib
from math import sin, cos

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

def text_3d(text, pos, direction=None, degrees=0.0, font='DejaVu Sans Mono for Powerline.ttf', font_size=20):
    """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degrees: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    # pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.colors = o3d.utility.Vector3dVector(np.asarray([[1, 0, 0]] * img.shape[0]))
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    
    trans = (
        Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
        Quaternion(axis=direction, degrees=degrees)
    ).transformation_matrix
    
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

def get_lineset(points: np.ndarray, color: list):
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return lineset

def get_points(xmin: np.float, xmax: np.float, ymin: np.float, ymax: np.float, zmin: np.float, zmax: np.float):
    points = [
        [xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]
    ]
    return np.asarray(points)

def rotz(angle: np.float):
    c = np.cos(angle)
    s = np.sin(angle)
    rotz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list, case: str, types: np.ndarray=None, scores: np.ndarray=None, thresh: np.float=0.5):
    for i, bbox in enumerate(bboxs):
        if (scores is not None) and scores[i] < thresh:
            continue
        if types is not None and types[i] == 3:
            continue
        
        x, y, z, l, w, h, heading = bbox
        points = get_points(-l / 2, l / 2, -w / 2, w / 2, -h / 2, h / 2)
        if scores is not None:
            # vis.add_geometry(text_3d('Test', [x, y, z + h], direction=(1, 0, 0)))
            print(f'[{bcolors.OKBLUE}{case}{bcolors.ENDC}] Score: {scores[i]:.2f}, Box: ({x:6.2f}, {y:6.2f}, {z:6.2f}, {l:5.2f}, {w:5.2f}, {h:5.2f}, {heading:5.2f})')
        
        points = rotz(heading) @ points.T + bbox[:3, np.newaxis]
        points = points.T
        
        lineset = get_lineset(points=points, color=color)
        vis.add_geometry(lineset)

def euler_to_so3(rpy: list):
    R_x = np.matrix([
        [1,           0,            0],
        [0, cos(rpy[0]), -sin(rpy[0])],
        [0, sin(rpy[0]),  cos(rpy[0])],
    ])
    R_y = np.matrix([
        [ cos(rpy[1]), 0, sin(rpy[1])],
        [           0, 1,           0],
        [-sin(rpy[1]), 0, cos(rpy[1])],
    ])
    R_z = np.matrix([
        [cos(rpy[2]), -sin(rpy[2]), 0],
        [sin(rpy[2]),  cos(rpy[2]), 0],
        [          0,            0, 1],
    ])
    return R_z * R_y * R_x

def build_se3_transform(xyzrpy: list):
    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3

def sort_detections(detections):
    indices = []
    for det in detections:
        indices.append(det['frame_id'])

    rank = list(np.argsort(np.array(indices)))
    detections = [detections[r] for r in rank]
    return detections

def reorganize_info(infos):
    new_info = {}
    for info in infos:
        token = info['token']
        new_info[token] = info
    return new_info

if __name__ == '__main__':
    # python tools/vis_pred.py --data data/Waymo/val --seq seq_0_frame_0 --infos data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl --pred1 work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/prediction.pkl --pred2 work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/static/one_box_est.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to data.')
    parser.add_argument('--seq', help='Sequence name.')
    parser.add_argument('--infos', help='Path to infos file.')
    parser.add_argument('--pred1', help='Path to predicted bounding box w/o temporal.')
    parser.add_argument('--pred2', help='Path to predicted bounding box w/  temporal.')
    args = parser.parse_args()

    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)
    infos = reorganize_info(infos)

    lidar_path = os.path.join(args.data, 'lidar', args.seq + '.pkl')
    with open(lidar_path, 'rb') as f:
        lidar = pickle.load(f)
    points = lidar['lidars']['points_xyz']
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Point cloud shape: {points.shape}')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=args.seq, left=0, top=40)

    # render_option = vis.get_render_option()
    # render_option.background_color = np.array([0, 0, 0], np.float32)
    # render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points.astype(np.float64)))
    vis.add_geometry(pcd)

    # Draw groundtruth bbox (red)
    annos_path = os.path.join(args.data, 'annos', args.seq + '.pkl')
    with open(annos_path, 'rb') as f:
        annos = pickle.load(f)
    objects = annos['objects']

    bboxs = np.array([obj['box'] for obj in objects])
    types = np.array([obj['label'] for obj in objects])
    bboxs = bboxs[:, [0, 1, 2, 3, 4, 5, -1]]
    draw_3dbbox(bboxs, vis, color=[255, 0, 0], case='GT', types=types)

    # Draw detection bbox w/o temporal (green)
    with open(args.pred1, 'rb') as f:
        pred1 = pickle.load(f)
    pred1 = pred1[args.seq + '.pkl']

    scores = pred1['scores'].numpy()
    label_preds = pred1['label_preds'].numpy()
    box3d_lidar = pred1['box3d_lidar'].numpy()
    
    box3d_lidar[:, -1] = -box3d_lidar[:, -1] - np.pi / 2
    box3d_lidar = box3d_lidar[:, [0, 1, 2, 4, 3, 5, -1]]
    draw_3dbbox(box3d_lidar, vis, color=[0, 255, 0], case='w/o temporal', scores=scores)

    # Draw detection bbox w/ temporal (blue)
    with open(args.pred2, 'rb') as f:
        pred2 = pickle.load(f)
    pred2 = sort_detections(pred2)
    
    annos2idx, token2idx = {}, {}
    for i, pred in enumerate(pred2):
        annos2idx[pred['frame_id']] = i
    for token in infos:
        scene_name = annos['scene_name']
        frame_id = annos['frame_id']
        token2idx[token] = annos2idx[f'segment-{scene_name}_with_camera_labels_{frame_id:03d}']
    
    scores = pred2[token2idx[args.seq + '.pkl']]['score']
    label_preds = pred2[token2idx[args.seq + '.pkl']]['name']
    box3d_lidar = pred2[token2idx[args.seq + '.pkl']]['boxes_lidar']
    draw_3dbbox(box3d_lidar, vis, color=[0, 0, 255], case='w/  temporal', scores=scores)
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 0, 10, np.pi / 2, -np.pi / 2, 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()