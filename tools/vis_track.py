import os
import pickle
import argparse
import numpy as np
import open3d as o3d
from math import sin, cos
import numpy.matlib as matlib

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

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

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list, scores: np.ndarray=None, thresh: np.float=0.5):
    for i, bbox in enumerate(bboxs):
        if (scores is not None) and scores[i] < thresh:
            continue
        
        x, y, z, l, w, h, heading = bbox
        points = get_points(-l / 2, l / 2, -w / 2, w / 2, -h / 2, h / 2)
        if scores is not None:
            print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Box: ({x:.2f}, {y:.2f}, {z:.2f}, {l:.2f}, {w:.2f}, {h:.2f}, {heading:.2f})')
        
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
    return transformed

if __name__ == '__main__':
    # python tools/vis_track.py --track work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/track.pkl --trackGT work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/val/trackGT.pkl --index 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', help='Path to track data.')
    parser.add_argument('--trackGT', help='Path to trackGT.pkl.')
    parser.add_argument('--index', help='Index to visulize.', type=int)
    args = parser.parse_args()

    with open(args.track, 'rb') as f:
        track = pickle.load(f)
    
    new_track = []
    for obj in track.values():
        match = obj['match'][-1]
        bbox = np.array(obj['bbox'])
        point = np.vstack(obj['point'])
        if match == None or bbox.shape[0] < 7 or obj['type'][0] == 2 or point.shape[0] == 0:
            continue
        new_track.append(obj)
    
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Total numebr of track: {len(new_track)}')

    with open(args.trackGT, 'rb') as f:
        trackGT = pickle.load(f)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Track visualization', left=0, top=40)

    render_option = vis.get_render_option()
    render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    data = new_track[args.index]
    match = data['match'][-1]
    pose = np.linalg.inv(trackGT[match]['pose'])

    # Draw groundtruth bbox (red)
    boxGT = transform_box(np.array(trackGT[match]['box']), pose)
    draw_3dbbox(boxGT, vis, color=[255, 0, 0])

    # Draw tracked bbox (green)
    bbox = transform_box(np.array(data['bbox']), pose)
    draw_3dbbox(bbox, vis, color=[0, 255, 0])

    match = trackGT[match]['static'] == 1
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Static or not: {match}')
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Numebr of bbox in track #{args.index}: {bbox.shape[0]}')
    
    lidars = np.vstack(data['point']).T
    lidars = pose @ np.concatenate([lidars, np.ones((1, lidars.shape[1]))], axis=0)
    lidars = lidars[:3, :].T
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Numebr of points in track #{args.index}: {lidars.shape[0]}')
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(lidars.astype(np.float64)))
    vis.add_geometry(pcd)
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 0, 10, np.pi / 2, -np.pi / 2, 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()