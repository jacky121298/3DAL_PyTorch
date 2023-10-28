import os
import pickle
import argparse
import numpy as np
import open3d as o3d
from math import sin, cos
import numpy.matlib as matlib

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

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list):
    for i, bbox in enumerate(bboxs):
        x, y, z, l, w, h, heading = bbox
        print(f'bbox: ({x:.2f}, {y:.2f}, {z:.2f}, {l:.2f}, {w:.2f}, {h:.2f}, {heading:.2f})')
        points = get_points(-l / 2, l / 2, -w / 2, w / 2, -h / 2, h / 2)

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

if __name__ == '__main__':
    # python3 vis_data.py --pts ./data/pts.npy --box ./data/box.npy --gt_box ./data/gt_box.npy --seg ./data/seg.npy
    parser = argparse.ArgumentParser(description='Input data visualization.')
    parser.add_argument('--pts', help='Path to pts.')
    parser.add_argument('--box', help='Path to box.')
    parser.add_argument('--gt_box', help='Path to gt box.')
    parser.add_argument('--seg', help='Path to seg.')
    parser.add_argument('--show_seg', action='store_true', help='Show seg.')
    args = parser.parse_args()

    pts = np.load(args.pts, allow_pickle=True)
    print('pts shape (org):', pts.shape)

    if args.show_seg:
        seg = np.load(args.seg, allow_pickle=True)
        seg = seg.astype(np.bool)
        pts = pts[seg]
        print('pts shape (seg):', pts.shape)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Frustum visualization', left=0, top=40)

    # render_option = vis.get_render_option()
    # render_option.background_color = np.array([0, 0, 0], np.float32)
    # render_option.point_color_option = o3d.visualization.PointColorOption.Default
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts[:, :3].astype(np.float64)))
    vis.add_geometry(pcd)

    # Draw groundtruth box (red)
    gt_box = np.load(args.gt_box, allow_pickle=True)
    gt_box = gt_box[np.newaxis, ...]
    draw_3dbbox(gt_box, vis, color=[255, 0, 0])

    # Draw input box sequence (blue)
    box = np.load(args.box, allow_pickle=True)
    for b in box:
        b = b[:7][np.newaxis, ...]
        draw_3dbbox(b, vis, color=[0, 0, 255])
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 0, 10, np.pi / 2, -np.pi / 2, 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()