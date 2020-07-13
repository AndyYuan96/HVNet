import open3d
import numpy as np
import os
from . import box3d_visual_utils

def create_boundingbox(locs, dims, rots, color):

    num_box = locs.shape[0]
    boxes = []
    linesets = []

    origin = [0.5,0.5,0.5]
    corners = box3d_visual_utils.corners_nd(dims)

    corners = box3d_visual_utils.rotation_3d_in_axis(corners, rots, axis=2)
    corners += locs.reshape([-1, 1, 3])
    lines = [[0,1],[0,3],[2,1],[2,3],
             [4,5],[4,7],[6,5],[6,7],
             [0,4],[1,5],[2,6],[3,7]]

    colors = [color for i in range(len(lines))]

    linesets = []

    for i in range(num_box):
        lineset = open3d.LineSet()
        lineset.points =open3d.Vector3dVector(corners[i])
        lineset.lines = open3d.Vector2iVector(lines)
        lineset.colors = open3d.Vector3dVector(colors)
        linesets.append(lineset)

    return linesets

def display_pcd(pcd_path, weishu = None, gt = None, detection = None, assigned_anchor = None, predict_anchor = None, gt_dis = True, dt_dis = True, ass_dis = True, predict_dis = True):
    if pcd_path[-4:] == ".pcd":
        pcd = open3d.io.read_point_cloud(pcd_path)
    else:
        if weishu is not None:
            points = np.fromfile(pcd_path, dtype=np.float32)
            points = points.reshape(-1, weishu)
            points = points[:,:3]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
        else:
            points = np.fromfile(pcd_path, dtype=np.float32)
            points = points.reshape(-1, 4)
            points = points[:,:3]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
    
    mesh_frame = open3d.create_mesh_coordinate_frame(size=3, origin=[0, 0, 0])
    vis = open3d.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)

    if gt is not None and gt_dis:
        locs_predict = gt[:,:3]
        dims_predict = gt[:,3:6]
        rots_predict = -gt[:,6]
        gt_vis = create_boundingbox(locs_predict,dims_predict,rots_predict,[1,0,0])
        for i in range(len(gt_vis)):
            vis.add_geometry(gt_vis[i])
    
    if detection is not None and dt_dis:
        locs_predict = detection[:,:3]
        dims_predict = detection[:,3:6]
        rots_predict = -detection[:,6]
        detection_vis = create_boundingbox(locs_predict,dims_predict,rots_predict,[0,1,0])
        for i in range(len(detection_vis)):
            vis.add_geometry(detection_vis[i])
    
    if assigned_anchor is not None and ass_dis:
        locs_predict = assigned_anchor[:,:3]
        dims_predict = assigned_anchor[:,3:6]
        rots_predict = -assigned_anchor[:,6]
        assigned_anchor_vis = create_boundingbox(locs_predict,dims_predict,rots_predict,[0,0,1])
        for i in range(len(assigned_anchor_vis)):
            vis.add_geometry(assigned_anchor_vis[i])
    
    if predict_anchor is not None and predict_dis:
        locs_predict = predict_anchor[:,:3]
        dims_predict = predict_anchor[:,3:6]
        rots_predict = -predict_anchor[:,6]
        predict_anchor_vis = create_boundingbox(locs_predict,dims_predict,rots_predict,[0,1,1])
        for i in range(len(predict_anchor_vis)):
            vis.add_geometry(predict_anchor_vis[i])

    vis.run()
    vis.destroy_window()