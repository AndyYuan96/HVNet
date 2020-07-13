import numba
from numba import cuda
from pathlib import Path
import numpy as np
# from open3d import *

# from configs.config import cfg

##-------------TEST---------------##
# def gen_dict_class_idx(cfg):
#         all_class = cfg.CLASS_NAME
#         class_idx_to_name = {}
#         class_name_to_idx = {}
#         for i, name in enumerate(all_class):
#                 class_idx_to_name[i] = name
#                 class_name_to_idx[name] = i
#         return class_idx_to_name, class_name_to_idx
# class_idx_to_name, class_name_to_idx = gen_dict_class_idx(cfg)
# def read_anno(info_path, configs):
#         with open(str(info_path), 'r') as f:
#                 lines=f.readlines()
#         annotation_contents = []
#         for line in lines:
#                 contents=line.split(' ')
#                 annotation_contents.append(contents)
#         annos = {}
#         annos['path'] = info_path
#         annos['cls'] = np.array([class_name_to_idx[x[0]] for x in annotation_contents]).reshape(-1,1)
#         # annos['idx'] = np.array([float(x[1]) for x in annotation_contents]).reshape(-1,1)
#         annos['idx']=np.array(np.arange(len(annos['cls'])),dtype=np.float32).reshape(-1,1)
#         annos['location'] = np.array([[float(elem) for elem in x[2:5]] for x in annotation_contents]).reshape(-1, 3)
#         # print(annos['location'].shape)
#         annos['dimensions'] = np.array([[float(elem) for elem in x[5:8]] for x in annotation_contents]).reshape(-1, 3)
#         annos['rotation'] = np.array([[float(elem) for elem in x[8:11]] for x in annotation_contents]).reshape(-1, 3)
#         annos['rotation'] = -annos['rotation'] 
#         if configs.TYPE == 'GT':
#                 # annos['difficulty'] = np.ones([len(annos['cls']),1])
#                 annos['difficulty'] = np.array([int(x[11]) for x in annotation_contents]).reshape(-1,1)
#                 # annos['difficulty'] = np.array([float(x[11]) for x in annotation_contents]).reshape(-1, 1)
#         if configs.TYPE == 'DT':
#                 # annos['score'] = np.ones([len(annos['cls']),1])
#                 annos['score'] = np.array([float(x[25]) for x in annotation_contents]).reshape(-1, 1)
#                 # annos['score'] = np.array([float(x[12]) for x in annotation_contents]).reshape(-1, 1)
#                 # print(annos['score'])   
#         annos['distance'] = (np.sqrt(np.sum(annos['location']**2, axis=-1))).reshape(-1, 1)
#         return annos
##------------TEST---------------##

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dims.dtype)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2**ndim, ndim])
    return corners

def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

@numba.jit(nopython=False)
def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners. 
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces

@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d

@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, 
            max_num_points_of_surface, 3] 
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces 
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]

    if isinstance(num_surfaces, type(None)):
    # if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)

    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

def points_in_rbbox(points, rbbox, lidar=True):
 
    h_axis = 2
    origin = [0.5, 0.5, 0.5]
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=h_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)

    return indices

@numba.jit(nopython=False)
def points_iou_box(gt_boxes_inds, dt_boxes_inds, opt = -1):

    # if opt == 0:
    #     iou_fn = lambda intersect, union, gt_inds, dt_inds: intersect/union
    # elif opt == 1:
    #     iou_fn = lambda intersect, union, gt_inds, dt_inds: intersect/gt_inds
    # elif opt == 2:
    #     iou_fn = lambda intersect, union, gt_inds, dt_inds: intersect/dt_inds

    num_gt = gt_boxes_inds.shape[1]
    num_dt = dt_boxes_inds.shape[1]

    gt_inds = []
    dt_inds = []

    iou = np.zeros((num_gt,num_dt),dtype=np.float32)
    for i in range(num_gt):
        gt_inds.append(np.where(gt_boxes_inds[:,i]!=False)[0])

    for j in range(num_dt):
        dt_inds.append(np.where(dt_boxes_inds[:,j]!=False)[0])

    for k in range(num_gt):
        for l in range(num_dt):
            one_gt_inds = gt_inds[k]
            one_dt_inds = dt_inds[l]
            intersect = np.intersect1d(one_gt_inds,one_dt_inds)
            union = np.union1d(one_gt_inds,one_dt_inds)
            
            # print(k,'   ',l,'   ',intersect.shape,'   ',union.shape)
            if union.shape[0]==0 or intersect.shape[0]==0:
                continue
            if opt == -1:
                iou[k,l] = intersect.shape[0]/union.shape[0]
            elif opt == 0:
                iou[k,l] = intersect.shape[0]/one_gt_inds.shape[0]
            elif opt == 1:
                iou[k,l] = intersect.shape[0]/one_dt_inds.shape[0]
            
            # print('gt_id:{}'.format(k),' points:{}'.format(one_gt_inds.shape[0]),' dt_id:{}'.format(l),' points:{}'.format(one_dt_inds.shape[0]), ' intersect:{}'.format(intersect.shape[0]), ' union:{}'.format(union.shape[0]), 'iou:{}'.format(iou[k,l]))

    return iou

def create_boundingbox(locs, dims, rots, color):

    num_box = locs.shape[0]
    boxes = []
    linesets = []

    origin = [0.5,0.5,0.5]
    corners = corners_nd(dims)
    corners = rotation_3d_in_axis(corners, rots, axis=2)
    corners += locs.reshape([-1, 1, 3])


    lines = [[0,1],[0,3],[2,1],[2,3],
             [4,5],[4,7],[6,5],[6,7],
             [0,4],[1,5],[2,6],[3,7]]
    
    colors = [color for i in range(len(lines))]

    linesets = []
    for i in range(num_box):
        lineset = LineSet()
        lineset.points = Vector3dVector(corners[i])
        lineset.lines = Vector2iVector(lines)
        lineset.colors = Vector3dVector(colors)
        linesets.append(lineset)

    return linesets

################################################
#                 test unit                    #
################################################
# gt_cfg = cfg.GT_ANNO
# dt_cfg = cfg.DT_ANNO

# pcd_file = '32_daxuecheng_01803191740_1352.pcd'
# txt_file = pcd_file.replace('pcd','txt')

# ## read data
# pcd_path = cfg.PCD_ROOT+'/'+pcd_file
# gt_path = cfg.GT_ANNO.ROOT_PATH+'/'+txt_file
# dt_path = cfg.DT_ANNO.ROOT_PATH+'/'+txt_file

# pcd = read_point_cloud(pcd_path)
# points = np.asarray(pcd.points)

# gt_anno = read_anno(gt_path,gt_cfg)
# dt_anno = read_anno(dt_path,dt_cfg)

# locs_gt = gt_anno['location']
# dims_gt = gt_anno['dimensions']
# rots_gt = gt_anno['rotation'][:,2]
# gt_boxes = np.concatenate([locs_gt, dims_gt, rots_gt[..., np.newaxis]], axis=1)

# locs_dt = dt_anno['location']
# dims_dt = dt_anno['dimensions']
# rots_dt = dt_anno['rotation'][:,2]
# dt_boxes = np.concatenate([locs_dt, dims_dt, rots_dt[..., np.newaxis]], axis=1)


# ## calculate overseg and underseg
# ou_gt_boxes = gt_boxes
# ou_gt_boxes[:,2] = 0
# ou_dt_boxes = dt_boxes
# ou_dt_boxes[:, 2] = 0

# centers_gt = locs_gt
# centers_gt[:,2]=0
# centers_dt = locs_dt
# centers_dt[:,2]=0

# gt_points_in_dt_boxes = points_in_rbbox(centers_gt, ou_dt_boxes)
# dt_points_in_gt_boxes = points_in_rbbox(centers_dt, ou_gt_boxes)

# overseg = 0
# underseg = 0
# for i in range(gt_boxes.shape[0]):
#         num = np.sum(dt_points_in_gt_boxes[:,i])
#         if num > 1:
#                 overseg+=1

# for i in range(dt_boxes.shape[0]):
#         num = np.sum(gt_points_in_dt_boxes[:,i])
#         print(i, num, locs_dt[i,:])
#         if num>1:
#                 underseg+=num

# print(overseg, underseg)
# centers = np.concatenate([centers_gt, centers_dt], axis = 0)
# pcd_centers = PointCloud()
# pcd_centers.points = Vector3dVector(centers)
# color_centers = np.ones((points.shape[0],3),dtype=np.float32)
# pcd_centers.colors = Vector3dVector(color_centers)

# ## 
# colors = np.ones((points.shape[0],3),dtype=np.float32)

# gt_boxes_inds = points_in_rbbox(points,gt_boxes)
# dt_boxes_inds = points_in_rbbox(points,dt_boxes)
# iou = points_iou_box(gt_boxes_inds,dt_boxes_inds,opt=cfg.IOU_OPT)

# # print(iou)
# num_gt = gt_boxes_inds.shape[1]
# num_dt = dt_boxes_inds.shape[1]
# gt_inds = []
# dt_inds = []
# for i in range(num_gt):
#     gt_inds.append(np.where(gt_boxes_inds[:,i]!=False)[0])
# for j in range(num_dt):
#     dt_inds.append(np.where(dt_boxes_inds[:,j]!=False)[0])
# gt_inds = np.concatenate(gt_inds,axis = 0)
# dt_inds = np.concatenate(dt_inds,axis = 0)
# inds_insect = np.intersect1d(gt_inds, dt_inds)
# colors[gt_inds,:]=np.array([0.0,0.0,0.0],dtype=np.float32)
# colors[dt_inds,:]=np.array([0.0,0.0,0.0],dtype=np.float32)
# colors[inds_insect,:]=np.array([0.0,0.0,0.0],dtype=np.float32)
# pcd.colors = Vector3dVector(colors)
# ## visualization
# gt_boxes_vis = create_boundingbox(locs_gt,dims_gt,rots_gt,[1,0,0])
# dt_boxes_vis = create_boundingbox(locs_dt,dims_dt,rots_dt,[0,1,0])
# # dt_boxes_vis = create_boundingbox(locs_dt[19:21,:],dims_dt[19:21,:],rots_dt[19:21],[0,1,0])
# # print(dt_boxes)
# mesh_frame = create_mesh_coordinate_frame(size = 5, origin = [0, 0, 0])
# vis = Visualizer()
# vis.create_window()
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 0, 0])
# # vis.add_geometry(pcd)
# vis.add_geometry(pcd_centers)
# vis.add_geometry(mesh_frame)
# for i in range(len(gt_boxes_vis)):
#     vis.add_geometry(gt_boxes_vis[i])
# for i in range(len(dt_boxes_vis)):
#     vis.add_geometry(dt_boxes_vis[i])
# vis.run()
# vis.destroy_window()
# print(gt_anno)