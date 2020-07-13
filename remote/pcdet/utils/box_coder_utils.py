import numpy as np
import torch
from . import common_utils
from . import box_utils
from ..config import cfg
from .box_utils import boxes3d_to_corners2d_lidar, boxes3d_to_corners2d_lidar_torch
import cv2

class ResidualCoder(object):
    def __init__(self, code_size=7):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=-1)

        # need to convert boxes to z-center format
        zg = zg + hg / 2
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 1.6
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
        rt = rg - ra
        cts = [g - a for g, a in zip(cgs, cas)]
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt, *cts], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = np.split(box_encodings, box_ndim, axis=-1)

        # need to convert box_encodings to z-bottom format
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        za = za + ha / 2
        zg = zg + hg / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        za = za + ha / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra

        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def decode_with_head_direction_torch(self, box_preds, anchors, dir_cls_preds,
                                         num_dir_bins, dir_offset, dir_limit_offset, use_binary_dir_classifier=False):
        """
        :param box_preds: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param dir_cls_preds: (batch_size, H, W, num_anchors_per_locations*2)
        :return:
        """
        batch_box_preds = self.decode_torch(box_preds, anchors)

        if dir_cls_preds is not None:
            dir_cls_preds = dir_cls_preds.view(box_preds.shape[0], box_preds.shape[1], -1)
            if use_binary_dir_classifier:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
                opp_labels = (batch_box_preds[..., -1] > 0) ^ dir_labels.byte()
                batch_box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(batch_box_preds),
                    torch.tensor(0.0).type_as(batch_box_preds)
                )
            else:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

                period = (2 * np.pi / num_dir_bins)
                dir_rot = common_utils.limit_period_torch(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                )
                batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_box_preds

class ResidualCoderSINCOS(object):
    def __init__(self, code_size=8):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=-1)

        # need to convert boxes to z-center format
        zg = zg + hg / 2
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 1.6
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
        rt = rg - ra
        cts = [g - a for g, a in zip(cgs, cas)]
        return np.concatenate([xt, yt, zt, wt, lt, ht, np.sin(2*rt), np.cos(2*rt), *cts], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xt, yt, zt, wt, lt, ht, rt_sin, rt_cos, *cts = np.split(box_encodings, box_ndim, axis=-1)

        # need to convert box_encodings to z-bottom format
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        
        rg = 0.5*np.arctan2(rt_sin, rt_cos) + ra
        zg = zg - hg / 2

        cgs = [t + a for t, a in zip(cts, cas)]

        return np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        za = za + ha / 2
        zg = zg + hg / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, wt, lt, ht, torch.sin(2*rt), torch.cos(2*rt),  *cts], dim=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt_sin, rt_cos, *cts = torch.split(box_encodings, 1, dim=-1)

        za = za + ha / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = 0.5 * torch.atan2(rt_sin, rt_cos) + ra

        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def decode_with_head_direction_torch(self, box_preds, anchors, dir_cls_preds,
                                         num_dir_bins, dir_offset, dir_limit_offset, use_binary_dir_classifier=False):
        """
        :param box_preds: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param dir_cls_preds: (batch_size, H, W, num_anchors_per_locations*2)
        :return:
        """
        batch_box_preds = self.decode_torch(box_preds, anchors)

        if dir_cls_preds is not None:
            dir_cls_preds = dir_cls_preds.view(box_preds.shape[0], box_preds.shape[1], -1)
            if use_binary_dir_classifier:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
                opp_labels = (batch_box_preds[..., -1] > 0) ^ dir_labels.byte()
                batch_box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(batch_box_preds),
                    torch.tensor(0.0).type_as(batch_box_preds)
                )
            else:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

                period = (2 * np.pi / num_dir_bins)
                dir_rot = common_utils.limit_period_torch(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                )
                batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_box_preds

class CornerCoder(object):
    def __init__(self, code_size=10):
        super().__init__()
        self.code_size = code_size
        # must use true angle to assign anchor, and calculate 4 corners
        assert  'LIDAR_ASSIGN' in cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG and cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG['LIDAR_ASSIGN']

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        # xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        # xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=-1)
        zg = boxes[:,2]
        hg = boxes[:,5]
        za = anchors[:,2]
        ha = anchors[:,5]

        gt = boxes
        gt_reverse_angle = boxes.copy()
        gt_reverse_angle[:,6] = gt_reverse_angle[:,6] + np.pi
        gt_corners = boxes3d_to_corners2d_lidar(gt, ry_flag=True)
        gt_reverse_corners = boxes3d_to_corners2d_lidar(gt_reverse_angle, ry_flag=True)

        anchor_corners = boxes3d_to_corners2d_lidar(anchors, ry_flag=False)

        # gt_anchor_dist = np.linalg.norm(gt_corners - anchor_corners)
        # gt_reverse_anchor_dist = np.linalg.norm(gt_reverse_corners - anchor_corners)

        # if gt_anchor_dist <= gt_reverse_anchor_dist:
        #     corner_target = gt_corners - anchor_corners
        # else:
        #     corner_target = gt_reverse_corners - anchor_corners

        # fix assign bug
        gt_anchor_dist = np.linalg.norm(gt_corners - anchor_corners, axis=1)
        gt_reverse_anchor_dist = np.linalg.norm(gt_reverse_corners - anchor_corners, axis=1)

        mask = gt_anchor_dist <= gt_reverse_anchor_dist
        mask = np.expand_dims(mask, axis=1)
        corner_target = np.where(mask, gt_corners - anchor_corners, gt_reverse_corners - anchor_corners)

        # corner_target = np.zeros(gt_corners.shape)

        # for i in range(gt_anchor_dist.shape[0]):
        #     if gt_anchor_dist[i] <= gt_reverse_anchor_dist[i]:
        #         corner_target[i] = gt_corners[i] - anchor_corners[i]
        #     else:
        #         corner_target[i] = gt_reverse_corners[i] - anchor_corners[i]
        
        #calculate z center
        # need to convert boxes to z-center format
        if cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG.DIRECT_H_AND_Z:
            h_target = hg
            z_target = zg
        else:
            zg = zg + hg / 2
            za = za + ha / 2

            z_target = zg - za
            h_target = hg - ha

        return np.concatenate([corner_target, np.expand_dims(z_target,1), np.expand_dims(h_target,1)], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        assert box_encodings.shape[1] == 10
        delta_corners = box_encodings[:,:8]
        delta_z = box_encodings[:,8]
        delta_h = box_encodings[:,9]

        # 1. convert anchor to corner
        # (N,8) (x0, y0, x1, y1, x2, y2, x3, y3)
        anchor_corners = boxes3d_to_corners2d_lidar(anchors, ry_flag=False)

        za = anchors[:,2]
        ha = anchors[:,5]

        if cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG.DIRECT_H_AND_Z:
            hg = delta_h
            zg = delta_z
        else:
            hg = delta_h + ha
            # need to convert box_encodings to z-bottom format
            za = za + ha / 2
            zg = za + delta_z
            # z center in z-bottom format
            zg = zg - hg / 2
        
        # convert to corner to x,y,w,l,r format
        # np.pi * 0.5 + theta is the 
        anchor_corners += delta_corners
        anchor_corners = anchor_corners.reshape(-1,4,2)

        objects = np.zeros((anchor_corners.shape[0], 5))
        for i in range(anchor_corners.shape[0]):
            ret = cv2.minAreaRect(anchor_corners[i])
            objects[i][0] = ret[0][0]
            objects[i][1] = ret[0][1]
            if ret[1][0] < ret[1][1]:
                #先碰到的是width
                objects[i][2] = ret[1][0]
                objects[i][3] = ret[1][1]
                objects[i][4] = ret[2] * np.pi / 180.
            else:
                #先碰到的是length
                objects[i][2] = ret[1][1]
                objects[i][3] = ret[1][0]
                objects[i][4] = ret[2] * np.pi / 180. + np.pi * 0.5
        #already in camera coordinate angle

        return np.concatenate([objects[:,:2], np.expand_dims(zg,axis=1), objects[:,2:4], np.expand_dims(hg, axis=1), np.expand_dims(objects[:,4],axis=1)], axis=-1)

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """

        box_ndim = anchors.shape[-1]
        # xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        # xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=-1)
        zg = boxes[:,2]
        hg = boxes[:,5]
        za = anchors[:,2]
        ha = anchors[:,5]

        gt = boxes
        gt_reverse_angle = boxes.clone()
        gt_reverse_angle[:,6] = gt_reverse_angle[:,6] + np.pi
        gt_corners = boxes3d_to_corners2d_lidar_torch(gt, ry_flag=True)
        gt_reverse_corners = boxes3d_to_corners2d_lidar_torch(gt_reverse_angle, ry_flag=True)

        anchor_corners = boxes3d_to_corners2d_lidar_torch(anchors, ry_flag=False)

        gt_anchor_dist = torch.norm(gt_corners - anchor_corners, dim=1)          
        gt_reverse_anchor_dist = torch.norm(gt_reverse_angle - anchor_corners, dim=1)

        corner_target = gt_corners.new_empty(gt_corners.shape)

        for i in range(gt_anchor_dist.shape[0]):
            if gt_anchor_dist[i] <= gt_reverse_anchor_dist[i]:
                corner_target[i] = gt_corners[i] - anchor_corners[i]
            else:
                corner_target[i] = gt_reverse_corners[i] - anchor_corners[i]
        
        #calculate z center
        # need to convert boxes to z-center format
        if cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG.DIRECT_H_AND_Z:
            h_target = hg
            z_target = zg
        else:
            zg = zg + hg / 2
            za = za + ha / 2

            z_target = zg - za
            h_target = hg - ha

        return torch.cat([corner_target, z_target.unsqueeze(dim=1), h_target.unsqueeze(dim=1)], axis=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        cuda_flag = False
        if box_encodings.is_cuda:
            cuda_flag = True
        box_encodings = box_encodings.cpu().numpy()
        anchors = anchors.cpu().numpy()

        batch_size = box_encodings.shape[0]
        box_encodings_last_dim = box_encodings.shape[2]
        anchors_last_dim = anchors.shape[2]

        box_encodings = box_encodings.reshape(-1, box_encodings_last_dim)
        anchors = anchors.reshape(-1, anchors_last_dim)

        assert box_encodings.shape[1] == 10
        delta_corners = box_encodings[:,:8]
        delta_z = box_encodings[:,8]
        delta_h = box_encodings[:,9]

        # 1. convert anchor to corner
        # (N,8) (x0, y0, x1, y1, x2, y2, x3, y3)
        anchor_corners = boxes3d_to_corners2d_lidar(anchors, ry_flag=False)

        za = anchors[:,2]
        ha = anchors[:,5]

        if cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG.DIRECT_H_AND_Z:
            hg = delta_h
            zg = delta_z
        else:
            hg = delta_h + ha
            # need to convert box_encodings to z-bottom format
            za = za + ha / 2
            zg = za + delta_z
            # z center in z-bottom format
            zg = zg - hg / 2

        # convert to corner to x,y,w,l,r format
        # np.pi * 0.5 + theta is the 
        anchor_corners += delta_corners
        anchor_corners = anchor_corners.reshape(-1,4,2)

        objects = np.zeros((anchor_corners.shape[0], 5))
        for i in range(anchor_corners.shape[0]):
            ret = cv2.minAreaRect(anchor_corners[i])
            objects[i][0] = ret[0][0]
            objects[i][1] = ret[0][1]
            if ret[1][0] < ret[1][1]:
                #先碰到的是width
                objects[i][2] = ret[1][0]
                objects[i][3] = ret[1][1]
                objects[i][4] = ret[2] * np.pi / 180.
            else:
                #先碰到的是length
                objects[i][2] = ret[1][1]
                objects[i][3] = ret[1][0]
                objects[i][4] = ret[2] * np.pi / 180. + np.pi * 0.5
        #already in camera coordinate angle

        ret = np.concatenate([objects[:,:2], np.expand_dims(zg,axis=1), objects[:,2:4], np.expand_dims(hg, axis=1), np.expand_dims(objects[:,4],axis=1)], axis=-1)
        ret = np.array(ret,dtype=np.float32)
        ret = torch.from_numpy(ret)

        ret = ret.reshape(batch_size, -1, anchors_last_dim)

        if cuda_flag:
            ret = ret.cuda()
        return ret

    def decode_with_head_direction_torch(self, box_preds, anchors, dir_cls_preds,
                                         num_dir_bins, dir_offset, dir_limit_offset, use_binary_dir_classifier=False):
        """
        :param box_preds: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param dir_cls_preds: (batch_size, H, W, num_anchors_per_locations*2)
        :return:
        """
        batch_box_preds = self.decode_torch(box_preds, anchors)

        if dir_cls_preds is not None:
            dir_cls_preds = dir_cls_preds.view(box_preds.shape[0], box_preds.shape[1], -1)
            if use_binary_dir_classifier:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
                opp_labels = (batch_box_preds[..., -1] > 0) ^ dir_labels.byte()
                batch_box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(batch_box_preds),
                    torch.tensor(0.0).type_as(batch_box_preds)
                )
            else:
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

                period = (2 * np.pi / num_dir_bins)
                dir_rot = common_utils.limit_period_torch(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                )
                batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_box_preds


if __name__ == '__main__':
    pass
