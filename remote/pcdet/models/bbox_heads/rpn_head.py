import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from ..model_utils.pytorch_utils import Empty, Sequential
from .anchor_target_assigner import AnchorGeneratorRange, TargetAssigner
from ...utils import box_coder_utils, common_utils, loss_utils
from ...config import cfg


class AnchorHead(nn.Module):
    def __init__(self, grid_size, anchor_target_cfg):
        super().__init__()

        anchor_cfg = anchor_target_cfg.ANCHOR_GENERATOR
        anchor_generators = []

        self.num_class = len(cfg.CLASS_NAMES)
        for cur_name in cfg.CLASS_NAMES:
            cur_cfg = None
            for a_cfg in anchor_cfg:
                if a_cfg['class_name'] == cur_name:
                    cur_cfg = a_cfg
                    break
            assert cur_cfg is not None, 'Not found anchor config: %s' % cur_name
            anchor_generator = AnchorGeneratorRange(
                anchor_ranges=cur_cfg['anchor_range'],
                sizes=cur_cfg['sizes'],
                rotations=cur_cfg['rotations'],
                class_name=cur_cfg['class_name'],
                match_threshold=cur_cfg['matched_threshold'],
                unmatch_threshold=cur_cfg['unmatched_threshold']
            )
            anchor_generators.append(anchor_generator)

        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)()

        self.target_assigner = TargetAssigner(
            anchor_generators=anchor_generators,
            pos_fraction=anchor_target_cfg.SAMPLE_POS_FRACTION,
            sample_size=anchor_target_cfg.SAMPLE_SIZE,
            region_similarity_fn_name=anchor_target_cfg.REGION_SIMILARITY_FN,
            box_coder=self.box_coder
        )
        self.num_anchors_per_location = self.target_assigner.num_anchors_per_location
        self.box_code_size = self.box_coder.code_size

        feature_map_size = grid_size[:2] // anchor_target_cfg.DOWNSAMPLED_FACTOR
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = self.target_assigner.generate_anchors(feature_map_size)
        anchors_dict = self.target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret['anchors'].reshape([-1, 7])
        self.anchor_cache = {
            'anchors': anchors,
            'anchors_dict': anchors_dict,
        }

        self.forward_ret_dict = None
        self.build_losses(cfg.MODEL.LOSSES)

    def build_losses(self, losses_cfg):
        # loss function definition
        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']

        rpn_code_weights = code_weights[3:7] if losses_cfg.RPN_REG_LOSS == 'bin-based' else code_weights
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=rpn_code_weights)
        self.dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()

    def assign_targets(self, gt_boxes):
        """
        :param gt_boxes: (B, N, 8)
        :return:
        """
        gt_boxes = gt_boxes.cpu().numpy()
        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, 7]
        gt_boxes = gt_boxes[:, :, :7]
        targets_dict_list = []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]

            cur_gt_classes = gt_classes[k][:cnt + 1]
            cur_gt_names = np.array(cfg.CLASS_NAMES)[cur_gt_classes.astype(np.int32) - 1]
            cur_target_dict = self.target_assigner.assign_v2(
                anchors_dict=self.anchor_cache['anchors_dict'],
                gt_boxes=cur_gt,
                gt_classes=cur_gt_classes,
                gt_names=cur_gt_names
            )
            targets_dict_list.append(cur_target_dict)

        targets_dict = {}
        for key in targets_dict_list[0].keys():
            val = np.stack([x[key] for x in targets_dict_list], axis=0)
            targets_dict[key] = val

        return targets_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim+1]) * torch.cos(boxes2[..., dim:dim+1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim+1]) * torch.sin(boxes2[..., dim:dim+1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim+1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim+1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period_torch(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict
        anchors = forward_ret_dict['anchors']
        box_preds = forward_ret_dict['box_preds']
        cls_preds = forward_ret_dict['cls_preds']

        box_dir_cls_preds = None 
        if 'dir_cls_preds' in forward_ret_dict.keys():
            box_dir_cls_preds = forward_ret_dict['dir_cls_preds']
        
        box_cls_labels = forward_ret_dict['box_cls_labels']
        box_reg_targets = forward_ret_dict['box_reg_targets']
        batch_size = int(box_preds.shape[0])
        
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        # rpn head losses
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        num_class = self.num_class

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), num_class + 1, dtype=box_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        if cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros']:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
            one_hot_targets = one_hot_targets[..., 1:]
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

        loss_weights_dict = loss_cfgs.LOSS_WEIGHTS
        cls_loss = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss_reduced = cls_loss.sum() / batch_size
        cls_loss_reduced = cls_loss_reduced * loss_weights_dict['rpn_cls_weight']

        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location)
        if loss_cfgs.RPN_REG_LOSS == 'smooth-l1':
            if 'SIN_DIFFERENCE' in loss_cfgs.keys() and not loss_cfgs.SIN_DIFFERENCE:
                loc_loss = self.reg_loss_func(box_preds, box_reg_targets, weights=reg_weights)  # [N, M]
            else:
                 # sin(a - b) = sinacosb-cosasinb
                box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
                loc_loss = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

            loc_loss_reduced = loc_loss.sum() / batch_size
        else:
            raise NotImplementedError

        loc_loss_reduced = loc_loss_reduced * loss_weights_dict['rpn_loc_weight']

        rpn_loss = loc_loss_reduced + cls_loss_reduced

        tb_dict = {
            'rpn_loss_loc': loc_loss_reduced.item(),
            'rpn_loss_cls': cls_loss_reduced.item()
        }
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS['dir_offset'],
                num_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins']
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins'])
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['rpn_dir_weight']
            rpn_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


class RPNV2(AnchorHead):
    def __init__(self, num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)
        if 'not_use_init' in args and args['not_use_init']:
            pass
        else:
            self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in, bev=None, **kwargs):
        ups = []
        x = x_in
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)
        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds

        ret_dict['anchors'] = torch.from_numpy(self.anchor_cache['anchors']).cuda()
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )

            ret_dict.update({
                'box_cls_labels': torch.from_numpy(targets_dict['labels']).cuda(),
                'box_reg_targets': torch.from_numpy(targets_dict['bbox_targets']).cuda(),
                'reg_src_targets': torch.from_numpy(targets_dict['bbox_src_targets']).cuda(),
                'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights']).cuda(),
            })

        self.forward_ret_dict = ret_dict
        return ret_dict

class RPNV3(AnchorHead):
    def __init__(self, num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        self.blocks1_first_block = Sequential(
                                    nn.ZeroPad2d(1),
                                    Conv2d(in_filters[0], args['num_filters'][0], 3, stride=args['layer_strides'][0]),
                                    BatchNorm2d(args['num_filters'][0]))
        blocks1 = []
        deblocks1 = []

        for i, layer_num in enumerate(args['layer_nums']):
            if i == 0:
                block = Sequential(
                    Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1),
                    BatchNorm2d(args['num_filters'][i]),
                    nn.ReLU(),
                )

                for j in range(layer_num-1):
                    block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                    block.add(BatchNorm2d(args['num_filters'][i]))
                    block.add(nn.ReLU())
                blocks1.append(block)

            else:
                block = Sequential(
                    nn.ZeroPad2d(1),
                    Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                    BatchNorm2d(args['num_filters'][i]),
                    nn.ReLU(),
                )
                for j in range(layer_num):
                    block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                    block.add(BatchNorm2d(args['num_filters'][i]))
                    block.add(nn.ReLU())
                
                blocks1.append(block)
            1
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks1.append(deblock)
        
        self.blocks1 = nn.ModuleList(blocks1)
        #2
        self.deblocks1 = nn.ModuleList(deblocks1)

        self.rfp_transfoms = Sequential(
            Conv2d(128*3, 128*3, 3, padding=1),
            BatchNorm2d(128*3),
            nn.ReLU(),
            Conv2d(128*3, 128*2, 3, padding=1),
            BatchNorm2d(128*2),
            nn.ReLU(),
            Conv2d(128*2, 128*1, 3, padding=1),
            BatchNorm2d(128*1),
            nn.ReLU(),
            Conv2d(128*1, 64, 3, padding=1),
            BatchNorm2d(64),
            nn.ReLU(),
        )

        self.rfp_weight = torch.nn.Conv2d(
            128*3,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)


        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']
        
        # 3
        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks1.append(deblock)
        

        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)
        if 'not_use_init' in args and args['not_use_init']:
            pass
        else:
            self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in, bev=None, **kwargs):
        ups = []
        x = x_in
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)
                
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)
        
        before_x = x
        #iterate
        rfp_feature = self.rfp_transfoms(x)
        first_block_feature = self.blocks1_first_block(x_in)
        first_block_feature += rfp_feature
        first_block_feature = F.relu(first_block_feature)
        x = first_block_feature

        new_ups = []
        for i in range(len(self.blocks1)):
            x = self.blocks1[i](x)
            #4
            new_ups.append(self.deblocks1[i](x))
            #new_ups.append(self.deblocks[i](x))
        
        new_x = torch.cat(new_ups, dim=1)
        add_weight = torch.sigmoid(self.rfp_weight(new_x))

        final_x = add_weight * new_x + (1 - add_weight) * before_x
        x = final_x

        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds

        ret_dict['anchors'] = torch.from_numpy(self.anchor_cache['anchors']).cuda()
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )

            ret_dict.update({
                'box_cls_labels': torch.from_numpy(targets_dict['labels']).cuda(),
                'box_reg_targets': torch.from_numpy(targets_dict['bbox_targets']).cuda(),
                'reg_src_targets': torch.from_numpy(targets_dict['bbox_src_targets']).cuda(),
                'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights']).cuda(),
            })

        self.forward_ret_dict = ret_dict
        return ret_dict

class RPNV4(AnchorHead):
    def __init__(self, num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)
        
        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)

            deblock = Sequential(
                ConvTranspose2d(
                    128, args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        
        self.P5_1 = nn.Conv2d(256, 128, 1, 1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(128, 128, 1, 1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(128,128,3,1,1)

        self.P3_1 = nn.Conv2d(64,128,1,1)
        self.P3_2 = nn.Conv2d(128,128,3,1,1)


        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        blocks1_first_block = []
        blocks1_first_block.append(Sequential(
                                    nn.ZeroPad2d(1),
                                    Conv2d(in_filters[0], args['num_filters'][0], 3, stride=args['layer_strides'][0]),
                                    BatchNorm2d(args['num_filters'][0])))
        blocks1_first_block.append(Sequential(
                                    nn.ZeroPad2d(1),
                                    Conv2d(in_filters[1], args['num_filters'][1], 3, stride=args['layer_strides'][1]),
                                    BatchNorm2d(args['num_filters'][1])))
        blocks1_first_block.append(Sequential(
                                    nn.ZeroPad2d(1),
                                    Conv2d(in_filters[2], args['num_filters'][2], 3, stride=args['layer_strides'][2]),
                                    BatchNorm2d(args['num_filters'][2])))
        
        self.blocks1_first_block = nn.ModuleList(blocks1_first_block)

        blocks1 = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                    Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1),
                    BatchNorm2d(args['num_filters'][i]),
                    nn.ReLU(),
                )

            for j in range(layer_num-1):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            
            blocks1.append(block)
        
        self.blocks1 = nn.ModuleList(blocks1)

        self.rfp_transfoms = Sequential(
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128),
            nn.ReLU(),
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128),
            nn.ReLU(),
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128),
            nn.ReLU()
        )

        self.dimension_conv1 = nn.Conv2d(128, 64, 1, bias=True)
        self.dimension_conv2 = nn.Conv2d(128, 128, 1, bias=True)
        self.dimension_conv3 = nn.Conv2d(128, 256, 1, bias=True)

        self.rfp_weight = torch.nn.Conv2d(
            128,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)


        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']
        
        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)
        if 'not_use_init' in args and args['not_use_init']:
            pass
        else:
            self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in, bev=None, **kwargs):
        x = x_in
        ret_dict = {}

        backbone_features = []

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            backbone_features.append(x)
        
        C3,C4,C5 = backbone_features

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        ups = [P3_x, P4_x, P5_x]
        
        aspp_ups = [None, None, None]
        for i in range(len(ups)):
            aspp_ups[i] = self.rfp_transfoms(ups[i])
        
        iterative_block = [None, None, None]
        
        f1 = self.blocks1_first_block[0](x_in)
        f1 += self.dimension_conv1(aspp_ups[0])
        f1 = F.relu(f1)
        f1 = self.blocks1[0](f1)
        iterative_block[0] = f1

        f2 = self.blocks1_first_block[1](f1)
        f2 += self.dimension_conv2(aspp_ups[1])
        f2 = F.relu(f2)
        f2 = self.blocks1[1](f2)
        iterative_block[1] = f2
        f3 = self.blocks1_first_block[2](f2)
        f3 += self.dimension_conv3(aspp_ups[2])
        f3 = F.relu(f3)
        f3 = self.blocks1[2](f3)
        iterative_block[2] = f3

        C3,C4,C5 = iterative_block
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        iterative_ups = [P3_x, P4_x, P5_x]

        final_fpn_ups = [None, None, None]
        for i in range(len(iterative_block)):
            add_weight = torch.sigmoid(self.rfp_weight(iterative_ups[i]))
            final_fpn_ups[i] = add_weight * iterative_ups[i] + (1 - add_weight) * ups[i]
        
        for i in range(len(self.deblocks)):
            final_fpn_ups[i] = self.deblocks[i](final_fpn_ups[i])

        x = torch.cat(final_fpn_ups, dim=1)

        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds

        ret_dict['anchors'] = torch.from_numpy(self.anchor_cache['anchors']).cuda()
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )

            ret_dict.update({
                'box_cls_labels': torch.from_numpy(targets_dict['labels']).cuda(),
                'box_reg_targets': torch.from_numpy(targets_dict['bbox_targets']).cuda(),
                'reg_src_targets': torch.from_numpy(targets_dict['bbox_src_targets']).cuda(),
                'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights']).cuda(),
            })

        self.forward_ret_dict = ret_dict
        return ret_dict

class AnchorHeadMultiFeature(nn.Module):
    def __init__(self, anchor_target_cfg):
        super().__init__()

        anchor_cfg = anchor_target_cfg.ANCHOR_GENERATOR
        anchor_generators = []
        
        self.num_class = len(cfg.CLASS_NAMES)
        for cur_name in cfg.CLASS_NAMES:
            cur_cfg = None
            for a_cfg in anchor_cfg:
                if a_cfg['class_name'] == cur_name:
                    cur_cfg = a_cfg
                    break

            assert  cur_cfg is not None, 'Not found anchor config: %s' % cur_name
            anchor_generator = AnchorGeneratorRange(
                anchor_ranges=cur_cfg['anchor_range'],
                sizes=cur_cfg['sizes'],
                rotations=cur_cfg['rotations'],
                class_name=cur_cfg['class_name'],
                match_threshold=cur_cfg['matched_threshold'],
                unmatch_threshold=cur_cfg['unmatched_threshold'],
                feature_map_size=cur_cfg['feature_map_size']
            )
            anchor_generators.append(anchor_generator)
        
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)()
        self.anchor_generators = anchor_generators

        self.target_assigner = TargetAssigner(
            anchor_generators=anchor_generators,
            pos_fraction=anchor_target_cfg.SAMPLE_POS_FRACTION,
            sample_size=anchor_target_cfg.SAMPLE_SIZE,
            region_similarity_fn_name=anchor_target_cfg.REGION_SIMILARITY_FN,
            box_coder=self.box_coder
        )

        self.box_code_size = self.box_coder.code_size

        anchors_dict = self.target_assigner.generate_anchors_dict(None,use_multi_head=True,zl=True)
        self.anchor_cache = {
            'anchors_dict': anchors_dict,
        }

        self.forward_ret_dict = None
        self.build_losses(cfg.MODEL.LOSSES)
    
    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim+1]) * torch.cos(boxes2[..., dim:dim+1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim+1]) * torch.sin(boxes2[..., dim:dim+1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim+1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim+1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period_torch(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets
    
    def build_losses(self, losses_cfg):
        # loss function definition
        if 'official_alpha' in  cfg.MODEL.RPN.RPN_HEAD.ARGS and cfg.MODEL.RPN.RPN_HEAD.ARGS['official_alpha']:
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=torch.tensor((0.75,0.75,0.25)).reshape(1,1,3), gamma=2.0)
        else:
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=code_weights)
        self.dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()
    
    def assign_targets(self, gt_boxes):
        """
        :param gt_boxes: (B, N, 8)
        :return:
        """
        multi_class_predict = False
        if 'multi_class_predict' in  cfg.MODEL.RPN.RPN_HEAD.ARGS and cfg.MODEL.RPN.RPN_HEAD.ARGS['multi_class_predict']:
                    multi_class_predict = True
        
        gt_boxes = gt_boxes.cpu().numpy()
        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, 7]
        gt_boxes = gt_boxes[:, :, :7]
        targets_dict_list = []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]

            cur_gt_classes = gt_classes[k][:cnt + 1]
            cur_gt_names = np.array(cfg.CLASS_NAMES)[cur_gt_classes.astype(np.int32) - 1]
            cur_target_dict = self.target_assigner.assign_v2(
                anchors_dict=self.anchor_cache['anchors_dict'],
                gt_boxes=cur_gt,
                gt_classes=cur_gt_classes, 
                gt_names=cur_gt_names,
                zl=True,
                multi_class_predict=multi_class_predict
            )   

            '''
            原来是:
                每一项都是一个numpy,包含了不同类别的属性
            现在是:
                每一项是个list, 每个元素代表不同类别的该属性
            最后生成的格式:
                属性 : [] : list每个元素是所有batch的不同类别的该属性
            cur_target_dict:
                'labels': [] : len是 不同类别的种类数
                'bbox_targets': [] 
                'bbox_src_targets': [],
                'bbox_outside_weights': []
            '''
            targets_dict_list.append(cur_target_dict)
        
        targets_dict = {}
        keys = list(targets_dict_list[0].keys())
        val_lens = len(targets_dict_list[0][keys[0]])

        for key in keys:
            targets_dict[key] = []
            for idx in range(val_lens):
                val = torch.from_numpy(np.stack([x[key][idx] for x in targets_dict_list], axis=0)).contiguous().cuda()
                targets_dict[key].append(val)
        '''
            返回的的每个属性是个list
            每个元素是一个batch的不同类别的该属性的numpy数组
        '''
        return targets_dict
    
    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict

        total_cls_loss = None
        total_loc_loss = None
        total_dir_loss = None
        tb_dict = {}

        for i, cur_class in enumerate(cfg.CLASS_NAMES):
            anchors = forward_ret_dict['anchors'][cur_class]
            cls_preds = forward_ret_dict[cur_class][0]
            box_preds = forward_ret_dict[cur_class][1]

            box_dir_cls_preds = None
            if len(forward_ret_dict[cur_class]) > 2:
                box_dir_cls_preds = forward_ret_dict[cur_class][2]

            box_cls_labels = forward_ret_dict['labels'][i]
            box_reg_targets = forward_ret_dict['bbox_targets'][i]

            batch_size = int(box_preds.shape[0])
            # print(anchors[0,0,0,0,:])
            # print("name: ", cur_class)
            # print("anchor: ",anchors.shape)
            # print("cls_preds: ",cls_preds.shape)
            # print("box_preds: ",box_preds.shape)
            # print("box_cls_labels: ",box_cls_labels.shape)
            # print("box_reg_targets: ",box_reg_targets.shape)

            anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
            # print("anchor_reshape : ",anchors.shape)
            
            cared = box_cls_labels >= 0
            positives = box_cls_labels > 0
            negative = box_cls_labels == 0
            negative_cls_weights = negative * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
            cls_targets = cls_targets.unsqueeze(dim=-1).float()

            loss_weights_dict = loss_cfgs.LOSS_WEIGHTS

            multi_class_predict = 1
            if 'multi_class_predict' in  cfg.MODEL.RPN.RPN_HEAD.ARGS and cfg.MODEL.RPN.RPN_HEAD.ARGS['multi_class_predict']:
                    multi_class_predict = self.num_class
            
            if multi_class_predict == 1:
                cls_preds = cls_preds.view(batch_size, -1, 1)
                cls_loss = self.cls_loss_func(cls_preds, cls_targets, weights=cls_weights)  # [N, M]
            
                cls_loss_reduced = cls_loss.sum() / batch_size
                cls_loss_reduced = cls_loss_reduced * loss_weights_dict['rpn_cls_weight']
            else:
                cls_targets = cls_targets.squeeze(dim=-1)
                num_class = self.num_class
                one_hot_targets = torch.zeros(
                *list(cls_targets.shape), num_class + 1, dtype=box_preds.dtype, device=cls_targets.device)

                one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

                if cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros']:
                    cls_preds = cls_preds.view(batch_size, -1, num_class)
                    one_hot_targets = one_hot_targets[..., 1:]
                else:
                    cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
                
                cls_loss = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
                cls_loss_reduced = cls_loss.sum() / batch_size
                cls_loss_reduced = cls_loss_reduced * loss_weights_dict['rpn_cls_weight']
            
            box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.anchor_generators[i].num_anchors_per_localization)
            # print("box_preds: ",box_preds.shape)
            # print("box_reg_targets: ",box_reg_targets.shape)
            
            if loss_cfgs.RPN_REG_LOSS == 'smooth-l1':
                coder = cfg.MODEL.RPN.RPN_HEAD.TARGET_CONFIG.BOX_CODER
                if coder != 'ResidualCoder':
                    loc_loss = self.reg_loss_func(box_preds, box_reg_targets, weights=reg_weights)  # [N, M]
                elif coder == 'ResidualCoder':
                    # sin(a - b) = sinacosb-cosasinb
                    box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
                    loc_loss = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M] 
            else:
                raise NotImplementedError

            loc_loss_reduced = loc_loss.sum() / batch_size
            loc_loss_reduced = loc_loss_reduced * loss_weights_dict['rpn_loc_weight']

            if total_cls_loss is None:
                total_cls_loss = cls_loss_reduced
            else:
                total_cls_loss += cls_loss_reduced
            
            if total_loc_loss is None:
                total_loc_loss = loc_loss_reduced
            else:
                total_loc_loss += loc_loss_reduced
            
            if box_dir_cls_preds is not None:
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS['dir_offset'],
                    num_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins']
                )
                dir_logits = box_dir_cls_preds.view(batch_size, -1, cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins'])
                weights = positives.type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * loss_weights_dict['rpn_dir_weight']
                if total_dir_loss is None:
                    total_dir_loss = dir_loss
                else:
                    total_dir_loss += dir_loss
            
            tb_dict.update({
                'rpn_loss_loc_{}'.format(cur_class): loc_loss_reduced.item(),
                'rpn_loss_cls_{}'.format(cur_class): cls_loss_reduced.item(),
                'rpn_loss_{}'.format(cur_class): loc_loss_reduced.item() + cls_loss_reduced.item() + (dir_loss.item() if total_dir_loss is not None else 0)
            })

            if total_dir_loss is not None:
                tb_dict.update({
                    'rpn_loss_dir_{}'.format(cur_class): dir_loss.item()
                })
        # fix bug, name is wrong
        # tb_dict.update({
        #     'rpn_loss_loc': total_cls_loss.item(),
        #     'rpn_loss_cls': total_loc_loss.item(),
        #     'rpn_loss': total_cls_loss.item() + total_loc_loss.item() + (total_dir_loss.item() if total_dir_loss is not None else 0)
        # })
        tb_dict.update({
            'rpn_loss_loc': total_loc_loss.item(),
            'rpn_loss_cls': total_cls_loss.item(),
            'rpn_loss': total_cls_loss.item() + total_loc_loss.item() + (total_dir_loss.item() if total_dir_loss is not None else 0)
        })

        if total_dir_loss is not None:
            tb_dict.update({
                'rpn_loss_dir':total_dir_loss.item()
            })
        if total_dir_loss is not None:
            return total_cls_loss + total_loc_loss + total_dir_loss, tb_dict
        else:
            return total_cls_loss + total_loc_loss, tb_dict

class FPNHead(AnchorHeadMultiFeature):
    def __init__(self, args, anchor_target_cfg,  **kwargs):
        super().__init__(anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)
        
        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        transform_blocks = []
        upsample_blocks = []
        feature_blocks = []
        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)

            transform_block =  Sequential(
                Conv2d(args['num_filters'][i], 128, 1, 1),
                BatchNorm2d(128),
                nn.ReLU(),
            )
            transform_blocks.append(transform_block)

            feature_block = Sequential(
                Conv2d(128, 128, 3, stride=1, padding=1),
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, stride=1, padding=1),
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, stride=1, padding=1),
                BatchNorm2d(128),
                nn.ReLU(),
            )
            feature_blocks.append(feature_block)
        
        for i in range(len(args['layer_nums']) - 1):
            upsample_block = Sequential(
                ConvTranspose2d(128, 128, 2, stride=2),
                BatchNorm2d(128),
                nn.ReLU(),
            )
            upsample_blocks.append(upsample_block)
        
        self.blocks = nn.ModuleList(blocks)
        self.transform_blocks = nn.ModuleList(transform_blocks)
        self.upsample_blocks = nn.ModuleList(upsample_blocks)
        self.feature_blocks = nn.ModuleList(feature_blocks)

        output_blocks = []
        for i in range(len(cfg.CLASS_NAMES)):
            output_block = []
            output_block.append(nn.Conv2d(128, 1*self.target_assigner.num_anchors_per_location_class(i), 1))
            output_block.append(nn.Conv2d(128, self.box_code_size * self.target_assigner.num_anchors_per_location_class(i), 1)) # self.box_coder.code_size
            if self._use_direction_classifier:
                output_block.append(nn.Conv2d(128*1, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(i), 1))
            output_blocks.append(output_block)
        self.output_blocks = output_blocks

        for i in range(len(self.output_blocks)):
            self.output_blocks[i] = nn.ModuleList(self.output_blocks[i])
        self.output_blocks = nn.ModuleList(self.output_blocks)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        for i in range(len(self.output_blocks)):
            nn.init.constant_(self.output_blocks[i][0].bias, -np.log((1 - pi) / pi))

    def forward(self, x, **kwargs):

        block_output = []

        for block in self.blocks:
            x = block(x)
            block_output.append(x)
        
        for i, block in enumerate(self.transform_blocks):
            block_output[i] = self.transform_blocks[i](block_output[i])
        
        output_features = [None] * len(self.feature_blocks)

        for i in range(len(self.feature_blocks) - 1, -1, -1):
            if i == len(self.feature_blocks) - 1:
                output_features[i] = self.feature_blocks[i](block_output[i])
            else:
                block_output[i] = block_output[i] + self.upsample_blocks[i](block_output[i+1])
                output_features[i] = self.feature_blocks[i](block_output[i])

        ret_dict = dict()
        
        if 'Pedestrian' in cfg.CLASS_NAMES:
            ret_dict['Pedestrian'] = []
            ret_dict['Pedestrian'].append(self.output_blocks[-3][0](output_features[-3]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Pedestrian'].append(self.output_blocks[-3][1](output_features[-3]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Pedestrian'].append(self.output_blocks[-3][2](output_features[-3]).permute(0,2,3,1).contiguous())
            
        if 'Cyclist' in cfg.CLASS_NAMES: #cfg.CLASS_NAMES:
            ret_dict['Cyclist'] = []
            ret_dict['Cyclist'].append(self.output_blocks[-2][0](output_features[-2]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Cyclist'].append(self.output_blocks[-2][1](output_features[-2]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Cyclist'].append(self.output_blocks[-2][2](output_features[-2]).permute(0,2,3,1).contiguous())
        
        if 'Car' in cfg.CLASS_NAMES:
            ret_dict['Car'] = []
            ret_dict['Car'].append(self.output_blocks[-1][0](output_features[-1]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Car'].append(self.output_blocks[-1][1](output_features[-1]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Car'].append(self.output_blocks[-1][2](output_features[-1]).permute(0,2,3,1).contiguous())
        
        ret_dict['anchors'] = dict()
        for key in self.anchor_cache['anchors_dict'].keys():
            ret_dict['anchors'][key] = torch.from_numpy(self.anchor_cache['anchors_dict'][key]['anchors']).cuda()
        
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )
            ret_dict.update(targets_dict)
        
        self.forward_ret_dict = ret_dict
      
        return ret_dict     

class FPNHeadV2(AnchorHeadMultiFeature):
    def __init__(self, args, anchor_target_cfg,  **kwargs):
        super().__init__(anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)
        
        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        transform_blocks = []
        upsample_blocks = []
        feature_blocks = []
        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)

            transform_block =  Sequential(
                Conv2d(args['num_filters'][i], 128, 1, 1),
                BatchNorm2d(128),
                nn.ReLU(),
            )
            transform_blocks.append(transform_block)

            feature_block = Sequential(
                Conv2d(128, 128, 3, stride=1, padding=1),
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, stride=1, padding=1),
                BatchNorm2d(128),
                nn.ReLU(),
                Conv2d(128, 128, 3, stride=1, padding=1),
                BatchNorm2d(128),
                nn.ReLU(),
            )
            feature_blocks.append(feature_block)
        
        for i in range(len(args['layer_nums']) - 1):
            upsample_block = Sequential(
                ConvTranspose2d(128, 128, 2, stride=2),
                BatchNorm2d(128),
                nn.ReLU(),
            )
            upsample_blocks.append(upsample_block)
        
        self.blocks = nn.ModuleList(blocks)
        self.transform_blocks = nn.ModuleList(transform_blocks)
        self.upsample_blocks = nn.ModuleList(upsample_blocks)
        self.feature_blocks = nn.ModuleList(feature_blocks)

        output_blocks = []
        for i in range(len(cfg.CLASS_NAMES)):
            output_block = []
            output_block.append(nn.Conv2d(128, 1*self.target_assigner.num_anchors_per_location_class(i), 1))
            output_block.append(nn.Conv2d(128, self.box_code_size * self.target_assigner.num_anchors_per_location_class(i), 1)) # self.box_coder.code_size
            if self._use_direction_classifier:
                output_block.append(nn.Conv2d(128*1, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(i), 1))
            output_blocks.append(output_block)
        self.output_blocks = output_blocks

        for i in range(len(self.output_blocks)):
            self.output_blocks[i] = nn.ModuleList(self.output_blocks[i])
        self.output_blocks = nn.ModuleList(self.output_blocks)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        for i in range(len(self.output_blocks)):
            nn.init.constant_(self.output_blocks[i][0].bias, -np.log((1 - pi) / pi))

    def forward(self, x, **kwargs):

        block_output = []

        for block in self.blocks:
            x = block(x)
            block_output.append(x)
        
        for i, block in enumerate(self.transform_blocks):
            block_output[i] = self.transform_blocks[i](block_output[i])
        
        output_features = [None] * len(self.feature_blocks)

        for i in range(len(self.feature_blocks) - 1, -1, -1):
            if i == len(self.feature_blocks) - 1:
                output_features[i] = self.feature_blocks[i](block_output[i])
            else:
                block_output[i] = block_output[i] + self.upsample_blocks[i](block_output[i+1])
                output_features[i] = self.feature_blocks[i](block_output[i])

        ret_dict = dict()
        
        if 'Pedestrian' in cfg.CLASS_NAMES:
            ret_dict['Pedestrian'] = []
            ret_dict['Pedestrian'].append(self.output_blocks[-3][0](output_features[-3]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Pedestrian'].append(self.output_blocks[-3][1](output_features[-3]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Pedestrian'].append(self.output_blocks[-3][2](output_features[-3]).permute(0,2,3,1).contiguous())
            
        if 'Cyclist' in cfg.CLASS_NAMES: #cfg.CLASS_NAMES:
            ret_dict['Cyclist'] = []
            ret_dict['Cyclist'].append(self.output_blocks[-2][0](output_features[-2]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Cyclist'].append(self.output_blocks[-2][1](output_features[-2]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Cyclist'].append(self.output_blocks[-2][2](output_features[-2]).permute(0,2,3,1).contiguous())
        
        if 'Car' in cfg.CLASS_NAMES:
            ret_dict['Car'] = []
            ret_dict['Car'].append(self.output_blocks[-1][0](output_features[-1]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Car'].append(self.output_blocks[-1][1](output_features[-1]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Car'].append(self.output_blocks[-1][2](output_features[-1]).permute(0,2,3,1).contiguous())
        
        ret_dict['anchors'] = dict()
        for key in self.anchor_cache['anchors_dict'].keys():
            ret_dict['anchors'][key] = torch.from_numpy(self.anchor_cache['anchors_dict'][key]['anchors']).cuda()
        
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )
            ret_dict.update(targets_dict)
        
        self.forward_ret_dict = ret_dict
      
        return ret_dict     

class HVHead(AnchorHeadMultiFeature):
    def __init__(self, args, anchor_target_cfg):
        super().__init__(anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])

        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)
        
        in_filters = [args['num_input_features'], *args['num_filters']]
        assert(len(in_filters) % 2 == 1)
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            if i == 0:
                block = []
                block.append(Sequential(
                    Conv2d(in_filters[0], in_filters[1], 3, stride=args['layer_strides'][0], padding=1),
                    BatchNorm2d(in_filters[1]),
                    nn.ReLU())
                )
                block[0].add(Conv2d(in_filters[1], in_filters[2], 3, padding=1))
                block[0].add(BatchNorm2d(in_filters[2]))
                block[0].add(nn.ReLU())

                for j in range(layer_num - 1):
                    block[0].add(Conv2d(in_filters[2], in_filters[2], 3, padding=1))
                    block[0].add(BatchNorm2d(in_filters[2]))
                    block[0].add(nn.ReLU())
                
                block.append(Sequential(
                    Conv2d(in_filters[2], 128, 1),
                    BatchNorm2d(in_filters[1]),
                    nn.ReLU())
                )

                blocks.append(block)
            else:
                block = []
                block.append(Sequential(
                    Conv2d(in_filters[i * 2], in_filters[i * 2 + 1], kernel_size=3 ,stride=args['layer_strides'][1], padding=1),
                    BatchNorm2d(in_filters[i * 2 + 1]),
                    nn.ReLU())
                )

                block.append(Sequential(
                    Conv2d(in_filters[i * 2 + 1] + in_filters[0], in_filters[i * 2 + 2], 3, padding=1),
                    BatchNorm2d(in_filters[i * 2 + 2]),
                    nn.ReLU())
                )

                for j in range(layer_num - 1):
                    block[1].add(Conv2d(in_filters[i * 2 + 2], in_filters[i * 2 + 2], 3, padding=1))
                    block[1].add(BatchNorm2d(in_filters[i * 2 + 2]))
                    block[1].add(nn.ReLU())
                
                block.append(Sequential(
                    Conv2d(in_filters[i * 2 + 2], 128, 1),
                    BatchNorm2d(in_filters[1]),
                    nn.ReLU())
                )

                blocks.append(block)
        
            if i == len(args['layer_nums']) - 1:
                deblock =  Sequential(
                                    Conv2d(128, 128, 3, padding=1),
                                    BatchNorm2d(128),
                                    nn.ReLU())
                deblock.add(ConvTranspose2d(128, args['num_upsample_filters'][i], args['upsample_strides'][i],
                                            stride=args['upsample_strides'][i]))
                deblock.add(BatchNorm2d(128))
                deblock.add(nn.ReLU())
            else:
                deblock = []
                deblock.append(Sequential(
                    ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                    BatchNorm2d(128),
                    nn.ReLU())
                )

                deblock.append(
                    Sequential(
                        Conv2d(128 * 2, 128, 3, padding=1),
                        BatchNorm2d(128),
                        nn.ReLU())
                )

                deblock[1].add(ConvTranspose2d(128, args['num_upsample_filters'][i], args['upsample_strides'][i],
                                            stride=args['upsample_strides'][i]))
                deblock[1].add(BatchNorm2d(128))
                deblock[1].add(nn.ReLU())

            deblocks.append(deblock)
        
        final_feature_blocks = []
        final_feature_blocks.append(
            Sequential(
                 Conv2d(128*3, 128, 3, stride=2, padding=1),
                 BatchNorm2d(128),
                 nn.ReLU())
        )

        final_feature_blocks.append(
            Sequential(
                 Conv2d(128, 128, 3, stride=2, padding=1),
                 BatchNorm2d(128),
                 nn.ReLU())
        )

        self.blocks = blocks
        self.deblocks = deblocks
        

        for i in range(len(self.blocks)):
            if isinstance(self.blocks[i], list):
                self.blocks[i] = nn.ModuleList(self.blocks[i])
        self.blocks = nn.ModuleList(self.blocks)
        
        for i in range(len(self.deblocks)):
            if isinstance(self.deblocks[i], list):
                self.deblocks[i] = nn.ModuleList(self.deblocks[i])
        self.deblocks = nn.ModuleList(self.deblocks)

        self.final_feature_blocks = nn.ModuleList(final_feature_blocks)

        output_blocks = []
        if len(cfg.CLASS_NAMES) == 1:
            output_block = []
            if args['final_33_conv']:
                output_block.append(nn.Conv2d(128, 1*self.target_assigner.num_anchors_per_location_class(0), kernel_size=3, stride=1, padding=1))
                output_block.append(nn.Conv2d(128, self.box_code_size * self.target_assigner.num_anchors_per_location_class(0), kernel_size=3, stride=1, padding=1)) # self.box_coder.code_size
                if self._use_direction_classifier:
                            output_block.append(nn.Conv2d(128, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(0), kernel_size=3, stride=1, padding=1))
            else:
                output_block.append(nn.Conv2d(128, 1*self.target_assigner.num_anchors_per_location_class(0), 1))
                output_block.append(nn.Conv2d(128, self.box_code_size * self.target_assigner.num_anchors_per_location_class(0), 1)) # self.box_coder.code_size
                if self._use_direction_classifier:
                            output_block.append(nn.Conv2d(128, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(0), 1))
            output_blocks.append(output_block)
        else:
            for i in range(len(self.deblocks)):
                output_block = []
                multi_class_predict = 1
                if 'multi_class_predict' in  cfg.MODEL.RPN.RPN_HEAD.ARGS and cfg.MODEL.RPN.RPN_HEAD.ARGS['multi_class_predict']:
                    multi_class_predict = self.num_class
                
                if i == 0:
                    if args['final_33_conv']:
                        output_block.append(nn.Conv2d(128*3, multi_class_predict*self.target_assigner.num_anchors_per_location_class(i), kernel_size=3, stride=1, padding=1))
                        output_block.append(nn.Conv2d(128*3, self.box_code_size * self.target_assigner.num_anchors_per_location_class(i), kernel_size=3, stride=1, padding=1)) # self.box_coder.code_size
                        if self._use_direction_classifier:
                            output_block.append(nn.Conv2d(128*3, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(i),kernel_size=3, stride=1, padding=1))
                    else:
                        output_block.append(nn.Conv2d(128*3, multi_class_predict*self.target_assigner.num_anchors_per_location_class(i), 1))
                        output_block.append(nn.Conv2d(128*3, self.box_code_size * self.target_assigner.num_anchors_per_location_class(i), 1)) # self.box_coder.code_size
                        if self._use_direction_classifier:
                            output_block.append(nn.Conv2d(128*3, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(i), 1))
                else:
                    if args['final_33_conv']:
                        output_block.append(nn.Conv2d(128, multi_class_predict*self.target_assigner.num_anchors_per_location_class(i), kernel_size=3, stride=1, padding=1))
                        output_block.append(nn.Conv2d(128, self.box_code_size * self.target_assigner.num_anchors_per_location_class(i), kernel_size=3, stride=1, padding=1)) # self.box_coder.code_size
                        if self._use_direction_classifier:
                            output_block.append(nn.Conv2d(128*1, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(i), kernel_size=3, stride=1, padding=1))
                    else:
                        output_block.append(nn.Conv2d(128, multi_class_predict*self.target_assigner.num_anchors_per_location_class(i), 1))
                        output_block.append(nn.Conv2d(128, self.box_code_size * self.target_assigner.num_anchors_per_location_class(i), 1)) # self.box_coder.code_size
                        if self._use_direction_classifier:
                            output_block.append(nn.Conv2d(128*1, args['num_direction_bins']*self.target_assigner.num_anchors_per_location_class(i), 1))
                        
                output_blocks.append(output_block)
        self.output_blocks = output_blocks

        for i in range(len(self.output_blocks)):
            self.output_blocks[i] = nn.ModuleList(self.output_blocks[i])
        self.output_blocks = nn.ModuleList(self.output_blocks)

        self.init_weights()
    
    def init_weights(self):
        pi = 0.01
        for i in range(len(self.output_blocks)):
            nn.init.constant_(self.output_blocks[i][0].bias, -np.log((1 - pi) / pi))
    
    def forward(self, scale_features, **kwargs):
        assert(len(scale_features) == len(self.blocks))
        assert(len(scale_features) == len(self.deblocks))
        
        middle_features = []
        output_features = []

        for i, blocks in enumerate(self.blocks):
            if i == 0:
                middle_features.append(blocks[0](scale_features[i]))
                output_features.append(blocks[1](middle_features[-1]))
            else:
                front_scale_feature = blocks[0](middle_features[-1])
                feature_cat = torch.cat((front_scale_feature, scale_features[i]), dim=1).contiguous()
                middle_features.append(blocks[1](feature_cat))
                output_features.append(blocks[2](middle_features[-1]))
        
        for i, blocks in enumerate(self.deblocks):
            if i == len(self.deblocks) - 1:
                output_features[i] = blocks(output_features[i])
            else:
                back_scale_feature = blocks[0](output_features[i+1])
                feature_cat = torch.cat((output_features[i], back_scale_feature), dim=1).contiguous()
                output_features[i] = blocks[1](feature_cat)
        
        output_feature = torch.cat(output_features, dim=1).contiguous()
        output_features[0] = output_feature
        
        for i, block in enumerate(self.final_feature_blocks):
            output_features[i + 1] = block(output_features[i])
        
        ret_dict = dict()
        
        if 'Pedestrian' in cfg.CLASS_NAMES:
            ret_dict['Pedestrian'] = []
            ret_dict['Pedestrian'].append(self.output_blocks[-3][0](output_features[-3]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Pedestrian'].append(self.output_blocks[-3][1](output_features[-3]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Pedestrian'].append(self.output_blocks[-3][2](output_features[-3]).permute(0,2,3,1).contiguous())
            
        if 'Cyclist' in cfg.CLASS_NAMES: #cfg.CLASS_NAMES:
            ret_dict['Cyclist'] = []
            ret_dict['Cyclist'].append(self.output_blocks[-2][0](output_features[-2]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Cyclist'].append(self.output_blocks[-2][1](output_features[-2]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Cyclist'].append(self.output_blocks[-2][2](output_features[-2]).permute(0,2,3,1).contiguous())
        
        if 'Car' in cfg.CLASS_NAMES:
            ret_dict['Car'] = []
            ret_dict['Car'].append(self.output_blocks[-1][0](output_features[-1]).permute(0,2,3,1).contiguous()) #cls
            ret_dict['Car'].append(self.output_blocks[-1][1](output_features[-1]).permute(0,2,3,1).contiguous()) #reg
            if self._use_direction_classifier:
                ret_dict['Car'].append(self.output_blocks[-1][2](output_features[-1]).permute(0,2,3,1).contiguous())

        # output_features[1] = self.final_feature_blocks[0](output_features[0])
        # ret_dict = dict()

        # if 'Car' in cfg.CLASS_NAMES:
        #     ret_dict['Car'] = []
        #     ret_dict['Car'].append(self.output_blocks[0][0](output_features[1]).permute(0,2,3,1).contiguous()) #cls
        #     ret_dict['Car'].append(self.output_blocks[0][1](output_features[1]).permute(0,2,3,1).contiguous()) #reg
        #     if self._use_direction_classifier:
        #         ret_dict['Car'].append(self.output_blocks[0][2](output_features[1]).permute(0,2,3,1).contiguous())
        
        ret_dict['anchors'] = dict()
        for key in self.anchor_cache['anchors_dict'].keys():
            ret_dict['anchors'][key] = torch.from_numpy(self.anchor_cache['anchors_dict'][key]['anchors']).cuda()
        
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=kwargs['gt_boxes'],
            )
            
            ret_dict.update(targets_dict)
        
        self.forward_ret_dict = ret_dict
      
        return ret_dict