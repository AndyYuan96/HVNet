import torch
import torch.nn as nn
import torch.nn.functional as F
from ...config import cfg
from ..model_utils.pytorch_utils import Empty
import math
from scatter_max import scatter_max,scatter_mean
from functools import partial
import skimage as Img
import numpy as np
import matplotlib.pyplot as plt

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

def dense(batch_size, spatial_shape, feature_dim, indices, features, channels_first=True):
    output_shape = [batch_size] + list(spatial_shape) + [feature_dim]
    res = scatter_nd(indices.long(), features, output_shape)
    if not channels_first:
        return res
    ndim = len(spatial_shape)
    trans_params = list(range(0, ndim + 1))
    trans_params.insert(1, ndim + 1)
    return res.permute(*trans_params).contiguous()

class ScatterMaxCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_index, output, output_index): 
        scatter_max(input, input_index, output, output_index, True)
        ctx.size = input.size()
        ctx.save_for_backward(output_index)
        # m1 = output_index.max()
        # m2 = output_index.min()
        # print("test grad")
        # print("input shape: ", input.shape)
        # print("max : ", m1)
        # print("min : ", m2)
        # print("points counts : ", ctx.size[0])
        # if m2 < 0 or m1 >= ctx.size[0]:
        #     print("do again")
        #     scatter_max(input, input_index, output, output_index, False)
        #     print("max : ", output_index.max())
        #     print("min : ", output_index.min())
        #     input.cpu().numpy().tofile("/root/input.bin")
        #     input_index.cpu().numpy().tofile("/root/input_index.bin")
        #     output.cpu().numpy().tofile("/root/output.bin")
        #     output_index.cpu().numpy().tofile("/root/output_index.bin")
        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        output_index = ctx.saved_tensors[0]
        grad_input = output_grad.new_zeros(ctx.size)
        grad_input.scatter_(0, output_index, output_grad)
       
        return grad_input, None, None, None

def scatterMax(input, input_index, voxel_nums, train):
    '''
        only accept two dimension tensor, and do maxpooing in first dimension
    '''
    output = input.new_full((voxel_nums, input.shape[1]), torch.finfo(input.dtype).min)
    output_index = input_index.new_empty((voxel_nums, input.shape[1]))
    
    if train:
        output = ScatterMaxCuda.apply(input, input_index, output, output_index)
    else:
        output = scatter_max(input, input_index, output, output_index, False)
    
    return output

# def scatterMean(input, input_index, voxel_nums):
#     output = input.new_full((voxel_nums, input.shape[1]), 0.0)
#     input_mean = input.new_empty(input.shape)

#     scatter_mean(input, input_index, output,input_mean)
#     return input_mean
class ScatterMeanCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_index, output, input_means, voxel_nums):
        scatter_mean(input, input_index, output, input_means)
        ctx.size = torch.Size([input.size()[0], input.size()[1], voxel_nums])
        ctx.save_for_backward(input_index)
        return input_means
    
    @staticmethod
    def backward(ctx, output_grad):
        input_index = ctx.saved_tensors[0]
        grad_input = output_grad.new_full(ctx.size[:2], 0.0)
        grad_input_tmp = output_grad.new_full((ctx.size[2],output_grad.shape[1]),0.0)
        tt = output_grad.new_full(output_grad.size(), 1.0)
        # must use clone, otherwise wrong
        tt = output_grad.clone()
        scatter_mean(tt, input_index, grad_input_tmp, grad_input)
        print("does bp")
        # print("output_grad")
        # print(output_grad)
        # print("input_index")
        # print(input_index)
        # print("grad_input_tmp")
        # print(grad_input_tmp)
        # print("grad_input")
        # print(grad_input)
        return grad_input, None, None, None,None

def scatterMean(input, input_index, voxel_nums):
    output = input.new_full((voxel_nums, input.shape[1]), 0.0)
    input_means = input.new_empty(input.shape)
    
    input_mean = ScatterMeanCuda.apply(input, input_index, output, input_means, voxel_nums)

    return input_mean

def scatterMeanCpu(input, input_index, voxel_nums):
    sums = input.new_full((voxel_nums, input.shape[1]), 0.0)
    counts = input.new_full((voxel_nums,1), 0.0)
    input_means = input.new_empty(input.shape)

    for i in range(input.shape[0]):
            sums[input_index[i]] += input[i]
            counts[input_index[i]] += 1.0
    
    sums /= counts
    for i in range(input.shape[0]):
        input_means[i] = sums[input_index[i]]
    
    return input_means





class VoxelFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError


class MeanVoxelFeatureExtractor(VoxelFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        return cfg.DATA_CONFIG.NUM_POINT_FEATURES['use']

    def forward(self, features, num_voxels, **kwargs):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param kwargs:
        :return:
        """
        points_mean = features[:, :, :].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            self.linear = nn.Linear(in_channels, self.units, bias=False)
            self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, self.units, bias=True)
            self.norm = Empty(self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        # x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        total_points, voxel_points, channels = x.shape
        x = self.norm(x.view(-1, channels)).view(total_points, voxel_points, channels)
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNetOld2(VoxelFeatureExtractor):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetOld2'
        assert len(num_filters) > 0
        num_input_features += 6
        if with_distance:
            num_input_features += 1
        self.with_distance = with_distance
        self.num_filters = num_filters
        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, features, num_voxels, coords):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param coors:
        :return:
        """
        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :3]
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (coords[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coords[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 2] = features[:, :, 2] - (coords[:, 1].to(dtype).unsqueeze(1) * self.vz + self.z_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
               
        return features.squeeze()

class MVFFeatureNetDVP(VoxelFeatureExtractor):
    def __init__(self,
                bev_h, bev_w):
        super().__init__()
        self.name = 'MVFFeatureNetDVP'
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_size = self.bev_h * self.bev_w

        self.bev_FC = nn.Sequential(
                nn.Linear(10,64,bias=False),
                nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, input_dict):
        batch_size = input_dict['batch_size']
        bev_coordinate = input_dict['bev_coordinate']
        bev_local_coordinate = input_dict['bev_local_coordinate']
        intensity = input_dict['intensity']
        bev_mapping_pv = input_dict['bev_mapping_pv']
        # throw z position
        bev_mapping_vf = input_dict['bev_mapping_vf'][:,:3].contiguous()

        point_mean = scatterMean(bev_coordinate, bev_mapping_pv, bev_mapping_vf.shape[0])
        feature = torch.cat((bev_coordinate, intensity.unsqueeze(1), (bev_coordinate - point_mean), bev_local_coordinate),dim=1).contiguous()

        bev_fc_output = self.bev_FC(feature)
        bev_maxpool = scatterMax(bev_fc_output, bev_mapping_pv, bev_mapping_vf.shape[0], True)
        bev_dense = dense(batch_size, [self.bev_h, self.bev_w], 64, bev_mapping_vf, bev_maxpool)

        return bev_dense


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.deconv0 = self._make_layer(block, 64, layers[0])
        self.conv1 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv2 = self._make_layer(block, 128, layers[2], stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)

        self.deconv1 =  nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                      nn.BatchNorm2d(64,eps=1e-3,momentum=0.01),
                                      nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=4, output_padding=1),
                                      nn.BatchNorm2d(64,eps=1e-3,momentum=0.01),
                                      nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 3, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(64,eps=1e-3,momentum=0.01),
                                   nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-3, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        deconv0_x = self.deconv0(x)

        conv1_x = self.conv1(x)
        conv2_x = self.conv2(conv1_x)

        deconv1_x = self.deconv1(conv1_x)
        deconv2_x = self.deconv2(conv2_x)

        final_x = torch.cat((deconv0_x,deconv1_x,deconv2_x),dim=1)
        final_x = self.conv3(final_x)

        return final_x


class MVFFeatureNet(VoxelFeatureExtractor):
    def __init__(self,
                bev_h, bev_w,
                fv_h, fv_w,
                with_tower):
        super().__init__()
        self.name = 'MVFFeatureNet'
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fv_h = fv_h
        self.fv_w = fv_w
        self.with_tower = with_tower

        self.bev_size = self.bev_h * self.bev_w
        self.fv_size = self.fv_h * self.fv_w

        self.shared_FC = nn.Sequential(
            nn.Linear(7, 128, bias=False),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.bev_FC = nn.Sequential(
            nn.Linear(3,64,bias=False),
            nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.fv_FC = nn.Sequential(
            nn.Linear(3,64,bias=False),
            nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.downsample_FC = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.fv_tower = None
        self.bev_tower = None
        
        if self.with_tower:
            self.fv_tower = ResNet(BasicBlock,[1,1,1])
            self.bev_tower = ResNet(BasicBlock,[1,1,1])
    
    def forward(self, input_dict):
        batch_size = input_dict['batch_size']
        
        bev_local_coordinate = input_dict['bev_local_coordinate']
        fv_local_coordiante = input_dict['fv_local_coordinate']
        intensity = input_dict['intensity']
        bev_mapping_pv = input_dict['bev_mapping_pv']
        bev_mapping_vf = input_dict['bev_mapping_vf']
        fv_mapping_pv = input_dict['fv_mapping_pv']
        fv_mapping_vf = input_dict['fv_mapping_vf']

        bev_fc_output = self.bev_FC(bev_local_coordinate)
        bev_maxpool = scatterMax(bev_fc_output, bev_mapping_pv, bev_mapping_vf.shape[0], True)
        
        fv_fc_output = self.fv_FC(fv_local_coordiante)
        fv_maxpool = scatterMax(fv_fc_output, fv_mapping_pv, fv_mapping_vf.shape[0], True)

        shared_fc_input = torch.cat((bev_local_coordinate, fv_local_coordiante, intensity.unsqueeze(1)), dim=1)
        shared_fc_output = self.shared_FC(shared_fc_input)

        bev_dense = dense(batch_size, [self.bev_h, self.bev_w], 64, bev_mapping_vf, bev_maxpool)
        fv_dense = dense(batch_size, [self.fv_h, self.fv_w], 64, fv_mapping_vf, fv_maxpool)

        bev_feature = None
        fv_feature = None

        if self.with_tower:
            bev_feature = self.bev_tower(bev_dense)
            fv_feature = self.fv_tower(fv_dense)
        else:
            bev_feature = bev_dense
            fv_feature = fv_dense
        
        # to (batch, h, w, c)
        fv_feature = fv_feature.permute(0,2,3,1).reshape(-1,64).contiguous()
        bev_feature = bev_feature.permute(0,2,3,1).reshape(-1,64).contiguous()
        
        # get each voxel's position in feature map
        # and then scatter those voxel
        bev_voxel_coordiante = bev_mapping_vf[:,0] * self.bev_size + bev_mapping_vf[:, 1] * self.bev_w + bev_mapping_vf[:, 2]
        fv_voxel_coordiante = fv_mapping_vf[:,0] * self.fv_size + fv_mapping_vf[:,1] * self.fv_w + fv_mapping_vf[:,2]
        
        # (64,M)
        bev_voxel_feature = torch.index_select(bev_feature,0,bev_voxel_coordiante)
        fv_voxel_feature = torch.index_select(fv_feature, 0, fv_voxel_coordiante)
        
        #bev_voxel_feature is (n1+n2+n3,3), bev_mapping_pv is (id + n1+id + n1+n2+id)
        bev_point_feature = torch.index_select(bev_voxel_feature, 0, bev_mapping_pv)
        fv_point_feature = torch.index_select(fv_voxel_feature, 0, fv_mapping_pv)

        final_point_feature = torch.cat((shared_fc_output, bev_point_feature, fv_point_feature),dim=1).contiguous()

        voxel_feature = scatterMax(final_point_feature, bev_mapping_pv, bev_mapping_vf.shape[0], True)
        
        final_voxel_feature = self.downsample_FC(voxel_feature)
        final_voxel_feature = dense(batch_size, [self.bev_h, self.bev_w], 64, bev_mapping_vf, final_voxel_feature)

        return final_voxel_feature

class HVFeatureNet(VoxelFeatureExtractor):
    def __init__(self,
                 bev_sizes,
                 use_norm=False,
                 input_scale_nums=3,
                 attention_knowledge_dim=8,
                 AVFE_feature_dim=64,
                 AVFEO_feature_dim=128):
        super().__init__()
        self.name = 'HVFeatureNet'      
        self.bev_sizes = bev_sizes

        if use_norm:
            BatchNorm1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            BatchNorm1d = Empty
        
        self.input_scale_nums = input_scale_nums
        self.output_scale_nums = len(bev_sizes) - self.input_scale_nums
        self.AVFE_feature_dim = AVFE_feature_dim
        self.attention_knowledge_dim = attention_knowledge_dim
        self.AVFEO_feature_dim = AVFEO_feature_dim

        self.AVFE_point_feature_fc = nn.Sequential(
                                                    nn.Linear(4,self.AVFE_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFE_feature_dim),
                                                    nn.ReLU())
        self.AVFE_Attention_feature_fc = nn.Sequential(
                                                    nn.Linear(self.attention_knowledge_dim,self.AVFE_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFE_feature_dim),
                                                    nn.ReLU())
        self.AVFEO_point_feature_fc = nn.Sequential(
                                                    nn.Linear(self.input_scale_nums * 2 * self.AVFE_feature_dim,self.AVFEO_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFEO_feature_dim),
                                                    nn.ReLU())
        
        self.AVFEO_Attention_feature_fc = nn.Sequential(
                                                    nn.Linear(self.attention_knowledge_dim,self.AVFEO_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFEO_feature_dim),
                                                    nn.ReLU())

    
    def forward(self, input_dict):

        batch_size = input_dict['batch_size']
        points = input_dict['bev_points']
        bev_mapping_pvs = input_dict['bev_mapping_pvs']
        bev_mapping_vfs = input_dict['bev_mapping_vfs']

        AVFE_features = []
        for i in range(self.input_scale_nums):
            point_feature_fc = self.AVFE_point_feature_fc(points)
            point_mean = scatterMean(points, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
            attention_feature_1 = points[:, :3] - point_mean[:, :3]
            attention_feature_2 = points[:, 3]
            attention_feature_3 = point_mean
            attention_feature = torch.cat((attention_feature_1, attention_feature_2.reshape(-1, 1), attention_feature_3), dim=1).contiguous()
            attention_feature_fc = self.AVFE_Attention_feature_fc(attention_feature)

            scatter_feature = point_feature_fc * attention_feature_fc
            voxel_feature = scatterMax(scatter_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
           
            point_voxel_feature = torch.index_select(voxel_feature, 0, bev_mapping_pvs[i])
            # origin
            # AVFE_features.append(torch.cat((attention_feature_fc, point_voxel_feature), dim=1).contiguous())
            AVFE_features.append(torch.cat((scatter_feature, point_voxel_feature), dim=1))
        
        final_AVFE_feature = torch.cat(AVFE_features, dim=1).contiguous()

        AVFEO_features = []
        for i in range(self.input_scale_nums, self.input_scale_nums + self.output_scale_nums):
            point_feature_fc = self.AVFEO_point_feature_fc(final_AVFE_feature)

            point_mean = scatterMean(points, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
            attention_feature_1 = points[:, :3] - point_mean[:, :3]
            attention_feature_2 = points[:, 3]
            attention_feature_3 = point_mean
            attention_feature = torch.cat((attention_feature_1, attention_feature_2.reshape(-1, 1), attention_feature_3), dim=1).contiguous()
            attention_feature_fc = self.AVFEO_Attention_feature_fc(attention_feature)

            scatter_feature = point_feature_fc * attention_feature_fc
            voxel_feature = scatterMax(scatter_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
           
            AVFEO_features.append(dense(batch_size, [self.bev_sizes[i][0], self.bev_sizes[i][1]], self.AVFEO_feature_dim, bev_mapping_vfs[i], voxel_feature))
        
        return AVFEO_features


class HVFeatureNetPaper(VoxelFeatureExtractor):
    def __init__(self,
                 bev_sizes,
                 bev_range,
                 use_norm=False,
                 input_scale_nums=3,
                 point_feature_dim=9,
                 AVFE_feature_dim=64,
                 AVFEO_feature_dim=128):
        super().__init__()
        self.name = 'HVFeatureNetPaper'      
        self.bev_sizes = bev_sizes
        self.bev_range = bev_range
        self.x_range = [self.bev_range[0], self.bev_range[1]]
        self.y_range = [self.bev_range[2], self.bev_range[3]]

        if use_norm:
            BatchNorm1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            BatchNorm1d = Empty
        
        self.input_scale_nums = input_scale_nums
        self.output_scale_nums = len(bev_sizes) - self.input_scale_nums
        self.AVFE_feature_dim = AVFE_feature_dim
        self.AVFEO_feature_dim = AVFEO_feature_dim

        self.AVFE_point_feature_fc = nn.Sequential(
                                                    nn.Linear(point_feature_dim,self.AVFE_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFE_feature_dim),
                                                    nn.ReLU())
        self.AVFE_Attention_feature_fc = nn.Sequential(
                                                    nn.Linear(point_feature_dim*2,self.AVFE_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFE_feature_dim),
                                                    nn.ReLU())
        self.AVFEO_point_feature_fc = nn.Sequential(
                                                    nn.Linear(self.input_scale_nums * 2 * self.AVFE_feature_dim,self.AVFEO_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFEO_feature_dim),
                                                    nn.ReLU())
        
        self.AVFEO_Attention_feature_fc = nn.Sequential(
                                                    nn.Linear(point_feature_dim*2,self.AVFEO_feature_dim,bias=False),
                                                    BatchNorm1d(self.AVFEO_feature_dim),
                                                    nn.ReLU())

    def forward(self, input_dict):

        batch_size = input_dict['batch_size']
        points = input_dict['bev_points']
        bev_mapping_pvs = input_dict['bev_mapping_pvs']
        bev_mapping_vfs = input_dict['bev_mapping_vfs']

        AVFE_features = []
        for i in range(self.input_scale_nums):
            point_mean = scatterMean(points, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
            feature1 = points[:, :3] - point_mean[:, :3]
            feature2 = points[:, 3]
            feature3 = point_mean

            cur_bev_size = self.bev_sizes[i] # (y,x)
            cur_voxel_size_x = (self.bev_range[1] - self.bev_range[0]) / cur_bev_size[1]
            cur_voxel_size_y = (self.bev_range[3] - self.bev_range[2]) / cur_bev_size[0]

            #(N,3) -> batch,y,x
            point_voxel_coordinates = torch.index_select(bev_mapping_vfs[i], 0, bev_mapping_pvs[i]).float()
            #throw batch,   y,x
            point_voxel_coordinates = point_voxel_coordinates[:,1:]
            point_voxel_coordinates[:,0] = point_voxel_coordinates[:, 0] * cur_voxel_size_y + self.bev_range[2] + cur_voxel_size_y * 0.5 
            point_voxel_coordinates[:,1] = point_voxel_coordinates[:, 1] * cur_voxel_size_x + self.bev_range[0] + cur_voxel_size_x * 0.5

            point_voxel_local_coordinate_y = points[:,1] - point_voxel_coordinates[:, 0]
            point_voxel_local_coordinate_x = points[:,0] - point_voxel_coordinates[:, 1]

            final_point_feature = torch.cat((points, feature1, point_voxel_local_coordinate_x.unsqueeze(1), point_voxel_local_coordinate_y.unsqueeze(1)), dim=1).contiguous()
            final_attention_feature = torch.cat((feature1, final_point_feature[:,3:], point_mean, torch.zeros((feature1.shape[0],3)).to(feature1.device), (point_mean[:,0] - point_voxel_coordinates[:, 1]).unsqueeze(dim=1), (point_mean[:,1] - point_voxel_coordinates[:, 0]).unsqueeze(1)),dim=1).contiguous()

            point_feature_fc = self.AVFE_point_feature_fc(final_point_feature)            
            attention_feature_fc = self.AVFE_Attention_feature_fc(final_attention_feature)

            scatter_feature = point_feature_fc * attention_feature_fc
            voxel_feature = scatterMax(scatter_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)

            point_voxel_feature = torch.index_select(voxel_feature, 0, bev_mapping_pvs[i])
            #1
            #AVFE_features.append(torch.cat((attention_feature_fc, point_voxel_feature), dim=1).contiguous())
            #now
            AVFE_features.append(torch.cat((scatter_feature, point_voxel_feature), dim=1))
        
        final_AVFE_feature = torch.cat(AVFE_features, dim=1).contiguous()

        AVFEO_features = []
        for i in range(self.input_scale_nums, self.input_scale_nums + self.output_scale_nums):
            point_feature_fc = self.AVFEO_point_feature_fc(final_AVFE_feature)
            point_mean = scatterMean(points, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
            feature1 = points[:, :3] - point_mean[:, :3]
            feature2 = points[:, 3]
            feature3 = point_mean

            cur_bev_size = self.bev_sizes[i] # (y,x)
            cur_voxel_size_x = (self.bev_range[1] - self.bev_range[0]) / cur_bev_size[1]
            cur_voxel_size_y = (self.bev_range[3] - self.bev_range[2]) / cur_bev_size[0]

            #(N,3) -> batch,y,x
            point_voxel_coordinates = torch.index_select(bev_mapping_vfs[i], 0, bev_mapping_pvs[i]).float()
            #throw batch,   y,x
            point_voxel_coordinates = point_voxel_coordinates[:,1:]
            point_voxel_coordinates[:,0] = point_voxel_coordinates[:, 0] * cur_voxel_size_y + self.bev_range[2] + cur_voxel_size_y * 0.5 
            point_voxel_coordinates[:,1] = point_voxel_coordinates[:, 1] * cur_voxel_size_x + self.bev_range[0] + cur_voxel_size_x * 0.5

            point_voxel_local_coordiante_y = points[:,1] - point_voxel_coordinates[:, 0]
            point_voxel_local_coordinate_x = points[:,0] - point_voxel_coordinates[:, 1]

            final_attention_feature = torch.cat((feature1, feature2.unsqueeze(1), feature1, point_voxel_local_coordinate_x.unsqueeze(1), point_voxel_local_coordiante_y.unsqueeze(1), point_mean, torch.zeros((feature1.shape[0],3)).to(feature1.device), 
                                                (point_mean[:,0] - point_voxel_coordinates[:, 1]).unsqueeze(1), (point_mean[:,1] - point_voxel_coordinates[:, 0]).unsqueeze(1)),dim=1).contiguous()
            
            attention_feature_fc = self.AVFEO_Attention_feature_fc(final_attention_feature)

            scatter_feature = point_feature_fc * attention_feature_fc
            voxel_feature = scatterMax(scatter_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
           
            AVFEO_features.append(dense(batch_size, [self.bev_sizes[i][0], self.bev_sizes[i][1]], self.AVFEO_feature_dim, bev_mapping_vfs[i], voxel_feature))
        
        return AVFEO_features

class HVFeatureNetFinal(VoxelFeatureExtractor):
    def __init__(self,
                 bev_sizes,
                 bev_range,
                 use_attention=True,
                 use_norm_point=False,
                 use_norm_attention=False,
                 use_norm_mid=False,
                 use_mid_relu=False,
                 use_local_z=False,
                 use_sigmoid=False,
                 use_point_relu=False,
                 use_attention_relu=False,
                 input_scale_nums=3,
                 point_feature_dim=9,
                 AVFE_feature_dim=64,
                 AVFEO_feature_dim=128):
        super().__init__()
        self.name = 'HVFeatureNetFinal'      
        self.bev_sizes = bev_sizes
        self.bev_range = bev_range
        self.use_attention = use_attention

        self.use_local_z = use_local_z
        if use_norm_point:
            BatchNorm1d_point = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            BatchNorm1d_point = Empty
        
        if use_norm_attention:
            BatchNorm1d_attention = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            BatchNorm1d_attention = Empty
        
        if use_norm_mid:
            BatchNorm1d_mid = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            BatchNorm1d_mid = Empty
        
        self.use_mid_relu = use_mid_relu
        self.use_sigmoid = use_sigmoid

        self.input_scale_nums = input_scale_nums
        self.output_scale_nums = len(bev_sizes) - self.input_scale_nums
        self.AVFE_feature_dim = AVFE_feature_dim
        self.AVFEO_feature_dim = AVFEO_feature_dim

        self.AVFE_point_feature_fc = nn.Sequential(
                                                    nn.Linear(point_feature_dim,self.AVFE_feature_dim,bias=False),
                                                    BatchNorm1d_point(self.AVFE_feature_dim),
                                                    nn.ReLU() if use_point_relu else Empty())
       
        self.AVFEO_point_feature_fc = nn.Sequential(
                                                    nn.Linear(self.input_scale_nums * 2 * self.AVFE_feature_dim,self.AVFEO_feature_dim,bias=False),
                                                    BatchNorm1d_point(self.AVFEO_feature_dim),
                                                    nn.ReLU() if use_point_relu else Empty())
        if self.use_attention:
            self.AVFE_Attention_feature_fc = nn.Sequential(
                                                        nn.Linear(point_feature_dim*2,self.AVFE_feature_dim,bias=False),
                                                        BatchNorm1d_attention(self.AVFE_feature_dim),
                                                        nn.ReLU() if use_attention_relu else Empty())
            self.AVFEO_Attention_feature_fc = nn.Sequential(
                                                        nn.Linear(point_feature_dim*2,self.AVFEO_feature_dim,bias=False),
                                                        BatchNorm1d_attention(self.AVFEO_feature_dim),
                                                        nn.ReLU() if use_attention_relu else Empty())
        
        self.AVFE_MID = nn.Sequential(
            BatchNorm1d_mid(self.AVFE_feature_dim),
            nn.ReLU()
        )

        self.AVFEO_MID = nn.Sequential(
            BatchNorm1d_mid(self.AVFEO_feature_dim),
            nn.ReLU()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        batch_size = input_dict['batch_size']
        points = input_dict['bev_points']
        voxel_local_coordinates = input_dict['voxel_local_coordinates']
        bev_mapping_pvs = input_dict['bev_mapping_pvs']
        bev_mapping_vfs = input_dict['bev_mapping_vfs']

        # bev_mapping_pvs[0] 640
        # bev_mapping_pvs[1] 320
        # bev_mapping_pvs[2] 160
        # bev_mapping_pvs[3] 320
        # bev_mapping_pvs[4] 160
        # bev_mapping_pvs[5] 80

        AVFE_features = []
        #iterate over input scale
        tts = []
        for i in range(self.input_scale_nums):
            # calcuate points's mean in each voxel
            point_mean = scatterMean(points, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
            # point - point_mean
            point_local_coordinate = points[:, :3] - point_mean[:, :3]
            # point intensity
            point_intensity = points[:, 3]
            point_voxel_mean = point_mean
            # point local coordinate in each voxel, (N,3) local_x, local_y, local_z
            voxel_local_coordinate = voxel_local_coordinates[i]

            #AVFE's point feature
            points_input_feature = torch.cat((points, point_local_coordinate, voxel_local_coordinate[:,:2]),dim=1)
            if self.use_local_z:
                points_input_feature = torch.cat((points_input_feature, voxel_local_coordinate[:,2].unsqueeze(1)),dim=1)
            
            points_input_feature = points_input_feature.contiguous()
            point_feature_fc = self.AVFE_point_feature_fc(points_input_feature)

            if self.use_attention:
                attention_feature_2 = scatterMean(points_input_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])

                # mean_cpu = scatterMeanCpu(points_input_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
                # error_t = (abs(mean_cpu - attention_feature_2))
                # max_error_t = error_t.max()
                # error_t = error_t.sum()
                # print("{} sum error {} max error {}".format(i, error_t, max_error_t))

                #first part of attention knowledge
                attention_feature_1 = torch.cat((point_local_coordinate, points_input_feature[:,3:]), dim=1)
                attention_feature = torch.cat((attention_feature_1, attention_feature_2), dim=1).contiguous()
                attention_feature_fc = self.AVFE_Attention_feature_fc(attention_feature)

                if self.use_sigmoid:
                    attention_feature_fc = self.sigmoid(attention_feature_fc)
    
                final_feature = attention_feature_fc * point_feature_fc
            else:
                final_feature = point_feature_fc

            if self.use_mid_relu:
                final_feature = self.AVFE_MID(final_feature)
            
            voxel_feature = scatterMax(final_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
            tts.append(dense(batch_size, [self.bev_sizes[i][0], self.bev_sizes[i][1]], self.AVFE_feature_dim, bev_mapping_vfs[i], voxel_feature))
            point_voxel_feature = torch.index_select(voxel_feature, 0, bev_mapping_pvs[i])
            AVFE_features.append(torch.cat((final_feature, point_voxel_feature), dim=1))
            
        final_AVFE_feature = torch.cat(AVFE_features, dim=1).contiguous()

        AVFEO_features = []
        point_feature_fc = self.AVFEO_point_feature_fc(final_AVFE_feature)
        tts = []
        for i in range(self.input_scale_nums, self.input_scale_nums + self.output_scale_nums):

            if self.use_attention:
                point_mean = scatterMean(points, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
                point_local_coordinate = points[:, :3] - point_mean[:, :3]
                point_intensity = points[:, 3]
                point_voxel_mean = point_mean
                voxel_local_coordinate = voxel_local_coordinates[i]

                points_input_feature = torch.cat((points, point_local_coordinate, voxel_local_coordinate[:,:2]),dim=1)
                if self.use_local_z:
                    points_input_feature = torch.cat((points_input_feature, voxel_local_coordinate[:,2].unsqueeze(1)),dim=1)
                
                points_input_feature = points_input_feature.contiguous()

                attention_feature_2 = scatterMean(points_input_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
                # mean_cpu = scatterMeanCpu(points_input_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])

                # error_t = (abs(mean_cpu - attention_feature_2))
                # max_error_t = error_t.max()
                # error_t = error_t.sum()
                # print("{} sum error {} max error {}".format(i, error_t, max_error_t))
                # assert (error_t) < 0.001, 'error {}, sum: {}'.format(i, error_t)

                attention_feature_1 = torch.cat((point_local_coordinate, points_input_feature[:,3:]), dim=1)
                attention_feature = torch.cat((attention_feature_1, attention_feature_2), dim=1).contiguous()
                attention_feature_fc = self.AVFEO_Attention_feature_fc(attention_feature)
                if self.use_sigmoid:
                    attention_feature_fc = self.sigmoid(attention_feature_fc)
                
                final_feature = attention_feature_fc * point_feature_fc
            else:
                final_feature = point_feature_fc

            if self.use_mid_relu:
                final_feature = self.AVFEO_MID(final_feature)
            
            voxel_feature = scatterMax(final_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
            tts.append(dense(batch_size, [self.bev_sizes[i][0], self.bev_sizes[i][1]], self.AVFEO_feature_dim, bev_mapping_vfs[i], scatterMax(point_feature_fc, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)))
            AVFEO_features.append(dense(batch_size, [self.bev_sizes[i][0], self.bev_sizes[i][1]], self.AVFEO_feature_dim, bev_mapping_vfs[i], voxel_feature))
        
        # for i in range(len(tts)):
        #     feature = tts[i]
        #     feature = feature[0]
        #     feature,_ = feature.max(dim=0)
        #     # feature = feature[0]#.sigmoid()
        #     feature = feature.cpu().detach().numpy()# * 255
        #     plt.imsave('/media/ovo/file3/detection/PCDet/debug/pics/AVFE_img_{}.jpg'.format(i),feature)

        for j in range(batch_size):
            cur_vf = bev_mapping_vfs[3][bev_mapping_vfs[3][:,0] == j]
            cur_vf = torch.cat((torch.zeros((cur_vf.shape[0],1)).long().cuda(),cur_vf[:,1:]),dim=1).contiguous()
            ff = torch.ones((cur_vf.shape[0],1)).cuda()
            tt = dense(1, [self.bev_sizes[3][0], self.bev_sizes[3][1]], 1, cur_vf ,ff)
            tt = tt.cpu().detach().numpy()[0,0]
            plt.imsave('/media/ovo/file3/detection/PCDet/debug/pics/img_batch_{}_bev.jpg'.format(j),tt)

        # for i in [2]: #range(len(AVFEO_features)):
        #     for j in range(batch_size):
        #         for m in range(AVFEO_features[i][j].shape[0]):
        #             feature = AVFEO_features[i]
        #             feature = feature[j]
        #             #feature,_ = feature.max(dim=0)
        #             feature = feature[m]#.sigmoid()
        #             feature = feature.cpu().detach().numpy()# * 255
        #             # min_val = np.min(feature)
        #             # if min_val < 0:
        #             #     feature -= min_val
        #             # gray_img = Img.color.gray2rgb(feature)
        #             plt.imsave('/media/ovo/file3/detection/PCDet/debug/pics/AVFEO_img_batch_{}_{}_{}.jpg'.format(j, i, m),feature)
        
        for i in [0,1,2]: #range(len(AVFEO_features)):
            for j in range(batch_size):
                for m in range(tts[i][j].shape[0]): #
                    feature = tts[i]
                    feature = feature[j]
                    #feature,_ = feature.max(dim=0)
                    feature = feature[m]#.sigmoid()
                    feature = feature.cpu().detach().numpy()# * 255
                    # min_val = np.min(feature)
                    # if min_val < 0:
                    #     feature -= min_val
                    # gray_img = Img.color.gray2rgb(feature)
                    plt.imsave('/media/ovo/file3/detection/PCDet/debug/pics/AVFEO_point_img_batch_{}_{}_{}.jpg'.format(j, i, m),feature)
        
        # input()
        # for i in range(len(tts)):
        #     for j in range(batch_size):
        #         feature = tts[i]
        #         feature = feature[j]
        #         feature = feature[0].sigmoid()
        #         feature = feature.cpu().detach().numpy()
                
        #         # min_val = np.min(feature)
        #         # if min_val < 0:
        #         #     feature -= min_val
        #         # gray_img = Img.color.gray2rgb(feature)
        #         plt.imsave('/root/tmp/point_img_batch_{}_{}.jpg'.format(j, i),feature,  cmap='viridis')
       
       

        return AVFEO_features




                






        






        



