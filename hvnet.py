class HVFeatureNetFinal(VoxelFeatureExtractor):
    def __init__(self,
                 bev_sizes,
                 bev_range,
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
        self.AVFE_Attention_feature_fc = nn.Sequential(
                                                    nn.Linear(point_feature_dim*2,self.AVFE_feature_dim,bias=False),
                                                    BatchNorm1d_attention(self.AVFE_feature_dim),
                                                    nn.ReLU() if use_attention_relu else Empty())
        self.AVFEO_point_feature_fc = nn.Sequential(
                                                    nn.Linear(self.input_scale_nums * 2 * self.AVFE_feature_dim,self.AVFEO_feature_dim,bias=False),
                                                    BatchNorm1d_point(self.AVFEO_feature_dim),
                                                    nn.ReLU() if use_point_relu else Empty())
        
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

        AVFE_features = []
        #iterate over input scale
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

            #AVFE's attention knowledge
            #second part of attention knowledge
            attention_feature_2 = scatterMean(points_input_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0])
            #first part of attention knowledge
            attention_feature_1 = torch.cat((point_local_coordinate, points_input_feature[:,3:]), dim=1)

            attention_feature = torch.cat((attention_feature_1, attention_feature_2), dim=1).contiguous()

            #point feature Linear
            point_feature_fc = self.AVFE_point_feature_fc(points_input_feature)
            #attention knowledge Linear
            attention_feature_fc = self.AVFE_Attention_feature_fc(attention_feature)

            if self.use_sigmoid:
                attention_feature_fc = self.sigmoid(attention_feature_fc)
            
            #attention
            final_feature = attention_feature_fc * point_feature_fc

            if self.use_mid_relu:
                final_feature = self.AVFE_MID(final_feature)
            
            # scatter max
            voxel_feature = scatterMax(final_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
            # scatter back to get point's cooresponding voxel feature
            point_voxel_feature = torch.index_select(voxel_feature, 0, bev_mapping_pvs[i])
            #concatenate point feature and cooresponding voxel feature
            AVFE_features.append(torch.cat((final_feature, point_voxel_feature), dim=1))
        
        final_AVFE_feature = torch.cat(AVFE_features, dim=1).contiguous()


        AVFEO_features = []
        for i in range(self.input_scale_nums, self.input_scale_nums + self.output_scale_nums):
            # same as AVFE, get attention knowledge
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
            attention_feature_1 = torch.cat((point_local_coordinate, points_input_feature[:,3:]), dim=1)
            attention_feature = torch.cat((attention_feature_1, attention_feature_2), dim=1).contiguous()

            #AVFEO point feature
            point_feature_fc = self.AVFEO_point_feature_fc(final_AVFE_feature)
            #AVFEO attention feature
            attention_feature_fc = self.AVFEO_Attention_feature_fc(attention_feature)

            if self.use_sigmoid:
                attention_feature_fc = self.sigmoid(attention_feature_fc)
            
            #attention feature
            final_feature = attention_feature_fc * point_feature_fc

            if self.use_mid_relu:
                final_feature = self.AVFEO_MID(final_feature)
            #voxel feature
            voxel_feature = scatterMax(final_feature, bev_mapping_pvs[i], bev_mapping_vfs[i].shape[0], True)
            AVFEO_features.append(dense(batch_size, [self.bev_sizes[i][0], self.bev_sizes[i][1]], self.AVFEO_feature_dim, bev_mapping_vfs[i], voxel_feature))

        return AVFEO_features
