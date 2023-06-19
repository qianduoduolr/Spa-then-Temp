from ..registry import COMPONENTS
from ..builder import build_components, build_loss, build_drop_layer
from mmcv.cnn import ConvModule, build_norm_layer, build_plugin_layer
import torch
import torch.nn as nn

def build_norm1d(cfg, num_features):
    if cfg['type'] == 'BN':
        return nn.BatchNorm1d(num_features=num_features)
    return build_norm_layer(cfg, num_features=num_features)[1]


@COMPONENTS.register_module()
class MlpHead(nn.Sequential):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.head_in = n_in
        self.head_out = n_out
        self.add_module("conv1", nn.Conv2d(n_in, n_in, 1, 1))
        self.add_module("bn1", nn.BatchNorm2d(n_in))
        self.add_module("relu", nn.ReLU(True))
        self.add_module("conv2", nn.Conv2d(n_in, n_out, 1, 1))


@COMPONENTS.register_module()
class DclHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]

@COMPONENTS.register_module()
class SimSiamHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_feat (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 conv_mid_channels=2048,
                 conv_out_channles=2048,
                 num_convs=0,
                 kernel_size=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=None,
                 drop_layer_cfg=None,
                 order=('pool', 'drop'),
                 num_projection_fcs=3,
                 projection_mid_channels=2048,
                 projection_out_channels=2048,
                 drop_projection_fc=False,
                 num_predictor_fcs=2,
                 predictor_mid_channels=512,
                 predictor_out_channels=2048,
                 drop_predictor_fc=False,
                 with_norm=True,
                 loss_feat=dict(type='CosineSimLoss', negative=False),
                 spatial_type='avg'):
        super().__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = with_norm
        self.loss_feat = build_loss(loss_feat)
        
        convs = []
        last_channels = in_channels
        for i in range(num_convs):
            is_last = i == num_convs - 1
            out_channels = conv_out_channles if is_last else conv_mid_channels
            convs.append(
                ConvModule(
                    last_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if not is_last else None,
                    act_cfg=self.act_cfg if not is_last else None))
            last_channels = out_channels
        if len(convs) > 0:
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = nn.Identity()

        projection_fcs = []
        for i in range(num_projection_fcs):
            is_last = i == num_projection_fcs - 1
            out_channels = projection_out_channels if is_last else \
                projection_mid_channels
            projection_fcs.append(nn.Linear(last_channels, out_channels))
            projection_fcs.append(build_norm1d(norm_cfg, out_channels))
            # no relu on output
            if not is_last:
                projection_fcs.append(nn.ReLU())
                if drop_projection_fc:
                    projection_fcs.append(build_drop_layer(drop_layer_cfg))
            last_channels = out_channels
        if len(projection_fcs):
            self.projection_fcs = nn.Sequential(*projection_fcs)
        else:
            self.projection_fcs = nn.Identity()

        predictor_fcs = []
        for i in range(num_predictor_fcs):
            is_last = i == num_predictor_fcs - 1
            out_channels = predictor_out_channels if is_last else \
                predictor_mid_channels
            predictor_fcs.append(nn.Linear(last_channels, out_channels))
            if not is_last:
                predictor_fcs.append(build_norm1d(norm_cfg, out_channels))
                predictor_fcs.append(nn.ReLU())
                if drop_predictor_fc:
                    predictor_fcs.append(build_drop_layer(drop_layer_cfg))
            last_channels = out_channels
        if len(predictor_fcs):
            self.predictor_fcs = nn.Sequential(*predictor_fcs)
        else:
            self.predictor_fcs = nn.Identity()

        assert spatial_type in ['avg', 'att', None]
        self.spatial_type = spatial_type
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = nn.Identity()
        if drop_layer_cfg is not None:
            self.dropout = build_drop_layer(drop_layer_cfg)
        else:
            self.dropout = nn.Identity()
        assert set(order) == {'pool', 'drop'}
        self.order = order

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward_projection(self, x):
        x = self.convs(x)
        for layer in self.order:
            if layer == 'pool':
                x = self.avg_pool(x)
                x = x.flatten(1)
            if layer == 'drop':
                x = self.dropout(x)
        z = self.projection_fcs(x)

        return z

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.convs(x)
        for layer in self.order:
            if layer == 'pool':
                x = self.avg_pool(x)
                x = x.flatten(1)
            if layer == 'drop':
                x = self.dropout(x)
        z = self.projection_fcs(x)
        p = self.predictor_fcs(z)

        return z, p
    
    def loss(self, p1, z1, p2, z2, mask12=None, mask21=None, weight=1.):
        assert mask12 is None
        assert mask21 is None

        losses = dict()

        loss_feat = self.loss_feat(p1, z2.detach()) * 0.5 + self.loss_feat(
            p2, z1.detach()) * 0.5
        losses['loss_feat'] = loss_feat * weight
        return losses

