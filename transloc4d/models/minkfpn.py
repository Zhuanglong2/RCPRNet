# Warsaw University of Technology

import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

from .RGFiLM import RGSGFMM
from .resnet import ResNetBase
import torch
from .interpolate_layer import Interpolate
from torchvision.models import resnet18, ResNet18_Weights

class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(
        self,
        in_channels,
        out_channels,
        num_top_down=1,
        conv0_kernel_size=5,
        block=BasicBlock,
        layers=(1, 1, 1),
        planes=(32, 64, 64),
    ):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)
        self.dropout = nn.Dropout(p=0.3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        # Bottom-up blocks (Conv1, Conv2 , Conv3 and Conv4)
        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
                )
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - i],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
            self.tconvs.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim,
                    self.lateral_dim,
                    kernel_size=2,
                    stride=2,
                    dimension=D,
                )
            )
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - self.num_top_down],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[0],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def random_dropout(self, tensor, dropout_prob=0.5, dim=3):
        # 确保dropout_prob在[0, 1]范围内
        dropout_prob = max(0, min(1, dropout_prob))

        # 如果输入张量不需要梯度，则直接返回未经dropout处理的输入
        if not tensor.requires_grad:
            return tensor

        # 生成与tensor相同形状的随机二进制掩码
        mask = torch.rand_like(tensor.select(dim, 0)) >= dropout_prob

        # 将第四维根据掩码进行dropout
        tensor_with_dropout = tensor.clone()
        tensor_with_dropout[:, :, :, :, mask] = 0.0

        return tensor_with_dropout

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger kernel)
        feature_maps = []
        # x.features.data=x.features[:,1:]
        # radom dropout
        # x.features.data = self.random_dropout(x.features.data,0.5,0)
        # x.features.data[:,:1] = self.dropout(x.features.data[:,:1])
        # x is in shape [N,2]
        x = self.conv0(x)  # [N,64]
        x = self.bn0(x)
        x = self.relu(x)
        x_conv0 = x
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
            x = bn(x)
            x = self.relu(x)
            x = block(x)  # [N,64] [N,128] [N,64] Top<--||-->Down [N,32]
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)  # [N,32] --> [N,256]

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)  # Upsample using transposed convolution # [N,256] [N,256]
            x = x + self.conv1x1[ndx + 1](feature_maps[-ndx - 1])  # [N,256] [N,256]

        return x, x_conv0

class MinkFPN2(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(
        self,
        in_channels,
        out_channels,
        num_top_down=1,
        conv0_kernel_size=5,
        block=BasicBlock,
        layers=(1, 1, 1),
        planes=(32, 64, 64),
    ):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)
        self.dropout = nn.Dropout(p=0.3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.inplanes = self.planes[0]
        self.conv1 = ME.MinkowskiConvolution(1, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        # Bottom-up blocks (Conv1, Conv2 , Conv3 and Conv4)
        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
                )
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - i],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
            self.tconvs.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim,
                    self.lateral_dim,
                    kernel_size=2,
                    stride=2,
                    dimension=D,
                )
            )
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - self.num_top_down],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[0],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )

        self.RGSGFMM = RGSGFMM(64)

        self.relu = ME.MinkowskiReLU(inplace=True)


    def random_dropout(self, tensor, dropout_prob=0.5, dim=3):
        # 确保dropout_prob在[0, 1]范围内
        dropout_prob = max(0, min(1, dropout_prob))

        # 如果输入张量不需要梯度，则直接返回未经dropout处理的输入
        if not tensor.requires_grad:
            return tensor

        # 生成与tensor相同形状的随机二进制掩码
        mask = torch.rand_like(tensor.select(dim, 0)) >= dropout_prob

        # 将第四维根据掩码进行dropout
        tensor_with_dropout = tensor.clone()
        tensor_with_dropout[:, :, :, :, mask] = 0.0

        return tensor_with_dropout

    def forward(self, x, x1):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger kernel)
        feature_maps = []
        x = self.conv0(x)  # [N,64]
        x = self.bn0(x)
        x = self.relu(x)

        x1 = self.conv1(x1)  # [N,64]
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        # 选择特征
        x = self.RGSGFMM(x, x1)  #RCS引导的特征调制模块 RCS-Guided Feature Modulation Module

        x_conv0 = x

        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
            x = bn(x)
            x = self.relu(x)
            x = block(x)  # [N,64] [N,128] [N,64] Top<--||-->Down [N,32]
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)  # [N,32] --> [N,256]
        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)  # Upsample using transposed convolution # [N,256] [N,256]
            x = x + self.conv1x1[ndx + 1](feature_maps[-ndx - 1])  # [N,256] [N,256]

        return x, x_conv0

class MinkFPN3(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(
        self,
        in_channels,
        out_channels,
        num_top_down=1,
        conv0_kernel_size=5,
        block=BasicBlock,
        layers=(1, 1, 1),
        planes=(32, 64, 64),
    ):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)
        self.dropout = nn.Dropout(p=0.3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.inplanes = self.planes[0]
        self.conv1 = ME.MinkowskiConvolution(1, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        # Bottom-up blocks (Conv1, Conv2 , Conv3 and Conv4)
        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
                )
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - i],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
            self.tconvs.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim,
                    self.lateral_dim,
                    kernel_size=2,
                    stride=2,
                    dimension=D,
                )
            )
        # There's one more lateral connection than top-down TConv blocks
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[-1 - self.num_top_down],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[0],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )

        self.RGSGFMM = RGSGFMM(64)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def random_dropout(self, tensor, dropout_prob=0.5, dim=3):
        # 确保dropout_prob在[0, 1]范围内
        dropout_prob = max(0, min(1, dropout_prob))

        # 如果输入张量不需要梯度，则直接返回未经dropout处理的输入
        if not tensor.requires_grad:
            return tensor

        # 生成与tensor相同形状的随机二进制掩码
        mask = torch.rand_like(tensor.select(dim, 0)) >= dropout_prob

        # 将第四维根据掩码进行dropout
        tensor_with_dropout = tensor.clone()
        tensor_with_dropout[:, :, :, :, mask] = 0.0

        return tensor_with_dropout

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger kernel)
        feature_maps = []
        x = self.conv0(x)  # [N,64]
        x = self.bn0(x)
        x = self.relu(x)

        x_conv0 = x

        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
            x = bn(x)
            x = self.relu(x)
            x = block(x)  # [N,64] [N,128] [N,64] Top<--||-->Down [N,32]
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down

        x = self.conv1x1[0](x)  # [N,32] --> [N,256]
        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)  # Upsample using transposed convolution # [N,256] [N,256]
            x = x + self.conv1x1[ndx + 1](feature_maps[-ndx - 1])  # [N,256] [N,256]

        return x, x_conv0



class FPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(
        self,
        in_channels,
        out_channels,
        num_top_down=1,
        conv0_kernel_size=5,
        block=BasicBlock,
        layers=(1, 1, 1),
        planes=(32, 64, 64),
    ):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)
        self.dropout = nn.Dropout(p=0.3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.convs1 = nn.ModuleList()
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.bn1 = nn.ModuleList()
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.tconvs1 = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections
        self.conv1x11 = nn.ModuleList()

        # Bottom-up blocks (Conv1, Conv2 , Conv3 and Conv4)
        m = 0
        for plane, layer in zip(self.planes, self.layers):
            self.inplanes = self.planes[m]
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D))
            self.convs1.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.bn1.append(ME.MinkowskiBatchNorm(self.inplanes))
            m += 1

        # Lateral connections
        for i in range(self.num_top_down):
            self.conv1x1.append(
                ME.MinkowskiConvolution(
                    self.planes[i],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
            self.conv1x11.append(
                ME.MinkowskiConvolution(
                    self.planes[i],
                    self.lateral_dim,
                    kernel_size=1,
                    stride=1,
                    dimension=D,
                )
            )
            self.tconvs.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim,
                    self.lateral_dim,
                    kernel_size=3,
                    stride=1,
                    dimension=D,
                )
            )
            self.tconvs1.append(
                ME.MinkowskiConvolutionTranspose(
                    self.lateral_dim,
                    self.lateral_dim,
                    kernel_size=2,
                    stride=2,
                    dimension=D,
                )
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

        #Inter
        self.Interpolate = Interpolate(64, 256)
        self.fusion_linear = ME.MinkowskiLinear(512, 256)

    def random_dropout(self, tensor, dropout_prob=0.5, dim=3):
        # 确保dropout_prob在[0, 1]范围内
        dropout_prob = max(0, min(1, dropout_prob))

        # 如果输入张量不需要梯度，则直接返回未经dropout处理的输入
        if not tensor.requires_grad:
            return tensor

        # 生成与tensor相同形状的随机二进制掩码
        mask = torch.rand_like(tensor.select(dim, 0)) >= dropout_prob

        # 将第四维根据掩码进行dropout
        tensor_with_dropout = tensor.clone()
        tensor_with_dropout[:, :, :, :, mask] = 0.0

        return tensor_with_dropout

    def forward(self, x_PRV):
        # # x_PR
        # PR_feartures = []
        # for ndx, (conv, bn, tconv) in enumerate(zip(self.convs, self.bn, self.tconvs)):
        #     x = conv(x_PR[self.num_top_down-ndx-1+2])  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
        #     x = bn(x)
        #     x = self.relu(x)
        #
        #     x = self.conv1x1[ndx](x)  # [N,M] --> [N,256]
        #     x = tconv(x)  # Upsample using transposed convolution
        #     PR_feartures.append(x)
        # x_PR_feartures = PR_feartures[0] + PR_feartures[1] + PR_feartures[2]
        # x_PR_feartures = self.Interpolate(x_PR_feartures, x_PR[0])

        #x_PRV
        PRV_features = []
        for ndx, (conv, bn, tconv) in enumerate(zip(self.convs1, self.bn1, self.tconvs1)):
            x = conv(x_PRV[self.num_top_down-ndx-1+2])  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
            x = bn(x)
            x = self.relu(x)

            x = self.conv1x11[ndx](x)  # [N,M] --> [N,256]
            x = tconv(x)  # Upsample using transposed convolution
            PRV_features.append(x)
        # x_PRV_feartures = PRV_features[0] + PRV_features[1] + PRV_features[2]
        Fusion_feature = self.Interpolate(PRV_features[2], x_PRV[0])

        # fused_feature = torch.cat([x_PR_feartures.F, x_PRV_feartures.F], dim=1)  # 只拼接 feature
        # Fusion_feature = ME.SparseTensor(
        #     fused_feature,  # 拼接后的特征
        #     coordinate_map_key=x_PRV_feartures.coordinate_map_key,
        #     coordinate_manager=x_PRV_feartures.coordinate_manager
        # )
        # Fusion_feature = self.fusion_linear(Fusion_feature)

        return Fusion_feature
class Backbone(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    def __init__(
        self,
        in_channels,
        out_channels,
        num_top_down=1,
        conv0_kernel_size=5,
        block=BasicBlock,
        layers=(1, 1, 1),
        planes=(32, 64, 64),
    ):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)
        self.dropout = nn.Dropout(p=0.3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs0 = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.convs1 = nn.ModuleList()
        self.bn0 = nn.ModuleList()  # Bottom-up BatchNorms
        self.bn1 = nn.ModuleList()
        self.blocks0 = nn.ModuleList()  # Bottom-up blocks
        self.blocks1 = nn.ModuleList()

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(1, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.conv1 = ME.MinkowskiConvolution(2, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn0_P = ME.MinkowskiBatchNorm(self.inplanes)
        self.bn0_V = ME.MinkowskiBatchNorm(self.inplanes)

        self.inplanes = self.planes[0]
        # Bottom-up blocks (Conv1, Conv2 , Conv3 and Conv4)
        for plane, layer in zip(self.planes, self.layers):
            self.convs0.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn0.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks0.append(self._make_layer(self.block, plane, layer))

        self.inplanes = self.planes[0]
        for plane, layer in zip(self.planes, self.layers):
            self.convs1.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn1.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks1.append(self._make_layer(self.block, plane, layer))


        self.relu = ME.MinkowskiReLU(inplace=True)

    def random_dropout(self, tensor, dropout_prob=0.5, dim=3):
        # 确保dropout_prob在[0, 1]范围内
        dropout_prob = max(0, min(1, dropout_prob))

        # 如果输入张量不需要梯度，则直接返回未经dropout处理的输入
        if not tensor.requires_grad:
            return tensor

        # 生成与tensor相同形状的随机二进制掩码
        mask = torch.rand_like(tensor.select(dim, 0)) >= dropout_prob

        # 将第四维根据掩码进行dropout
        tensor_with_dropout = tensor.clone()
        tensor_with_dropout[:, :, :, :, mask] = 0.0

        return tensor_with_dropout

    def process(self, x, coordinate_manager=None):
        # 处理 SparseTensor 并返回结果
        if coordinate_manager is not None:
            x = ME.SparseTensor(x.F, coordinates=x.C, coordinate_manager=coordinate_manager)
        return x

    def forward(self, x_PRV):
        # *** x_P ***
        # First bottom-up convolution is special (with bigger kernel)
        # x_PR_feature_maps = []
        #
        # x = self.conv0(x_PR)  # [N,64]
        # x = self.bn0_P(x)
        # x = self.relu(x)
        # x_PR_feature_maps.append(x)
        #
        # # BOTTOM-UP PASS
        # for ndx, (conv, bn, block) in enumerate(zip(self.convs0, self.bn0, self.blocks0)):
        #     x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
        #     x = bn(x)
        #     x = self.relu(x)
        #     x = block(x)  # [N,64] [N,128] [N,64] Top<--||-->Down [N,32]
        #     x_PR_feature_maps.append(x)

        # *** x_PRV ***
        x_PRV_feature_maps = []

        x = self.conv1(x_PRV)  # [N,64]
        x = self.bn0_V(x)
        x = self.relu(x)
        x_PRV_feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs1, self.bn1, self.blocks1)):
            x = conv(x)  # Downsample (conv stride=2 with 2x2x2 kernel) # [N,64] [N,64] [N,128] Top<--||-->Down [N,64]
            x = bn(x)
            x = self.relu(x)
            x = block(x)  # [N,64] [N,128] [N,64] Top<--||-->Down [N,32]
            x_PRV_feature_maps.append(x)

        return x_PRV_feature_maps
