# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from ikan import ChebyKANLinear

from .kan import KanLinear
from .layers.pooling_wrapper import PoolingWrapper
from .transformer import LocalFeatureTransformer, NonlinearFeatureAggregation, SparseLocalSelfAttention
from .cbam import SpatialAttGate

from .interpolate_layer import Interpolate
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ..datasets.quantization import CartesianQuantizer


class MinkLoc(torch.nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: PoolingWrapper,
        normalize_embeddings: bool = False,
        self_att: bool = False,
        spatial_att: bool = False,
        add_FTU: bool = False,
    ): 
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        self.linearselfatt = LocalFeatureTransformer() if self_att else None
        # self.linearselfatt = asvt_module() if self_att else None
        self.spatialatt = SpatialAttGate() if spatial_att else None
        self.FTU = Interpolate(64, 256) if add_FTU else None

    def forward(self, batch):
        x = ME.SparseTensor(batch["features"][:, :2], coordinates=batch["coords"])

        x, x_conv0 = self.backbone(x)
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )
        if self.FTU is not None:
            x = self.FTU(x, x_conv0) #(115898 256) (202618 64)  --->  (202618 256)
        if self.linearselfatt is not None:
            x = self.linearselfatt(x, x)
        x = self.pooling(x)
        if hasattr(self.pooling, "stats"):
            self.stats.update(self.pooling.stats)

        # x = x.flatten(1)
        assert (x.dim() == 2), f"Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions."
        assert x.shape[1] == self.pooling.output_dim, (
            f"Output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.output_dim}"
        )

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        return {"global": x}

    def print_info(self):
        print("Model class: MinkLoc")
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f"Total parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f"Backbone: {type(self.backbone).__name__} #parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f"Pooling method: {self.pooling.pool_method}   #parameters: {n_params}")
        print("# channels from the backbone: {}".format(self.pooling.in_dim))
        print("# output channels : {}".format(self.pooling.output_dim))
        print(f"Embedding normalization: {self.normalize_embeddings}")

    def forward_local(self, batch):
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x = self.backbone(x)[0].features
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )

        x = F.normalize(x, p=2, dim=1)

        return {"local": x}

class MinkLoc2(torch.nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: PoolingWrapper,
        normalize_embeddings: bool = False,
        self_att: bool = False,
        spatial_att: bool = False,
        add_FTU: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        self.NonlinearFeatureAggregation = NonlinearFeatureAggregation()
        self.spatialatt = SpatialAttGate() if spatial_att else None
        self.FTU = Interpolate(64, 256) if add_FTU else None

    def forward(self, batch):
        x = ME.SparseTensor(batch["features"][:, :2], coordinates=batch["coords"])
        x1 = ME.SparseTensor(batch["features"][:, 1].unsqueeze(1), coordinates=batch["coords"])

        x, x_conv0 = self.backbone(x, x1)
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )
        if self.FTU is not None:
            x = self.FTU(x, x_conv0) #(115898 256) (202618 64)  --->  (202618 256)

        x = self.NonlinearFeatureAggregation(x, x) #非线性特征聚合模块

        x = self.pooling(x)
        if hasattr(self.pooling, "stats"):
            self.stats.update(self.pooling.stats)

        assert (
            x.dim() == 2
        ), f"Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions."
        assert x.shape[1] == self.pooling.output_dim, (
            f"Output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.output_dim}"
        )

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        return {"global": x}

    def print_info(self):
        print("Model class: MinkLoc")
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f"Total parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f"Backbone: {type(self.backbone).__name__} #parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f"Pooling method: {self.pooling.pool_method}   #parameters: {n_params}")
        print("# channels from the backbone: {}".format(self.pooling.in_dim))
        print("# output channels : {}".format(self.pooling.output_dim))
        print(f"Embedding normalization: {self.normalize_embeddings}")

    def forward_local(self, batch):
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x = self.backbone(x)[0].features
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )

        x = F.normalize(x, p=2, dim=1)

        return {"local": x}


class MinkLoc3(torch.nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: PoolingWrapper,
        normalize_embeddings: bool = False,
        self_att: bool = False,
        spatial_att: bool = False,
        add_FTU: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        self.local_attn = SparseLocalSelfAttention(in_channels=256, nhead=8, k=5, extra_k=5)

        self.spatialatt = SpatialAttGate() if spatial_att else None
        self.FTU = Interpolate(64, 256) if add_FTU else None

        self.linear_last = nn.Linear(512,256)

        self.quantizer = CartesianQuantizer(quant_step=0.01)

        self.rgb_proj = nn.Linear(3, 2)
        self.learnable_sigma = nn.Parameter(torch.tensor([1.0]))  # 初始化 sigma

    def learnable_gaussian_fusion(self, radar_feat, rgb_feat, sigma_param):
        """
        使用可训练的高斯融合将 radar 和 RGB 特征融合。

        参数:
            radar_feat: Tensor, shape (N, C1), e.g., [RCS, Doppler]
            rgb_feat: Tensor, shape (N, C2), e.g., [R, G, B]
            sigma_param: nn.Parameter, 可训练的 sigma (标量)

        返回:
            fused_feat: Tensor, shape (N, C1), 融合后的特征（匹配 radar_feat 的维度）
        """

        # Step 1: 归一化两个模态（防止幅值不同）
        radar_proj = F.normalize(radar_feat, dim=1)
        rgb_proj = F.normalize(rgb_feat[:, :radar_proj.shape[1]], dim=1)

        # Step 2: 计算欧式距离并计算高斯权重 alpha
        diff = torch.norm(radar_proj - rgb_proj, dim=1)  # (N,)
        alpha = torch.exp(-diff ** 2 / (2 * sigma_param.clamp(min=1e-4) ** 2)).unsqueeze(1)  # (N, 1)

        # Step 3: 加权融合
        fused = alpha * radar_feat + (1 - alpha) * rgb_feat[:, :radar_feat.shape[1]]

        return fused

    def forward(self, batch):
        radar_feat = batch["features"][:, :2]  # (N, 2)
        rgb_feat = batch["features"][:, 2:]  # (N, 3)
        rgb_feat_proj = self.rgb_proj(rgb_feat)  # (N, 2)
        fused_feat = self.learnable_gaussian_fusion(radar_feat, rgb_feat_proj, self.learnable_sigma)
        x = ME.SparseTensor(fused_feat, coordinates=batch["coords"])

        x, x_conv0  = self.backbone(x)
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )
        if self.FTU is not None:
            x = self.FTU(x, x_conv0) #(115898 256) (202618 64)  --->  (202618 256)

        x = self.local_attn(x)

        x = self.pooling(x)
        if hasattr(self.pooling, "stats"):
            self.stats.update(self.pooling.stats)

        assert (
            x.dim() == 2
        ), f"Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions."
        assert x.shape[1] == self.pooling.output_dim, (
            f"Output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.output_dim}"
        )

        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        return {"global": x}

    def print_info(self):
        print("Model class: MinkLoc")
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f"Total parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f"Backbone: {type(self.backbone).__name__} #parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f"Pooling method: {self.pooling.pool_method}   #parameters: {n_params}")
        print("# channels from the backbone: {}".format(self.pooling.in_dim))
        print("# output channels : {}".format(self.pooling.output_dim))
        print(f"Embedding normalization: {self.normalize_embeddings}")

    def forward_local(self, batch):
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x = self.backbone(x)[0].features
        assert x.shape[1] == self.pooling.in_dim, (
            f"Backbone output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.in_dim}"
        )

        x = F.normalize(x, p=2, dim=1)

        return {"local": x}

