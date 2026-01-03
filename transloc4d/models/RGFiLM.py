import torch
from torch import nn
import MinkowskiEngine as ME
from ikan import ChebyKANLinear, KANLinear

class RGSGFMM(nn.Module):
    """
    # RCS引导的特征调制模块 RCS-Guided Feature Modulation Module

    改进版空间注意力模块，使用上下文建模（Context Modeling）+ 残差机制
    输入 PR 特征，输出点级别权重，对 PRV 特征进行加权增强
    """

    def __init__(self, in_channels, reduction=4):
        super(RGSGFMM, self).__init__()
        hidden_dim = max(8, in_channels // reduction)

        self.context_encoder = nn.Sequential(
            ChebyKANLinear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.attn_generator = nn.Sequential(
            ChebyKANLinear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制增强强度

    def forward(self, x_PRV, x_PR):
        PR_feat = x_PR.F    # [N, C]
        PRV_feat = x_PRV.F  # [N, C]

        # Encode context
        context = self.context_encoder(PR_feat)  # [N, hidden_dim]

        # Generate attention weight per point
        attn_weights = self.attn_generator(context)  # [N, 1]

        # Apply spatial attention to PRV features
        enhanced = PRV_feat * attn_weights  # 点级别缩放

        # Residual fusion
        fused = self.alpha * enhanced + PRV_feat

        return ME.SparseTensor(
            features=fused,
            coordinate_map_key=x_PRV.coordinate_map_key,
            coordinate_manager=x_PRV.coordinate_manager
        )