import copy
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from ikan import ChebyKANLinear, KANLinear

from .linearselfatt import LinearAttention, FullAttention

from torch_cluster import knn
import torch_scatter


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear", dropout_p=0.05):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.act = nn.ReLU()

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim) - x)  # [N, L, C]
        message = self.norm1(message)
        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = 256
        self.nhead = 2
        self.layer_names = ["self"] * 1
        encoder_layer = LoFTREncoderLayer(256, self.nhead, "linear")
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        features = feat0.decomposed_features
        batch_size = len(features)

        for i in range(batch_size):
            features_single = features[i].unsqueeze(0)
            assert self.d_model == features_single.size(
                2
            ), "the feature number of src and transformer must be equal"

            for layer, name in zip(self.layers, self.layer_names):
                features_single = layer(features_single, features_single, mask0, mask0)

            features[i] = features_single[0]


        feat0 = ME.SparseTensor(
            features=torch.cat(features, 0),
            coordinates=feat0.coordinates,
            coordinate_manager=feat0.coordinate_manager,
        )

        return feat0

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear", dropout_p=0.05):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = ChebyKANLinear(d_model, d_model)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.act = nn.ReLU()

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim) - x)  # [N, L, C]
        message = self.norm1(message)

        return x + message

class NonlinearFeatureAggregation(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(NonlinearFeatureAggregation, self).__init__()

        self.d_model = 256
        self.nhead = 2
        self.layer_names = ["self"] * 1
        encoder_layer = EncoderLayer(256, self.nhead, "linear")
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        features = feat0.decomposed_features #这可以把稀疏tensor转成正常卷积处理的tensor
        batch_size = len(features)

        for i in range(batch_size):
            features_single = features[i].unsqueeze(0)
            assert self.d_model == features_single.size(2), "the feature number of src and transformer must be equal"

            for layer, name in zip(self.layers, self.layer_names):
                features_single = layer(features_single, features_single, mask0, mask1)

            features[i] = features_single[0]

        feat0 = ME.SparseTensor(
            features=torch.cat(features, 0),
            coordinates=feat0.coordinates,
            coordinate_manager=feat0.coordinate_manager,
        )

        return feat0





#随即挑选5个点
class SparseLocalSelfAttention(nn.Module):
    def __init__(self, in_channels=256, nhead=2, k=10, extra_k=5):
        super().__init__()
        assert in_channels % nhead == 0, "in_channels must be divisible by nhead"
        self.in_channels = in_channels
        self.nhead = nhead
        self.dim = in_channels // nhead
        self.k = k
        self.extra_k = extra_k  # 额外随机邻居数量

        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.scale = self.dim ** -0.5

        self.pos_proj = nn.Linear(3, self.nhead)

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: ME.SparseTensor):
        features = x.F  # [N, C]
        coords = x.C[:, 1:].float()  # [N, 3]
        N = coords.size(0)

        # 投影并reshape为[N, H, D]
        Q = self.q_proj(features).reshape(-1, self.nhead, self.dim)
        K = self.k_proj(features).reshape(-1, self.nhead, self.dim)
        V = self.v_proj(features).reshape(-1, self.nhead, self.dim)

        # knn获取邻居索引
        neighbor_idx, query_idx = knn(coords, coords, self.k)  # [M,], [M,]

        # 生成额外随机邻居对
        if self.extra_k > 0:
            extra_query_idx = torch.randint(0, N, (N * self.extra_k,), device=coords.device)
            extra_neighbor_idx = torch.randint(0, N, (N * self.extra_k,), device=coords.device)

            # 拼接原邻居和额外邻居索引
            neighbor_idx = torch.cat([neighbor_idx, extra_neighbor_idx], dim=0)
            query_idx = torch.cat([query_idx, extra_query_idx], dim=0)

        # 计算相对位置编码
        rel_pos = coords[query_idx] - coords[neighbor_idx]  # [M + M_extra, 3]
        pos_enc = self.pos_proj(rel_pos)  # [M + M_extra, H]

        # 按索引取q,k,v
        q = Q[query_idx]  # [M + M_extra, H, D]
        k = K[neighbor_idx]  # [M + M_extra, H, D]
        v = V[neighbor_idx]  # [M + M_extra, H, D]

        # 点积注意力分数 (M + M_extra, H)
        attn_scores = torch.einsum('mhd,mhd->mh', q, k) * self.scale + pos_enc

        # softmax按query分组
        attn_weights = torch_scatter.scatter_softmax(attn_scores, query_idx, dim=0)  # [M + M_extra, H]

        # 加权value
        weighted_values = v * attn_weights.unsqueeze(-1)  # [M + M_extra, H, D]

        # 聚合邻居信息到query节点
        out = torch_scatter.scatter_add(weighted_values, query_idx, dim=0, dim_size=Q.size(0))  # [N, H, D]

        out = out.reshape(-1, self.in_channels)
        out_features = self.out_proj(out)

        # 残差连接 + LayerNorm
        out_features = self.norm(out_features + features)

        return ME.SparseTensor(
            features=out_features,
            coordinates=x.C,
            coordinate_manager=x.coordinate_manager
        )

#不随机挑选
# class SparseLocalSelfAttention(nn.Module):
#     def __init__(self, in_channels=256, nhead=2, k=10):
#         super().__init__()
#         self.in_channels = in_channels
#         self.nhead = nhead
#         self.dim = in_channels // nhead
#         self.k = k
#
#         self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
#         self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
#         self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
#         self.out_proj = nn.Linear(in_channels, in_channels, bias=False)
#
#         self.scale = self.dim ** -0.5
#
#         self.pos_proj = nn.Linear(3, self.nhead)
#
#         self.norm = nn.LayerNorm(in_channels)  # 增加LayerNorm层
#
#     def forward(self, x: ME.SparseTensor):
#         features = x.F  # [N, C]
#         coords = x.C[:, 1:].float()  # [N, 3]
#
#         # 投影并reshape为[N, H, D]
#         Q = self.q_proj(features).reshape(-1, self.nhead, self.dim)
#         K = self.k_proj(features).reshape(-1, self.nhead, self.dim)
#         V = self.v_proj(features).reshape(-1, self.nhead, self.dim)
#
#         # knn获取邻居索引，batch_x和batch_y为None时按全图处理
#         neighbor_idx, query_idx = knn(coords, coords, self.k)
#
#         # 计算相对位置编码
#         rel_pos = coords[query_idx] - coords[neighbor_idx]  # [M, 3]
#         pos_enc = self.pos_proj(rel_pos)  # [M, H]
#
#         # 按索引取q,k,v
#         q = Q[query_idx]  # [M, H, D]
#         k = K[neighbor_idx]  # [M, H, D]
#         v = V[neighbor_idx]  # [M, H, D]
#
#         # 点积注意力分数 (M,H)
#         attn_scores = torch.einsum('mhd,mhd->mh', q, k) * self.scale + pos_enc
#
#         # softmax按query分组
#         attn_weights = torch_scatter.scatter_softmax(attn_scores, query_idx, dim=0)  # [M, H]
#
#         # 加权value
#         weighted_values = v * attn_weights.unsqueeze(-1)  # [M, H, D]
#
#         # 聚合邻居信息到query节点
#         out = torch_scatter.scatter_add(weighted_values, query_idx, dim=0, dim_size=Q.size(0))  # [N, H, D]
#
#         out = out.reshape(-1, self.in_channels)
#         out_features = self.out_proj(out)
#
#         # 残差连接 + LayerNorm
#         out_features = self.norm(out_features + features)
#
#         return ME.SparseTensor(
#             features=out_features,
#             coordinates=x.C,
#             coordinate_manager=x.coordinate_manager
#         )


