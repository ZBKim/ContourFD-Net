import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .modules import MLP, PositionEmbeddingRandom, LayerNorm2d
from .SCSA2 import SCSA
from configs.base import Config


class DirectionalGradient(nn.Module):
    def __init__(self):
        super(DirectionalGradient, self).__init__()
        # 水平 X
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        # 垂直 Y（X 的转置）
        sobel_y = sobel_x.T
        # 主对角线 (↘)
        sobel_d1 = torch.tensor([[2, 1, 0],
                                 [1, 0, -1],
                                 [0, -1, -2]], dtype=torch.float32)
        # 副对角线 (↙)
        sobel_d2 = torch.tensor([[0, 1, 2],
                                 [-1, 0, 1],
                                 [-2, -1, 0]], dtype=torch.float32)
        self.register_buffer('sobel_kernel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_kernel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('sobel_kernel_d1', sobel_d1.view(1, 1, 3, 3))
        self.register_buffer('sobel_kernel_d2', sobel_d2.view(1, 1, 3, 3))

    def forward(self, x):
        B, C, H, W = x.shape
        kx = self.sobel_kernel_x.expand(C, 1, 3, 3)
        ky = self.sobel_kernel_y.expand(C, 1, 3, 3)
        kd1 = self.sobel_kernel_d1.expand(C, 1, 3, 3)
        kd2 = self.sobel_kernel_d2.expand(C, 1, 3, 3)

        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        gd1 = F.conv2d(x, kd1, padding=1, groups=C)
        gd2 = F.conv2d(x, kd2, padding=1, groups=C)

        grad = torch.cat([gx, gy, gd1, gd2], dim=1)  # [B, 4*C, H, W]
        return grad


class DynamicChannelAdapter(nn.Module):
    """
    把 (B, C, L) -> (B, out_channels, L) 的 1D 映射器。
    这里使用两层 1x1 conv1d（实现在 __init__ 中），避免在 forward 动态构造层。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, hidden: int = None):
        super().__init__()
        if hidden is None:
            hidden = max(in_channels // 2, 4)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, L)
        return self.net(x)  # (B, out_channels, L)


class MaskDecoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.num_mask_tokens = cfg.num_classes
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, cfg.encoder_out_features[-1])
        self.indices = list(range(len(cfg.encoder_out_features)))[::-1]
        self.gradient_extractor = DirectionalGradient()

        # 原有 grad_projs（用于 cross-attn 的投影）
        self.grad_projs = nn.ModuleDict()

        # 新增：用于残差融合的 conv (4*C -> C)
        self.grad_residual_convs = nn.ModuleDict()

        # 新增：用于将 conv1d adapter 按 block 存储，输出维度为 dim（方便直接作为 MHA 的 key/value）
        self.adpt_channels = nn.ModuleDict()

        # 可学习融合系数 alpha（每个尺度一个）
        self.alpha = nn.ParameterDict()

        for block_index in self.indices:
            dim = cfg.encoder_out_features[block_index]

            setattr(
                self,
                f"SCSA_s{block_index}",
                SCSA(
                    dim=dim,
                    head_num=cfg.mask_num_head,
                    window_size=7,
                ),
            )

            out_dim = cfg.encoder_out_features[block_index - 1] if block_index > 0 else dim

            setattr(
                self,
                f"MLP_s{block_index}",
                nn.ModuleList([
                    MLP(dim, dim, out_dim, 3) for _ in range(self.num_mask_tokens)
                ])
            )

            if block_index > 0:
                setattr(
                    self,
                    f"SUB_s{block_index}",
                    nn.Sequential(
                        nn.ConvTranspose2d(dim, out_dim, kernel_size=2, stride=2),
                        LayerNorm2d(out_dim),
                        nn.GELU(),
                    ),
                )
            else:
                mid_dim = dim // 2
                setattr(
                    self,
                    f"SUB_s{block_index}",
                    nn.Sequential(
                        nn.ConvTranspose2d(dim, mid_dim, kernel_size=2, stride=2),
                        LayerNorm2d(mid_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(mid_dim, dim, kernel_size=2, stride=2),
                        LayerNorm2d(dim),
                        nn.GELU(),
                    ),
                )

            setattr(
                self,
                f"TokenCrossAttn_s{block_index}",
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=cfg.mask_num_head,
                    batch_first=True,
                ),
            )

            # grad_projs: 将 [B, HW, 4*dim] -> [B, HW, dim]，用于 cross-attn 的线性映射
            self.grad_projs[str(block_index)] = nn.Linear(4 * dim, dim)

            # grad_residual_convs: 将 [B, 4*dim, H, W] -> [B, dim, H, W]
            self.grad_residual_convs[str(block_index)] = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True)

            # adpt_channels: 将 [B, dim, L] -> [B, dim, L] 的 adapter（这里 in_channels=dim, out_channels=dim）
            # 注意：grad_projs 输出是 (B, HW, dim)，我们会转为 (B, dim, HW) 送入 adapter
            self.adpt_channels[str(block_index)] = DynamicChannelAdapter(in_channels=dim, out_channels=dim, kernel_size=1, hidden=max(dim // 2, 8))

            # alpha：可学习融合系数，初始化为小一点的值（例如 0.5）
            self.alpha[str(block_index)] = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, image_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image_embeddings: list/tuple of features from encoder, each shape [B, C, H, W],
                          and indices iterate from deepest -> shallowest
        """
        query_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            image_embeddings[-1].size(0), -1, -1
        )

        previous_feat = torch.zeros_like(image_embeddings[-1])
        current_feat = torch.zeros_like(image_embeddings[-1])

        for block_index in self.indices:
            feat = image_embeddings[block_index]
            b, c, h, w = feat.shape

            # 基线特征（融合上一个尺度的上采样输出）
            current_feat = feat + previous_feat

            # SCSA 模块处理
            current_feat = getattr(self, f"SCSA_s{block_index}")(current_feat)

            # --- 1) 提取梯度特征（4*C, H, W）
            grad_feat = self.gradient_extractor(current_feat)  # [B, 4*C, H, W]

            # --- 2) 残差式融合：通过 1x1 conv 映射回 C 通道，然后相加（带可学习权重 alpha）
            grad_residual = self.grad_residual_convs[str(block_index)](grad_feat)  # [B, C, H, W]
            # 注意：alpha 是标量，广播加权
            current_feat = current_feat + self.alpha[str(block_index)] * grad_residual

            # --- 3) 保留辅助通路（cross-attention 的 key/value）
            # Flatten grad_feat -> (B, HW, 4*C)
            grad_feat_flat = grad_feat.flatten(2).permute(0, 2, 1)  # [B, HW, 4*C]
            # 投影到 dim -> (B, HW, dim)
            grad_feat_flat2 = self.grad_projs[str(block_index)](grad_feat_flat)  # [B, HW, dim]
            # 转为 (B, dim, HW) 以便 conv1d adapter 使用 (B, C, L)
            grad_feat_for_adapter = grad_feat_flat2.permute(0, 2, 1).contiguous()  # [B, dim, HW]
            # adapter -> (B, dim, HW)
            grad_feat_adapted = self.adpt_channels[str(block_index)](grad_feat_for_adapter)  # [B, dim, HW]
            # 转回 (B, HW, dim) 给 MultiheadAttention 作为 key/value
            kv = grad_feat_adapted.permute(0, 2, 1).contiguous()  # [B, HW, dim]

            # --- 4) Cross-attention: query_tokens 作为 query，kv 作为 key/value
            # MultiheadAttention expects: (query, key, value) each as (B, L, E) when batch_first=True
            query_tokens, _ = getattr(self, f"TokenCrossAttn_s{block_index}")(
                query_tokens, kv, kv
            )

            # --- 5) 上采样 / SUB
            current_feat = previous_feat = getattr(self, f"SUB_s{block_index}")(current_feat)

            # --- 6) 用 MLP 更新 query_tokens（保持原有逻辑）
            query_tokens = torch.stack([
                getattr(self, f"MLP_s{block_index}")[i](query_tokens[:, i, :])
                for i in range(self.num_mask_tokens)
            ], dim=1)

        # 最终 mask 生成（和原来保持一致）
        b, c, h, w = current_feat.shape
        masks = (query_tokens @ current_feat.view(b, c, h * w)).view(b, -1, h, w)

        return masks, current_feat
