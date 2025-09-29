import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

class AdaptiveGradReducer(nn.Module):
    """自适应梯度降维模块"""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EnhancedSpatialAttention(nn.Module):
    """增强的空间注意力模块"""
    def __init__(self, channels):
        super().__init__()
        
        # 改进的水平和垂直注意力
        self.conv_h = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),  # 增加容量
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.conv_w = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        '''
        # 添加坐标注意力
        self.coord_conv = nn.Sequential(
            nn.Conv2d(2, channels // 8, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 7, padding=3),
            nn.Sigmoid()
        )
        '''
        # 改进的全局上下文
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),  # 增加容量
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 空间注意力
        x_h = torch.mean(x, dim=3, keepdim=False)  # (B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=False)  # (B, C, W)
        
        attn_h = self.conv_h(x_h).unsqueeze(-1)  # (B, C, H, 1)
        attn_w = self.conv_w(x_w).unsqueeze(-2)  # (B, C, 1, W)
        '''
        # 坐标注意力
        x_coord = torch.linspace(-1, 1, w, device=x.device)
        y_coord = torch.linspace(-1, 1, h, device=x.device)
        y_coord, x_coord = torch.meshgrid(y_coord, x_coord, indexing='ij')
        coord_map = torch.stack([x_coord, y_coord], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        coord_attn = self.coord_conv(coord_map)  # (B, 1, H, W)
        '''
        # 全局上下文注意力
        global_attn = self.global_conv(self.global_pool(x))  # (B, C, 1, 1)
        
        # 组合注意力 - 加权融合
        spatial_attn = (attn_h + attn_w  + global_attn) / 3
        
        return x * spatial_attn

class SCSA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        qkv_bias: bool = False,
        down_sample_mode: str = 'avg_pool',
        attn_drop_ratio: float = 0.1,
        gate_layer: str = 'sigmoid',
        use_gradient: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size
        self.use_gradient = use_gradient


        if use_gradient:
            self.grad_extractor = DirectionalGradient()
            self.grad_reducer = AdaptiveGradReducer(dim * 4, dim)
        

        self.spatial_attn = EnhancedSpatialAttention(dim)
               
        # 归一化层 - 使用LayerNorm替代GroupNorm
        self.norm = nn.LayerNorm(dim)
        
        # QKV投影 - 使用更高效的实现
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        
        # 注意力dropout
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        
        # 输出投影
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(attn_drop_ratio)
        
        # 通道注意力门控
        self.ca_gate = nn.Sigmoid() if gate_layer == 'sigmoid' else nn.Softmax(dim=1)
        
        # 下采样函数
        self._setup_downsample(window_size, down_sample_mode)

    def _setup_downsample(self, window_size, down_sample_mode):
        """设置下采样函数"""
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
            self.conv_d = nn.Identity()
        else:
            if down_sample_mode == 'avg_pool':
                
                self.down_func = nn.AvgPool2d(
                    kernel_size=window_size, 
                    stride=window_size,
                    padding=0
                )
            elif down_sample_mode == 'max_pool':
                
                self.down_func = nn.MaxPool2d(
                    kernel_size=window_size,
                    stride=window_size,
                    padding=0
                )
            else:  # adaptive
                
                self.down_func = nn.AdaptiveAvgPool2d((4, 4))  # 固定大小
            
            self.Identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. 梯度增强（保留残差）
        if self.use_gradient:
            grad_feat = self.grad_extractor(x)  # (B, 4C, H, W)
            grad_feat = self.grad_reducer(grad_feat)  # (B, C, H, W)
            x = x + grad_feat  # 残差连接

        # 2. 空间注意力 (pre-norm + residual)
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # NHWC->NCHW
        x = self.spatial_attn(x_norm)
        
        # 4. 通道自注意力机制
        y_Identity = self.Identity(x)
        y = self.down_func(x)
        B_y, C_y, H_y, W_y = y.shape
        
        # 归一化 - Pre-LN: LayerNorm前置
        y_norm = y.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        y_norm = self.norm(y_norm)
        y_norm = y_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        # QKV投影 + Attention
        qkv = self.qkv(y_norm)
        qkv = qkv.reshape(B, 3, self.head_num, self.head_dim, H_y * W_y)
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaler
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.reshape(B, H_y * W_y, C_y)
        out = out.permute(0, 2, 1).contiguous().reshape(B, C_y, H_y, W_y)
        # 输出投影 + dropout
        out_proj = self.proj(out)
        out_proj = out_proj.mean((2, 3), keepdim=True)  # 求平均
        # 5. 通道注意力门控
        out_attn=torch.sigmoid(out_proj)
        result = y_Identity * out_attn # 用通道注意力微调最终的特征
        return result
