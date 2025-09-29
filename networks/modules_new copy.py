import torch
import torch.nn as nn
from .encoder.modules import Conv, Concat, ConvTranspose

class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, feature_channels_list, out_channels):
        super().__init__()
        self.feature_channels = feature_channels_list
        total_channels = sum(feature_channels_list)
        
        # 为每个特征分支生成权重
        self.weight_conv = nn.Sequential(
            Conv(total_channels, total_channels // 4, 1, 1),
            nn.ReLU(inplace=True),
            Conv(total_channels // 4, len(feature_channels_list), 1, 1),
            nn.Softmax(dim=1)
        )
        
        # 统一各分支通道数
        self.align_convs = nn.ModuleList([
            Conv(ch, out_channels // len(feature_channels_list), 1, 1) 
            for ch in feature_channels_list
        ])
        
        self.fusion_conv = Conv(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, features):
        # features: [f1, f2, f3]
        concat_features = torch.cat(features, dim=1)
        weights = self.weight_conv(concat_features)  # [B, 3, H, W]
        
        # 对齐通道数并加权融合
        aligned_features = []
        for i, (feat, align_conv) in enumerate(zip(features, self.align_convs)):
            weight = weights[:, i:i+1, :, :]  # [B, 1, H, W]
            aligned_feat = align_conv(feat)
            aligned_features.append(aligned_feat * weight)
        
        fused = torch.cat(aligned_features, dim=1)
        return self.fusion_conv(fused)

class CrossScaleFeatureEnhancement(nn.Module):
    """跨尺度特征增强模块"""
    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = Conv(channels, channels, 3, 1, 1)
        self.enhance_conv = nn.Sequential(
            Conv(channels * 2, channels, 1, 1),
            nn.ReLU(inplace=True),
            Conv(channels, channels, 3, 1, 1)
        )
    
    def forward(self, x):
        # 全局特征
        global_feat = self.global_pool(x)
        global_feat = global_feat.expand_as(x)
        
        # 局部特征
        local_feat = self.local_conv(x)
        
        # 融合增强
        enhanced = self.enhance_conv(torch.cat([global_feat, local_feat], dim=1))
        return x + enhanced

class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 3
        
        self.branch_1x1 = Conv(in_channels, mid_channels, 1, 1)
        self.branch_3x3 = Conv(in_channels, mid_channels, 3, 1, 1)
        self.branch_5x5 = nn.Sequential(
            Conv(in_channels, mid_channels, 3, 1, 1),
            Conv(mid_channels, mid_channels, 3, 1, 1)
        )
        self.fusion_conv = Conv(out_channels, out_channels, 1, 1)
    
    def forward(self, x):
        branch1 = self.branch_1x1(x)
        branch2 = self.branch_3x3(x)
        branch3 = self.branch_5x5(x)
        
        concat = torch.cat([branch1, branch2, branch3], dim=1)
        return self.fusion_conv(concat)

# 更简单稳定的版本
class MLFF(nn.Module):
    """轻量"""
    def __init__(self, cfg) -> None:
        super().__init__()
        n = cfg.n_channel 
        # 特征融合权重
        self.fusion_weights_0 = nn.Parameter(torch.ones(3))
        self.fusion_weights_1 = nn.Parameter(torch.ones(3))
        self.fusion_weights_2 = nn.Parameter(torch.ones(3))
        
        # 高层处理
        self.neck_0_0 = Conv(n * 4, n * 2, 1, 1)
        self.neck_0_1 = ConvTranspose(n * 8, n * 2, 2, 2)
        self.neck_0_2 = ConvTranspose(n * 16, n * 2, 4, 4)
        self.neck_0_fusion = Conv(n * 6, n * 4, 3, 1, 1)
        
        # 中层处理
        self.neck_1_0 = Conv(n * 4, n * 2, 3, 2)
        self.neck_1_1 = Conv(n * 8, n * 4, 1, 1)
        self.neck_1_2 = ConvTranspose(n * 16, n * 2, 2, 2)
        self.neck_1_fusion = Conv(n * 8, n * 8, 3, 1, 1)
        
        # 低层处理
        self.neck_2_0 = Conv(n * 4, n * 4, 3, 4)
        self.neck_2_1 = Conv(n * 8, n * 4, 3, 2)
        self.neck_2_2 = Conv(n * 16, n * 8, 1, 1)
        self.neck_2_fusion = Conv(n * 16, n * 16, 3, 1, 1)
    
    def forward(self, inputs):
        c2f_15, c2f_18, c2f_21 = inputs
        
        # 归一化融合权重
        w0 = torch.softmax(self.fusion_weights_0, dim=0)
        w1 = torch.softmax(self.fusion_weights_1, dim=0)
        w2 = torch.softmax(self.fusion_weights_2, dim=0)
        
        # 高层特征
        high_features = [
            self.neck_0_0(c2f_15) * w0[0],
            self.neck_0_1(c2f_18) * w0[1],
            self.neck_0_2(c2f_21) * w0[2]
        ]
        high_concat = torch.cat(high_features, dim=1)
        high_level_features = self.neck_0_fusion(high_concat)
        
        # 中层特征
        mid_features = [
            self.neck_1_0(c2f_15) * w1[0],
            self.neck_1_1(c2f_18) * w1[1],
            self.neck_1_2(c2f_21) * w1[2]
        ]
        mid_concat = torch.cat(mid_features, dim=1)
        mid_level_features = self.neck_1_fusion(mid_concat)
        
        # 低层特征
        low_features = [
            self.neck_2_0(c2f_15) * w2[0],
            self.neck_2_1(c2f_18) * w2[1],
            self.neck_2_2(c2f_21) * w2[2]
        ]
        low_concat = torch.cat(low_features, dim=1)
        low_level_features = self.neck_2_fusion(low_concat)
        
        return high_level_features, mid_level_features, low_level_features