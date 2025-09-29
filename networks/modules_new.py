import torch
import torch.nn as nn
from .encoder.modules import Conv, Concat, ConvTranspose

class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

class MLFF(nn.Module):
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
        print(low_level_features.shape, mid_level_features.shape, high_level_features.shape)
        return high_level_features, mid_level_features, low_level_features