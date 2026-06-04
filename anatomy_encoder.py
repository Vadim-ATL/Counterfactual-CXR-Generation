import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.models import swin_t, Swin_T_Weights



class AnatomyEncoder(nn.Module):

    """

    Swin-T backbone → spatial anatomy feature map (B, 256, 16, 16)

    + segmentation head for lung/heart region supervision

    """

    def __init__(self, out_channels=256, img_size=512):

        super().__init__()

        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

        # Swin-T stages output at strides 4,8,16,32

        # We take the first 3 stages → stride 16, keep spatial resolution

        self.stage0 = nn.Sequential(backbone.features[0], backbone.features[1])  # stride 4,  C=96

        self.stage1 = nn.Sequential(backbone.features[2], backbone.features[3])  # stride 8,  C=192

        self.stage2 = nn.Sequential(backbone.features[4], backbone.features[5])  # stride 16, C=384



        # Project to fixed anatomy latent channel dim

        self.proj = nn.Sequential(

            nn.Linear(384, out_channels),

            nn.LayerNorm(out_channels),

        )



        # Segmentation head: predicts 3 classes (background, lung, heart)

        # Input: (B, H, W, C) → rearrange → (B, C, H, W) then upsample to 512

        self.seg_head = nn.Sequential(

            nn.Conv2d(out_channels, 128, 3, padding=1),

            nn.ReLU(),

            nn.Conv2d(128, 3, 1),  # 3 classes

        )



    def forward(self, x):

        # x: (B, 3, 512, 512)

        f = self.stage0(x)   # (B, 128, 128, 96)

        f = self.stage1(f)   # (B,  64,  64, 192)

        f = self.stage2(f)   # (B,  32,  32, 384)



        z = self.proj(f)     # (B, 32, 32, 256)



        # rearrange to (B, C, H, W) for conv seg head

        z_spatial = z.permute(0, 3, 1, 2).contiguous()  # (B, 256, 32, 32)



        seg_logits = self.seg_head(z_spatial)            # (B, 3, 32, 32)

        seg_logits = F.interpolate(seg_logits, size=512, mode="bilinear", align_corners=False)



        return z_spatial, seg_logits   # anatomy map + seg prediction





class PathologyEncoder(nn.Module):

    def __init__(self, in_channels=256, embed_dim=64):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(in_channels, embed_dim),

            nn.LayerNorm(embed_dim),

            nn.Tanh(),

        )



    def forward(self, z_spatial):

        # z_spatial: (B, 256, 32, 32)

        return self.fc(self.pool(z_spatial))  # (B, 64) 

