# Backbone CSP blocks
# Neck PANet
# Head Decoupled Head attached to the outputs of that PANet

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x_passed_through_convs = self.conv_layers(x)
        if self.shortcut:
            return x + x_passed_through_convs
        else:
            return x_passed_through_convs


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, shortcut=True):
        super().__init__()

        block = []
        for i in range(num_blocks):
            block.append(
                Bottleneck(
                    in_channels // 2,
                    out_channels // 2,
                    shortcut=True,
                )
            )
        self.dense_block = nn.Sequential(*block)

    def forward(self, x):
        c = x.shape[1]
        half = c // 2
        part_1 = x[:, half:, :, :]  # output of the right-side stack
        part_2 = x[:, :half, :, :]  # output of the left-side stack
        part_2_passed = self.dense_block(part_2)
        return torch.cat((part_1, part_2_passed), dim=1)


class YOLOXNeck(nn.Module):
    def __init__(self):
        super().__init__()

        self.up_upsample_block_1 = CSPBlock(
            in_channels=512,
            out_channels=256,
            num_blocks=5,
        )

        self.up_upsample_block_2 = CSPBlock(
            in_channels=512,
            out_channels=256,
            num_blocks=5,
        )
        # PAN Neck Downsample
        self.downsample_1 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
        )

        self.downsample_block_1 = CSPBlock(
            in_channels=512, out_channels=256, num_blocks=5, shortcut=False
        )

        self.downsample_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        self.down_block_2 = CSPBlock(
            in_channels=512, out_channels=256, num_blocks=5, shortcut=False
        )

    def forward(self, backbone_outputs):
        shallow, mid, deep = (
            backbone_outputs  # [1, 256, 80, 80],[1, 256, 40, 40],[1, 256, 20, 20]
        )

        deep_upsampled = F.interpolate(deep, scale_factor=2, mode="nearest")

        fused_mid = torch.cat((deep_upsampled, mid), dim=1)

        mid_processed = self.up_upsample_block_1(fused_mid)

        mid_processed_upwnsampled = F.interpolate(
            mid_processed, scale_factor=2, mode="nearest"
        )

        fused_shallow = torch.cat((mid_processed_upwnsampled, shallow), dim=1)

        shallow_processed = self.up_upsample_block_2(fused_shallow)

        shallow_downsamplled = self.downsample_1(shallow_processed)

        shallow_downsamplled_concated = torch.cat(
            (shallow_downsamplled, mid_processed), dim=1
        )

        mid_pan_out = self.downsample_block_1(shallow_downsamplled_concated)

        mid_downsamplled = self.downsample_2(mid_pan_out)

        mid_downsamplled_concated = torch.cat((mid_downsamplled, deep), dim=1)

        deep_processed = self.down_block_2(mid_downsamplled_concated)

        return shallow_processed, mid_pan_out, deep_processed


class DecopledHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()

        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.cls_pred = nn.Conv2d(out_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(out_channels, 4, kernel_size=3, padding=1)
        self.obj_pred = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)

        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)

        return cls_out, reg_out, obj_out


class CSPDarknet_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2)

        # Stage 1: 64 → 128
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.csp1 = CSPBlock(128, 128, num_blocks=3)

        # Stage 2: 128 → 256  ← shallow output
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.csp2 = CSPBlock(256, 256, num_blocks=3)

        # Stage 3: stays 256  ← mid output
        self.down3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.csp3 = CSPBlock(256, 256, num_blocks=3)

        # Stage 4: stays 256  ← deep output
        self.down4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.csp4 = CSPBlock(256, 256, num_blocks=3)

    def forward(self, x):
            # Image comes in at 640x640
            x = self.stem(x) # Drops to 320x320
            
            # Stage 1: We don't need this feature map, it's too big (160x160). 
            # So we just keep calling it 'x' and pass it along.
            x = self.down1(x)
            x = self.csp1(x)
            
            # Stage 2: The Intern! (80x80). We want to save this!
            x = self.down2(x)
            shallow = self.csp2(x)
            
            # Stage 3: The Manager! (40x40). We want to save this!
            x = self.down3(shallow)
            mid = self.csp3(x)
            
            # Stage 4: The CEO! (20x20). We want to save this!
            x = self.down4(mid)
            deep = self.csp4(x)
            
            return shallow, mid, deep

class YOLOX(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = CSPDarknet_Backbone()
        self.neck = YOLOXNeck()

        self.head_0 = DecopledHead(
            in_channels=256, out_channels=256, num_classes=num_classes
        )
        self.head_1 = DecopledHead(
            in_channels=256, out_channels=256, num_classes=num_classes
        )
        self.head_2 = DecopledHead(
            in_channels=256, out_channels=256, num_classes=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        outputs = self.neck(x)

        shallow_out = self.head_0(outputs[0])
        mid_out = self.head_1(outputs[1])
        deep_out = self.head_2(outputs[2])

        return shallow_out, mid_out, deep_out
