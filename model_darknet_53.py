import torch
import torch.nn as nn

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x        


class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        #Bottleneck Architecture
        self.conv_1=BaseConv(in_channels=channels,
                            out_channels=channels//2,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            )
        
        self.conv_2=BaseConv(in_channels=channels//2,
                            out_channels=channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            )
    def forward(self,x):
        x_orig=x
        x=self.conv_1(x)
        x=self.conv_2(x)

        return x_orig+x       


def make_group_of_conv(in_channels,num_blocks):
    layers=[
        BaseConv(
            in_channels=in_channels,
            out_channels=in_channels*2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    ]

    for _ in range(num_blocks):
        layers.append(ResBlock(channels=in_channels*2))
    return nn.Sequential(*layers)


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem=BaseConv(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        
        self.stage1 = make_group_of_conv(in_channels=32,num_blocks=1)   
        self.stage2 = make_group_of_conv(in_channels=64,num_blocks=2)
    
        self.stage3 = make_group_of_conv(in_channels=128,num_blocks=8)#small object detection 
        self.stage4 = make_group_of_conv(in_channels=256,num_blocks=8)#medium object detection 
        self.stage5 = make_group_of_conv(in_channels=512,num_blocks=4)#large object detection
        
    
    def forward(self,x):
        x=self.stem(x)
        x=self.stage1(x)
        x=self.stage2(x)

        out_feature_1=self.stage3(x)
        out_feature_2=self.stage4(out_feature_1)
        out_feature_3=self.stage5(out_feature_2)

        return out_feature_1,out_feature_2,out_feature_3    


class SPP(nn.Module):#Spatial Pyramid Pooling
    def __init__(self,in_channels,out_channels,):    
        super().__init__()

        hidden_channels=in_channels//2

        self.conv_1=BaseConv(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)

        self.pool_1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.pool_2 = nn.MaxPool2d(kernel_size=9,stride=1,padding=4)
        self.pool_3 = nn.MaxPool2d(kernel_size=13,stride=1,padding=6)

        self.conv_2 = BaseConv(hidden_channels*4,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x=self.conv_1(x)

        y1=self.pool_1(x)
        y2=self.pool_2(x)
        y3=self.pool_3(x)

        out=torch.cat([y1,y2,y3,x],dim=1)
        return self.conv_2(out)



#the decopuled head yolo uses copled head ie classification and reg in same conv head
#but in yolox we have de coupled head


class DecopledHead(nn.Module):
    def __init__(self,num_classes,in_channels=256):
        super().__init__()

        self.stem=BaseConv(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
        
        
        self.cls_convs=nn.Sequential(
            BaseConv(256,256,kernel_size=3,stride=1,padding=1),
            BaseConv(256,256,kernel_size=3,stride=1,padding=1),
        
        )

        self.cls_pred=nn.Conv2d(256,num_classes,kernel_size=1,stride=1,padding=0)

        self.reg_convs = nn.Sequential(
            BaseConv(256, 256, kernel_size=3, stride=1, padding=1),
            BaseConv(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.reg_preds = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)       

        self.obj_preds = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)  


    def forward(self,x):
        x=self.stem(x)
        cls_feat=self.cls_convs(x)
        reg_feat=self.reg_convs(x)
        cls_pred=self.cls_pred(cls_feat)
        reg_pred=self.reg_preds(reg_feat)
        obj_pred=self.obj_preds(reg_feat)

        return cls_pred,reg_pred,obj_pred
    

class YoloFPN(nn.Module):
    def __init__(self,in_channels=(256,512,1024),out_channels=256):
        super().__init__()

        self.upsample=nn.Upsample(scale_factor=2,mode="nearest")

        # processing P5 (out_feature_3 + SPP)
        self.conv_out_5=BaseConv(in_channels[2],out_channels,kernel_size=1,stride=1,padding=0)

        # Merging P5 into P4 (out_feature_2)
        self.conv_4_1x1 = BaseConv(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)
        # After concat, channels double (256 + 256 = 512), so we mix them and output 256
        self.conv_out_4 = BaseConv(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

        # Merging P4 into P3 (out_feature_1)
        self.conv_3_1x1 = BaseConv(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)
        # After concat, channels double again, mix them and output 256
        self.conv_out_3 = BaseConv(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self,outfeature_1,outfeature_2,outfeature_3):

        x5=self.conv_out_5(outfeature_3)

        x5_up=self.upsample(x5)

        x4_prepped=self.conv_4_1x1(outfeature_2)

        x4_combined=torch.cat([x4_prepped,x5_up],dim=1)

        x4_out=self.conv_out_4(x4_combined)

        x4_up=self.upsample(x4_out)

        x3_prepped=self.conv_3_1x1(outfeature_1)

        x3_combined=torch.cat([x3_prepped,x4_up],dim=1)

        x3_out=self.conv_out_3(x3_combined)

        return x3_out,x4_out,x5
        



class YOLOx(nn.Module):
    def __init__(self,num_classes=80):
        super().__init__()

        #Instantiate the Darknet53 backbone (640×640×3 -> 20×20×1024)
        self.backbone=Darknet53()

        #Instantiate the SPP block (Attaches to the deepest layer: 1024 channels)
        self.spp=SPP(in_channels=1024,out_channels=1024)

        self.fpn=YoloFPN(in_channels=(256,512,1024),out_channels=256)

        self.head_small = DecopledHead(num_classes, 256)
        self.head_medium = DecopledHead(num_classes, 256)
        self.head_large = DecopledHead(num_classes, 256)



    def forward(self,x):

        out_feature_1,out_feature_2,out_feature_3=self.backbone(x)
        
        #Pass the deepest map (out_3) through the SPP block to get global context        
        out_feature_3 = self.spp(out_feature_3)

        # Pass the features through FPN
        fpn_small, fpn_medium, fpn_large = self.fpn(out_feature_1, out_feature_2, out_feature_3)
        
        # cls_small, reg_small, obj_small = self.head_small(fpn_small)
        # cls_medium, reg_medium, obj_medium = self.head_medium(fpn_medium)
        # cls_large, reg_large, obj_large = self.head_large(fpn_large)

        pred_small = self.head_small(fpn_small)
        pred_medium = self.head_medium(fpn_medium)
        pred_large = self.head_large(fpn_large)

        return pred_small, pred_medium, pred_large

        
# if __name__ == "__main__":
#     # 1. Create a fake image tensor (Batch Size of 1, 3 Color Channels, 640x640 resolution)
#     dummy_image = torch.randn(1, 3, 640, 640)
    
#     # 2. Instantiate your shiny new model!
#     model = YOLOx(num_classes=80)
    
#     # 3. Pass the fake image through the model
#     print("Feeding image to YOLOX...")
#     preds_small, preds_medium, preds_large = model(dummy_image)
    
#     # 4. Print out the shapes of the predictions!
#     print("Success! Here are the output shapes:")
    
#     # Each prediction is a tuple of (Classes, Bounding Boxes, Objectness Score)
#     cls_s, reg_s, obj_s = preds_small
#     print(f"Small Object Head -> Classes: {cls_s.shape}, Boxes: {reg_s.shape}, Obj: {obj_s.shape}")
    
#     cls_m, reg_m, obj_m = preds_medium
#     print(f"Medium Object Head -> Classes: {cls_m.shape}, Boxes: {reg_m.shape}, Obj: {obj_m.shape}")
    
#     cls_l, reg_l, obj_l = preds_large
#     print(f"Large Object Head -> Classes: {cls_l.shape}, Boxes: {reg_l.shape}, Obj: {obj_l.shape}")