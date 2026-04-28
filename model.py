import torch
import torch.nn as nn

class BaseConv(nn.Module):
    def __init__(self, in_channles, out_channels, kernel_size, stride, padding,):
        super().__init__()
        self.conv = nn.Conv2d(in_channles, out_channels, kernel_size, stride, padding, bias=False)
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

