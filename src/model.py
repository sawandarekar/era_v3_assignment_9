import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50_Model(nn.Module):
    def __init__(self):
        super(ResNet50_Model, self).__init__()

        drop_out_value = 0.1

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),# affine=False),
            nn.Dropout(drop_out_value)
        ) # output_size =    RF=

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),# affine=False),
            nn.Dropout(drop_out_value)
        ) # output_size =   RF=
        
        self.skip1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),# affine=False),
            nn.Dropout(drop_out_value)
        ) # output_size =    RF=

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output_size =   

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),# affine=False),
            nn.Dropout(drop_out_value)
        ) # output_size =    RF=

        #self.skip2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False)

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),# affine=False),
            #nn.Dropout(drop_out_value)
        ) # output_size =   RF=


        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            # nn.ReLU() NEVER!
        ) # output_size =   RF=

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5) 
        )

        #self.dropout = nn.Dropout(drop_out_value)

        



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.skip1(x)
        x = self.convblock3(x)
        #x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        #x = self.skip2(x)
        x = self.convblock7(x)
        #x = self.dropout(x)
        x = self.convblock8(x)
        x = self.gap(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)