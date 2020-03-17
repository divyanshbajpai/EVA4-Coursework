from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


dropout_value = 0.05
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1_1 = nn.Sequential(
          nn.Conv2d(3, 32, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.convblock1_2 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.convblock1_3 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(dropout_value)
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock2_1 = nn.Sequential(
          nn.Conv2d(32, 64, 1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(dropout_value)
        )
        self.convblock2_2 = nn.Sequential(
	      nn.Conv2d(32, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )

        self.convblock2_3 = nn.Sequential(
        nn.Conv2d(64, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )
        
        self.convblock2_4 = nn.Sequential(
        nn.Conv2d(64, 64, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
        )

        self.pool2 = nn.MaxPool2d(2, 2)


        self.convblock3_1 = nn.Sequential(
        nn.Conv2d(64, 128, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
        )
        self.convblock3_2 = nn.Sequential(
        nn.Conv2d(64, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
        )

        self.convblock3_3 = nn.Sequential(
        nn.Conv2d(128, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)      
        )

        self.convblock3_4 = nn.Sequential(
        nn.Conv2d(128, 128, 3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)      
        )

        self.pool3 = nn.MaxPool2d(2, 2)

       
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4))
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 




    def forward(self, x):
        x1 = self.convblock1_1(x) # input conv
        x2 = self.convblock1_2(x1)
        x2 = x1+x2
        x3 = self.convblock1_3(x2)
        x3 = x2+x3
        x4 = self.pool1(x3)
        x5 = self.convblock2_1(x4)
        x4_1 = self.convblock2_2(x4)
        x5 = x4_1+x5
        x6 = self.convblock2_3(x5)
        x6 = x5+x6
        x7 = self.convblock2_4(x6)
        x8 = self.pool2(x7)
        x8_1 = self.convblock3_1(x8)
        x9 = self.convblock3_2(x8)
        x9 = x8_1+x9
        x10 = self.convblock3_3(x9)
        x10 = x9+x10
        x11 = self.convblock3_4(x10)
        x12 = F.adaptive_avg_pool2d(x11, 1)        
        y = self.convblock4(x12)
        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)