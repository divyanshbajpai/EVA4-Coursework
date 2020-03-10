# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import network_blocks as nb

class Net(nn.Module):
    def __init__(self,drop_val=0):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nb.Conv2dBnDr( inc=3, opc=16, ks=(3,3), padding=1, drop_val=drop_val),#3, 32*32*16
            nb.Conv2dBnDr( inc=16, opc=16, ks=(3,3), padding=1, drop_val=drop_val),#5 32*32*16
            nb.Conv2dBnDr( inc=16, opc=16, ks=(3,3), padding=1, drop_val=drop_val)#7 32*32*16
        )

        self.convSep1 = nn.Sequential(
            nb.Conv2d_Sep(inc=16, opc=32, ks=(3,3), padding=1, dilation=1,drop_val=drop_val)#9 32*32*32
        )

        self.convMaxPoint1 = nn.Sequential(
            nb.MaxPoint(inc=32, opc=16)#9 16*16*16
        )

        self.convblock2 = nn.Sequential(
            nb.Conv2dBnDr( inc=16, opc=32, ks=(3,3), padding=1, drop_val=drop_val),#13 16*16*32
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val),#17 16*16*32
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, dilation=2, drop_val=drop_val)#25 14*14*32
        )

        self.convSep2 = nn.Sequential(
            nb.Conv2d_Sep(inc=32, opc=32, ks=(3,3), padding=1, dilation=1,drop_val=drop_val)#33 14*14*32
        )

#        self.convMaxPoint2 = nn.Sequential(
#            nb.MaxPoint(inc=64, opc=32)
#       )

        self.convblock3 = nn.Sequential(
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val),#41 14*14*32
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val),#49 14*14*32
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val)#57 14*14*32
        )
		
        self.convSep3 = nn.Sequential(
            nb.Conv2d_Sep(inc=32, opc=64, ks=(3,3), padding=1, dilation=1,drop_val=drop_val)#65 14*14*64
        )
		
        self.convMaxPoint3 = nn.Sequential(
            nb.MaxPoint(inc=64, opc=32)#65 7*7*32
        )

        self.convblock4 = nn.Sequential(
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val),#71 7*7*32
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val),#87 7*7*32
            nb.Conv2dBnDr( inc=32, opc=32, ks=(3,3), padding=1, drop_val=drop_val)#103 7*7*32
        )
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) #1*1*32
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)#1*1*10
        ) 


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convSep1(x)
        x = self.convMaxPoint1(x)
        x = self.convblock2(x)
        x = self.convSep2(x)
        #x = self.convMaxPoint2(x)
        x = self.convblock3(x)
        x = self.convSep3(x)
        x = self.convMaxPoint3(x)
        x = self.convblock4(x)
        x = self.gap(x)        
        x = self.convblock5(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)