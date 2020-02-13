import torch
import torch.nn as nn
import torch.nn.functional as F

class FireBlock(nn.Module):
    def __init__(self,n_in,f1x1,e1x1,e3x3):
        super(FireBlock,self).__init__()
        self.conv1x1_1 = nn.Conv2d(n_in,f1x1, kernel_size=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(f1x1,e1x1,kernel_size=1,bias=False)
        self.conv3x3 = nn.Conv2d(e1x1,e3x3,kernel_size=3,padding=1,bias=False)

    def forward(self,x):
        out = F.relu(self.conv1x1_1(x))
        out = self.conv1x1_2(out)
        out = F.relu(self.conv3x3(out))
        return out

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,96,kernel_size=3,padding=1)
        self.Fire1 = self._make_layer(96,[16,64,64,16,64,64,32,128,128])
        self.Fire2 = self._make_layer(256,[32,128,128,48,192,192,48,192,192,64,256,256])
        self.Fire9 = self._make_layer(512,[64,256,256])
        self.Conv2 = nn.Conv2d(512,100,kernel_size=1,stride=1)
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self,n_in,params):
        layers = []
        for i in range(int(len(params)/3)):
            layers.append(FireBlock(n_in,params[i*3],params[i*3+1],params[i*3+2]))
            n_in = 2*params[i*3+1]
        return nn.Sequential(*layers)
    def forward(self,x):
        out = self.conv1(x)
        out = self.Fire1(out)
        out = F.max_pool2d(out, 3, stride=2)
        out = self.Fire2(out)
        F.max_pool2d()
        out = F.max_pool2d(out, 3, stride=2)
        out = self.dropout(self.Fire9(out))
        out = self.conv2(out)
        out = F.avg_pool2d(out, 8)
        return out
