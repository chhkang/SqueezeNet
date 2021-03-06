import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class FireBlock(nn.Module):
    def __init__(self, n_in, f1x1, e1x1, e3x3):
        super(FireBlock,self).__init__()
        self.conv1x1_1 = nn.Conv2d(n_in, f1x1, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(f1x1, e1x1, kernel_size=1)
        self.conv3x3 = nn.Conv2d(f1x1, e3x3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(f1x1)
        self.bn2 = nn.BatchNorm2d(e1x1)
        self.bn3 = nn.BatchNorm2d(e3x3)
    def forward(self, x):
        out = self.bn1(F.relu(self.conv1x1_1(x)))
        out_1 = self.bn2(self.conv1x1_2(out))
        out_2 = self.bn3(self.conv3x3(out))
        out = F.relu(torch.cat([out_1, out_2], 1))
        return out

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,96,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.Fire1 = self._make_layer(96,[16,64,64,16,64,64,32,128,128])
        self.Fire2 = self._make_layer(256,[32,128,128,48,192,192,48,192,192,64,256,256])
        self.Fire3 = self._make_layer(512,[64,256,256])
        self.conv2 = nn.Conv2d(512,100,kernel_size=1,stride=1)
        self.bn2 = nn.BatchNorm2d(100)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.conv2:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, n_in, params):
        layers = []
        for i in range(int(len(params)/3)):
            layers.append(FireBlock(n_in,params[i*3],params[i*3+1],params[i*3+2]))
            n_in = 2*params[i*3+1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.Fire1(out)
        out = F.max_pool2d(out, 3, stride=2, ceil_mode=True)
        out = self.Fire2(out)
        out = F.max_pool2d(out, 3, stride=2, ceil_mode=True)
        out = self.dropout(self.Fire3(out))
        out = self.conv2(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out
