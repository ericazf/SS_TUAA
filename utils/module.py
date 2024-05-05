import torch 
import torch.nn as nn 
from torch.nn import Parameter
 
# input 224 * 224 * 3            
#===============ProS-GAN PrototypeNet==============================
class PrototypeNet(nn.Module):
    def __init__(self, bits, num_classes):
        super(PrototypeNet, self).__init__()
        self.bit = bits
        self.feature = nn.Sequential(
            nn.Linear(num_classes, 4096),
            nn.ReLU(True),
            nn.Linear(4096,512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        self.hashcode = nn.Sequential(
            nn.Linear(512, self.bit),
            nn.Tanh()
        )
        
    def forward(self,input):
        f = self.feature(input)
        c = self.classifier(f)
        h = self.hashcode(f)
        return f, h, c  