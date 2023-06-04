import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.activation = activation
        self.fc2 = nn.Linear(out_features, out_features)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += residual
        out = self.activation(out)
        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet1D(nn.Module):
    def __init__(self, num_blocks, hidden_size):
        super(ResNet1D, self).__init__()
        self.activation = nn.ReLU()
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(hidden_size, hidden_size, activation=self.activation))
        if len(layers) > 0:
            self.blocks = nn.Sequential(*layers)
        else:
            self.blocks = Identity()       
        
    def forward(self, x):
        out = self.blocks(x)
        return out
