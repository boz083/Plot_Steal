# custom models
import torch
from torch import nn
import torch.nn.functional as F

class CNN_custom(nn.Module):
    def __init__(self, ks = 3, conv_num = 2, fc_num = 2, drop_val = 0, max_pool = True, act = 'relu', channel_num = 64, img_size = 32):
        super().__init__()
        self.conv_num = conv_num
        self.fc_num = fc_num
        self.max_pool = max_pool
        self.act = act #'relu', 'tanh'
        
        chs = channel_num
        og_size = img_size
        
        self.conv1 = nn.Conv2d(3, chs, ks, 1) # input channel; output channel; kernel size; stride
        self.conv1_bn=nn.BatchNorm2d(chs)
        
        self.conv2 = nn.Conv2d(chs, chs, ks, 1)
        self.conv2_bn=nn.BatchNorm2d(chs)
        
        self.conv3 = nn.Conv2d(chs, chs, ks, 1)
        self.conv3_bn=nn.BatchNorm2d(chs)
        
        self.conv4 = nn.Conv2d(chs, chs, ks, 1)
        self.conv4_bn=nn.BatchNorm2d(chs)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        input_size = og_size
        for i in range(conv_num):
            input_size = input_size - ks + 1
        if max_pool:
            input_size = int(input_size/2)
            
        self.fc1 = nn.Linear(chs * input_size * input_size, 512) 
        self.fc2 = nn.Linear(512, 512) # input; output
        self.fc3 = nn.Linear(512, 512) # input; output
        self.fc4 = nn.Linear(512, 10)
        self.drop_layer = nn.Dropout(p=drop_val)

    def forward(self, x):
        act = self.act
        
        if act == 'relu':
            activate = F.relu
        elif act == 'prelu':
            activate = F.prelu
        elif act == 'elu':
            activate = F.elu
        elif act == 'tanh':
            activate = F.tanh
        
        x = self.conv1(x)
        x = activate(self.conv1_bn(x))
        x = self.conv2(x)
        x = activate(self.conv2_bn(x))
            
        if self.conv_num > 2:
            x = self.conv3(x)
            x = activate(self.conv3_bn(x))
            
        if self.conv_num > 3:
            x = self.conv4(x)
            x = activate(self.conv4_bn(x))
        
        if self.max_pool:
            x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = activate(self.fc1(x))
        x = self.drop_layer(x)
        
        if self.fc_num > 2:
            x = activate(self.fc2(x))
            x = self.drop_layer(x)
            
        if self.fc_num > 3:    
            x = activate(self.fc3(x))
            x = self.drop_layer(x)
        
        x = F.softmax(self.fc4(x), dim = 1)
        return x

