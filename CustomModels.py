# custom models
import torch
from torch import nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1) # input channel; output channel; kernel size; stride
        self.conv1_bn=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.conv2_bn=nn.BatchNorm2d(128)
        #self.conv2 = nn.Conv2d(64, 256, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*5*5, 1024) # input; output # 6 for 32, 5 for 28 
        self.fc2 = nn.Linear(1024, 256) # input; output
        self.fc3 = nn.Linear(256, 10)
        self.drop_layer = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x

class CNN11(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1) # input channel; output channel; kernel size; stride
        self.conv1_bn=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.conv2_bn=nn.BatchNorm2d(128)
        #self.conv2 = nn.Conv2d(64, 256, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*5*5, 2048) # input; output
        self.fc2 = nn.Linear(2048, 256) # input; output
        self.fc3 = nn.Linear(256, 10)
        self.drop_layer = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x

class CNN12(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1) # input channel; output channel; kernel size; stride
        self.conv1_bn=nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.conv2_bn=nn.BatchNorm2d(128)
        #self.conv2 = nn.Conv2d(64, 256, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*5*5, 512) # input; output
        self.fc2 = nn.Linear(512, 256) # input; output
        self.fc3 = nn.Linear(256, 10)
        self.drop_layer = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x
    
    
    
drop_prob = 0.4
class CNN1_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1) # input channel; output channel; kernel size; stride 
        self.conv1_bn=nn.BatchNorm2d(32) # 32-5+1=28 28-4 = 24
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.conv2_bn=nn.BatchNorm2d(128) # 28-3+1=26 24-2 = 22
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1) # input channel; output channel; kernel size; stride
        self.conv3_bn=nn.BatchNorm2d(256) # 26-3+1=24/2=12  22 - 2 = 20/2 = 10
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.conv4_bn=nn.BatchNorm2d(512)# (12-3+1)/2 = 5  10-2 = 8/2 = 4
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512*4*4, 1024) # input; output 32->5 28->4
        self.fc2 = nn.Linear(1024, 256) # input; output
        self.fc3 = nn.Linear(256, 10)
        self.drop_layer = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = self.conv3(x)
        x = self.pool(F.relu(self.conv3_bn(x)))
        x = self.conv4(x)
        x = self.pool(F.relu(self.conv4_bn(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x

class CNN2_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1) # input channel; output channel; kernel size; stride 
        self.conv1_bn=nn.BatchNorm2d(64) # 32-5+1=28 28-4=24
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.conv2_bn=nn.BatchNorm2d(128) # 28-5+1=24 24-4 = 20
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1) # input channel; output channel; kernel size; stride
        self.conv3_bn=nn.BatchNorm2d(256) # 24-3+1=22/2=11 20-2 = 18/2 = 9
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.conv4_bn=nn.BatchNorm2d(512)# (11-3+1)/2 = 4 19-2)/2 = 3
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512*3*3, 1024) # input; output 32->4 28->3
        self.fc2 = nn.Linear(1024, 256) # input; output
        self.fc3 = nn.Linear(256, 10)
        self.drop_layer = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = self.conv3(x)
        x = self.pool(F.relu(self.conv3_bn(x)))
        x = self.conv4(x)
        x = self.pool(F.relu(self.conv4_bn(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x
    
class CNN3_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1) # input channel; output channel; kernel size; stride 
        self.conv1_bn=nn.BatchNorm2d(64) # 32-3+1=30 28-2 = 26
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv2_bn=nn.BatchNorm2d(128) # 30-3+1=28 26-2 = 24
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1) # input channel; output channel; kernel size; stride
        self.conv3_bn=nn.BatchNorm2d(256) # 28-3+1=26/2=13 24-2= 22/2 = 11
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.conv4_bn=nn.BatchNorm2d(256)# (13-3+1)/2 = 5 11-2 = 9/2 = 4
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*4*4, 1024) # input; output 32->5 28->4
        self.fc2 = nn.Linear(1024, 256) # input; output
        self.fc3 = nn.Linear(256, 10)
        self.drop_layer = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv1_bn(x))
        x = self.conv2(x)
        x = F.relu(self.conv2_bn(x))
        x = self.conv3(x)
        x = self.pool(F.relu(self.conv3_bn(x)))
        x = self.conv4(x)
        x = self.pool(F.relu(self.conv4_bn(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x

