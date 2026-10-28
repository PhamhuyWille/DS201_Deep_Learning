import torch.nn as nn
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int=1, padding: int=0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

        
class Inception(nn.Module):
    def __init__(self, in_channels, out_block_1, out_block_21, out_block_22,
                 out_block_31, out_block_32, out_block_4):         
        super().__init__()

        self.block1 = Block(in_channels, out_block_1, kernel_size=1)
        self.block2 = nn.Sequential(
            Block(in_channels, out_block_21, kernel_size=1),
            Block(out_block_21, out_block_22, kernel_size=3, padding=1)
        )
        self.block3 = nn.Sequential(
            Block(in_channels, out_block_31, kernel_size=1),
            Block(out_block_31, out_block_32, kernel_size=5, padding=2)
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Block(in_channels, out_block_4, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)

class GoogleNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=21):
        super().__init__()

        self.conv1 = Block(in_channels=input_channel,
                           out_channels=32,
                           kernel_size=7,
                           stride=2,
                           padding=3)                   # output = [B, 32, 32, 32]
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,
                                  stride=2,
                                  padding=1)            # output = [B, 16, 16, 32]
        
        
        self.conv2 = Block(in_channels=32,
                           out_channels=64,
                           kernel_size=1,
                           stride=1)                    # output = [B, 16, 16, 64]
        
        self.conv3 = Block(in_channels=64,
                           out_channels=192,
                           kernel_size=3,
                           stride=1,
                           padding=1)                   # output = [B, 16, 16, 192]
        
        self.pool2 = nn.MaxPool2d(kernel_size=3,
                                  stride=2,
                                  padding=1)            # output = [B, 8, 8, 192]

        self.branch1a = Inception(in_channels=192, 
                                  out_block_1=120, 
                                  out_block_21=96, 
                                  out_block_22=120,
                                  out_block_31=96, 
                                  out_block_32=120, 
                                  out_block_4=120)          # output = [B, 8, 8, 480]

        self.branch1b = Inception(in_channels=480, 
                                  out_block_1=192, 
                                  out_block_21=96, 
                                  out_block_22=192,
                                  out_block_31=96, 
                                  out_block_32=192, 
                                  out_block_4=192)          # output = [B, 8, 8, 768]

        self.pool3 = nn.MaxPool2d(kernel_size=3, 
                                  stride=2, 
                                  padding=1)                # output = [B, 4, 4, 768]

        self.branch2a = Inception(in_channels=768, 
                                  out_block_1=240, 
                                  out_block_21=120, 
                                  out_block_22=240,
                                  out_block_31=120, 
                                  out_block_32=240, 
                                  out_block_4=240)          # output = [B, 4, 4, 960]

        self.branch2b = Inception(in_channels=960, 
                                  out_block_1=256, 
                                  out_block_21=128, 
                                  out_block_22=256,
                                  out_block_31=128, 
                                  out_block_32=256, 
                                  out_block_4=256)          # output = [B, 4, 4, 1024]

        self.branch2c = Inception(in_channels=1024, 
                                  out_block_1=385, 
                                  out_block_21=128, 
                                  out_block_22=385,
                                  out_block_31=128, 
                                  out_block_32=385, 
                                  out_block_4=385)          # output = [B, 4, 4, 1540]

        self.branch2d = Inception(in_channels=1540, 
                                  out_block_1=449, 
                                  out_block_21=256, 
                                  out_block_22=449,
                                  out_block_31=256, 
                                  out_block_32=449, 
                                  out_block_4=449)          # output = [B, 4, 4, 1796]

        self.branch2e = Inception(in_channels=1796, 
                                  out_block_1=512, 
                                  out_block_21=256, 
                                  out_block_22=512,
                                  out_block_31=256, 
                                  out_block_32=512, 
                                  out_block_4=512)          # output = [B, 4, 4, 2048]
        
        self.pool4 = nn.MaxPool2d(kernel_size=3,
                                  stride=2, 
                                  padding=1)                # output = [B, 2, 2, 2048]

        self.branch3a = Inception(in_channels=2048, 
                                  out_block_1=768, 
                                  out_block_21=512, 
                                  out_block_22=768,
                                  out_block_31=512, 
                                  out_block_32=768, 
                                  out_block_4=768)          # output = [B, 2, 2, 3072]

        self.branch3b = Inception(in_channels=3072, 
                                  out_block_1=1024, 
                                  out_block_21=512, 
                                  out_block_22=1024,
                                  out_block_31=512, 
                                  out_block_32=1024, 
                                  out_block_4=1024)          # output = [B, 2, 2, 4096]
        
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))       # output = [B, 1, 1, 4096]
        self.flatten = nn.Flatten(start_dim=1)               # output = [B, 4096]
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(4096, 512) 
        self.fc2 = nn.Linear(512, 120)
        self.fc3 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)

        x = self.branch1a(x)
        x = self.branch1b(x)
        x = self.pool3(x)

        x = self.branch2a(x)
        x = self.branch2b(x)
        x = self.branch2c(x)
        x = self.branch2d(x)
        x = self.branch2e(x)
        x = self.pool4(x)

        x = self.branch3a(x)
        x = self.branch3b(x)

        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
