import torch.nn as nn
import torch

class LeNet5(nn.Module):
    def __init__(self, input_channel=3, image_size=64, num_classes=21):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=6, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2)

        n_flatten = self._flat_dim(input_channel, image_size)
        self.f5 = nn.Linear(n_flatten, out_features=120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)


    def _flat_dim(self, in_ch, img):
        with torch.no_grad():
            tensor = torch.zeros(1, in_ch, img, img)

            tensor = self.s2(torch.relu(self.conv1(tensor)))
            tensor = self.s4(torch.relu(self.conv3(tensor)))
            tensor = tensor.flatten(1)
            
        return tensor.shape[1]
        
    def forward(self, x):
        x = self.s2(torch.relu(self.conv1(x)))
        x = self.s4(torch.relu(self.conv3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)