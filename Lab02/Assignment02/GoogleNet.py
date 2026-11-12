import torch
import torch.nn as nn


class Block(nn.Module):
    """Conv + ReLU, including convolution biases as in Lab 2 B.4."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU())

    def forward(self, x):
        return self.block(x)


class Inception(nn.Module):
    def __init__(self, in_channels, out_block_1, out_block_21, out_block_22,
                 out_block_31, out_block_32, out_block_4):
        super().__init__()
        self.branch_channels = (out_block_1, out_block_22, out_block_32, out_block_4)
        self.out_channels = sum(self.branch_channels)
        self.block1 = Block(in_channels, out_block_1, 1)
        self.block2 = nn.Sequential(
            Block(in_channels, out_block_21, 1),
            Block(out_block_21, out_block_22, 3, padding=1))
        self.block3 = nn.Sequential(
            Block(in_channels, out_block_31, 1),
            Block(out_block_31, out_block_32, 5, padding=2))
        self.block4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), Block(in_channels, out_block_4, 1))

    def forward(self, x):
        return torch.cat([self.block1(x), self.block2(x),
                          self.block3(x), self.block4(x)], dim=1)


class GoogleNet(nn.Module):
    """Reduced GoogLeNet: three Inception blocks for RGB 64x64 images."""

    def __init__(self, input_channel=3, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            Block(input_channel, 32, 3, padding=1), nn.MaxPool2d(2, 2),
            Block(32, 64, 3, padding=1), nn.MaxPool2d(2, 2))
        self.inception1 = Inception(64, 16, 16, 24, 4, 8, 8)  # 56 channels
        self.inception2 = Inception(self.inception1.out_channels, 32, 32, 48, 8, 16, 16)  # 112
        self.pool = nn.MaxPool2d(2, 2)
        self.inception3 = Inception(self.inception2.out_channels, 64, 32, 64, 16, 32, 32)  # 192
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.inception3.out_channels, num_classes)

    def stages(self):
        return [('stem', self.stem), ('inception1', self.inception1),
                ('inception2', self.inception2), ('pool', self.pool),
                ('inception3', self.inception3), ('globalpool', self.globalpool)]

    def forward(self, x):
        for _, layer in self.stages():
            x = layer(x)
        return self.classifier(x.flatten(1))


@torch.no_grad()
def layer_table(model, img=64, in_ch=3):
    parameter = next(model.parameters())
    x = torch.zeros(1, in_ch, img, img, device=parameter.device, dtype=parameter.dtype)
    layers, channels = [], []
    for name, layer in model.stages():
        x = layer(x)
        parameters = sum(p.numel() for p in layer.parameters())
        layers.append((name, 'x'.join(map(str, x.shape[1:])), parameters))
        if isinstance(layer, Inception):
            expected = sum(layer.branch_channels)
            assert x.shape[1] == expected
            channels.append({'block': name,
                             'formula': ' + '.join(map(str, layer.branch_channels)),
                             'calculated_channels': expected, 'actual_channels': x.shape[1],
                             'parameters': parameters})
    x = model.classifier(x.flatten(1))
    layers.append(('classifier', str(x.shape[1]), sum(p.numel() for p in model.classifier.parameters())))
    total = sum(p for _, _, p in layers)
    assert total == sum(p.numel() for p in model.parameters())
    print(f'{"Layer":<15}{"Output (C,H,W)":<20}{"Parameters":>12}')
    for name, shape, parameters in layers:
        print(f'{name:<15}{shape:<20}{parameters:>12,}')
    print(f'Total parameters: {total:,}')
    for row in channels:
        print(f'{row["block"]}: {row["formula"]} = {row["calculated_channels"]}; actual={row["actual_channels"]}')
    return layers, channels, total
