import torch
from torch import nn
from transformers import ResNetForImageClassification

MODEL_ID = 'microsoft/resnet-50'

class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        base = ResNetForImageClassification.from_pretrained(MODEL_ID)
        self.resnet = base.resnet
        self.classifier = nn.Linear(2048, num_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.resnet.eval()
        return self

    def forward(self, images):
        features = self.resnet(images).pooler_output.flatten(1)
        return self.classifier(features)
