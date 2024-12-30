import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet50_Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_Model, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)