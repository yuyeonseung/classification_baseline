import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class BaseModel(nn.Module):
    def __init__(self, num_classes=3):
        super(BaseModel_2D, self).__init__()
        self.backbone = models.efficientnet_b7(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.sigmoid(self.classifier(x))
        return x