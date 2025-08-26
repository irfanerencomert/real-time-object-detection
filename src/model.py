#import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_CONFIG


class CustomResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CustomResNet, self).__init__()

        # Pre-trained ResNet50 yükle
        self.backbone = models.resnet50(pretrained=True)

        # Feature extraction için tüm katmanları dondur
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Son birkaç katmanı fine-tune için aç
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Custom classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )

        # Ağırlıkları initialize et
        self._initialize_weights()

    def _initialize_weights(self):
        """Yeni katmanların ağırlıklarını initialize et"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetModel(nn.Module):
    """Alternatif model: EfficientNet"""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(EfficientNetModel, self).__init__()

        # EfficientNet-B0 yükle
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=True)

        # Feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Son katmanları fine-tune için aç
        for param in self.backbone.features[-3:].parameters():
            param.requires_grad = True

        # Classifier değiştir
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def get_model(model_type='resnet', num_classes=10):
    """Model factory function"""
    if model_type == 'resnet':
        return CustomResNet(num_classes=num_classes,
                            dropout_rate=MODEL_CONFIG['dropout_rate'])
    elif model_type == 'efficientnet':
        return EfficientNetModel(num_classes=num_classes,
                                 dropout_rate=MODEL_CONFIG['dropout_rate'])
    else:
        raise ValueError(f"Desteklenmeyen model türü: {model_type}")