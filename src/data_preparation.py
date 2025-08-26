import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
#import numpy as np
from config import MODEL_CONFIG


class CIFAR10DataModule:
    def __init__(self, batch_size=64, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = MODEL_CONFIG['img_size']

    def get_transforms(self):
        """Eğitim ve validasyon için transform'ları döndür"""

        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def get_dataloaders(self):
        """DataLoader'ları döndür"""

        train_transform, val_transform = self.get_transforms()

        # Tam veri setini yükle
        full_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=val_transform
        )

        # Eğitim setini train/validation olarak böl
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Validation için transform'u değiştir
        val_dataset.dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=val_transform
        )

        # DataLoader'ları oluştur
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader


def get_class_distribution(dataset):
    """Sınıf dağılımını analiz et"""
    class_counts = torch.zeros(10)
    for _, label in dataset:
        class_counts[label] += 1
    return class_counts