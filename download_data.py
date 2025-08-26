import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import os


def download_cifar10():
    """CIFAR-10 veri setini indir ve hazırla"""

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Transform tanımla
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # CIFAR-10 indir
    print("CIFAR-10 veri seti indiriliyor...")

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"Eğitim seti boyutu: {len(train_dataset)}")
    print(f"Test seti boyutu: {len(test_dataset)}")
    print("Veri seti başarıyla indirildi!")


if __name__ == "__main__":
    download_cifar10()