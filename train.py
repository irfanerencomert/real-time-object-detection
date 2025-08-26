#!/usr/bin/env python3
"""
Real-time Object Detection Model Training Script
"""

import argparse
import torch
#import torch.nn as nn
#from pathlib import Path

from src.data_preparation import CIFAR10DataModule
from src.model import get_model
from src.train import Trainer, plot_training_history
from src.utils import evaluate_model, plot_confusion_matrix
from config import MODEL_CONFIG, TRAIN_CONFIG, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description='Train Object Detection Model')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=MODEL_CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=MODEL_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')

    args = parser.parse_args()

    # Device seçimi
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)

    # Veri yükleme
    print("Veri yükleniyor...")
    data_module = CIFAR10DataModule(batch_size=args.batch_size)
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model oluşturma
    print("Model oluşturuluyor...")
    model = get_model(args.model, num_classes=MODEL_CONFIG['num_classes'])

    # Model özeti
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Trainer oluşturma
    trainer = Trainer(model, train_loader, val_loader, device=device)

    # Checkpoint'tan devam et
    if args.resume:
        print(f"Checkpoint yükleniyor: {args.resume}")
        trainer.load_model(args.resume)

    # Eğitim
    print("Eğitim başlıyor...")
    history = trainer.train(args.epochs)

    # Eğitim geçmişini görselleştir
    print("Eğitim geçmişi görselleştiriliyor...")
    plot_training_history(history)

    # Test seti değerlendirmesi
    print("Test seti değerlendiriliyor...")
    trainer.load_model('best_model.pth')  # En iyi modeli yükle
    report, cm = evaluate_model(model, test_loader, device)

    # Sonuçları yazdır
    print("\nTest Seti Sonuçları:")
    print("-" * 30)
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"{class_name:12}: "
                  f"Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, "
                  f"F1: {metrics['f1-score']:.3f}")

    print(f"\nGenel Doğruluk: {report['accuracy']:.3f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")

    # Confusion matrix
    plot_confusion_matrix(cm, data_module.get_class_names())

    print(f"\nEğitim tamamlandı! En iyi model: {MODELS_DIR}/best_model.pth")


if __name__ == "__main__":
    main()