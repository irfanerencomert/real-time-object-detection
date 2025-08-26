import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import MODEL_CONFIG, TRAIN_CONFIG, MODELS_DIR


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Model'i device'a taşı
        self.model.to(device)

        # Loss function ve optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=MODEL_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=3,
            factor=0.5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Early stopping
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = TRAIN_CONFIG['early_stopping_patience']

    def train_epoch(self):
        """Bir epoch eğitim"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Progress bar güncelle
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc='Validation'):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs):
        """Ana eğitim döngüsü"""
        print(f"Eğitim başlıyor: {num_epochs} epoch, Device: {self.device}")

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            # Eğitim
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate()

            # Learning rate güncelle
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # History güncelle
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')

            # En iyi modeli kaydet
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print(f'Yeni en iyi model kaydedildi! Val Acc: {val_acc:.2f}%')
            else:
                self.patience_counter += 1

            # Checkpoint kaydet
            if (epoch + 1) % TRAIN_CONFIG['save_every'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')

            # Early stopping kontrolü
            if self.patience_counter >= self.early_stopping_patience:
                print(f'Early stopping! {self.early_stopping_patience} epoch boyunca iyileşme yok.')
                break

        print(f'\nEğitim tamamlandı! En iyi val accuracy: {self.best_val_acc:.2f}%')
        return self.history

    def save_model(self, filename):
        """Modeli kaydet"""
        MODELS_DIR.mkdir(exist_ok=True)
        filepath = MODELS_DIR / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'model_config': MODEL_CONFIG
        }, filepath)

    def load_model(self, filename):
        """Modeli yükle"""
        filepath = MODELS_DIR / filename
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']

        print(f'Model yüklendi: {filename}, Best Val Acc: {self.best_val_acc:.2f}%')


def plot_training_history(history):
    """Eğitim geçmişini görselleştir"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    # Learning Rate
    ax3.plot(history['lr'])
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('LR')

    # Val Accuracy zoomed
    ax4.plot(history['val_acc'])
    ax4.set_title('Validation Accuracy (Detailed)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()