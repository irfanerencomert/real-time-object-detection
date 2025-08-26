import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from config import CLASS_NAMES, MODEL_CONFIG


def predict_single_image(model, image_path, device='cuda'):
    """Tek bir görüntü için tahmin yap"""
    transform = transforms.Compose([
        transforms.Resize((MODEL_CONFIG['img_size'], MODEL_CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Görüntüyü yükle ve işle
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path

    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(outputs, 1)

    return predicted.item(), probabilities.cpu().numpy()


def evaluate_model(model, test_loader, device='cuda'):
    """Model performansını değerlendir"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Classification report
    report = classification_report(all_targets, all_preds,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    return report, cm


def plot_confusion_matrix(cm, class_names):
    """Confusion matrix'i görselleştir"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, device='cuda', num_samples=16):
    """Model tahminlerini görselleştir"""
    model.eval()

    # Denormalizasyon için
    denormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i * len(data) >= num_samples:
                break

            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            for j in range(min(len(data), num_samples - i * len(data))):
                idx = i * len(data) + j
                if idx >= num_samples:
                    break

                # Görüntüyü denormalize et
                img = denormalize(data[j]).cpu()
                img = torch.clamp(img, 0, 1)

                axes[idx].imshow(img.permute(1, 2, 0))
                axes[idx].set_title(f'True: {CLASS_NAMES[targets[j]]}\n'
                                    f'Pred: {CLASS_NAMES[predicted[j]]}\n'
                                    f'Conf: {probabilities[j][predicted[j]]:.2f}')
                axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook'ları kaydet
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward(retain_graph=True)

        # CAM hesapla
        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam / torch.max(cam)

        return cam.cpu().numpy()