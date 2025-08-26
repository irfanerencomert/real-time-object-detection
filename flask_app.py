from flask import Flask, render_template, request, jsonify, Response
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import base64
import io
import json
from src.model import get_model
from src.utils import predict_single_image
from config import CLASS_NAMES, MODEL_CONFIG, MODELS_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model
model = None


def load_model():
    """Modeli yükle"""
    global model
    try:
        model = get_model('resnet', num_classes=MODEL_CONFIG['num_classes'])
        model_path = MODELS_DIR / 'best_model.pth'

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model başarıyla yüklendi!")
            return True
        else:
            print("Model dosyası bulunamadı!")
            return False
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        return False


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html', class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """Görüntü tahmin API'si"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya bulunamadı'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400

        # Görüntüyü yükle
        image = Image.open(file.stream).convert('RGB')

        # Tahmin yap
        pred_class, probabilities = predict_single_image(model, image, 'cpu')

        # Sonuçları hazırla
        result = {
            'predicted_class': CLASS_NAMES[pred_class],
            'predicted_index': int(pred_class),
            'confidence': float(probabilities[pred_class]),
            'all_probabilities': {
                CLASS_NAMES[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Base64 encoded görüntü için tahmin"""
    try:
        data = request.get_json()
        image_data = data['image']

        # Base64 decode
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Tahmin yap
        pred_class, probabilities = predict_single_image(model, image, 'cpu')

        result = {
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': float(probabilities[pred_class]),
            'all_probabilities': [float(p) for p in probabilities]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Sistem durumu kontrolü"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })


if __name__ == '__main__':
    # Model yükle
    if load_model():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Model yüklenemedi, uygulama başlatılamıyor!")