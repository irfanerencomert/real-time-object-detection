import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.model import get_model
from src.utils import predict_single_image
from config import CLASS_NAMES, MODEL_CONFIG, STREAMLIT_CONFIG, MODELS_DIR
import time

# Sayfa konfigürasyonu
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B35;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.prediction-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #FF6B35;
    margin: 1rem 0;
}

.confidence-bar {
    background-color: #e0e0e0;
    border-radius: 10px;
    padding: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Modeli yükle ve cache'le"""
    try:
        model = get_model('resnet', num_classes=MODEL_CONFIG['num_classes'])
        model_path = MODELS_DIR / 'best_model.pth'

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, checkpoint.get('best_val_acc', 0)
        else:
            st.error("Model dosyası bulunamadı! Önce modeli eğitmeniz gerekiyor.")
            return None, 0
    except Exception as e:
        st.error(f"Model yükleme hatası: {str(e)}")
        return None, 0


def create_prediction_chart(probabilities, class_names):
    """Tahmin sonuçları için interaktif grafik oluştur"""
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities * 100
    }).sort_values('Probability', ascending=True)

    fig = px.bar(df, x='Probability', y='Class', orientation='h',
                 title='Sınıf Olasılıkları',
                 color='Probability',
                 color_continuous_scale='viridis',
                 height=400)

    fig.update_layout(
        xaxis_title="Olasılık (%)",
        yaxis_title="Sınıflar",
        showlegend=False
    )

    return fig


def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🔍 Real-time Object Detection</h1>',
                unsafe_allow_html=True)

    # Model yükle
    model, best_acc = load_model()
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.title("⚙️ Ayarlar")
    st.sidebar.success(f"Model başarıyla yüklendi!\nEn iyi doğruluk: {best_acc:.2f}%")

    # Input method seçimi
    input_method = st.sidebar.selectbox(
        "Giriş yöntemi seçin:",
        ["📁 Görüntü Yükle", "📷 Kamera", "🎥 Real-time Webcam"]
    )

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Güven Eşiği (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="Bu değerin üzerindeki tahminler vurgulanır"
    )

    # Sınıf bilgileri
    with st.sidebar.expander("📋 Sınıf Bilgileri"):
        st.write("Model aşağıdaki 10 sınıfı tanıyabilir:")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i}: {class_name}")

    # Ana içerik
    if input_method == "📁 Görüntü Yükle":
        handle_image_upload(model, confidence_threshold)
    elif input_method == "📷 Kamera":
        handle_camera_input(model, confidence_threshold)
    elif input_method == "🎥 Real-time Webcam":
        handle_webcam_input(model, confidence_threshold)


def handle_image_upload(model, confidence_threshold):
    """Görüntü yükleme işlemi"""
    st.header("📁 Görüntü Yükle ve Sınıflandır")

    uploaded_file = st.file_uploader(
        "Bir görüntü seçin:",
        type=STREAMLIT_CONFIG['supported_formats'],
        help=f"Maksimum dosya boyutu: {STREAMLIT_CONFIG['max_file_size']}MB"
    )

    if uploaded_file is not None:
        # Görüntüyü göster
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Yüklenen Görüntü")
            st.image(image, caption="Orijinal Görüntü", use_container_width=True)

        # Tahmin yap
        if st.button("🚀 Sınıflandır", type="primary", use_container_width=True):
            with st.spinner("Tahmin yapılıyor..."):
                try:
                    pred_class, probabilities = predict_single_image(model, image, 'cpu')

                    with col2:
                        st.subheader("Tahmin Sonucu")

                        # Ana tahmin
                        confidence = probabilities[pred_class] * 100
                        predicted_class = CLASS_NAMES[pred_class]

                        # Confidence'a göre renk
                        if confidence >= confidence_threshold:
                            st.success(f"🎯 **{predicted_class}**")
                            st.success(f"Güven: **{confidence:.2f}%**")
                        else:
                            st.warning(f"⚠️ **{predicted_class}**")
                            st.warning(f"Güven: **{confidence:.2f}%** (Düşük)")

                    # Detaylı sonuçlar
                    st.subheader("📊 Detaylı Analiz")

                    # Interaktif grafik
                    fig = create_prediction_chart(probabilities, CLASS_NAMES)
                    st.plotly_chart(fig, use_container_width=True)

                    # Top 3 tahmin
                    st.subheader("🏆 En İyi 3 Tahmin")
                    top3_idx = np.argsort(probabilities)[-3:][::-1]

                    for i, idx in enumerate(top3_idx):
                        conf = probabilities[idx] * 100
                        class_name = CLASS_NAMES[idx]

                        if i == 0:
                            st.markdown(f"🥇 **{class_name}**: {conf:.2f}%")
                        elif i == 1:
                            st.markdown(f"🥈 **{class_name}**: {conf:.2f}%")
                        else:
                            st.markdown(f"🥉 **{class_name}**: {conf:.2f}%")

                        # Progress bar
                        st.progress(conf / 100)

                except Exception as e:
                    st.error(f"Tahmin sırasında hata oluştu: {str(e)}")


def handle_camera_input(model, confidence_threshold):
    """Kamera ile tek görüntü çekme"""
    st.header("📷 Kamera ile Görüntü Çek")

    camera_image = st.camera_input("Bir fotoğraf çekin:")

    if camera_image is not None:
        # Görüntüyü işle
        image = Image.open(camera_image).convert('RGB')

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Çekilen Fotoğraf")
            st.image(image, caption="Kamera Görüntüsü", use_container_width=True)

        # Otomatik tahmin
        with st.spinner("Tahmin yapılıyor..."):
            try:
                pred_class, probabilities = predict_single_image(model, image, 'cpu')

                with col2:
                    st.subheader("Anlık Tahmin")

                    confidence = probabilities[pred_class] * 100
                    predicted_class = CLASS_NAMES[pred_class]

                    if confidence >= confidence_threshold:
                        st.success(f"🎯 **{predicted_class}**")
                        st.success(f"Güven: **{confidence:.2f}%**")
                    else:
                        st.warning(f"⚠️ **{predicted_class}**")
                        st.warning(f"Güven: **{confidence:.2f}%** (Düşük)")

                # Grafik
                fig = create_prediction_chart(probabilities, CLASS_NAMES)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Tahmin sırasında hata oluştu: {str(e)}")


def handle_webcam_input(model, confidence_threshold):
    """Real-time webcam işlemi - Yerel ortam için"""
    st.header("🎥 Real-time Webcam Sınıflandırma (Yerel Ortam)")
    st.warning("⚠️ Bu özellik sadece yerel ortamda çalışır. Web ortamında çalıştırmak için ek ayarlar gerekir.")

    # Webcam için OpenCV kullanımı
    run_webcam = st.checkbox('🔴 Webcam\'i Başlat (Yerel Ortam)')

    if run_webcam:
        st.info("Webcam başlatılıyor... Yerel ortamda OpenCV ile çalışır.")

        # OpenCV ile webcam yakalama
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Webcam açılamadı!")
            return

        # Görüntüleme için placeholders
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        stop_button = st.button('⏹️ Durdur')

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Görüntü alınamadı!")
                break

            # Görüntüyü göster
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            # Belirli aralıklarla tahmin yap
            if int(time.time()) % 3 == 0:  # Her 3 saniyede bir
                with result_placeholder.container():
                    with st.spinner("Tahmin yapılıyor..."):
                        try:
                            # Görüntüyü PIL formatına çevir
                            pil_img = Image.fromarray(frame)
                            pred_class, probabilities = predict_single_image(model, pil_img, 'cpu')

                            confidence = probabilities[pred_class] * 100
                            predicted_class = CLASS_NAMES[pred_class]

                            if confidence >= confidence_threshold:
                                st.success(f"🎯 **{predicted_class}**")
                                st.success(f"Güven: **{confidence:.2f}%**")
                            else:
                                st.warning(f"⚠️ **{predicted_class}**")
                                st.warning(f"Güven: **{confidence:.2f}%** (Düşük)")

                        except Exception as e:
                            st.error(f"Tahmin hatası: {str(e)}")

            if stop_button:
                break

            time.sleep(0.1)  # CPU kullanımını azaltmak için

        cap.release()
        st.success("Webcam durduruldu.")


if __name__ == "__main__":
    main()