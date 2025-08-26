// Global değişkenler
let currentStream = null;
let isWebcamActive = false;
// const CLASS_NAMES = {{ class_names | tojson }};

// Sayfa yüklendiğinde
document.addEventListener('DOMContentLoaded', function() {
    setupDropZone();
    checkModelHealth();
});

// Model sağlık kontrolü
async function checkModelHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (!data.model_loaded) {
            showError('Model yüklenmedi! Lütfen sunucuyu yeniden başlatın.');
        }
    } catch (error) {
        showError('Sunucu bağlantı hatası!');
    }
}

// Drag & Drop ayarları
function setupDropZone() {
    const dropZone = document.getElementById('dropZone');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.getElementById('dropZone').classList.add('dragover');
}

function unhighlight(e) {
    document.getElementById('dropZone').classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            loadImageFromFile(file);
        }
    }
}

// Görüntü yükleme
function loadImage(event) {
    const file = event.target.files[0];
    if (file) {
        loadImageFromFile(file);
    }
}

function loadImageFromFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('previewImage');
        const dropZone = document.getElementById('dropZone');

        img.src = e.target.result;
        img.style.display = 'block';
        dropZone.style.display = 'none';

        // Tahmin butonunu aktif et
        document.getElementById('predictBtn').disabled = false;

        // Webcam'i durdur
        stopWebcam();
    };
    reader.readAsDataURL(file);
}

// Kamera başlatma
async function startCamera() {
    try {
        // Webcam'i durdur
        stopWebcam();

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });

        const video = document.getElementById('cameraStream');
        const img = document.getElementById('previewImage');
        const dropZone = document.getElementById('dropZone');

        video.srcObject = stream;
        video.style.display = 'block';
        img.style.display = 'none';
        dropZone.style.display = 'none';

        currentStream = stream;

        // Fotoğraf çek butonu ekle
        addCaptureButton();

    } catch (error) {
        showError('Kamera erişimi reddedildi: ' + error.message);
    }
}

function addCaptureButton() {
    // Eğer buton yoksa ekle
    if (!document.getElementById('captureBtn')) {
        const button = document.createElement('button');
        button.id = 'captureBtn';
        button.className = 'btn btn-warning mt-2';
        button.innerHTML = '<i class="fas fa-camera"></i> Fotoğraf Çek';
        button.onclick = capturePhoto;

        document.querySelector('#imageContainer').appendChild(button);
    }
}

function capturePhoto() {
    const video = document.getElementById('cameraStream');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    // Canvas'ı görüntüye çevir
    canvas.toBlob(function(blob) {
        const img = document.getElementById('previewImage');
        img.src = URL.createObjectURL(blob);
        img.style.display = 'block';

        // Video'yu gizle
        video.style.display = 'none';

        // Tahmin butonunu aktif et
        document.getElementById('predictBtn').disabled = false;

        // Fotoğraf çek butonunu kaldır
        const captureBtn = document.getElementById('captureBtn');
        if (captureBtn) {
            captureBtn.remove();
        }

        // Stream'i durdur
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
    });
}

// Webcam başlatma
function startWebcam() {
    if (isWebcamActive) {
        stopWebcam();
        return;
    }

    startCamera().then(() => {
        isWebcamActive = true;
        document.querySelector('[onclick="startWebcam()"]').innerHTML =
            '<i class="fas fa-stop"></i> Webcam Durdur';

        // Real-time tahmin başlat
        startRealTimePrediction();
    });
}

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }

    isWebcamActive = false;
    document.querySelector('[onclick="startWebcam()"]').innerHTML =
        '<i class="fas fa-video"></i> Webcam Başlat';

    // UI elemanlarını gizle
    document.getElementById('cameraStream').style.display = 'none';
    document.getElementById('dropZone').style.display = 'flex';

    const captureBtn = document.getElementById('captureBtn');
    if (captureBtn) {
        captureBtn.remove();
    }
}

// Real-time tahmin
function startRealTimePrediction() {
    if (!isWebcamActive) return;

    const video = document.getElementById('cameraStream');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');

    function predict() {
        if (!isWebcamActive) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        // Canvas'ı base64'e çevir
        const imageData = canvas.toDataURL('image/jpeg');

        // Tahmin API'sini çağır
        fetch('/predict_base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Prediction error:', data.error);
            } else {
                updateRealTimeResults(data);
            }
        })
        .catch(error => {
            console.error('Network error:', error);
        });

        // 500ms sonra tekrar tahmin yap
        setTimeout(predict, 500);
    }

    // İlk tahmini başlat
    setTimeout(predict, 1000);
}

function updateRealTimeResults(data) {
    const resultDiv = document.getElementById('predictionResult');
    const classElement = document.getElementById('predictedClass');
    const confidenceElement = document.getElementById('confidence');

    if (classElement && confidenceElement) {
        classElement.textContent = data.predicted_class;
        confidenceElement.textContent = (data.confidence * 100).toFixed(1);
        resultDiv.style.display = 'block';

        // Güven seviyesine göre renk değiştir
        const alertDiv = resultDiv.querySelector('.alert');
        alertDiv.className = data.confidence > 0.7 ?
            'alert alert-success' : 'alert alert-warning';
    }
}

// Görüntü tahmini
async function predictImage() {
    const img = document.getElementById('previewImage');
    if (!img.src || img.style.display === 'none') {
        showError('Lütfen önce bir görüntü yükleyin!');
        return;
    }

    showLoading();

    try {
        // Görüntüyü canvas'a çiz ve base64'e çevir
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);

        const imageData = canvas.toDataURL('image/jpeg');

        const response = await fetch('/predict_base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            showPredictionResult(data);
        }

    } catch (error) {
        showError('Tahmin sırasında hata oluştu: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Sonuçları göster
function showPredictionResult(data) {
    const resultDiv = document.getElementById('predictionResult');
    const classElement = document.getElementById('predictedClass');
    const confidenceElement = document.getElementById('confidence');
    const barsDiv = document.getElementById('probabilityBars');

    // Ana tahmin
    classElement.textContent = data.predicted_class;
    confidenceElement.textContent = (data.confidence * 100).toFixed(1);

    // Olasılık çubukları
    barsDiv.innerHTML = '';
    const sortedProbs = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5); // Top 5

    sortedProbs.forEach(([className, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-item';

        barContainer.innerHTML = `
            <div class="probability-label d-flex justify-content-between">
                <span>${className}</span>
                <span>${percentage}%</span>
            </div>
            <div class="progress">
                <div class="progress-bar bg-primary" role="progressbar"
                     style="width: ${percentage}%" aria-valuenow="${percentage}"
                     aria-valuemin="0" aria-valuemax="100"></div>
            </div>
        `;

        barsDiv.appendChild(barContainer);
    });

    // Sonuç divini göster
    resultDiv.style.display = 'block';
    resultDiv.classList.add('fade-in');

    // Hata mesajını gizle
    document.getElementById('errorMessage').style.display = 'none';
}

// Yardımcı fonksiyonlar
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('predictionResult').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    errorText.textContent = message;
    errorDiv.style.display = 'block';

    document.getElementById('predictionResult').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
}

// Responsive tasarım için
window.addEventListener('resize', function() {
    const video = document.getElementById('cameraStream');
    if (video.style.display !== 'none') {
        // Video boyutunu ayarla
        const container = document.getElementById('imageContainer');
        const containerWidth = container.offsetWidth;
        video.style.maxWidth = containerWidth + 'px';
    }
});