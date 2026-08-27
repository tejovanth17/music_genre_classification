# Symphony AI — Music Genre Classification & Audio Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![SOTA Accuracy](https://img.shields.io/badge/Top%20Accuracy-91.19%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, end-to-end deep audio processing and music genre classification platform built on the **GTZAN Music Dataset** (1,000 tracks, 9,990 slices, 10 musical genres). Combines Convolutional Neural Networks (CNNs), SOTA Computer Vision backbones (ResNet, YOLO, EfficientNet, MobileNet), high-speed gradient boosting (LightGBM, XGBoost), and an acoustic similarity recommendation engine with a dual-theme (Light/Dark) web interface.

---

## 🌟 Key Features

- **Multi-Model Architecture Suite**:
  - **SOTA Tabular Boosting**: LightGBM (**91.19%** Accuracy) and XGBoost (**89.43%** Accuracy).
  - **Deep Neural Networks**: ResNet-18 (**90.19%**), EfficientNet-B0 (**89.63%**), MobileNetV3-Small (**89.51%**), and Multi-Layer Perceptron.
  - **Computer Vision SpecNet**: 2D Mel-Spectrogram CNN (`models/Trained_model.h5`) with input shape `(210, 210, 1)`.
- **Acoustic Feature Extraction Pipeline**:
  - 128-band Mel-Spectrograms, STFT, 20 MFCCs, Spectral Centroids, Spectral Bandwidth, Rolloff, Zero-Crossing Rate, and Harmony/Percussion separation.
- **AI Sound-Alike Recommender**:
  - Cosine-similarity nearest neighbor recommendation engine finding acoustic matches across 57 spectral attributes.
- **Enterprise Web Interface**:
  - High-contrast solid design with **1-click Light/Dark Theme Switcher** and `localStorage` persistence.
  - Drag-and-drop audio uploader (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
  - Embedded HTML5 audio playback with real-time animated frequency wave bars.
  - Interactive preset sandbox for instant 1-click GTZAN benchmark track testing.
  - Real-time 3-panel audio spectrum visualization (Waveform, Mel-Spectrogram dB, and Spectral Centroid trajectory).
- **Production Ready**:
  - Configurable via environment variables (`PORT`, `HOST`, `FLASK_DEBUG`).
  - Zero hardcoded dataset paths with dynamic path resolution.

---

## 🏆 SOTA Benchmarking Leaderboard

Evaluated on **9,990 audio segments** (3-second slices, 57 acoustic features):

| Rank | Model Architecture | Architecture Category | Accuracy | Weighted F1 | Inference Latency | Parameters | Memory Footprint |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **LightGBM (LGBMClassifier)** | SOTA GBDT Tree Engine | **91.19%** | **0.9119** | 0.046 ms | 300 Trees | Very Low (~4 MB) |
| 🥈 | **ResNet-18 (Deep Residuals)** | Deep Residual CNN | **90.19%** | **0.9018** | 0.018 ms | 83,658 | Low (~1.2 MB) |
| 🥉 | **EfficientNet-B0** | SOTA Compound Scaled | **89.63%** | **0.8963** | **0.013 ms** | 28,986 | Low (~0.9 MB) |
| 4 | **MobileNetV3-Small** | Inverted Residuals + SE | **89.51%** | **0.8948** | 0.015 ms | **17,474 (Ultra-small)** | **Ultra-Low (~0.4 MB)** |
| 5 | **XGBoost Classifier** | Extreme Gradient Boost | **89.43%** | **0.8943** | 0.082 ms | 300 Trees | Low (~5 MB) |
| 6 | **Random Forest (500 Trees)** | Ensemble Bagging | **83.71%** | **0.8363** | 0.120 ms | 500 Trees | Medium (~12 MB) |
| 7 | **Neural Net (MLP 128/64)** | Multi-Layer Perceptron | **83.27%** | **0.8325** | 0.022 ms | 16,522 | Ultra-Low (~0.3 MB) |
| 8 | **K-Nearest Neighbors (KNN)** | Distance Metric | **81.06%** | **0.8113** | 1.990 ms | Non-parametric | Medium (~8 MB) |
| 9 | **Support Vector Machine (SVM)** | RBF Kernel | **75.58%** | **0.7534** | 0.310 ms | Kernel Boundary | Low (~2 MB) |
| 10 | **YOLOv8-cls (CSPNet Nano)** | Cross-Stage Partial Net | **75.54%** | **0.7534** | 0.015 ms | 53,514 | Ultra-Low (~0.6 MB) |

---

## 📂 Project Architecture

```
music_genre_classification/
├── app.py                      # Production Flask web application & REST API
├── train.py                    # Modular training and smoke test CLI entry point
├── train_benchmarks.py         # 10 Classical ML models benchmarking suite
├── train_sota_models.py        # SOTA deep learning benchmark (ResNet, YOLO, EfficientNet, LightGBM)
├── requirements.txt            # Production dependencies
├── README.md                   # System documentation
│
├── src/                        # Modular Core Python Package
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Dynamic dataset & path configuration
│   ├── data_loader.py          # Data ingestion, scaling, train/test split, PCA
│   ├── audio_processing.py     # Librosa spectrogram extraction & visualizations
│   ├── models.py               # Classifier definitions, evaluation & feature importance
│   ├── recommender.py          # Cosine-similarity acoustic recommendation engine
│   ├── eda.py                  # Exploratory Data Analysis & correlation plots
│   ├── smoke_test.py           # Automated 6-step end-to-end smoke verification
│   └── main.py                 # Core CLI dispatcher
│
├── models/                     # Saved model artifacts
│   └── Trained_model.h5        # Pre-trained 2D Mel-Spectrogram CNN
│
├── templates/                  # Responsive Jinja2 HTML Templates
│   ├── index.html              # Homepage, hero, genres, SOTA leaderboard & pipeline
│   ├── upload.html             # Drag & drop uploader with live audio preview & sample test
│   └── result.html             # Classification card, gauge, probability meters & spectrum plot
│
├── static/                     # Static Web Assets
│   ├── css/
│   │   └── style.css           # Solid production design system with Light/Dark CSS variables
│   ├── js/
│   │   └── script.js           # Client-side theme switcher, uploader & audio controller
│   ├── spectrograms/           # Generated real-time spectrogram plots (.gitkeep)
│   └── waveforms/              # Generated waveform visualizers (.gitkeep)
│
└── outputs/                    # Exported Benchmark Reports & Visualizations
    ├── models/                 # Saved Joblib / PyTorch model weights
    ├── plots/                  # Confusion matrices & feature importance plots
    └── reports/                # Benchmark CSV & JSON reports
```

---

## ⚡ Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/tejovanth17/music_genre_classification.git
cd music_genre_classification
pip install -r requirements.txt
```

### 2. Run the Automated Smoke Test
Verify dataset discovery, feature scaling, model training, audio feature extraction, and recommender in seconds:
```bash
python train.py --smoke
```

### 3. Launch the Web Application
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

### 4. Run SOTA Benchmark Comparison
Train and evaluate ResNet-18, YOLOv8-cls, MobileNetV3-Small, EfficientNet-B0, and LightGBM:
```bash
python train_sota_models.py
```

### 5. Run Classical 10-Model Benchmark
```bash
python train_benchmarks.py
```

---

## 🌐 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main interactive web application |
| `GET` | `/upload` | File upload interface |
| `POST` | `/predict` | Upload audio file (`multipart/form-data`) and receive classification |
| `GET` | `/predict-sample/<name>` | 1-click inference for preset GTZAN audio samples |
| `GET` | `/audio-file/<path>` | Stream raw audio slices directly to the browser player |
| `GET` | `/model-info` | Inspect CNN tensor architecture, layers, and class labels |
| `POST` | `/api/predict` | JSON REST API for automated audio genre classification |

---

## 🚀 Production Deployment

### Using Gunicorn (Linux / Container)
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Using Waitress (Windows / Cross-Platform)
```bash
pip install waitress
waitress-serve --port=5000 app:app
```

### Environment Variables
- `PORT`: Server port (default: `5000`)
- `HOST`: Server host binding (default: `0.0.0.0`)
- `FLASK_DEBUG`: Debug mode (`True`/`False`, default: `False`)

---

## 📜 License
Released under the [MIT License](LICENSE).
