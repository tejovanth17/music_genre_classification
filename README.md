# Music Genre Classification & Audio Analysis Web Application

An end-to-end deep learning system and web application for classifying music audio into 10 genres using TensorFlow, Keras, Librosa, and Flask.

---

## 🎧 Features

- **Multi-Genre Prediction**: Classifies tracks into 10 GTZAN genres: *Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock*.
- **Audio Processing Pipeline**: Computes 128-band Mel-frequency spectrograms and audio features with Librosa.
- **Interactive Visualizations**: Real-time generation of audio Waveform, Mel-Spectrogram, and Spectral Centroid plots for uploaded tracks.
- **Web UI & REST API**: Modern web interface and programmatic `/api/predict` endpoint for headless predictions.
- **Model Debugging**: `/model-info` endpoint for inspecting model architecture and tensor shapes.

---

## 🛠️ Project Structure

```
├── app.py                                                   # Main Flask web application & REST API
├── requirements.txt                                         # Python dependencies
├── work-w-audio-data-visualise-classify-recommend.ipynb     # Jupyter notebook for analysis & training
├── models/
│   └── Trained_model.h5                                     # Pre-trained CNN model
├── templates/
│   ├── index.html                                           # Landing page
│   ├── upload.html                                          # Upload interface
│   └── result.html                                          # Results & visualizations page
└── static/
    ├── css/style.css                                        # UI stylesheet
    └── js/script.js                                         # Frontend scripts
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/tejovanth17/music_genre_classification.git
cd music_genre_classification
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`.

---

## 📡 API Usage

You can send a `POST` request to `/api/predict` with an audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`):

```bash
curl -X POST -F "file=@sample_audio.wav" http://localhost:5000/api/predict
```

**Sample Response:**
```json
{
  "predicted_genre": "rock",
  "confidence": 0.892,
  "all_predictions": {
    "blues": 0.015,
    "classical": 0.001,
    "country": 0.021,
    "disco": 0.012,
    "hiphop": 0.008,
    "jazz": 0.003,
    "metal": 0.045,
    "pop": 0.003,
    "reggae": 0.002,
    "rock": 0.892
  },
  "timestamp": "2026-08-27T16:35:00.000000"
}
```

---

## 📜 License
MIT License
