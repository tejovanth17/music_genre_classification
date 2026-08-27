import os
import tempfile
import uuid
import logging
import traceback
import json
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, render_template, jsonify, flash,
    redirect, url_for, send_from_directory, abort
)
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from src.config import resolve_dataset_dir, GENRES, get_output_dir
from src.recommender import MusicRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'symphony-ai-music-genre-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

# Global model state
model = None
model_input_shape = None
recommender = None


def get_recommender():
    """Lazily initializes and returns the music recommendation engine"""
    global recommender
    if recommender is None:
        try:
            recommender = MusicRecommender()
            recommender.fit()
            logger.info("Music Recommender engine initialized.")
        except Exception as e:
            logger.warning(f"Recommender initialization notice: {e}")
    return recommender


def load_and_validate_model():
    """Load deep learning model and validate architecture"""
    global model, model_input_shape
    try:
        model_path = Path('models/Trained_model.h5')
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            return False
            
        model = load_model(str(model_path))
        model_input_shape = model.input_shape
        
        logger.info(f"Model loaded successfully! Input shape: {model_input_shape}, Output shape: {model.output_shape}")
        
        # Test model with dummy data
        test_input_shape = model_input_shape[1:]
        dummy_input = np.random.random((1,) + test_input_shape)
        test_prediction = model.predict(dummy_input, verbose=0)
        logger.info(f"Test prediction verification completed. Outputs: {test_prediction.shape[-1]}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio(file_path):
    """Preprocesses audio into normalized spectrogram tensor for model inference"""
    try:
        logger.info(f"Processing audio file: {file_path}")
        y, sr = librosa.load(file_path, duration=30, sr=22050)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
            
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Z-score standardization
        mel_spec_mean = np.mean(mel_spec_db)
        mel_spec_std = np.std(mel_spec_db)
        if mel_spec_std == 0:
            mel_spec_std = 1e-8
            
        mel_spec_norm = (mel_spec_db - mel_spec_mean) / mel_spec_std
        
        # Determine target shape
        target_height = model_input_shape[1] if model_input_shape else 210
        target_width = model_input_shape[2] if model_input_shape else 210
        
        if mel_spec_norm.shape != (target_height, target_width):
            from scipy.ndimage import zoom
            height_scale = target_height / mel_spec_norm.shape[0]
            width_scale = target_width / mel_spec_norm.shape[1]
            mel_spec_resized = zoom(mel_spec_norm, (height_scale, width_scale), order=1)
        else:
            mel_spec_resized = mel_spec_norm
            
        if len(model_input_shape) == 4:
            mel_spec_final = mel_spec_resized.reshape(1, target_height, target_width, 1)
        else:
            mel_spec_final = mel_spec_resized.reshape(1, target_height, target_width)
            
        return mel_spec_final, mel_spec_db, y, sr
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None


def generate_visualizations(y, sr, mel_spec_db, session_id):
    """Generates multi-panel spectrogram, waveform, and spectral feature plots"""
    try:
        os.makedirs('static/spectrograms', exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#6366f1', alpha=0.85)
        axes[0].set_title('Audio Amplitude Waveform', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Mel-Spectrogram
        img1 = librosa.display.specshow(
            mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=axes[1], cmap='viridis'
        )
        axes[1].set_title('Mel-Frequency Spectrogram (Energy dB)', fontsize=12, fontweight='bold')
        plt.colorbar(img1, ax=axes[1], format='%+2.0f dB')
        
        # 3. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr)
        axes[2].plot(t, spectral_centroids, color='#ec4899', alpha=0.9, linewidth=1.5)
        axes[2].set_title('Spectral Centroid (Brightness Trajectory)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        spectrogram_path = f'static/spectrograms/{session_id}_analysis.png'
        plt.savefig(spectrogram_path, dpi=120, bbox_inches='tight', facecolor='#ffffff')
        plt.close(fig)
        return spectrogram_path
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return None


# --- Web Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/model-info')
def model_info():
    """Returns metadata about the active deep learning model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    return jsonify({
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'number_of_parameters': model.count_params(),
        'genres': GENRES,
        'number_of_genres': len(GENRES),
        'status': 'active'
    })


@app.route('/audio-file/<path:filename>')
def serve_audio(filename):
    """Streams audio samples from dataset genres folder for in-browser playback"""
    dataset_dir = resolve_dataset_dir()
    genres_dir = dataset_dir / 'genres_original'
    return send_from_directory(str(genres_dir), filename)


@app.route('/predict-sample/<sample_name>')
def predict_sample(sample_name):
    """1-Click prediction handler for GTZAN benchmark samples"""
    dataset_dir = resolve_dataset_dir()
    # Find the audio file inside genres_original
    matching_files = list((dataset_dir / 'genres_original').glob(f"**/{sample_name}"))
    if not matching_files:
        flash(f"Sample audio {sample_name} not found.")
        return redirect(url_for('upload_page'))
        
    sample_file_path = matching_files[0]
    genre_folder = sample_file_path.parent.name
    session_id = str(uuid.uuid4())
    
    try:
        processed_audio, mel_spec_db, y, sr = preprocess_audio(str(sample_file_path))
        if processed_audio is None:
            flash("Error processing sample audio.")
            return redirect(url_for('upload_page'))
            
        visualization_path = generate_visualizations(y, sr, mel_spec_db, session_id)
        prediction = model.predict(processed_audio, verbose=0)
        
        if not np.allclose(np.sum(prediction), 1.0, atol=0.1):
            prediction = tf.nn.softmax(prediction).numpy()
            
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = float(prediction[0][predicted_index]) * 100
        
        all_predictions = [
            {'genre': GENRES[i], 'confidence': float(prediction[0][i]) * 100}
            for i in range(len(GENRES))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get recommendations
        rec_engine = get_recommender()
        recommendations = []
        if rec_engine:
            try:
                recs_df = rec_engine.recommend(sample_name, top_n=3)
                recommendations = recs_df.to_dict(orient='records')
            except Exception as e:
                logger.warning(f"Recommender query error: {e}")
                
        audio_url = f"/audio-file/{genre_folder}/{sample_name}"
        
        return render_template(
            'result.html',
            predicted_genre=predicted_genre,
            confidence=confidence,
            all_predictions=all_predictions,
            filename=sample_name,
            visualization_path=visualization_path,
            session_id=session_id,
            audio_url=audio_url,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error predicting sample: {e}")
        flash(f"Error analyzing sample: {e}")
        return redirect(url_for('upload_page'))


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded. Please check the model file.')
        return redirect(url_for('upload_page'))
    
    # Check if sample preset was chosen
    sample_name = request.form.get('sample_name')
    if sample_name:
        return redirect(url_for('predict_sample', sample_name=sample_name))
        
    if 'file' not in request.files:
        flash('No audio file selected')
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No audio file selected')
        return redirect(url_for('upload_page'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            processed_audio, mel_spec_db, y, sr = preprocess_audio(temp_path)
            if processed_audio is None:
                flash('Error processing audio file. Please check format.')
                os.unlink(temp_path)
                return redirect(url_for('upload_page'))
                
            visualization_path = generate_visualizations(y, sr, mel_spec_db, session_id)
            prediction = model.predict(processed_audio, verbose=0)
            
            if not np.allclose(np.sum(prediction), 1.0, atol=0.1):
                prediction = tf.nn.softmax(prediction).numpy()
            
            predicted_index = np.argmax(prediction)
            predicted_genre = GENRES[predicted_index]
            confidence = float(prediction[0][predicted_index]) * 100
            
            all_predictions = [
                {'genre': GENRES[i], 'confidence': float(prediction[0][i]) * 100}
                for i in range(len(GENRES))
            ]
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Recommendations
            rec_engine = get_recommender()
            recommendations = []
            if rec_engine and rec_engine.is_fitted:
                try:
                    # Pick sample from top predicted genre
                    candidate_songs = rec_engine.list_available_songs(genre=predicted_genre, limit=1)
                    if candidate_songs:
                        recs_df = rec_engine.recommend(candidate_songs[0], top_n=3)
                        recommendations = recs_df.to_dict(orient='records')
                except Exception as e:
                    logger.warning(f"Recommender notice: {e}")
                    
            os.unlink(temp_path)
            
            return render_template(
                'result.html',
                predicted_genre=predicted_genre,
                confidence=confidence,
                all_predictions=all_predictions,
                filename=filename,
                visualization_path=visualization_path,
                session_id=session_id,
                recommendations=recommendations,
                audio_url=None
            )
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error(traceback.format_exc())
            flash(f"Error processing audio: {str(e)}")
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            return redirect(url_for('upload_page'))
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, M4A, or OGG.')
        return redirect(url_for('upload_page'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Headless JSON REST API for audio prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        file.save(tmp_file.name)
        temp_path = tmp_file.name
        
    try:
        processed_audio, _, _, _ = preprocess_audio(temp_path)
        if processed_audio is None:
            return jsonify({'error': 'Error processing audio'}), 400
            
        prediction = model.predict(processed_audio, verbose=0)
        if not np.allclose(np.sum(prediction), 1.0, atol=0.1):
            prediction = tf.nn.softmax(prediction).numpy()
            
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = float(prediction[0][predicted_index])
        
        os.unlink(temp_path)
        return jsonify({
            'predicted_genre': predicted_genre,
            'confidence': confidence,
            'all_predictions': {GENRES[i]: float(prediction[0][i]) for i in range(len(GENRES))},
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    if not load_and_validate_model():
        logger.error("Failed to load model. Exiting.")
        exit(1)
        
    os.makedirs('static/spectrograms', exist_ok=True)
    get_recommender()
    
    logger.info("Application started successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)
