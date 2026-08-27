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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'symphony-ai-music-genre-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

# Ensure necessary static directories exist
os.makedirs('static/spectrograms', exist_ok=True)
os.makedirs('static/waveforms', exist_ok=True)

# Global model state
_global_model = None
_global_model_input_shape = None
_global_recommender = None


def get_model():
    """Returns the loaded Keras model, initializing it if necessary."""
    global _global_model, _global_model_input_shape
    if _global_model is None:
        load_and_validate_model()
    return _global_model


def get_recommender():
    """Lazily initializes and returns the music recommendation engine"""
    global _global_recommender
    if _global_recommender is None:
        try:
            _global_recommender = MusicRecommender()
            _global_recommender.fit()
            logger.info("Music Recommender engine initialized.")
        except Exception as e:
            logger.warning(f"Recommender initialization notice: {e}")
    return _global_recommender


def load_and_validate_model():
    """Load deep learning model and validate architecture"""
    global _global_model, _global_model_input_shape
    try:
        # Search for model path relative to app root
        model_paths = [
            Path('models/Trained_model.h5'),
            Path(__file__).resolve().parent / 'models' / 'Trained_model.h5'
        ]
        
        target_path = None
        for p in model_paths:
            if p.exists():
                target_path = p
                break
                
        if not target_path:
            logger.error(f"Model file not found at any of {[str(p) for p in model_paths]}")
            return False
            
        logger.info(f"Loading model from {target_path}...")
        _global_model = load_model(str(target_path))
        _global_model_input_shape = _global_model.input_shape
        
        logger.info(f"Model loaded successfully! Input shape: {_global_model_input_shape}, Output shape: {_global_model.output_shape}")
        
        # Test model with dummy data
        test_input_shape = _global_model_input_shape[1:]
        dummy_input = np.random.random((1,) + test_input_shape)
        test_prediction = _global_model.predict(dummy_input, verbose=0)
        logger.info(f"Test prediction verification completed. Outputs: {test_prediction.shape[-1]}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return False


# Eager warmup on module import for Gunicorn workers
try:
    load_and_validate_model()
except Exception as e:
    logger.warning(f"Eager model warmup warning: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio(file_path):
    """Preprocesses audio into normalized spectrogram tensor for model inference"""
    try:
        logger.info(f"Processing audio file: {file_path}")
        y, sr = librosa.load(file_path, duration=30, sr=22050)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
            
        # Compute 128-band Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=2048, 
            hop_length=512, 
            n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Generate clean spectrogram image matrix for CNN tensor
        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        
        temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_img_path = temp_img.name
        temp_img.close()
        
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        
        # Load and resize for model input shape (210, 210, 1)
        img = tf.keras.preprocessing.image.load_img(
            temp_img_path, 
            target_size=(210, 210), 
            color_mode='grayscale'
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 210, 210, 1)
        
        try:
            os.unlink(temp_img_path)
        except Exception:
            pass
            
        return img_tensor, mel_spec_db, y, sr
        
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None


def generate_visualizations(y, sr, mel_spec_db, session_id):
    """Generates audio waveforms and Mel-Spectrogram plots for UI presentation"""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), facecolor='#0f172a')
        
        # 1. Waveform
        time_axis = np.linspace(0, len(y) / sr, len(y))
        axes[0].plot(time_axis, y, color='#38bdf8', linewidth=0.7)
        axes[0].set_title('Raw Audio Waveform Amplitude', color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        axes[0].set_facecolor('#0b0f19')
        axes[0].tick_params(colors='#94a3b8')
        for spine in axes[0].spines.values():
            spine.set_color('#334155')
        
        # 2. Mel-Spectrogram (dB)
        img = librosa.display.specshow(
            mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', 
            ax=axes[1], cmap='magma'
        )
        axes[1].set_title('128-Band Mel-Frequency Spectrogram (dB)', color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        axes[1].tick_params(colors='#94a3b8')
        for spine in axes[1].spines.values():
            spine.set_color('#334155')
        
        # 3. Spectral Centroid
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        frames = range(len(centroids))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=512)
        axes[2].plot(t, centroids, color='#a855f7', linewidth=1.5, label='Spectral Centroid (Brightness)')
        axes[2].set_title('Spectral Centroid Trajectory', color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        axes[2].set_facecolor('#0b0f19')
        axes[2].tick_params(colors='#94a3b8')
        axes[2].set_xlabel('Time (s)', color='#94a3b8')
        for spine in axes[2].spines.values():
            spine.set_color('#334155')
        
        plt.tight_layout()
        
        spec_filename = f"spec_{session_id}.png"
        spec_path = Path('static/spectrograms') / spec_filename
        plt.savefig(spec_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=120)
        plt.close(fig)
        
        return f"spectrograms/{spec_filename}"
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/model-info')
def model_info():
    """Returns metadata about the active deep learning model"""
    active_model = get_model()
    if active_model is None:
        return jsonify({'error': 'Model not loaded', 'status': 'unavailable'})
    return jsonify({
        'input_shape': str(active_model.input_shape),
        'output_shape': str(active_model.output_shape),
        'number_of_parameters': active_model.count_params(),
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
    active_model = get_model()
    if active_model is None:
        flash("Model is initializing. Please try again in a moment.")
        return redirect(url_for('upload_page'))
        
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
        prediction = active_model.predict(processed_audio, verbose=0)
        
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
    active_model = get_model()
    if active_model is None:
        flash('Model is loading or unavailable. Please check the model file.')
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
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{session_id}_{filename}")
        
        try:
            file.save(temp_path)
            processed_audio, mel_spec_db, y, sr = preprocess_audio(temp_path)
            
            if processed_audio is None:
                flash('Error preprocessing audio file. Please ensure it is a valid, uncorrupted audio track.')
                return redirect(url_for('upload_page'))
                
            visualization_path = generate_visualizations(y, sr, mel_spec_db, session_id)
            
            # Predict
            prediction = active_model.predict(processed_audio, verbose=0)
            
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
            
            # AI Recommendations
            rec_engine = get_recommender()
            recommendations = []
            if rec_engine:
                try:
                    recs_df = rec_engine.recommend(filename, top_n=3)
                    recommendations = recs_df.to_dict(orient='records')
                except Exception as e:
                    logger.warning(f"Recommender query note: {e}")
                    
            try:
                os.unlink(temp_path)
            except Exception:
                pass
                
            return render_template(
                'result.html',
                predicted_genre=predicted_genre,
                confidence=confidence,
                all_predictions=all_predictions,
                filename=filename,
                visualization_path=visualization_path,
                session_id=session_id,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Prediction pipeline error: {e}")
            logger.error(traceback.format_exc())
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            flash(f"Error processing audio track: {str(e)}")
            return redirect(url_for('upload_page'))
            
    else:
        flash('Invalid file format. Allowed formats: .WAV, .MP3, .FLAC, .OGG, .M4A')
        return redirect(url_for('upload_page'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API Endpoint for programmatic audio classification"""
    active_model = get_model()
    if active_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
        
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{secure_filename(file.filename)}")
    
    try:
        file.save(temp_path)
        processed_audio, _, _, _ = preprocess_audio(temp_path)
        os.unlink(temp_path)
        
        if processed_audio is None:
            return jsonify({'error': 'Failed to process audio spectrum'}), 400
            
        prediction = active_model.predict(processed_audio, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = float(prediction[0][predicted_index]) * 100
        
        return jsonify({
            'success': True,
            'filename': file.filename,
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
    logger.info("Starting Flask development server...")
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    
    logger.info(f"Application serving on http://{host}:{port} (debug={debug})")
    app.run(debug=debug, host=host, port=port)
