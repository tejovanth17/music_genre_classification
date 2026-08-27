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
import joblib
import librosa
import numpy as np
import pandas as pd
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
_sota_artifact = None
_global_recommender = None


def get_sota_model_artifact():
    """Returns the loaded SOTA LightGBM model artifact (model, scaler, label_encoder, feature_names)."""
    global _sota_artifact
    if _sota_artifact is None:
        load_and_validate_sota_model()
    return _sota_artifact


def load_and_validate_sota_model():
    """Loads our newly trained best SOTA LightGBM model (91.19% Accuracy)"""
    global _sota_artifact
    try:
        model_paths = [
            Path('models/sota_lightgbm.joblib'),
            Path('outputs/models/sota_lightgbm.joblib'),
            Path(__file__).resolve().parent / 'models' / 'sota_lightgbm.joblib'
        ]
        
        target_path = None
        for p in model_paths:
            if p.exists():
                target_path = p
                break
                
        if not target_path:
            logger.error(f"SOTA Model file not found at any of {[str(p) for p in model_paths]}")
            return False
            
        logger.info(f"Loading SOTA LightGBM model from {target_path}...")
        _sota_artifact = joblib.load(str(target_path))
        logger.info(f"SOTA LightGBM model loaded successfully! Model: {type(_sota_artifact['model'])}")
        return True
    except Exception as e:
        logger.error(f"Error loading SOTA model: {e}")
        logger.error(traceback.format_exc())
        return False


# Eager warmup on module import for Gunicorn workers
try:
    load_and_validate_sota_model()
except Exception as e:
    logger.warning(f"Eager model warmup warning: {e}")


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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_acoustic_features_and_spec(file_path):
    """
    Extracts 57 statistical acoustic features for SOTA LightGBM classifier
    and generates Mel-Spectrogram matrix for visual inspection.
    """
    try:
        logger.info(f"Extracting acoustic features from: {file_path}")
        y, sr = librosa.load(file_path, duration=30, sr=22050)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
            
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harm, perc = librosa.effects.hpss(y)
        
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
        except Exception:
            tempo_val = 120.0
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 128-band Mel Spectrogram for spectrum visualization
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        feats = {
            'chroma_stft_mean': float(np.mean(chroma_stft)), 'chroma_stft_var': float(np.var(chroma_stft)),
            'rms_mean': float(np.mean(rms)), 'rms_var': float(np.var(rms)),
            'spectral_centroid_mean': float(np.mean(spec_cent)), 'spectral_centroid_var': float(np.var(spec_cent)),
            'spectral_bandwidth_mean': float(np.mean(spec_bw)), 'spectral_bandwidth_var': float(np.var(spec_bw)),
            'rolloff_mean': float(np.mean(rolloff)), 'rolloff_var': float(np.var(rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zcr)), 'zero_crossing_rate_var': float(np.var(zcr)),
            'harmony_mean': float(np.mean(harm)), 'harmony_var': float(np.var(harm)),
            'perceptr_mean': float(np.mean(perc)), 'perceptr_var': float(np.var(perc)),
            'tempo': tempo_val
        }
        for i in range(1, 21):
            feats[f'mfcc{i}_mean'] = float(np.mean(mfcc[i-1]))
            feats[f'mfcc{i}_var'] = float(np.var(mfcc[i-1]))
            
        df_feats = pd.DataFrame([feats])
        return df_feats, mel_spec_db, y, sr
        
    except Exception as e:
        logger.error(f"Error in extract_acoustic_features_and_spec: {e}")
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
    """Returns metadata about the active SOTA deep learning / ML model"""
    art = get_sota_model_artifact()
    if art is None:
        return jsonify({'error': 'Model not loaded', 'status': 'unavailable'})
    return jsonify({
        'model_name': 'SOTA LightGBM Classifier',
        'benchmark_accuracy': '91.19%',
        'weighted_f1': 0.9119,
        'number_of_features': len(art.get('feature_names', [])),
        'genres': GENRES,
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
    """1-Click prediction handler for GTZAN benchmark samples using SOTA model"""
    art = get_sota_model_artifact()
    if art is None:
        flash("Model is initializing. Please try again in a moment.")
        return redirect(url_for('upload_page'))
        
    dataset_dir = resolve_dataset_dir()
    matching_files = list((dataset_dir / 'genres_original').glob(f"**/{sample_name}"))
    if not matching_files:
        flash(f"Sample audio {sample_name} not found.")
        return redirect(url_for('upload_page'))
        
    sample_file_path = matching_files[0]
    genre_folder = sample_file_path.parent.name
    session_id = str(uuid.uuid4())
    
    try:
        df_feats, mel_spec_db, y, sr = extract_acoustic_features_and_spec(str(sample_file_path))
        if df_feats is None:
            flash("Error processing sample audio.")
            return redirect(url_for('upload_page'))
            
        visualization_path = generate_visualizations(y, sr, mel_spec_db, session_id)
        
        # Predict using SOTA LightGBM
        model = art['model']
        scaler = art['scaler']
        le = art['label_encoder']
        feature_names = art['feature_names']
        
        df_scaled = scaler.transform(df_feats[feature_names])
        probs = model.predict_proba(df_scaled)[0]
        pred_idx = np.argmax(probs)
        predicted_genre = le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx]) * 100
        
        all_predictions = [
            {'genre': le.classes_[i], 'confidence': float(probs[i]) * 100}
            for i in range(len(le.classes_))
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
            recommendations=recommendations,
            model_name="SOTA LightGBM Classifier (91.19% Accuracy)"
        )
    except Exception as e:
        logger.error(f"Error predicting sample: {e}")
        flash(f"Error analyzing sample: {e}")
        return redirect(url_for('upload_page'))


@app.route('/predict', methods=['POST'])
def predict():
    art = get_sota_model_artifact()
    if art is None:
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
            df_feats, mel_spec_db, y, sr = extract_acoustic_features_and_spec(temp_path)
            
            if df_feats is None:
                flash('Error preprocessing audio file. Please ensure it is a valid, uncorrupted audio track.')
                return redirect(url_for('upload_page'))
                
            visualization_path = generate_visualizations(y, sr, mel_spec_db, session_id)
            
            # Predict using SOTA LightGBM
            model = art['model']
            scaler = art['scaler']
            le = art['label_encoder']
            feature_names = art['feature_names']
            
            df_scaled = scaler.transform(df_feats[feature_names])
            probs = model.predict_proba(df_scaled)[0]
            pred_idx = np.argmax(probs)
            predicted_genre = le.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx]) * 100
            
            all_predictions = [
                {'genre': le.classes_[i], 'confidence': float(probs[i]) * 100}
                for i in range(len(le.classes_))
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
                recommendations=recommendations,
                model_name="SOTA LightGBM Classifier (91.19% Accuracy)"
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
    """REST API Endpoint for programmatic audio classification using SOTA model"""
    art = get_sota_model_artifact()
    if art is None:
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
        df_feats, _, _, _ = extract_acoustic_features_and_spec(temp_path)
        os.unlink(temp_path)
        
        if df_feats is None:
            return jsonify({'error': 'Failed to process audio spectrum'}), 400
            
        model = art['model']
        scaler = art['scaler']
        le = art['label_encoder']
        feature_names = art['feature_names']
        
        df_scaled = scaler.transform(df_feats[feature_names])
        probs = model.predict_proba(df_scaled)[0]
        pred_idx = np.argmax(probs)
        predicted_genre = le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx]) * 100
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'predicted_genre': predicted_genre,
            'confidence': confidence,
            'model_used': 'SOTA LightGBM Classifier (91.19% Accuracy)',
            'all_predictions': {le.classes_[i]: float(probs[i]) * 100 for i in range(len(le.classes_))},
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
