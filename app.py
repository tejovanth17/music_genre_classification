from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from werkzeug.utils import secure_filename
import tempfile
import uuid
import logging
from datetime import datetime
import traceback

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
app.secret_key = 'your-secret-key-here-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# GTZAN dataset genres - CRITICAL: This order must match your model's training labels!
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

# Global variables for model
model = None
model_input_shape = None

def load_and_validate_model():
    """Load model and validate its architecture"""
    global model, model_input_shape
    
    try:
        model = load_model('models/Trained_model.h5')
        model_input_shape = model.input_shape
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Model input shape: {model_input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        logger.info(f"Number of model outputs: {model.output_shape[-1]}")
        
        # Validate model output matches number of genres
        if model.output_shape[-1] != len(GENRES):
            logger.warning(f"Model output dimension ({model.output_shape[-1]}) doesn't match number of genres ({len(GENRES)})")
        
        # Test model with dummy data
        test_input_shape = model_input_shape[1:]  # Remove batch dimension
        dummy_input = np.random.random((1,) + test_input_shape)
        test_prediction = model.predict(dummy_input, verbose=0)
        logger.info(f"Test prediction shape: {test_prediction.shape}")
        logger.info(f"Test prediction values: {test_prediction[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path):
    """Preprocess audio for model prediction with extensive logging"""
    try:
        logger.info(f"Processing audio file: {file_path}")
        
        # Load audio - these parameters should match your training data!
        y, sr = librosa.load(file_path, duration=30, sr=22050)
        logger.info(f"Audio loaded: duration={len(y)/sr:.2f}s, sample_rate={sr}, shape={y.shape}")
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        # Extract Mel spectrogram with consistent parameters
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )
        
        logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        logger.info(f"Mel spectrogram dB range: [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")
        
        # Standardization (Z-score normalization)
        mel_spec_mean = np.mean(mel_spec_db)
        mel_spec_std = np.std(mel_spec_db)
        
        if mel_spec_std == 0:
            logger.warning("Zero standard deviation in spectrogram, using small epsilon")
            mel_spec_std = 1e-8
        
        mel_spec_norm = (mel_spec_db - mel_spec_mean) / mel_spec_std
        logger.info(f"Normalized spectrogram stats: mean={np.mean(mel_spec_norm):.4f}, std={np.std(mel_spec_norm):.4f}")
        
        # Determine target shape based on model input
        if model_input_shape is None:
            target_height, target_width = 128, 128
            logger.warning("Model input shape unknown, using default (128, 128)")
        else:
            # Extract spatial dimensions (assuming format: batch, height, width, channels)
            target_height, target_width = model_input_shape[1], model_input_shape[2]
            logger.info(f"Target shape from model: ({target_height}, {target_width})")
        
        # Resize spectrogram to match model input
        if mel_spec_norm.shape != (target_height, target_width):
            # Use proper interpolation instead of np.resize
            from scipy.ndimage import zoom
            height_scale = target_height / mel_spec_norm.shape[0]
            width_scale = target_width / mel_spec_norm.shape[1]
            mel_spec_resized = zoom(mel_spec_norm, (height_scale, width_scale), order=1)
            logger.info(f"Resized spectrogram from {mel_spec_norm.shape} to {mel_spec_resized.shape}")
        else:
            mel_spec_resized = mel_spec_norm
        
        # Add batch and channel dimensions
        if len(model_input_shape) == 4:  # (batch, height, width, channels)
            mel_spec_final = mel_spec_resized.reshape(1, target_height, target_width, 1)
        elif len(model_input_shape) == 3:  # (batch, height, width)
            mel_spec_final = mel_spec_resized.reshape(1, target_height, target_width)
        else:
            raise ValueError(f"Unsupported model input shape: {model_input_shape}")
        
        logger.info(f"Final input shape: {mel_spec_final.shape}")
        logger.info(f"Final input stats: mean={np.mean(mel_spec_final):.4f}, std={np.std(mel_spec_final):.4f}")
        
        return mel_spec_final, mel_spec_db, y, sr
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None

def generate_visualizations(y, sr, mel_spec_db, session_id):
    """Generate spectrogram and waveform visualizations"""
    try:
        os.makedirs('static/spectrograms', exist_ok=True)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.8)
        axes[0].set_title('Audio Waveform', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Mel spectrogram
        img1 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', 
                                       fmax=8000, ax=axes[1], cmap='viridis')
        axes[1].set_title('Mel-frequency Spectrogram', fontsize=14, fontweight='bold')
        plt.colorbar(img1, ax=axes[1], format='%+2.0f dB')
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        
        axes[2].plot(t, spectral_centroids, color='b', alpha=0.8)
        axes[2].set_title('Spectral Centroid', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Hz')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        spectrogram_path = f'static/spectrograms/{session_id}_analysis.png'
        plt.savefig(spectrogram_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Visualization saved: {spectrogram_path}")
        return spectrogram_path
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return None

def validate_prediction(prediction, filename):
    """Validate and log prediction details"""
    logger.info(f"Raw prediction for {filename}: {prediction}")
    logger.info(f"Prediction shape: {prediction.shape}")
    logger.info(f"Prediction sum: {np.sum(prediction):.6f}")
    
    # Check if prediction looks like probabilities
    if np.any(prediction < 0):
        logger.warning("Prediction contains negative values - might need softmax")
    
    if not np.allclose(np.sum(prediction), 1.0, atol=0.1):
        logger.warning(f"Prediction doesn't sum to 1.0: {np.sum(prediction):.6f}")
    
    # Log individual genre probabilities
    for i, genre in enumerate(GENRES):
        prob = prediction[0][i] * 100
        logger.info(f"{genre}: {prob:.2f}%")
    
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/model-info')
def model_info():
    """Debug endpoint to check model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        # Get model information
        info = {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'number_of_parameters': model.count_params(),
            'genres': GENRES,
            'number_of_genres': len(GENRES)
        }
        
        # Test prediction with random data
        test_input = np.random.random((1,) + model_input_shape[1:])
        test_pred = model.predict(test_input, verbose=0)
        info['test_prediction'] = test_pred.tolist()
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded. Please check the model file.')
        return redirect(url_for('upload_page'))
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('upload_page'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        
        logger.info(f"Processing file: {filename} (session: {session_id})")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Preprocess audio
            processed_audio, mel_spec_db, y, sr = preprocess_audio(temp_path)
            
            if processed_audio is None:
                flash('Error processing audio file. Please check the file format.')
                os.unlink(temp_path)
                return redirect(url_for('upload_page'))
            
            # Generate visualizations
            visualization_path = generate_visualizations(y, sr, mel_spec_db, session_id)
            
            # Make prediction with detailed logging
            logger.info("Making prediction...")
            prediction = model.predict(processed_audio, verbose=0)
            
            # Validate prediction
            validate_prediction(prediction, filename)
            
            # Apply softmax if needed (in case model doesn't have softmax as last layer)
            if not np.allclose(np.sum(prediction), 1.0, atol=0.1):
                logger.info("Applying softmax to normalize probabilities")
                prediction = tf.nn.softmax(prediction).numpy()
            
            # Get predicted genre
            predicted_index = np.argmax(prediction)
            predicted_genre = GENRES[predicted_index]
            confidence = float(prediction[0][predicted_index]) * 100
            
            logger.info(f"Final prediction: {predicted_genre} with {confidence:.2f}% confidence")
            
            # Get all predictions for visualization
            all_predictions = [
                {
                    'genre': GENRES[i],
                    'confidence': float(prediction[0][i]) * 100
                }
                for i in range(len(GENRES))
            ]
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Log final results
            logger.info(f"Top 3 predictions for {filename}:")
            for i, pred in enumerate(all_predictions[:3]):
                logger.info(f"{i+1}. {pred['genre']}: {pred['confidence']:.2f}%")
            
            return render_template('result.html',
                                 predicted_genre=predicted_genre,
                                 confidence=confidence,
                                 all_predictions=all_predictions,
                                 filename=filename,
                                 visualization_path=visualization_path,
                                 session_id=session_id)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error(traceback.format_exc())
            flash(f'Error processing audio: {str(e)}')
            try:
                os.unlink(temp_path)
            except:
                pass
            return redirect(url_for('upload_page'))
    
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, M4A, or OGG files.')
        return redirect(url_for('upload_page'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
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
        
        # Apply softmax if needed
        if not np.allclose(np.sum(prediction), 1.0, atol=0.1):
            prediction = tf.nn.softmax(prediction).numpy()
        
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = float(prediction[0][predicted_index])
        
        os.unlink(temp_path)
        
        return jsonify({
            'predicted_genre': predicted_genre,
            'confidence': confidence,
            'all_predictions': {
                GENRES[i]: float(prediction[0][i]) 
                for i in range(len(GENRES))
            },
            'raw_prediction': prediction.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup/<session_id>')
def cleanup_files(session_id):
    """Clean up generated files"""
    try:
        visualization_path = f'static/spectrograms/{session_id}_analysis.png'
        if os.path.exists(visualization_path):
            os.remove(visualization_path)
            logger.info(f"Cleaned up: {visualization_path}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# Initialize model on startup
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    
    # Load and validate model
    if not load_and_validate_model():
        logger.error("Failed to load model. Please check the model file.")
        exit(1)
    
    # Create necessary directories
    os.makedirs('static/spectrograms', exist_ok=True)
    
    logger.info("Application started successfully!")
    app.run(debug=True, host='0.0.0.0', port=5000)
