"""
Audio Processing & Feature Extraction Module.
Uses Librosa to extract time-domain, spectral, and timbral audio representations.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn.preprocessing

from .config import DEFAULT_SAMPLE_RATE, DEFAULT_DURATION, get_output_dir


def load_and_trim_audio(
    audio_path: Union[str, Path],
    duration: Optional[float] = DEFAULT_DURATION,
    sr: int = DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Loads an audio file and trims leading/trailing silence.
    
    Returns:
        (raw_signal, sample_rate, trimmed_signal)
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    y, sr = librosa.load(str(audio_path), duration=duration, sr=sr)
    trimmed_audio, _ = librosa.effects.trim(y)
    return y, sr, trimmed_audio


def extract_features_from_audio(
    audio_signal: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 20
) -> Dict[str, Any]:
    """
    Extracts core musical and audio features from an audio array.
    """
    # Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    
    # Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Harmonic and Percussive Components
    y_harm, y_perc = librosa.effects.hpss(audio_signal)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio_signal)[0]
    
    # Spectral Features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr)[0]
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sr, hop_length=hop_length)
    
    # Tempo estimation
    try:
        tempo, _ = librosa.beat.beat_track(y=audio_signal, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo[0]) if len(tempo) > 0 else float(tempo)
        else:
            tempo_val = float(tempo)
    except Exception:
        tempo_val = 0.0

    return {
        "sample_rate": sr,
        "duration_sec": len(audio_signal) / sr,
        "tempo": tempo_val,
        "stft_db": stft_db,
        "mel_db": mel_db,
        "y_harm": y_harm,
        "y_perc": y_perc,
        "zero_crossings_count": int(np.sum(librosa.zero_crossings(audio_signal, pad=False))),
        "spectral_centroids": spectral_centroids,
        "spectral_rolloff": spectral_rolloff,
        "mfccs": mfccs,
        "chroma": chroma
    }


def generate_audio_visualizations(
    audio_path: Union[str, Path],
    save_filename: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generates a multi-panel visual report of the audio's physical and acoustic properties.
    """
    audio_path = Path(audio_path)
    _, sr, audio_file = load_and_trim_audio(audio_path)
    feats = extract_features_from_audio(audio_file, sr=sr)
    
    if output_dir is None:
        plots_dir = get_output_dir("plots")
    else:
        plots_dir = Path(output_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
    if save_filename is None:
        save_filename = f"{audio_path.stem}_audio_analysis.png"
        
    out_path = plots_dir / save_filename
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. Waveform
    librosa.display.waveshow(audio_file, sr=sr, ax=axes[0], color="#6366f1", alpha=0.8)
    axes[0].set_title(f"Audio Waveform ({audio_path.name}) - Tempo: {feats['tempo']:.1f} BPM", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Mel-Spectrogram
    img_mel = librosa.display.specshow(feats["mel_db"], sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="viridis")
    axes[1].set_title("Mel-frequency Spectrogram (dB)", fontsize=12, fontweight="bold")
    fig.colorbar(img_mel, ax=axes[1], format="%+2.0f dB")
    
    # 3. Spectral Centroid & Roll-Off Over Time
    frames = range(len(feats["spectral_centroids"]))
    t = librosa.frames_to_time(frames, sr=sr)
    norm_centroid = sklearn.preprocessing.minmax_scale(feats["spectral_centroids"])
    norm_rolloff = sklearn.preprocessing.minmax_scale(feats["spectral_rolloff"])
    
    librosa.display.waveshow(audio_file, sr=sr, ax=axes[2], alpha=0.3, color="#94a3b8")
    axes[2].plot(t, norm_centroid, color="#f59e0b", label="Normalized Spectral Centroid", linewidth=1.5)
    axes[2].plot(t, norm_rolloff, color="#ec4899", label="Normalized Spectral Roll-off", linewidth=1.5)
    axes[2].set_title("Spectral Centroid & Roll-off Trajectory", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Normalized Energy")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    # 4. MFCCs
    img_mfcc = librosa.display.specshow(feats["mfccs"], sr=sr, x_axis="time", ax=axes[3], cmap="coolwarm")
    axes[3].set_title("MFCCs (Mel-Frequency Cepstral Coefficients)", fontsize=12, fontweight="bold")
    fig.colorbar(img_mfcc, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    
    return out_path
