"""
Automated Smoke Test & Verification Suite.
Validates dataset loading, audio feature processing, rapid model training, and recommendation engine.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Union

# Handle Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .config import resolve_dataset_dir, get_output_dir
from .data_loader import load_feature_data, prepare_train_test_data
from .audio_processing import load_and_trim_audio, extract_features_from_audio, generate_audio_visualizations
from .models import get_classifier_suite, evaluate_model, plot_confusion_matrix
from .recommender import MusicRecommender
from .eda import run_full_eda
from sklearn.preprocessing import LabelEncoder


def run_smoke_test(dataset_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Executes an end-to-end smoke test across the entire music genre pipeline.
    """
    print("\n" + "=" * 65)
    print(">>> STARTING MUSIC GENRE CLASSIFICATION SMOKE TEST <<<")
    print("=" * 65)
    
    start_total = time.time()
    steps_passed = 0
    total_steps = 6

    # 1. Dataset Resolution
    print("\n[Step 1/6] Resolving Dataset Directory...")
    resolved_dir = resolve_dataset_dir(dataset_dir)
    print(f"  -> Resolved Dataset Path: {resolved_dir}")
    if not resolved_dir.exists():
        print(f"  [FAIL] Dataset directory does not exist: {resolved_dir}")
        return False
    print("  [OK] Dataset directory found.")
    steps_passed += 1

    # 2. Data Loading & Preprocessing
    print("\n[Step 2/6] Loading Feature Datasets (30s & 3s)...")
    try:
        df_30 = load_feature_data(dataset_dir=resolved_dir, feature_type="30_sec")
        df_3 = load_feature_data(dataset_dir=resolved_dir, feature_type="3_sec")
        print(f"  -> 30s dataset shape: {df_30.shape}")
        print(f"  -> 3s dataset shape:  {df_3.shape}")
        
        # Split a fast 15% sample for smoke training
        X_tr, X_te, y_tr, y_te, scaler = prepare_train_test_data(
            df_3, test_size=0.25, sample_frac=0.15, scale=True
        )
        print(f"  -> Smoke train split: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
        print(f"  -> Smoke test split:  {X_te.shape[0]} samples")
        print("  [OK] Datasets successfully loaded and normalized.")
        steps_passed += 1
    except Exception as e:
        print(f"  [FAIL] Error during dataset loading: {e}")
        return False

    # 3. Audio Processing & Visualizer
    print("\n[Step 3/6] Testing Audio Feature Extraction & Plots...")
    try:
        audio_files = list(resolved_dir.glob("genres_original/**/*.wav"))
        if not audio_files:
            print("  [WARN] No raw .wav files found in genres_original/, skipping raw audio test.")
        else:
            sample_wav = audio_files[0]
            print(f"  -> Testing on audio sample: {sample_wav.name}")
            y, sr, trimmed = load_and_trim_audio(sample_wav, duration=10)
            feats = extract_features_from_audio(trimmed, sr=sr)
            print(f"  -> Audio loaded: sr={sr}, tempo={feats['tempo']:.1f} BPM, STFT shape={feats['stft_db'].shape}")
            plot_out = generate_audio_visualizations(sample_wav, save_filename="smoke_audio_analysis.png")
            print(f"  -> Saved test visualization: {plot_out.name}")
        print("  [OK] Audio processing passed.")
        steps_passed += 1
    except Exception as e:
        print(f"  [FAIL] Error in audio processing: {e}")
        return False

    # 4. Smoke Training Across Models
    print("\n[Step 4/6] Running Smoke Model Training & Benchmark...")
    try:
        models = get_classifier_suite(fast_mode=True)
        le = LabelEncoder()
        le.fit(y_tr)
        
        trained_results = []
        for name, model in list(models.items())[:6]:  # Test first 6 key models in smoke mode
            t0 = time.time()
            res = evaluate_model(model, X_tr, y_tr, X_te, y_te, label_encoder=le)
            elapsed = time.time() - t0
            print(f"  -> Model '{name}': Accuracy = {res['accuracy']*100:.2f}% (trained in {elapsed:.2f}s)")
            trained_results.append((name, res))
            
        # Save a sample confusion matrix for top model
        best_name, best_res = trained_results[0]
        cm_path = plot_confusion_matrix(best_res["confusion_matrix"], best_res["labels"], f"Smoke_{best_name}")
        print(f"  -> Saved smoke confusion matrix: {cm_path.name}")
        print("  [OK] Smoke training passed.")
        steps_passed += 1
    except Exception as e:
        print(f"  [FAIL] Error during model training: {e}")
        return False

    # 5. Music Recommender Engine
    print("\n[Step 5/6] Testing Music Recommender System...")
    try:
        recommender = MusicRecommender(dataset_dir=resolved_dir)
        recommender.fit()
        sample_song = recommender.list_available_songs(limit=1)[0]
        recs = recommender.recommend(sample_song, top_n=3)
        print(f"  -> Query song: {sample_song}")
        print("  -> Top Recommendations:")
        for idx, row in recs.iterrows():
            print(f"     {idx+1}. {row['recommended_song']} ({row['genre']}) - similarity: {row['similarity_score']}")
        print("  [OK] Recommender system passed.")
        steps_passed += 1
    except Exception as e:
        print(f"  [FAIL] Error in recommender: {e}")
        return False

    # 6. EDA Visualizations
    print("\n[Step 6/6] Testing EDA Generator...")
    try:
        eda_res = run_full_eda(dataset_dir=resolved_dir)
        print(f"  -> Generated: {eda_res['correlation_heatmap'].name}")
        print(f"  -> Generated: {eda_res['bpm_boxplot'].name}")
        print(f"  -> Generated: {eda_res['pca_scatter'].name} (PCA Explained: {eda_res['pca_variance_explained_pct']:.2f}%)")
        print("  [OK] EDA visualization passed.")
        steps_passed += 1
    except Exception as e:
        print(f"  [FAIL] Error in EDA generator: {e}")
        return False

    total_time = time.time() - start_total
    print("\n" + "=" * 65)
    print(f"*** ALL SMOKE TESTS PASSED ({steps_passed}/{total_steps}) in {total_time:.2f}s! ***")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
