"""
Unified CLI Entry Point for Music Genre Classification.
Allows training models, benchmarking, music recommendations, EDA, and smoke tests.
"""

import argparse
import sys
from pathlib import Path
import joblib

# Handle Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .config import resolve_dataset_dir, get_output_dir, RANDOM_STATE
from .data_loader import load_feature_data, prepare_train_test_data
from .models import (
    get_classifier_suite, evaluate_model, benchmark_all_models,
    plot_confusion_matrix, compute_feature_importance
)
from .audio_processing import generate_audio_visualizations
from .recommender import MusicRecommender
from .eda import run_full_eda
from .smoke_test import run_smoke_test
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser(
        description="Music Genre Classification & Audio Analysis Toolkit"
    )
    
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to Music_Data folder (auto-detected if omitted)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run fast smoke test & smoke training pipeline"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train a specific model"
    )
    parser.add_argument(
        "--model", type=str, default="XGBoost",
        choices=[
            "XGBoost", "Random Forest", "Support Vector Machine",
            "Neural Net (MLP)", "KNN", "Logistic Regression",
            "Decision Tree", "Naive Bayes", "Stochastic Gradient Descent", "XGBoost RF"
        ],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark all classifiers and display comparison table"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use lightweight hyperparameters for fast training"
    )
    parser.add_argument(
        "--eda", action="store_true",
        help="Run full Exploratory Data Analysis and generate plots"
    )
    parser.add_argument(
        "--recommend", type=str, default=None,
        help="Song filename to find similar recommendations for (e.g. pop.00019.wav)"
    )
    parser.add_argument(
        "--audio-analysis", type=str, default=None,
        help="Path to .wav audio file to visualize"
    )
    parser.add_argument(
        "--feature-type", type=str, default="3_sec", choices=["3_sec", "30_sec"],
        help="Feature dataset granularity (default: 3_sec)"
    )

    args = parser.parse_args()

    # If no flags provided, default to --smoke test
    if not any([args.smoke, args.train, args.benchmark, args.eda, args.recommend, args.audio_analysis]):
        print("[INFO] No specific command specified. Running full smoke test by default...")
        args.smoke = True

    # 1. Smoke test
    if args.smoke:
        success = run_smoke_test(dataset_dir=args.data_dir)
        sys.exit(0 if success else 1)

    # 2. EDA
    if args.eda:
        print("\n[EDA] Running Exploratory Data Analysis...")
        results = run_full_eda(dataset_dir=args.data_dir)
        print(f"  [OK] Correlation Heatmap: {results['correlation_heatmap']}")
        print(f"  [OK] BPM Boxplot:         {results['bpm_boxplot']}")
        print(f"  [OK] PCA Scatter:         {results['pca_scatter']} ({results['pca_variance_explained_pct']:.2f}% variance)")

    # 3. Audio Analysis
    if args.audio_analysis:
        print(f"\n[AUDIO] Generating audio analysis for: {args.audio_analysis}")
        out_plot = generate_audio_visualizations(args.audio_analysis)
        print(f"  [OK] Saved visualization to: {out_plot}")

    # 4. Music Recommendation
    if args.recommend:
        print(f"\n[RECOMMENDER] Querying recommendations for: '{args.recommend}'...")
        rec = MusicRecommender(dataset_dir=args.data_dir)
        recs_df = rec.recommend(args.recommend, top_n=5)
        print("\nTop 5 Similar Songs:")
        print(recs_df.to_string(index=False))

    # 5. Benchmark All Models
    if args.benchmark:
        print(f"\n[BENCHMARK] Loading {args.feature_type} dataset for model benchmarking...")
        df = load_feature_data(dataset_dir=args.data_dir, feature_type=args.feature_type)
        X_train, X_test, y_train, y_test, _ = prepare_train_test_data(
            df, scale=True, random_state=RANDOM_STATE
        )
        print(f"Dataset split: {len(X_train)} train, {len(X_test)} test samples.")
        print(f"Running benchmark across all classifiers (fast_mode={args.fast})...\n")
        bench_df = benchmark_all_models(X_train, y_train, X_test, y_test, fast_mode=args.fast)
        print(bench_df.to_string(index=False))

    # 6. Train Selected Model
    if args.train:
        print(f"\n[TRAIN] Training model '{args.model}' on {args.feature_type} features...")
        df = load_feature_data(dataset_dir=args.data_dir, feature_type=args.feature_type)
        X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(
            df, scale=True, random_state=RANDOM_STATE
        )
        
        models_suite = get_classifier_suite(fast_mode=args.fast)
        model = models_suite[args.model]
        
        le = LabelEncoder()
        le.fit(y_train)
        
        eval_res = evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder=le)
        print(f"\n[DONE] Training Complete!")
        print(f"Accuracy: {eval_res['accuracy'] * 100:.2f}%\n")
        
        # Save model artifact
        models_dir = get_output_dir("models")
        model_filename = models_dir / f"{args.model.lower().replace(' ', '_')}_model.joblib"
        joblib.dump({
            "model": eval_res["model"],
            "scaler": scaler,
            "label_encoder": le,
            "accuracy": eval_res["accuracy"],
            "feature_names": list(X_train.columns)
        }, model_filename)
        print(f"  [SAVED] Model artifact saved to: {model_filename}")
        
        # Confusion matrix
        cm_path = plot_confusion_matrix(eval_res["confusion_matrix"], eval_res["labels"], args.model)
        print(f"  [SAVED] Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
