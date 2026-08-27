"""
Comprehensive Model Benchmark & Training Script.
Trains all 10 machine learning models on both 3-sec and 30-sec feature datasets,
generates detailed metrics, confusion matrices, and serialized model artifacts.
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import resolve_dataset_dir, get_output_dir, RANDOM_STATE, GENRES
from src.data_loader import load_feature_data, prepare_train_test_data
from src.models import (
    get_classifier_suite, evaluate_model, plot_confusion_matrix, compute_feature_importance
)


def run_full_benchmark():
    dataset_dir = resolve_dataset_dir()
    reports_dir = get_output_dir("reports")
    plots_dir = get_output_dir("plots")
    models_dir = get_output_dir("models")
    
    print("=" * 70)
    print(">>> STARTING COMPREHENSIVE MUSIC GENRE CLASSIFIER BENCHMARK <<<")
    print("=" * 70)
    
    # 1. Benchmark on 3-Second Granularity (9,990 samples)
    print("\n[Step 1/2] Loading 3-Second Feature Dataset (features_3_sec.csv)...")
    df_3sec = load_feature_data(dataset_dir=dataset_dir, feature_type="3_sec")
    X_train_3, X_test_3, y_train_3, y_test_3, scaler_3 = prepare_train_test_data(
        df_3sec, scale=True, test_size=0.25, random_state=RANDOM_STATE
    )
    print(f"  -> Train samples: {X_train_3.shape[0]}, Test samples: {X_test_3.shape[0]}, Features: {X_train_3.shape[1]}")
    
    # Label encoder
    le = LabelEncoder()
    le.fit(y_train_3)
    
    models = get_classifier_suite(fast_mode=False)
    
    results = []
    trained_models = {}
    
    print("\nTraining & Benchmarking Models on 3-Second Features:")
    print("-" * 70)
    for model_name, model in models.items():
        print(f"  Training [{model_name:<28}]...", end="", flush=True)
        t0 = time.time()
        eval_res = evaluate_model(model, X_train_3, y_train_3, X_test_3, y_test_3, label_encoder=le)
        elapsed = time.time() - t0
        
        preds_labels = eval_res["predictions"]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_3, preds_labels, average="weighted", zero_division=0
        )
        
        acc = eval_res["accuracy"]
        print(f" -> Acc: {acc*100:6.2f}% | F1: {f1:.4f} | Time: {elapsed:5.2f}s")
        
        results.append({
            "Dataset": "3-Second Granularity",
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (Weighted)": round(prec, 4),
            "Recall (Weighted)": round(rec, 4),
            "F1-Score (Weighted)": round(f1, 4),
            "Training Time (s)": round(elapsed, 2)
        })
        
        trained_models[model_name] = eval_res
        
        # Save individual confusion matrix
        plot_confusion_matrix(
            eval_res["confusion_matrix"], eval_res["labels"],
            model_name=f"{model_name}_3sec",
            output_path=plots_dir / f"cm_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_3sec.png"
        )
        
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # 2. Benchmark on 30-Second Granularity (1,000 samples)
    print("\n[Step 2/2] Loading 30-Second Feature Dataset (features_30_sec.csv)...")
    df_30sec = load_feature_data(dataset_dir=dataset_dir, feature_type="30_sec")
    X_train_30, X_test_30, y_train_30, y_test_30, scaler_30 = prepare_train_test_data(
        df_30sec, scale=True, test_size=0.25, random_state=RANDOM_STATE
    )
    
    print("\nTraining & Benchmarking Models on 30-Second Features:")
    print("-" * 70)
    results_30 = []
    for model_name, model in models.items():
        print(f"  Training [{model_name:<28}]...", end="", flush=True)
        t0 = time.time()
        fresh_model = get_classifier_suite(fast_mode=False)[model_name]
        eval_res_30 = evaluate_model(fresh_model, X_train_30, y_train_30, X_test_30, y_test_30, label_encoder=le)
        elapsed = time.time() - t0
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_30, eval_res_30["predictions"], average="weighted", zero_division=0
        )
        acc = eval_res_30["accuracy"]
        print(f" -> Acc: {acc*100:6.2f}% | F1: {f1:.4f} | Time: {elapsed:5.2f}s")
        
        results_30.append({
            "Dataset": "30-Second Granularity",
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (Weighted)": round(prec, 4),
            "Recall (Weighted)": round(rec, 4),
            "F1-Score (Weighted)": round(f1, 4),
            "Training Time (s)": round(elapsed, 2)
        })
        
    results_30_df = pd.DataFrame(results_30).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # Combine & Save Benchmark Reports
    combined_df = pd.concat([results_df, results_30_df], ignore_index=True)
    csv_report_path = reports_dir / "model_benchmark_report.csv"
    json_report_path = reports_dir / "model_benchmark_report.json"
    
    combined_df.to_csv(csv_report_path, index=False)
    
    with open(json_report_path, "w") as f:
        json.dump({
            "benchmark_3sec": results_df.to_dict(orient="records"),
            "benchmark_30sec": results_30_df.to_dict(orient="records"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
        
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK LEADERBOARD (3-Second Features):")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Save best models
    best_model_name = results_df.iloc[0]["Model"]
    best_eval = trained_models[best_model_name]
    
    best_artifact_path = models_dir / "best_model_artifact.joblib"
    joblib.dump({
        "model_name": best_model_name,
        "model": best_eval["model"],
        "scaler": scaler_3,
        "label_encoder": le,
        "accuracy": best_eval["accuracy"],
        "feature_names": list(X_train_3.columns),
        "genres": GENRES
    }, best_artifact_path)
    
    # Feature Importance for best model
    print(f"\nComputing Feature Importance for Top Model ({best_model_name})...")
    feat_df, feat_plot = compute_feature_importance(
        best_eval["model"], X_test_3, y_test_3, label_encoder=le,
        output_path=plots_dir / "best_model_feature_importance.png"
    )
    
    print(f"\n[SAVED] Benchmark Report: {csv_report_path}")
    print(f"[SAVED] Best Model Artifact: {best_artifact_path}")
    print(f"[SAVED] Feature Importance Plot: {feat_plot}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_benchmark()
