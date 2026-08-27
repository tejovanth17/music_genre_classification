"""
Machine Learning Models Module.
Implements benchmark suite of classifiers for audio genre classification, evaluation, and serialization.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRFClassifier

from .config import GENRES, RANDOM_STATE, get_output_dir


def get_classifier_suite(fast_mode: bool = False) -> Dict[str, Any]:
    """
    Returns a dictionary of ML classification models.
    
    Args:
        fast_mode: If True, uses reduced iterations/estimators for quick testing & smoke tests.
    """
    rf_trees = 30 if fast_mode else 500
    xgb_trees = 30 if fast_mode else 300
    mlp_max_iter = 100 if fast_mode else 400
    sgd_max_iter = 500 if fast_mode else 3000

    models = {
        "Naive Bayes": GaussianNB(),
        "Stochastic Gradient Descent": SGDClassifier(max_iter=sgd_max_iter, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7 if fast_mode else 19),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_trees, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Machine": SVC(decision_function_shape="ovo", probability=True, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, solver="lbfgs", max_iter=300
        ),
        "Neural Net (MLP)": MLPClassifier(
            solver="adam", alpha=1e-4, hidden_layer_sizes=(128, 64),
            random_state=RANDOM_STATE, max_iter=mlp_max_iter
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_trees, learning_rate=0.08, random_state=RANDOM_STATE,
            eval_metric="mlogloss", n_jobs=-1
        ),
        "XGBoost RF": XGBRFClassifier(
            n_estimators=rf_trees, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1
        )
    }
    return models


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: Optional[LabelEncoder] = None
) -> Dict[str, Any]:
    """
    Trains and evaluates a model, returning accuracy, predictions, and report.
    """
    # If using XGBoost or models requiring numerical labels
    is_xgb = "XGB" in model.__class__.__name__
    
    if is_xgb and label_encoder is not None:
        y_train_fit = label_encoder.transform(y_train)
        y_test_eval = label_encoder.transform(y_test)
    else:
        y_train_fit = y_train
        y_test_eval = y_test

    model.fit(X_train, y_train_fit)
    preds = model.predict(X_test)
    
    acc = float(accuracy_score(y_test_eval, preds))
    
    if is_xgb and label_encoder is not None:
        preds_labels = label_encoder.inverse_transform(preds)
        target_names = [str(cls) for cls in label_encoder.classes_]
    else:
        preds_labels = preds
        target_names = sorted(list(set(y_test)))
        
    report = classification_report(y_test, preds_labels, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, preds_labels, labels=GENRES if set(GENRES).issubset(set(target_names)) else target_names)
    
    return {
        "accuracy": acc,
        "predictions": preds_labels,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "labels": GENRES if set(GENRES).issubset(set(target_names)) else target_names,
        "model": model
    }


def benchmark_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fast_mode: bool = False
) -> pd.DataFrame:
    """
    Runs full benchmark over all classifiers and returns comparison table.
    """
    models = get_classifier_suite(fast_mode=fast_mode)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    results = []
    for name, model in models.items():
        eval_res = evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder=label_encoder)
        results.append({
            "Model": name,
            "Accuracy": eval_res["accuracy"],
            "Accuracy (%)": round(eval_res["accuracy"] * 100, 2)
        })
        
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return results_df


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    labels: list,
    model_name: str,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Plots and saves a formatted confusion matrix heatmap.
    """
    if output_path is None:
        output_path = get_output_dir("plots") / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=True
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Genre", fontsize=11)
    plt.ylabel("True Genre", fontsize=11)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def compute_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: Optional[LabelEncoder] = None,
    top_n: int = 15,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Computes permutation feature importance for model explainability.
    """
    is_xgb = "XGB" in model.__class__.__name__
    y_eval = label_encoder.transform(y_test) if (is_xgb and label_encoder is not None) else y_test
    
    perm = permutation_importance(
        model, X_test, y_eval, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)
    
    plot_path = None
    if output_path is not None:
        plot_path = Path(output_path)
    else:
        plot_path = get_output_dir("plots") / "feature_importance.png"
        
    top_df = importance_df.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_df, x="Importance_Mean", y="Feature", palette="viridis")
    plt.title(f"Top {top_n} Features (Permutation Importance)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Importance (Drop in Accuracy)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    
    return importance_df, plot_path
