"""
Exploratory Data Analysis (EDA) Module.
Generates correlation heatmaps, BPM boxplots, and PCA scatter visual plots.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import resolve_dataset_dir, get_output_dir, RANDOM_STATE
from .data_loader import load_feature_data, compute_pca, prepare_train_test_data


def generate_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generates and saves correlation heatmap for mean acoustic features.
    """
    if output_path is None:
        output_path = get_output_dir("plots") / "correlation_heatmap.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    spike_cols = [col for col in df.columns if "mean" in col]
    if not spike_cols:
        spike_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "length"]

    corr = df[spike_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.8, center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.6}, ax=ax
    )
    plt.title("Audio Feature Correlation Heatmap (Mean Variables)", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def generate_bpm_boxplot(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generates and saves BPM / Tempo distribution boxplot per music genre.
    """
    if output_path is None:
        output_path = get_output_dir("plots") / "bpm_boxplot.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    tempo_col = "tempo" if "tempo" in df.columns else None
    if not tempo_col or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'label' and 'tempo' columns")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="label", y="tempo", hue="label", data=df, palette="husl", legend=False, ax=ax)

    plt.title("BPM (Beats Per Minute) Distribution by Music Genre", fontsize=14, fontweight="bold")
    plt.xlabel("Genre", fontsize=11, fontweight="bold")
    plt.ylabel("Tempo (BPM)", fontsize=11, fontweight="bold")
    plt.xticks(fontsize=10, rotation=30)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def generate_pca_scatter(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[Path, float]:
    """
    Generates 2D PCA projection of acoustic feature space colored by genre.
    """
    if output_path is None:
        output_path = get_output_dir("plots") / "pca_scatter.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, _, y_train, _, _ = prepare_train_test_data(df, scale=True, test_size=0.2, random_state=RANDOM_STATE)
    pca_df, pca = compute_pca(X_train, y=y_train, n_components=2)
    var_exp = float(np.sum(pca.explained_variance_ratio_) * 100)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="principal_component_1", y="principal_component_2",
        data=pca_df, hue="label", alpha=0.75, s=60, palette="tab10", ax=ax
    )

    plt.title(f"PCA Projection of Music Genres (Explained Variance: {var_exp:.2f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Genre")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path, var_exp


def run_full_eda(dataset_dir: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Runs all exploratory analyses and saves visual plots.
    """
    df_30 = load_feature_data(dataset_dir=dataset_dir, feature_type="30_sec")
    
    plots_dir = Path(output_dir) if output_dir else get_output_dir("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = generate_correlation_heatmap(df_30, output_path=plots_dir / "correlation_heatmap.png")
    bpm_path = generate_bpm_boxplot(df_30, output_path=plots_dir / "bpm_boxplot.png")
    pca_path, var_exp = generate_pca_scatter(df_30, output_path=plots_dir / "pca_scatter.png")

    return {
        "correlation_heatmap": heatmap_path,
        "bpm_boxplot": bpm_path,
        "pca_scatter": pca_path,
        "pca_variance_explained_pct": var_exp
    }
