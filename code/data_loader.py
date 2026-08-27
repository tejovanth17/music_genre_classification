"""
Data Loading and Preprocessing Module.
Handles loading GTZAN CSV features, normalization, train/test splitting, and dimensionality reduction.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from .config import resolve_dataset_dir, RANDOM_STATE


def load_feature_data(
    dataset_dir: Optional[Union[str, Path]] = None,
    feature_type: str = "3_sec",
    drop_unnamed: bool = True
) -> pd.DataFrame:
    """
    Loads GTZAN tabular feature dataset from CSV.
    
    Args:
        dataset_dir: Path to Music_Data folder (auto-resolved if None)
        feature_type: '3_sec' for 3-second segments or '30_sec' for full tracks
        drop_unnamed: Whether to drop Unnamed columns if present
        
    Returns:
        pd.DataFrame containing audio features and labels
    """
    base_dir = resolve_dataset_dir(dataset_dir)
    filename = f"features_{feature_type}.csv"
    file_path = base_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Feature dataset file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    if drop_unnamed:
        unnamed_cols = [c for c in df.columns if "Unnamed" in c]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            
    return df


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = "label",
    drop_cols: Tuple[str, ...] = ("filename", "length"),
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
    scale: bool = True,
    sample_frac: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[preprocessing.MinMaxScaler]]:
    """
    Splits features and labels into training and testing datasets with optional normalization.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target label column
        drop_cols: Metadata columns to exclude from training
        test_size: Proportion of dataset to include in the test split
        random_state: Random seed for reproducibility
        scale: If True, applies MinMaxScaler (0 to 1) to numerical features
        sample_frac: Optional fraction (e.g. 0.1) for quick smoke testing
        
    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    data = df.copy()
    
    if sample_frac and 0.0 < sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=random_state)
        
    y = data[target_col]
    
    cols_to_drop = [c for c in drop_cols if c in data.columns] + [target_col]
    X = data.drop(columns=cols_to_drop)
    
    scaler = None
    if scale:
        scaler = preprocessing.MinMaxScaler()
        X_scaled_np = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled_np, columns=X.columns, index=X.index)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler


def compute_pca(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    n_components: int = 2
) -> Tuple[pd.DataFrame, PCA]:
    """
    Applies Principal Component Analysis (PCA) on normalized features.
    
    Returns:
        (principal_df, pca_model)
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    col_names = [f"principal_component_{i+1}" for i in range(n_components)]
    principal_df = pd.DataFrame(data=principal_components, columns=col_names, index=X.index)
    
    if y is not None:
        principal_df["label"] = y.values
        
    return principal_df, pca
