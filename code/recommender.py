"""
Music Recommendation Engine.
Calculates cosine similarity across extracted acoustic features to find sonically similar songs.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

from .config import resolve_dataset_dir


class MusicRecommender:
    """
    Content-based music recommendation engine based on audio feature embeddings.
    """

    def __init__(self, dataset_dir: Optional[Union[str, Path]] = None):
        self.dataset_dir = resolve_dataset_dir(dataset_dir)
        self.feature_df: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.DataFrame] = None
        self.sim_matrix_df: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def fit(self, feature_file: str = "features_30_sec.csv") -> "MusicRecommender":
        """
        Loads dataset, scales audio features, and computes the pairwise cosine similarity matrix.
        """
        file_path = self.dataset_dir / feature_file
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found at: {file_path}")

        # Read features with filename as index
        df = pd.read_csv(file_path, index_col="filename")

        # Extract labels
        self.labels = df[["label"]].copy()

        # Drop non-feature columns
        drop_cols = [c for c in ["length", "label"] if c in df.columns]
        features = df.drop(columns=drop_cols)

        # Scale features using StandardScaler
        scaled_features = preprocessing.scale(features)

        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(scaled_features)

        # Build similarity DataFrame with song filenames as row & column index
        self.sim_matrix_df = pd.DataFrame(
            sim_matrix,
            index=df.index,
            columns=df.index
        )
        self.feature_df = features
        self.is_fitted = True
        return self

    def recommend(self, song_name: str, top_n: int = 5) -> pd.DataFrame:
        """
        Returns top_n most similar songs for a query song filename.
        
        Args:
            song_name: Song identifier (e.g. 'pop.00019.wav' or 'pop.00019')
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommended songs, genres, and similarity scores.
        """
        if not self.is_fitted or self.sim_matrix_df is None:
            self.fit()

        # Handle missing extension if provided without .wav
        if song_name not in self.sim_matrix_df.index:
            candidate = f"{song_name}.wav"
            if candidate in self.sim_matrix_df.index:
                song_name = candidate
            else:
                matches = [s for s in self.sim_matrix_df.index if song_name.lower() in s.lower()]
                if matches:
                    song_name = matches[0]
                else:
                    available_sample = list(self.sim_matrix_df.index[:5])
                    raise KeyError(
                        f"Song '{song_name}' not found in database. Examples: {available_sample}"
                    )

        series = self.sim_matrix_df[song_name].sort_values(ascending=False)
        
        # Drop the query song itself
        series = series.drop(song_name)
        
        top_songs = series.head(top_n)
        
        results = []
        for match_name, score in top_songs.items():
            genre = self.labels.loc[match_name, "label"] if self.labels is not None else "unknown"
            results.append({
                "query_song": song_name,
                "recommended_song": match_name,
                "genre": genre,
                "similarity_score": round(float(score), 4)
            })

        return pd.DataFrame(results)

    def list_available_songs(self, genre: Optional[str] = None, limit: int = 10) -> List[str]:
        """Lists song filenames in database, optionally filtered by genre."""
        if not self.is_fitted:
            self.fit()
            
        if self.labels is None:
            return []
            
        if genre:
            filtered = self.labels[self.labels["label"].str.lower() == genre.lower()]
            return list(filtered.index[:limit])
        return list(self.labels.index[:limit])
