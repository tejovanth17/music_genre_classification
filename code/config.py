"""
Configuration and Dynamic Path Management.
Centralizes dataset location resolution, directory creation, constants, and seeds.
"""

import os
from pathlib import Path
from typing import Optional

# Standard GTZAN 10 Genres
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

RANDOM_STATE = 42
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_DURATION = 30


def get_project_root() -> Path:
    """Returns project root directory based on file hierarchy."""
    # Since config.py is in <root>/code/config.py
    current_file_dir = Path(__file__).resolve().parent
    return current_file_dir.parent


def resolve_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """
    Dynamically resolves dataset folder without hardcoded absolute paths.
    
    Priority:
    1. Explicit custom_path argument
    2. MUSIC_DATA_DIR environment variable
    3. Music_Data relative to current working directory
    4. Music_Data relative to project root
    """
    candidates = []
    
    if custom_path:
        candidates.append(Path(custom_path).resolve())
    
    env_path = os.environ.get("MUSIC_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path).resolve())
    
    cwd = Path.cwd()
    candidates.append(cwd / "Music_Data")
    candidates.append(cwd / "music" / "Music_Data")
    
    root = get_project_root()
    candidates.append(root / "Music_Data")
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
            
    # Default fallback to project root Music_Data
    default_path = root / "Music_Data"
    return default_path


def get_output_dir(subfolder: Optional[str] = None) -> Path:
    """Gets or creates the output directory for plots, models, reports."""
    root = get_project_root()
    out_dir = root / "outputs"
    if subfolder:
        out_dir = out_dir / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
