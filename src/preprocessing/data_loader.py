import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import glob

EMOTION_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprise'
}
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse RAVDESS filename to extract metadata.
    Filename format: XX-XX-XX-XX-XX-XX-XX.ext
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    """
    basename = os.path.basename(filename)
    if '.' in basename:
        name_without_ext = basename.rsplit('.', 1)[0]
    else:
        name_without_ext = basename
    parts = name_without_ext.split('-')
    if len(parts) < 7:
        raise ValueError(f"Invalid RAVDESS filename format: {filename}")
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion_code': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    }


def get_emotion_label(filename: str) -> str:
    """Extract emotion label from RAVDESS filename."""
    metadata = parse_filename(filename)
    emotion_code = metadata['emotion_code']
    return EMOTION_map.get(emotion_code, 'unknown')


def get_emotion_label_idx(filename: str) -> int:
    """Get emotion index from filename for model training."""
    emotion = get_emotion_label(filename)
    if emotion in EMOTION_LABELS:
        return EMOTION_LABELS.index(emotion)
    return -1


def load_ravdess_dataset(data_dir: str, audio_only: bool = True, speech_only: bool = True) -> Tuple[List[str], List[int]]:
    """Load RAVDESS dataset from directory.
    
    Args:
        data_dir: Directory containing RAVDESS audio files
        audio_only: If True, only load audio-only files (03-*)
        speech_only: If True, only load speech files (vocal_channel=01)
    
    Returns:
        Tuple of (file_paths, emotion_labels)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    audio_files = list(data_path.glob('**/*.wav'))
    if not audio_files:
        audio_files = list(data_path.glob('**/*.mp4'))
    if not audio_files:
        audio_files = list(data_path.glob('**/*.flac'))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {data_dir}")
    file_paths = []
    emotion_labels = []
    for audio_file in audio_files:
        try:
            metadata = parse_filename(str(audio_file))
            if audio_only and metadata['modality'] != '03':
                continue
            if speech_only and metadata['vocal_channel'] != '01':
                continue
            if metadata['emotion_code'] in EMOTION_map:
                file_paths.append(str(audio_file))
                emotion_labels.append(EMOTION_LABELS.index(EMOTION_map[metadata['emotion_code']]))
        except ValueError:
            continue
    return file_paths, emotion_labels


def create_dataset_csv(data_dir: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Create CSV file with file paths and emotion labels."""
    file_paths, emotion_labels = load_ravdess_dataset(data_dir)
    df = pd.DataFrame({
        'file_path': file_paths,
        'emotion_label': [EMOTION_LABELS[e] for e in emotion_labels],
        'emotion_idx': emotion_labels
    })
    if output_path:
        df.to_csv(output_path, index=False)
    return df


class RAVDESSDataset:
    """PyTorch-style dataset for RAVDESS."""
    
    def __init__(self, file_paths: List[str], emotion_labels: List[int], 
                 extract_features_fn, transform=None):
        self.file_paths = file_paths
        self.emotion_labels = np.array(emotion_labels)
        self.extract_features = extract_features_fn
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        file_path = self.file_paths[idx]
        features = self.extract_features(file_path)
        label = self.emotion_labels[idx]
        if self.transform:
            features = self.transform(features)
        return features, label


def get_train_test_split(file_paths: List[str], emotion_labels: List[int], 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Split dataset into train and test sets, stratified by emotion."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        file_paths, emotion_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=emotion_labels
    )