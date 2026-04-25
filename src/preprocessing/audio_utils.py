import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


SAMPLE_RATE = 22050
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_DURATION = 2.5


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr


def pad_or_truncate(audio: np.ndarray, fixed_length: int) -> np.ndarray:
    """Pad or truncate audio to fixed length in samples."""
    if len(audio) > fixed_length:
        return audio[:fixed_length]
    elif len(audio) < fixed_length:
        padding = fixed_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant')
    return audio


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Extract MFCC features from audio signal."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return mfcc_features


def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> np.ndarray:
    """Extract Mel Spectrogram features from audio signal."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_all_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract combined MFCC and Mel Spectrogram features."""
    mfcc = extract_mfcc(audio, sr)
    mel_spec = extract_mel_spectrogram(audio, sr)
    combined = np.vstack([mfcc, mel_spec])
    return combined


def extract_features_from_file(file_path: str, use_mel: bool = True, use_mfcc: bool = True) -> np.ndarray:
    """Extract features from audio file."""
    audio, sr = load_audio(file_path, sr=SAMPLE_RATE)
    fixed_length = int(FIXED_DURATION * sr)
    audio = pad_or_truncate(audio, fixed_length)
    if use_mel and use_mfcc:
        features = extract_all_features(audio, sr)
    elif use_mel:
        features = extract_mel_spectrogram(audio, sr)
    elif use_mfcc:
        features = extract_mfcc(audio, sr)
    else:
        raise ValueError("At least one feature type must be selected")
    return features


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance."""
    mean = np.mean(features)
    std = np.std(features)
    if std > 0:
        return (features - mean) / std
    return features


def get_feature_shape(use_mel: bool = True, use_mfcc: bool = True) -> Tuple[int, int]:
    """Get the expected feature shape for given configuration."""
    if use_mel and use_mfcc:
        n_frames = int(FIXED_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
        return (n_frames, N_MELS + N_MFCC * 3)
    elif use_mel:
        n_frames = int(FIXED_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
        return (n_frames, N_MELS)
    else:
        n_frames = int(FIXED_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
        return (n_frames, N_MFCC * 3)