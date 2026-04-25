import os
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Callable
import tensorflow as tf
from tqdm import tqdm

from src.preprocessing.audio_utils import extract_features_from_file, normalize_features, get_feature_shape
from src.preprocessing.data_loader import (
    load_ravdess_dataset, 
    get_train_test_split,
    EMOTION_LABELS
)
from src.models.cnn_lstm import (
    build_cnn_lstm_model, 
    build_cnn_model, 
    build_lstm_model,
    compile_model,
    get_callbacks
)


class EmotionDataGenerator(tf.keras.utils.Sequence):
    """Keras data generator for emotion recognition."""
    
    def __init__(self, file_paths: list, labels: list, 
                 batch_size: int = 32,
                 extract_features_fn: Callable = extract_features_from_file,
                 shuffle: bool = True):
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.extract_features = extract_features_fn
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        X = np.array([normalize_features(self.extract_features(self.file_paths[i])) 
                   for i in batch_indices])
        y = self.labels[batch_indices]
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def prepare_data(data_dir: str, test_size: float = 0.2,
                 random_state: int = 42) -> Tuple:
    """Prepare train and test datasets."""
    print("Loading RAVDESS dataset...")
    file_paths, labels = load_ravdess_dataset(data_dir)
    print(f"Found {len(file_paths)} audio files")
    print(f"Emotion distribution: {np.bincount(labels)}")
    
    X_train_paths, X_test_paths, y_train, y_test = get_train_test_split(
        file_paths, labels, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train_paths)}")
    print(f"Test samples: {len(X_test_paths)}")
    
    return X_train_paths, X_test_paths, y_train, y_test


def extract_all_features(file_paths: list, labels: list) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for all files."""
    features = []
    for fp in tqdm(file_paths, desc="Extracting features"):
        feat = extract_features_from_file(fp)
        feat = normalize_features(feat)
        features.append(feat)
    return np.array(features), np.array(labels)


def train_model(data_dir: str,
               model_type: str = 'cnn_lstm',
               epochs: int = 100,
               batch_size: int = 32,
               learning_rate: float = 0.001,
               test_size: float = 0.2,
               model_save_path: str = 'models/emotion_model.keras',
               use_mel: bool = True,
               use_mfcc: bool = True) -> tf.keras.Model:
    """Train emotion recognition model."""
    
    X_train_paths, X_test_paths, y_train, y_test = prepare_data(
        data_dir, test_size=test_size
    )
    
    feature_shape = get_feature_shape(use_mel=use_mel, use_mfcc=use_mfcc)
    print(f"Feature shape: {feature_shape}")
    
    print("Extracting features for training set...")
    X_train, y_train = extract_all_features(X_train_paths, y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    
    print("Extracting features for test set...")
    X_test, y_test = extract_all_features(X_test_paths, y_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(EMOTION_LABELS)
    
    if model_type == 'cnn_lstm':
        model = build_cnn_lstm_model(input_shape, num_classes)
    elif model_type == 'cnn':
        model = build_cnn_model(input_shape, num_classes)
    elif model_type == 'lstm':
        model = build_lstm_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model, learning_rate)
    
    print(f"\nModel: {model_type}")
    print(f"Input shape: {input_shape}")
    model.summary()
    
    callbacks = get_callbacks(checkpoint_path=model_save_path)
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, history


def load_trained_model(model_path: str, input_shape: tuple, 
                    num_classes: int = 8) -> tf.keras.Model:
    """Load a trained model."""
    model = tf.keras.models.load_model(model_path)
    return model


def predict_emotion(model: tf.keras.Model, audio_file: str) -> dict:
    """Predict emotion from audio file."""
    features = extract_features_from_file(audio_file)
    features = normalize_features(features)
    features = features.reshape(1, features.shape[1], features.shape[0])
    
    predictions = model.predict(features, verbose=0)[0]
    emotion_probs = {EMOTION_LABELS[i]: float(predictions[i]) 
                    for i in range(len(EMOTION_LABELS))}
    predicted_emotion = EMOTION_LABELS[np.argmax(predictions)]
    
    return {
        'predicted_emotion': predicted_emotion,
        'probabilities': emotion_probs
    }