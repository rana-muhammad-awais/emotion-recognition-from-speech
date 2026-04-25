import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    f1_score
)
from typing import List, Dict
import tensorflow as tf

from src.preprocessing.data_loader import EMOTION_LABELS


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, 
                 y_test: np.ndarray) -> Dict:
    """Evaluate model on test set."""
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'y_pred': y_pred,
        'y_true': y_test
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Emotion Recognition')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print classification report."""
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, 
                               target_names=EMOTION_LABELS))


def plot_training_history(history, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()