#!/usr/bin/env python3
"""
Emotion Recognition from Speech
Main entry point for training and prediction.
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf

from src.training.train import train_model, predict_emotion, prepare_data, extract_all_features
from src.preprocessing.audio_utils import extract_features_from_file, normalize_features
from src.preprocessing.data_loader import EMOTION_LABELS
from src.models.cnn_lstm import build_cnn_lstm_model, compile_model
from src.evaluation.evaluate import (
    evaluate_model, 
    plot_confusion_matrix,
    print_classification_report,
    plot_training_history
)


def main():
    parser = argparse.ArgumentParser(
        description='Emotion Recognition from Speech'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to RAVDESS dataset'
    )
    train_parser.add_argument(
        '--model-type', type=str, default='cnn_lstm',
        choices=['cnn_lstm', 'cnn', 'lstm'],
        help='Model architecture'
    )
    train_parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size'
    )
    train_parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test split ratio'
    )
    train_parser.add_argument(
        '--model-save-path', type=str, default='models/emotion_model.keras',
        help='Model save path'
    )
    
    predict_parser = subparsers.add_parser('predict', help='Predict emotion')
    predict_parser.add_argument(
        '--audio-file', type=str, required=True,
        help='Path to audio file'
    )
    predict_parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model'
    )
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to RAVDESS dataset'
    )
    eval_parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test split ratio'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("=" * 60)
        print("EMOTION RECOGNITION FROM SPEECH - TRAINING")
        print("=" * 60)
        
        os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
        
        model, history = train_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            test_size=args.test_size,
            model_save_path=args.model_save_path
        )
        
        plot_training_history(history, save_path='models/training_history.png')
        print("\nTraining complete!")
        print(f"Model saved to: {args.model_save_path}")
        
    elif args.command == 'predict':
        print("=" * 60)
        print("EMOTION RECOGNITION FROM SPEECH - PREDICTION")
        print("=" * 60)
        
        model = tf.keras.models.load_model(args.model_path)
        result = predict_emotion(model, args.audio_file)
        
        print(f"\nAudio file: {args.audio_file}")
        print(f"Predicted emotion: {result['predicted_emotion']}")
        print("\nProbabilities:")
        for emotion, prob in sorted(
            result['probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"  {emotion}: {prob:.4f}")
            
    elif args.command == 'evaluate':
        print("=" * 60)
        print("EMOTION RECOGNITION FROM SPEECH - EVALUATION")
        print("=" * 60)
        
        _, X_test_paths, _, y_test = prepare_data(
            args.data_dir, test_size=args.test_size
        )
        
        X_test, y_test = extract_all_features(X_test_paths, y_test)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        model = tf.keras.models.load_model(args.model_path)
        
        metrics = evaluate_model(model, X_test, y_test)
        
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        
        print_classification_report(
            metrics['y_true'], 
            metrics['y_pred']
        )
        
        plot_confusion_matrix(
            metrics['y_true'], 
            metrics['y_pred'],
            save_path='models/confusion_matrix.png'
        )
        print("\nConfusion matrix saved to: models/confusion_matrix.png")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()