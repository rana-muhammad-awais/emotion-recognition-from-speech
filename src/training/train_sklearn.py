import numpy as np
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
from tqdm import tqdm


SAMPLE_RATE = 22050
N_MFCC = 40
FIXED_DURATION = 2.5
HOP_LENGTH = 512


EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprise'
}


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio


def pad_or_truncate(audio, fixed_length):
    if len(audio) > fixed_length:
        return audio[:fixed_length]
    elif len(audio) < fixed_length:
        return np.pad(audio, (0, fixed_length - len(audio)), mode='constant')
    return audio


def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, mel_spec_db])
    
    features_mean = np.mean(features, axis=1)
    features_std = np.std(features, axis=1)
    
    return np.concatenate([features_mean, features_std])


def parse_filename(filename):
    basename = os.path.basename(filename)
    name = basename.rsplit('.', 1)[0]
    parts = name.split('-')
    if len(parts) >= 7:
        return parts[2]
    return None


def load_dataset(data_dir):
    print("Loading dataset...")
    file_paths = []
    labels = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion_code = parse_filename(file_path)
                if emotion_code and emotion_code in EMOTION_MAP:
                    file_paths.append(file_path)
                    labels.append(EMOTION_LABELS.index(EMOTION_MAP[emotion_code]))
    
    print(f"Found {len(file_paths)} audio files in {data_dir}")
    return file_paths, labels


def extract_all_features(file_paths):
    features = []
    for fp in tqdm(file_paths, desc="Extracting features"):
        audio = load_audio(fp)
        audio = pad_or_truncate(audio, int(FIXED_DURATION * SAMPLE_RATE))
        features.append(extract_features(audio))
    return np.array(features)


def train_model(data_dir, model_path='models/emotion_model.pkl', test_size=0.2):
    file_paths, labels = load_dataset(data_dir)
    
    if len(file_paths) == 0:
        print("No audio files found! Please add RAVDESS files to data/raw/")
        return None
    
    print(f"Found {len(file_paths)} files")
    
    X = extract_all_features(file_paths)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"Model saved to {model_path}")
    
    return model


def load_trained_model(model_path='models/emotion_model.pkl'):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']


def predict_emotion(audio_file, model_path='models/emotion_model.pkl'):
    model, scaler = load_trained_model(model_path)
    
    audio = load_audio(audio_file)
    audio = pad_or_truncate(audio, int(FIXED_DURATION * SAMPLE_RATE))
    features = extract_features(audio)
    features = scaler.transform(features.reshape(1, -1))
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    result = {
        'predicted_emotion': EMOTION_LABELS[prediction],
        'probabilities': {EMOTION_LABELS[i]: float(probabilities[i]) for i in range(8)}
    }
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    args = parser.parse_args()
    
    train_model(args.data_dir)