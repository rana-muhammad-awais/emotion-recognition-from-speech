# Emotion Recognition from Speech

A deep learning system that recognizes human emotions (happy, sad, angry, calm, neutral, fearful, disgust, surprise) from speech audio using MFCC and Mel Spectrogram features.

## Features

- **Feature Extraction**: MFCC (40 coefficients) + Mel Spectrogram (128 bands)
- **Machine Learning Model**: Random Forest Classifier (200 estimators)
- **Web Interface**: Streamlit-based UI for easy usage
- **8 Emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprise

## Dataset

Uses [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) (Ryerson Audio-Visual Database of Emotional Speech and Song):
- 24 actors (12 male, 12 female)
- 1440 speech audio files
- 8 emotional expressions

## Installation

```bash
pip install streamlit librosa scikit-learn soundfile numpy scipy pandas matplotlib tqdm
```

## Usage

### Train Model
```bash
python src/training/train_sklearn.py --data-dir data/raw
```

### Run Web Interface
```bash
streamlit run app.py
```
Or double-click `run_app.bat`

The web interface opens at http://localhost:8501

## Project Structure

```
emotion-recognition-from-speech/
├── app.py                 # Streamlit web interface
├── gui.py                 # Desktop GUI (Tkinter)
├── main.py                # CLI entry point
├── requirements.txt      # Dependencies
├── run_app.bat           # Windows launcher
├── .gitignore
└── src/
    ├── preprocessing/
    │   ├── audio_utils.py    # MFCC & Mel Spectrogram extraction
    │   └── data_loader.py  # RAVDESS data loading
    ├── models/
    │   └── cnn_lstm.py    # Deep learning models
    ├── training/
    │   └── train_sklearn.py # Model training
    └── evaluation/
        └── evaluate.py    # Metrics & evaluation
```

## Results

- **Test Accuracy**: ~70-85% (varies with data split)
- **Best performing emotions**: Happy, Sad, Angry
- **Challenging emotions**: Surprise, Calm (context-dependent)

## Tech Stack

- **Python 3.11+**
- **Librosa** - Audio feature extraction
- **Scikit-learn** - Machine learning
- **Streamlit** - Web interface
- **TensorFlow/Keras** - Deep learning (optional)

## License

MIT License

## Author

[Rana Muhammad Awais](https://github.com/rana-muhammad-awais)