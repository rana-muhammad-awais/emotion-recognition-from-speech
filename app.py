import streamlit as st
import tempfile
import os
import numpy as np
import pickle

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="mic",
    layout="centered"
)


@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']


def load_audio(file_path):
    import librosa
    audio, sr = librosa.load(file_path, sr=22050)
    return audio


def extract_features(audio):
    import librosa
    
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, mel_spec_db])
    
    features_mean = np.mean(features, axis=1)
    features_std = np.std(features, axis=1)
    
    return np.concatenate([features_mean, features_std])


def predict_emotion(model_info, audio_path):
    model, scaler = model_info
    
    audio = load_audio(audio_path)
    fixed_length = int(2.5 * 22050)
    if len(audio) > fixed_length:
        audio = audio[:fixed_length]
    elif len(audio) < fixed_length:
        audio = np.pad(audio, (0, fixed_length - len(audio)), mode='constant')
    
    features = extract_features(audio)
    features = scaler.transform(features.reshape(1, -1))
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
    
    return {
        'predicted_emotion': EMOTION_LABELS[prediction],
        'probabilities': {EMOTION_LABELS[i]: float(probabilities[i]) for i in range(8)}
    }


st.markdown("""
<style>
    .main-title { font-size: 36px; text-align: center; color: #4CAF50; }
    .result-box { background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 15px; color: white; text-align: center; margin: 20px 0; }
    .stButton > button { width: 100%; padding: 20px; font-size: 16px; }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown("<h1 class='main-title'>Emotion Recognition from Speech</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model Path", value="models/emotion_model.pkl")
        
        if st.button("Load Model", type="primary"):
            if os.path.exists(model_path):
                try:
                    st.session_state.model_info = load_model(model_path)
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model file not found! Train the model first.")
        
        st.markdown("---")
        st.markdown("""
        **Instructions:**
        1. Train the model first using train_model.bat
        2. Upload an audio file (wav, mp3, ogg)
        3. Click 'Analyze Emotion'
        
        Recognizes 8 emotions:
        - Happy, Sad, Angry, Fearful
        - Neutral, Calm, Disgust, Surprise
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Voice Recording")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'webm', 'flac']
        )
        
        audio_path = None
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name
            st.success(f"Loaded: {uploaded_file.name}")
            st.audio(uploaded_file)
    
    with col2:
        if "model_info" in st.session_state and st.session_state.get("model_loaded"):
            st.markdown("### Recognition Result")
            
            if st.button("Analyze Emotion", type="primary"):
                if audio_path:
                    with st.spinner("Analyzing voice..."):
                        try:
                            result = predict_emotion(
                                st.session_state.model_info, 
                                audio_path
                            )
                            
                            emotion = result['predicted_emotion']
                            confidence = result['probabilities'][emotion]
                            
                            st.markdown(f"""
                            <div class="result-box">
                                <h2 style="margin:0; text-transform:uppercase;">{emotion}</h2>
                                <p style="font-size:18px;">Confidence: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### All Emotion Probabilities:")
                            
                            sorted_probs = sorted(
                                result['probabilities'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            for emo, prob in sorted_probs:
                                bar_color = "#4CAF50" if emo == emotion else "#2196F3"
                                st.markdown(
                                    f"""<div style="margin:8px 0;">
                                    <span style="font-weight:bold;">{emo.capitalize()}</span>
                                    <div style="background:#eee; border-radius:5px; height:20px;">
                                        <div style="background:{bar_color}; width:{prob*100}%; border-radius:5px; height:100%; text-align:center; color:white; font-size:12px;">{prob*100:.1f}%</div>
                                    </div>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                            
                        except Exception as e:
                            st.error(f"Error analyzing: {e}")
                else:
                    st.warning("Please upload an audio file first!")
        else:
            st.warning("Please load a model first!")
    
    st.markdown("---")
    st.caption("Emotion Recognition System | Using Random Forest")


if __name__ == "__main__":
    main()