import streamlit as st
import tempfile
import os
import numpy as np
import pickle

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)


@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']


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


def predict_from_audio_file(audio_path):
    import librosa
    
    audio, sr = librosa.load(audio_path, sr=22050)
    
    fixed_length = int(2.5 * 22050)
    if len(audio) > fixed_length:
        audio = audio[:fixed_length]
    elif len(audio) < fixed_length:
        audio = np.pad(audio, (0, fixed_length - len(audio)), mode='constant')
    
    features = extract_features(audio)
    model, scaler = st.session_state.model_info
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
    .main-title { font-size: 42px; text-align: center; color: #4CAF50; }
    .result-box { background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 15px; color: white; text-align: center; margin: 20px 0; }
    .stButton > button { padding: 15px 30px; font-size: 16px; }
    .prompt-box { font-size: 20px; text-align: center; padding: 20px; background: linear-gradient(135deg, #4CAF50, #45a049); color: white; border-radius: 15px; margin: 10px 0; }
    .instruction-box { padding: 20px; background: #f5f5f5; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown("<h1 class='main-title'>🎤 Real-Time Emotion Recognition</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input("Model Path", value="models/emotion_model.pkl")
        
        if st.button("Load Model", type="primary"):
            if os.path.exists(model_path):
                try:
                    st.session_state.model_info = load_model(model_path)
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model not found!")
        
        st.markdown("---")
        st.markdown("""
        **Instructions:**
        1. Record your voice using phone/PC recorder
        2. Upload the audio file
        3. View your emotion result
        
        **Works best with:**
        - Short phrases (2-5 seconds)
        - Clear speech
        - Normal speaking volume
        """)
    
    if "model_info" not in st.session_state:
        st.warning("⚠️ Load the model first!")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎤 Record Your Voice")
        
        st.markdown("""
        <div class="prompt-box">
            📱 Use your phone's voice recorder<br>
            or Windows Voice Recorder (Win+Shift+R)<br><br>
            <b>Speak any sentence:</b>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Example phrases:")
        phrases = [
            "I am so happy today!",
            "This makes me really sad",
            "I am very angry!",
            "Everything is fine",
            "I am scared",
            "What a wonderful day!",
            "I love this",
            "I feel great!"
        ]
        for phrase in phrases:
            st.markdown(f"- *{phrase}*")
        
        st.markdown("---")
        st.markdown("### 📎 Upload Recording")
        
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=['wav', 'mp3', 'ogg', 'webm', 'm4a', 'flac']
        )
        
        audio_path = None
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name
            st.success(f"✅ Loaded: {uploaded_file.name}")
            st.audio(uploaded_file)
    
    with col2:
        st.markdown("### 🎯 Result")
        
        if audio_path:
            if st.button("🎯 Analyze Emotion", type="primary", use_container_width=True):
                with st.spinner("Analyzing your voice..."):
                    try:
                        result = predict_from_audio_file(audio_path)
                        
                        emotion = result['predicted_emotion']
                        confidence = result['probabilities'][emotion]
                        
                        emoji_map = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠',
                            'fearful': '😨', 'neutral': '😐', 'calm': '😌',
                            'disgust': '🤢', 'surprise': '😲'
                        }
                        emoji = emoji_map.get(emotion, '🎭')
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h1 style="margin:0; font-size:60px;">{emoji}</h1>
                            <h2 style="margin:0;">{emotion.upper()}</h2>
                            <p style="font-size:20px;">Confidence: {confidence*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### All Emotions:")
                        
                        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                        for emo, prob in sorted_probs:
                            color = "#4CAF50" if emo == emotion else "#2196F3"
                            st.markdown(
                                f"""<div style="margin:8px 0;">
                                <span style="font-weight:bold;">{emo.capitalize()}</span>
                                <div style="background:#eee; border-radius:5px; height:24px;">
                                    <div style="background:{color}; width:{prob*100}%; border-radius:5px; height:100%; text-align:center; color:white; font-size:14px; line-height:24px;">{prob*100:.1f}%</div>
                                </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("👆 Upload a voice recording to analyze")
    
    st.markdown("---")
    st.caption("🎤 Real-Time Emotion Recognition | Built with Random Forest + MFCC")


if __name__ == "__main__":
    main()