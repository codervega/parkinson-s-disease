import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Function to extract features from audio files
def extract_features(file_path, mfcc, chroma, mel):
    audio_data, _ = librosa.load(file_path, sr=22050)
    if mfcc:
        result = np.mean(librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40).T, axis=0)
    if chroma:
        result = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=22050).T, axis=0)
    if mel:
        result = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=22050).T, axis=0)
    return result

# Load the trained model
model = load_model('parkisions_classification_model.h5')

# Define Streamlit app
def main():
    st.title('Voice as an assisting tool for detection of Parkinson’s disease ')
    st.sidebar.title('Upload Audio File')
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav'])
    
    if uploaded_file is not None:
        # Display the uploaded audio file
        st.audio(uploaded_file)
        
        # Extract features from the uploaded audio file
        features = extract_features(uploaded_file, mfcc=True, chroma=False, mel=False)
        features = features.reshape(1, features.shape[0], 1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        
        # Map prediction index to class label
        class_labels = ['MildParkisions', 'Normal', 'Parkisions']
        predicted_label = class_labels[predicted_class]
        
        st.write(f"Predicted Class: {predicted_label}")

if __name__ == "__main__":
    main()
