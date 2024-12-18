# parkinson-s-disease
Parkinson's Disease is a neurological disorder affecting movement and coordination. This repository focuses on early detection using audio analysis and machine learning techniques.


-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective:
The project identifies Parkinson's Disease by analyzing audio data, specifically targeting vocal characteristics affected by the disease. It provides an easy-to-use interface for healthcare professionals, researchers, or individuals to assess the presence and severity of Parkinson's Disease.

Key Technologies:
Python for feature extraction, model training, and application development.
Librosa for audio processing and feature extraction.
Streamlit for building an interactive web interface.
TensorFlow/Keras and Scikit-learn for training and loading models.
NumPy and Pandas for numerical operations and data manipulation.

Workflow:
Upload .wav audio file via the Streamlit UI.
Extract MFCC (Mel-Frequency Cepstral Coefficients) from the audio.
Preprocess the extracted features to feed into a pre-trained machine learning model.
Predict the disease type or severity and display the result.

Models Used:
CNN (Convolutional Neural Network): Extract spatial features from audio data.
RNN (Recurrent Neural Network): Capture sequential dependencies in the audio data.
DNN (Deep Neural Network): Perform classification on the extracted features.
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Pre-trained Model

The futer_extract() function loads a pre-trained model (model_A) for predictions. This model is trained on MFCC features extracted from audio samples.

Pre-trained Model Assumptions:

Architecture:
A combination of CNN, RNN, and DNN layers.
CNN captures spatial features, RNN captures temporal dependencies, and DNN performs classification.
Input Shape:
40 MFCC features for a fixed duration of 3 seconds.

---------------------------------------------------------------------------------------------------------------------------------------------------------------
Backend Workflow

Input: The .wav file uploaded by the user.
Feature Extraction:
MFCC features are extracted using the extract_mfcc() function.
Feature Preprocessing:
Features are reshaped and expanded to match the model's expected input shape.
Prediction:
The pre-trained model predicts the probability distribution across multiple classes.
The class with the highest probability is mapped to the disease type.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

How to Use the Application
  #Install the required dependencies:
    pip install streamlit numpy pandas librosa scikit-learn tensorflow
  #Run the Streamlit app:
    streamlit run app.py
Upload an audio file in .wav format.
View the prediction results and listen to the uploaded audio.

----------------------------------------------------------------------------------------------------------------------------------------------------------------



