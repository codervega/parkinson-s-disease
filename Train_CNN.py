import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to extract features from audio files
def extract_features(file_path, mfcc, chroma, mel):
    audio_data, _ = librosa.load(file_path, sr=22050)
    if mfcc:
        result = np.mean(librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40).T, axis=0)
    elif chroma:
        result = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=22050).T, axis=0)
    elif mel:
        result = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=22050).T, axis=0)
    return result

# Define the main function for loading data and training the model
def main(data_dir):
    # Constants
    MFCC = True
    CHROMA = False
    MEL = False
    LABELS = ['mildparkisions', 'normal', 'parkisions']
    NUM_CLASSES = len(LABELS)
    # Load data
    data = []
    labels = []
    for i, label in enumerate(LABELS):
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path, MFCC, CHROMA, MEL)
            data.append(features)
            labels.append(i)
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Reshape data for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # Build the CNN model
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Train the model and record training history
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    # Save the model
    model.save('parkisions_classification_model.h5')
    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main('Dataset')
