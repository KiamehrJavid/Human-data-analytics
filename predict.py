import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("/home/kia/Desktop/PoD/HDA/dumped_models/test_rnn_model.h5")


int2lab = {
    0: 'down',
    1: 'go',
    2: 'left',
    3: 'no',
    4: 'right',
    5: 'stop',
    6: 'up',
    7: 'yes'}

sr = 16000
n_mels = 24
win_length=int(sr*0.05)
hop_length=int(sr*0.03)



def process_audio(file_path, sr = 16000, n_mels = 24, win_length=int(16000*0.05), hop_length=int(16000*0.03)):

    audio, sr = librosa.load(file_path, sr=sr)
    length = len(audio)

    if length > sr:
        return np.random.random(sr)  # Skip this file if it doesn't meet the length criteria

    # Padding if necessary
    if length < sr:
        padded = np.zeros(sr)
        start = (16000 - length) // 2
        end = start + length
        padded[start:end] = audio
        audio = padded


    audio = librosa.util.normalize(audio)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, win_length=win_length, hop_length=hop_length)
    mel = librosa.power_to_db(mel, ref=np.max)

    return np.expand_dims(mel.T, axis=0)

def predict(file_path):
    # Preprocess the audio file
    mel_spectrogram = process_audio(file_path)

    # Predict
    predictions = model.predict(mel_spectrogram)
    predicted_label = np.argmax(predictions, axis=1)[0]


    return int2lab[predicted_label]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    label = predict(file_path)
    print(f"Predicted Label: {label}")
