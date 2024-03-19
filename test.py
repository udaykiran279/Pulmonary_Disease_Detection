
import os
import time
import shutil
import numpy as np
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io.wavfile import write
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder



# Function to get list of audio files from a folder
def get_audio_files(folder_path):
    audio_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp3', '.wav')):
            audio_files.append(os.path.join(folder_path, filename))
    return audio_files

def save_uploaded_file(uploaded_file, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'audio.wav'), "wb") as f:
        f.write(uploaded_file.getbuffer())


def showplot(audio_file):
    signal, sample_rate = librosa.load(audio_file, sr=None)

    # Display the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.title('Sound Wave')
    plt.tight_layout()
    return plt


def Denoise(raw_audio, sample_rate=4000, filter_order=5, filter_lowcut=50, filter_highcut=1800, btype="bandpass"):
    b, a = 0,0
    if btype == "bandpass":
        b, a = signal.butter(filter_order, [filter_lowcut/(sample_rate/2), filter_highcut/(sample_rate/2)], btype=btype)

    if btype == "highpass":
        b, a = signal.butter(filter_order, filter_lowcut, btype=btype, fs=sample_rate)

    audio = signal.lfilter(b, a, raw_audio)

    return audio

# Function to generate spectrogram
def generate_spectrogram(audio_file):
    y, sr = librosa.load(audio_file)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    return plt

def generate_mfcc(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    return plt

def build_mfcc(file_path):
    plt.interactive(False)
    file_audio_series,sr = librosa.load(file_path,sr=None)    
    spec_image = plt.figure(figsize=[1,1])
    ax = spec_image.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectogram = librosa.feature.melspectrogram(y=file_audio_series, sr=sr)
    logmel=librosa.power_to_db(spectogram, ref=np.max)
    librosa.display.specshow(librosa.feature.mfcc(S=logmel, n_mfcc=30))
    
    image_name  = 'image/mfccfile.png'
    plt.savefig(image_name, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    spec_image.clf()
    plt.close(spec_image)
    plt.close('all')


def model_predict():
    model = load_model('mfccall.h5')
    labels = np.load('labels.npy')
    # Load the VGG16 model without the top classification layer
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Load and preprocess the test image
    test_image_path = 'image/mfccfile.png'
    img = load_img(test_image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Extract features from the test image using VGG16
    features = vgg16.predict(img_array)


    # Make prediction using the trained LSTM model
    prediction = model.predict(features.reshape((features.shape[0], -1, features.shape[-1])))
    # Convert prediction probabilities to class labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

    st.write(predicted_class[0])

# Streamlit app title
st.title("🫁 Lung Disease Detection 🫁")

# Folder path where audio files are stored
folder_path = 'audio'  # Replace 'path_to_your_folder' with the actual folder path

# Get the list of audio files from the folder
audio_files = get_audio_files(folder_path)

# Display buttons to either upload or browse audio files
option = st.radio("Select an option", ("Upload Manually", "Browse List"))

# If user chooses to upload manually
if option == "Upload Manually":
    upfile= st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if upfile:
        save_uploaded_file(upfile, "uploaded_audio")
        uploaded_file='uploaded_audio/audio.wav'
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            st.success('Audio File Uploaded')
            if st.button("Generate"):
                st.pyplot(showplot(uploaded_file))
                st.success('Denoising of Audio File Started')
                time.sleep(3)
                raw_audio, sample_rate = librosa.load(uploaded_file, sr=4000)
                #raw_audio, sample_rate = librosa.load(uploaded_file, sr=4000)
                adfile=Denoise(raw_audio)
                path='denaudio'
                save_path=os.path.join(path,'denoise.wav')
                write(save_path, sample_rate, adfile)
                file='denaudio/denoise.wav'
                build_mfcc(file)
                st.pyplot(showplot(file))
                st.success('Generating Log-Mel Spectograms..')
                time.sleep(3)
                st.pyplot(generate_spectrogram(file))
                st.success('Generating MFCC..')
                time.sleep(3)
                st.pyplot(generate_mfcc(file))
                model_predict()

# If user chooses to browse the list
elif option == "Browse List":
    # Display dropdown menu to select an audio file
    selected_audio_file = st.selectbox("Select an audio file", audio_files)
    # Display the selected audio file
    if selected_audio_file:
        st.audio(selected_audio_file, format='audio/*')
        st.success('Audio File Uploaded')
        if st.button("Generate"):
            st.pyplot(showplot(selected_audio_file))
            st.success('Denoising of Audio File Started')
            time.sleep(3)
            raw_audio, sample_rate = librosa.load(selected_audio_file, sr=4000)
            adfile=Denoise(raw_audio)
            path='denaudio'
            save_path=os.path.join(path,'denoise.wav')
            write(save_path, sample_rate, adfile)
            file='denaudio/denoise.wav'
            build_mfcc(file)
            st.pyplot(showplot(file))
            st.success('Generating Log-Mel Spectograms..')
            time.sleep(3)
            st.pyplot(generate_spectrogram(file))
            st.success('Generating MFCC..')
            time.sleep(3)
            st.pyplot(generate_mfcc(file))
            model_predict()

