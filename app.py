import os
import time
import shutil
import numpy as np
import pandas as pd
from io import BytesIO
from xhtml2pdf import pisa
import librosa.display
from jinja2 import Template
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io.wavfile import write
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
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


def generate_html(patient_info, test_results,data):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {{
      font-family: Arial, sans-serif;
      font-size: 15px;
      margin: 0;
      padding: 0;
    }}

    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px;
      background-color: #f2f2f2;
      color: #000; /* Black */
      animation: glow 1s ease-in-out infinite alternate;
    }}
    .logo {{
      max-width: 100px;
      max-height: 100px;
    }}
    @keyframes glow {{
        from {{
            box-shadow: 0 0 10px #45B8AC; /* Black shadow */
        }}
        to {{
            box-shadow: 0 0 20px #45B8AC; /* Larger black shadow */
        }}
    }}

    .container {{
      width: 100%;
      background-color: #f2f2f2;
      padding: 0px;
    }}

    .content {{
      width: 50%;
      margin: 0 auto;
      background-color: #fff;
      padding: 3px;
      box-sizing: border-box;
    }}

    
    table {{
      border-collapse: collapse;
      table-layout: auto;
      width: 100%;
    }}

    th, td {{
      border: 1px solid #dddddd;
      padding: 5px;
      text-align: left;
    }}

    th {{
      background-color: #dddddd;
      border: 1px solid #dddddd;
      font-weight: bold;
    }}

    .positive-result td {{
      background-color: #ffbdbd;
    }}
    footer {{
      color: black;
      padding: 30px;
      text-align: center;
      width: 100%;
    }}
    </style>
    </head>
    <body>
    <header>
        <div>
            <img src="image/left.png" alt="Left Logo" class="logo">
        </div>
        <h1>Medical Report</h1>
        <div>
            <img src="image/left.png" alt="Right Logo" class="logo">
        </div>
    </header>
    <div class="container">
        <div class="content">
          <h2>Patient Information:</h2>
          <p><b>Name&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp:</b> {patient_info['name']}</p>
          <p><b>Age&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp:</b> {patient_info['age']}</p>
          <p><b>Gender&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp:</b> {patient_info['gender']}</p>
          <p><b>Mobile Number&nbsp:</b> {patient_info['mobile']}</p>

      <h2>Test Results:</h2>
      <table id="testResults">
        <thead>
          <tr>
            <th>S.NO</th>
            <th>Disease</th>
            <th>Result</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
    """

    for index, result in enumerate(test_results, start=1):
        html += f"""
          <tr{' class="positive-result"' if result['result'] == "Positive" else ""}>
            <td>{index}</td>
            <td>{result['disease']}</td>
            <td>{result['result']}</td>
            <td>{result['severity']}</td>
          </tr>
        """

    html += f"""
        </tbody>
      </table>

      <h2>Remedies:</h2>
      <p>{data['rem']}</p>
      <h2>Medicines:</h2>
      <p>{data['med']}</p>
        </div>
    </div>
    <br>
    <footer>
      <p><b>Disclaimer: This medical report is Machine predicted.</b></p>
    </footer>

    </body>
    </html>
    """

    return html

def generate_pdf(html_content):
    # Save HTML content to a file
    html_file = "medical_report.html"
    with open(html_file, "w") as f:
        f.write(html_content)
    
    # Convert HTML to PDF
    result = BytesIO()
    pisa.pisaDocument(BytesIO(html_content.encode('utf-8')), result)
    with open('Report/Report.pdf', "wb") as f:
        f.write(result.getvalue())


def send_email_with_attachment(mail_id,name,pdf_path):
    sender_email = "udaylabs27@gmail.com"
    sender_password = "qpku hfol hlsm ddqw"
    receiver_email = mail_id
    current_datetime = datetime.now()
    subject = f"Medical Report of {name}"
    body = f"""
    Dear {name},
    
    Thank you for visiting Uday Labs. Please find the attached medical report for {name} tested on {current_datetime.strftime("%Y-%m-%d")} at {current_datetime.strftime("%H:%M:%S")}. If you have any questions or require further information, please do not hesitate to contact me.
    
    Best regards,
    Uday Labs
    """
    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = mail_id
    message["Subject"] = subject
    
    # Attach body
    message.attach(MIMEText(body, "plain"))
    
    # Open PDF file to be attached
    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    
    # Encode the attachment
    encoders.encode_base64(part)
    
    # Add header
    part.add_header("Content-Disposition", f"attachment; filename= {pdf_path.split('/')[1]}")
    
    # Add attachment to message
    message.attach(part)
    
    # Connect to SMTP server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, mail_id, message.as_string())
    
    st.success("Report Sent to Your mail")


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

    scaler = StandardScaler()
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    feat=features.reshape(features.shape[0], -1)
    new_feat= scaler.fit_transform(feat)
    new_clusters = kmeans.predict(new_feat)[0]
    severity={
        1:'Mild',
        2:'Moderate',
        3:'Severe'
    }
    return [predicted_class[0],severity[new_clusters]]

# storing remedies and medicines
rem={
    'COPD':'Quit smoking, Healthy diet, Manage Stress, Stay Hydrated',
    'Pneumonia':'Rest, Hydration,Control Fever and Pain,Use a Humidifier,Avoid Smoke and Irritants',
    'LungFibrosis':'Oxygen Therapy, Nutrition, Hydration, palliative care, Emotional support, Lung transplantation',
    'URTI':'Gargle with Warm salt water, steam,rest, hydration, use honey,warm compresses, elevate your head, take garlic.',
    'Bronchiectasis':'Airway clearance techniques, Immunizations,Nutritious diet, stay hydrated, Regular exercise',
    'Bronchiolitis':'elevate head position, avoid smoke exposure, practice good hand hygiene,use a humidifier and maintain Hydration.',
    'Asthma':'stay active, maintain a healthy weight, quit smoking. Use asthma inhalers correctly, monitor peak flow'
}
med={
    'COPD':'bronchodilators,Corticosteroids,Methylxanthines,Roflumilast',
    'Pneumonia':'Fluoroquinolones, Cephalosporins, Macrolides, Monobactams, Antibiotics, Lincosamide, Tetracyclines, Carbapenems, Oxazolidinones, Aminoglycosides, Penicillins, Amino, Penicillins.',
    'LungFibrosis':'nintedanib,pirfenidone',
    'URTI':'Penicillin VK, Amoxicillin,pencillin G, Cefadroxil, Erythromycin and clavulanate',
    'Bronchiectasis':'Amoxicillin 500mg ,Clarithromycin 500mg,doxycycline and ciprofloxacin',
    'Bronchiolitis':'aerosolized ribavirin,bronchodilators and corticosteroids',
    'Asthma':'Fluticasone (Flovent HFA, Arnuity Ellipta, others),Budesonide (Pulmicort Flexhaler), Mometasone (Asmanex Twisthaler),Beclomethasone (Qvar RediHaler),Ciclesonide (Alvesco)'
}

# Streamlit app title
st.title("Lung 🫁 Disease Detection")

# Folder path where audio files are stored
folder_path = 'audio'  # Replace 'path_to_your_folder' with the actual folder path

# Get the list of audio files from the folder
audio_files = get_audio_files(folder_path)
name = st.text_input("Enter your Name")
age = st.text_input("Enter your Age")
gender = st.selectbox("Select your Gender", ["---","Male", "Female", "Other"])
mobile_number = st.text_input("Enter your Mobile Number")
mail_id=st.text_input("Enter Your mail ID")
mail_id=mail_id.lower()

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
                model_pred=model_predict()
                disease=model_pred[0]
                sever=model_pred[1]
                patient_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "mobile": mobile_number
                }
                test_results = [
                    {"disease": "COPD", "result": "", "severity": "--"},
                    {"disease": "Asthma", "result": "", "severity": "--"},
                    {"disease": "Bronchiectasis", "result": "", "severity": "--"},
                    {"disease": "Bronchiolitis", "result": "", "severity": "--"},
                    {"disease": "URTI", "result": "", "severity": "--"},
                    {"disease": "LungFibrosis", "result": "", "severity": "--"},
                    {"disease": "Pneumonia", "result": "", "severity": "--"}
                ]
                for test in test_results:
                    if test['disease']==disease:
                        test['result']='Positive'
                        test['severity']=sever
                    else:
                        test['result']='Negative'
                if disease!='Healthy':
                    remedi=rem[disease]
                    medicine=med[disease]
                else:
                    remedi="You are healthy,Maintain same diet and also be far from smoking as you are now."
                    medicine="-- Not Applicable --"
                    
                data={
                    "rem":remedi,
                    "med":medicine
                }
                html = generate_html(patient_info, test_results,data)
                generate_pdf(html)

                pdf_path = "Report/Report.pdf"
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                send_email_with_attachment(mail_id,name,pdf_path)
                time.sleep(2)
                st.download_button(label="Download Report", data=pdf_bytes, file_name=f"{name}_Report.pdf", mime="application/pdf")
                
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
            model_pred=model_predict()
            disease=model_pred[0]
            sever=model_pred[1]
            patient_info = {
                "name": name,
                "age": age,
                "gender": gender,
                "mobile": mobile_number
            }
            test_results = [
                {"disease": "COPD", "result": "", "severity": "--"},
                {"disease": "Asthma", "result": "", "severity": "--"},
                {"disease": "Bronchiectasis", "result": "", "severity": "--"},
                {"disease": "Bronchiolitis", "result": "", "severity": "--"},
                {"disease": "URTI", "result": "", "severity": "--"},
                {"disease": "LungFibrosis", "result": "", "severity": "--"},
                {"disease": "Pneumonia", "result": "", "severity": "--"}
            ]
            for test in test_results:
                if test['disease']==disease:
                    test['result']='Positive'
                    test['severity']=sever
                else:
                    test['result']='Negative'
            if disease!='Healthy':
                remedi=rem[disease]
                medicine=med[disease]
            else:
                remedi="You are healthy,Maintain same diet and also be far from smoking as you are now."
                medicine="-- Not Applicable --"
                
            data={
                "rem":remedi,
                "med":medicine
            }
            html = generate_html(patient_info, test_results,data)
            generate_pdf(html)

            pdf_path = "Report/Report.pdf"
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            send_email_with_attachment(mail_id,name,pdf_path)
            time.sleep(2)
            st.download_button(label="Download Report", data=pdf_bytes, file_name=f"{name}_Report.pdf", mime="application/pdf")

