import streamlit as st
import os

def save_uploaded_file(uploaded_file, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

def main():
    st.title("Upload and Store Audio")

    # Define the folder where uploaded files will be stored
    folder_path = "uploaded_audio"

    # File uploader for audio
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the uploaded file into the specified folder
        save_uploaded_file(uploaded_file, folder_path)
        
        st.success(f"Audio file '{uploaded_file.name}' saved successfully in folder '{folder_path}'.")

if __name__ == "__main__":
    main()
