import streamlit as st
import os
import sys
sys.path.append(os.getenv('PROJECT_ROOT'))
import dotenv
dotenv.load_dotenv()
from src.inference import prediction
from src.database import MongoDBService
import time
from datetime import datetime
import zlib
import numpy as np
import matplotlib.pyplot as plt

# MongoDB service
db_service = MongoDBService(os.getenv('MONGO_URI'), 'PatientData')

# Initialise session state
if 'mri_file' not in st.session_state:
    st.session_state.mri_file = None
if 'patient_reference' not in st.session_state:
    st.session_state.patient_reference = ''
if 'modality' not in st.session_state:
    st.session_state.modality = None
if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

if 'image' not in st.session_state:
    st.session_state.image = None
if 'anomaly' not in st.session_state:
    st.session_state.anomaly = None

if 'patient_reference2' not in st.session_state:
    st.session_state.patient_reference2 = ''
if 'modality2' not in st.session_state:
    st.session_state.modality2 = None
if 'disabled2' not in st.session_state:
    st.session_state["disabled2"] = False

# Function to validate input and process MRI
def validate_input():
    if st.session_state.mri_file is None:   # Check mri                               
        st.toast(body='Please upload a file.', icon='⚠️')

    if st.session_state.patient_reference == '': # Check patient reference
        st.toast(body='Please enter a patient reference.', icon='⚠️')

    if st.session_state.modality is None:   # Check modality
        st.toast(body='Please select a modality.', icon='⚠️')
                
    if st.session_state.mri_file is not None and st.session_state.patient_reference != '' and st.session_state.modality is not None:
        # Disable submit button
        st.session_state["disabled"] = True
        st.session_state["disabled2"] = False

        # Save mri file to local for processing
        with st.spinner('Processing MRI...', show_time=True):
            save_dir = 'interface/temp_file_storage'
            save_path = os.path.join(save_dir, st.session_state.mri_file.name)
            with open(save_path, "wb") as f:
                f.write(st.session_state.mri_file.getbuffer())

            # Process MRI
            if st.session_state.modality == 'T1':
                mdl = 1
            else:
                mdl = 2

            image, anomaly = prediction(save_path, mdl)
            image = image.squeeze()  
            image_np = image.cpu().numpy()
            anomaly_np = anomaly.cpu().numpy()
            st.session_state.image = image_np
            st.session_state.anomaly = anomaly_np

            # Remove the file after processing
            os.remove(save_path)

            # Compress arrays for storage
            image_compressed = zlib.compress(image_np.tobytes())
            anomaly_compressed = zlib.compress(anomaly_np.tobytes())

            # Insert into MongoDB
            db_service.insert("mri_records", {
                "patient_reference": st.session_state.patient_reference,
                "modality": mdl,
                "image": image_compressed,
                "anomaly": anomaly_compressed,
                "timestamp": datetime.now()
            })

        
    # Display success message
    success_placeholder = st.empty()
    success_placeholder.success('Done Processing. Please navigate to Visualise Tab.')
    time.sleep(3)
    success_placeholder.empty()

# Function to retrieve MRI result
def retrieve_mri_result():
    if st.session_state.patient_reference2 == '': # Check patient reference
        st.toast(body='Please enter a patient reference.', icon='⚠️')
    
    if st.session_state.modality2 is None:   # Check modality
        st.toast(body='Please select a modality.', icon='⚠️')

    if st.session_state.patient_reference2 != '' and st.session_state.modality2 is not None:
        if st.session_state.modality2 == 'T1':
            mdl = 1
        else:
            mdl = 2

        data = {
            "patient_reference": st.session_state.patient_reference2,
            "modality": mdl
        }

        # Retrieve MRI result from MongoDB
        doc = db_service.find("mri_records", data)
        if doc is None:
            st.toast(body='No patient found.', icon='⚠️')
        else:
            # Disable submit button
            st.session_state["disabled"] = False
            st.session_state["disabled2"] = True

            # Retrieve data
            with st.spinner('Retrieving MRI...', show_time=True):
                image = doc["image"]
                anomaly = doc["anomaly"]

                # To decompress and reshape
                decompressed_image = zlib.decompress(image)
                decompressed_anomaly = zlib.decompress(anomaly)

                decompressed_image_array = np.frombuffer(decompressed_image, dtype=np.float32)
                decompressed_anomaly_array = np.frombuffer(decompressed_anomaly, dtype=np.float32)

                decompressed_image_array = decompressed_image_array.reshape(128, 128, 128)
                decompressed_anomaly_array = decompressed_anomaly_array.reshape(128, 128, 128)

                st.session_state.image = decompressed_image_array
                st.session_state.anomaly = decompressed_anomaly_array

            # Display success message
            success_placeholder = st.empty()
            success_placeholder.success('Patient record found. Please navigate to Visualise Tab.')
            time.sleep(3)
            success_placeholder.empty()

# Streamlit app
tab1, tab2, tab3 = st.tabs(["Upload MRI", "Retreive MRI Result", "Visualise MRI Result"])

with tab1:
    # Welcome message
    st.markdown("<h1 style='font-size: 40px;'>Abnormalities Detection in Brain MRI</h1>", unsafe_allow_html=True)
    st.caption('This is a web application that detects abnormalities in brain MRI images.')
    st.caption('Please upload a 3D MRI image to get started.')

    # Nessassary uploads
    mri_file = st.file_uploader(label='The file should be in .nii.gz format.', type=['nii.gz'])
    if mri_file is not None:
        st.session_state.mri_file = mri_file
    patient_reference = st.text_input(label='Patient Reference: ', value='', placeholder='Enter patient reference here', key='patient_reference')
    modality = st.selectbox(label='Modality: ', options=['T1', 'T2'], index=None, key='modality')

    # Validation and Processing
    submit_btn = st.button(label='Submit', type='primary', key='submit_btn', disabled=st.session_state["disabled"], on_click=validate_input)
        
with tab2:
    # Welcome message
    st.markdown("<h1 style='font-size: 40px;'>Abnormalities Detection in Brain MRI</h1>", unsafe_allow_html=True)
    st.caption('This is a web application that detects abnormalities in brain MRI images.')
    st.caption('To check a MRI result, please insert a patient refrence and modality to get started.')

    # Nessassary uploads
    patient_reference = st.text_input(label='Patient Reference: ', value='', placeholder='Enter patient reference here', key='patient_reference2')
    modality = st.selectbox(label='Modality: ', options=['T1', 'T2'], index=None, key='modality2')

    # Validation and Retreival
    submit_button = st.button(label='Submit', type='primary', key='submit_btn2', disabled=st.session_state["disabled2"], on_click=retrieve_mri_result)

with tab3:
    if st.session_state.image is not None and st.session_state.anomaly is not None:    
        # Welcome message
        st.markdown("<h1 style='font-size: 40px;'>Abnormalities Detection in Brain MRI</h1>", unsafe_allow_html=True)
        st.caption('This is a web application that detects abnormalities in brain MRI images.')
        st.caption('To visualise a MRI result, please click on desired plane and adjust the slider to get started.')

        if st.session_state.disabled: # Navigate from upload tab
            st.markdown(f'**Patient Reference:** {st.session_state.patient_reference}')
            st.markdown(f'**Modality:** {st.session_state.modality}')
        else: # Navigate from retrieve tab
            st.markdown(f'**Patient Reference:** {st.session_state.patient_reference2}')
            st.markdown(f'**Modality:** {st.session_state.modality2}')

        st.markdown(f'**Planes:**')

        # Display the original image and anomaly
        with st.expander('Axial Plane'):
            # Slider for axial slice
            axial_slice = st.slider(label='Slices', min_value=0, max_value=124, value=44, step=1, format='%d', key='axial_slice')
            
            # Display the image and anomaly slices
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            image_slice = st.session_state.image[:, :, axial_slice]
            anomaly_slice = st.session_state.anomaly[:, :, axial_slice]
            
            axes[0].imshow(image_slice, cmap="gray")
            axes[1].imshow(anomaly_slice, cmap="gray")
            axes[2].imshow(image_slice, cmap="gray")
            axes[2].imshow(anomaly_slice, cmap="hot", alpha=0.5)
 
            for col in range(3):
                axes[col].axis("off")

            axes[0].set_title("MRI")
            axes[1].set_title("Anomaly")
            axes[2].set_title("MRI + Anomaly")

            plt.tight_layout()
            st.pyplot(fig)

        with st.expander('Coronal Plane'):
            # Slider for coronal slice
            coronal_slice = st.slider(label='Slices', min_value=0, max_value=124, value=62, step=1, format='%d', key='coronal_slice')

            # Display the image and anomaly slices
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            image_slice = st.session_state.image[:, coronal_slice, :]
            anomaly_slice = st.session_state.anomaly[:, coronal_slice, :]
            
            axes[0].imshow(image_slice, cmap="gray")
            axes[1].imshow(anomaly_slice, cmap="gray")
            axes[2].imshow(image_slice, cmap="gray")
            axes[2].imshow(anomaly_slice, cmap="hot", alpha=0.5)
 
            for col in range(3):
                axes[col].axis("off")

            axes[0].set_title("MRI")
            axes[1].set_title("Anomaly")
            axes[2].set_title("MRI + Anomaly")

            plt.tight_layout()
            st.pyplot(fig)

        with st.expander('Sagittal Plane'):
            # Slider for sagittal slice
            sagittal_slice = st.slider(label='Slices', min_value=0, max_value=124, value=62, step=1, format='%d', key='sagittal_slice')

            # Display the image and anomaly slices
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            image_slice = st.session_state.image[sagittal_slice, :, :]
            anomaly_slice = st.session_state.anomaly[sagittal_slice, :, :]
            
            axes[0].imshow(image_slice, cmap="gray")
            axes[1].imshow(anomaly_slice, cmap="gray")
            axes[2].imshow(image_slice, cmap="gray")
            axes[2].imshow(anomaly_slice, cmap="hot", alpha=0.5)
 
            for col in range(3):
                axes[col].axis("off")

            axes[0].set_title("MRI")
            axes[1].set_title("Anomaly")
            axes[2].set_title("MRI + Anomaly")

            plt.tight_layout()
            st.pyplot(fig)
        
        
