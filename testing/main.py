import os
import sys
sys.path.append(os.getenv('PROJECT_ROOT'))
import dotenv
dotenv.load_dotenv()
import subprocess
import numpy as np
import nibabel as nib
import streamlit as st
from src.database import MongoDBService
from src.inference import prediction
from datetime import datetime

# Initialise session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'mri_file' not in st.session_state:
    st.session_state.mri_file = None
if 'patient_reference' not in st.session_state:
    st.session_state.patient_reference = ''
if 'modality' not in st.session_state:
    st.session_state.modality = None
if 'submit_btn' not in st.session_state:
    st.session_state.submit_btn = False

# Viewer
napari_script = os.path.join(os.getenv('PROJECT_ROOT'), r'src\view_with_napari.py')
db_service = MongoDBService(os.getenv('MONGO_URI'), 'PatientData')

# Functions
def save_as_nii_gz(array, save_path, affine=None):
    if affine is None:
        affine = np.eye(4) 

    nii_image = nib.Nifti1Image(array, affine)
    nib.save(nii_image, save_path)

def save_patient(client, patient_reference, modality, uploaded_file, anomaly_mask):
    with open(uploaded_file, 'rb') as f:
        original_file_content = f.read()
    original_mri_file_id = client.save_file(os.path.basename(uploaded_file), original_file_content)

    with open(anomaly_mask, 'rb') as f:
        anomaly_mask_content = f.read()
    anomaly_mask_file_id = client.save_file('anomaly_mask.nii.gz', anomaly_mask_content)

    data = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_reference": patient_reference,
        "modality": modality,
        "ori_mri_id": original_mri_file_id,
        "anomaly_mask_id": anomaly_mask_file_id  
    }
    
    client.insert_patient("mri_records", data)

def process_mri(client, mri_path, patient_reference, modality, insert=False):
    if st.session_state.modality == 'T1':
        modality = 1
    else:
        modality = 2

    reconstruction, anomaly, result = prediction(mri_path, modality)
    
    anomaly_save_path = os.path.join(os.getenv('PROJECT_ROOT'), r'interface\temp_file_storage', 'anomaly_mask.nii.gz')
    result_save_path = os.path.join(os.getenv('PROJECT_ROOT'), r'interface\temp_file_storage', 'result.nii.gz')

    save_as_nii_gz(anomaly, anomaly_save_path)
    save_as_nii_gz(result, result_save_path)

    if insert:
        save_patient(client, patient_reference, st.session_state.modality, mri_path, anomaly_save_path)
    
    return anomaly_save_path, result_save_path
    
def delete_files_in_folder(folder_path=os.path.join(os.getenv('PROJECT_ROOT'), r'interface\temp_file_storage')):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) and filename.endswith('.nii.gz'):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Navigation
with st.sidebar:
    st.title('Navigation')
    selection = st.radio('Go to:', ['Home', 'Retrieve', 'Delete'])

    st.session_state.page = selection

# Home Page
if st.session_state.page == 'Home':
    st.markdown("<h1 style='font-size: 40px;'>Abnormalities Detection in Brain MRI</h1>", unsafe_allow_html=True)
    st.caption('This is a web application that detects abnormalities in brain MRI images.')
    st.caption('Please upload a 3D MRI image to get started.')

    mri_file = st.file_uploader(label='The file should be in .nii.gz format.', type=['nii.gz'])
    if mri_file is not None:
        st.session_state.mri_file = mri_file

        # Save the file to disk
        save_path = os.path.join(os.getenv('PROJECT_ROOT'), r'interface\temp_file_storage', mri_file.name)
        with open(save_path, "wb") as f:
            f.write(mri_file.getbuffer())  

    save_toggle = st.toggle(label='Save Result', value=False)
    if save_toggle:
        with st.container(key='patient_form', border=True):
            patient_reference = st.text_input(label='Patient Reference: ', value='', placeholder='Enter patient reference here', key='patient_reference')
            modality = st.selectbox(label='Modality: ', options=['T1', 'T2'], index=None, key='modality')

    submit_btn = st.button(label='Submit', type='primary', disabled=st.session_state.submit_btn)
    if submit_btn:
        if st.session_state.mri_file is None:
            st.toast(body='Please upload a file.', icon='⚠️')
        else: 
            if save_toggle:
                if st.session_state.patient_reference == '':
                    st.toast(body='Please enter a patient reference.', icon='⚠️')
                if st.session_state.modality is None:
                    st.toast(body='Please select a modality.', icon='⚠️')
                if st.session_state.mri_file is not None and st.session_state.patient_reference != '' and st.session_state.modality is not None:
                    st.session_state.submit_btn = True

                    with st.spinner('Processing MRI...', show_time=True):
                        anomaly, result = process_mri(db_service, save_path, st.session_state.patient_reference, st.session_state.modality, True)
                        viewer = subprocess.run(['python', napari_script, result], shell=True)

                    st.success('Processing complete.')
                    delete_files_in_folder()

            else:
                st.session_state.submit_btn = True
                with st.spinner('Processing MRI...', show_time=True):
                        anomaly, result = process_mri(db_service, save_path, st.session_state.patient_reference, st.session_state.modality, True)
                        viewer = subprocess.run(['python', napari_script, result], shell=True)

                st.success('Processing complete.')
                delete_files_in_folder()  
                        
# Retrieve Page
elif st.session_state.page == 'Retrieve':
    patient_reference = st.text_input(label='Patient Reference: ', value='', placeholder='Enter patient reference here', key='patient_reference')
    
    original_file, anomaly_mask, modality = db_service.find_patient('mri_records', patient_reference)

    if original_file is None:
        st.warning('No patient found.')
    else:
        st.success('Patient found.')

# Delete Page
elif st.session_state.page == 'Delete':
    st.title('Delete')
    st.caption('This is the delete page.')






# st.title('Abnormalities Detection in Brain MRI')

# if 'complete_input' not in st.session_state:
#     st.caption('This is a web application that detects abnormalities in brain MRI images. Please upload a 3D MRI image to get started. The file should be in .nii.gz format.')

#     mri_file = st.file_uploader(label='Upload a 3D MRI image', type=['nii.gz'])
#     if mri_file is not None:
#         st.session_state.mri_file = mri_file

#     save_toggle = st.toggle(label='Save Result', value=False)
#     if save_toggle:
#         with st.container(key='patient_form', border=True):
#             patient_reference = st.text_input(label='Patient Reference: ', value='', placeholder='Enter patient reference here', key='patient_reference')
#             modality = st.selectbox(label='Modality: ', options=['T1', 'T2'], index=None, key='modality')

#     submit_btn = st.button(label='Submit', type='primary')
#     if submit_btn:
#         if st.session_state.mri_file is None:
#             st.toast(body='Please upload a file.', icon='⚠️')
#         else: 
#             if save_toggle:
#                 if st.session_state.patient_reference == '':
#                     st.toast(body='Please enter a patient reference.', icon='⚠️')
#                 if st.session_state.modality is None:
#                     st.toast(body='Please select a modality.', icon='⚠️')
#                 if st.session_state.mri_file is not None and st.session_state.patient_reference != '' and st.session_state.modality is not None:
#                     confirmation_dialog()
#                     st.session_state.complete_input = True
#             else:
#                 st.session_state.complete_input = True
#                 st.rerun()
# else:
#     st.write('**MRI File:** ', st.session_state.mri_file.name)
#     st.write(f'**Patient Reference:** {st.session_state.patient_reference}')
#     st.write(f'**Modality:** {st.session_state.modality}')

#     db_service = MongoDBService(os.getenv('MONGO_URI'), 'PatientData')

#     # Load .npz file
#     data = np.load(npz_path)

# # Extract arrays (adjust the key based on your file)
# image = data["data"]  # or 'ground_truth', 'mask' if needed

# # Open Napari viewer
# viewer = napari.Viewer()
# viewer.add_image(image, name="MRI Scan")

# # Start the Napari event loop
# napari.run()
    

    
    



    
        




    