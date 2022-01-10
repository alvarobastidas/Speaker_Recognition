import os
import streamlit as st
import shutil
from constants import Constants as cts
from Audio_treatment.audio_treatment import save_audio_dataset


def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)
        answer = f'Folder {name} created succesfully'
    else:
        answer = f"The folder {name} already exists"
    return answer


def move_temporal_to_audio_dataset_folder(folder, recording):
    if st.session_state.myrecording:
        dataset_audio_path = os.path.join(cts.DATASET_AUDIO_PATH, folder)

        if not os.path.exists(dataset_audio_path):
            os.makedirs(dataset_audio_path)
            save_audio_dataset(dataset_audio_path, recording, duration=cts.RECORD_SECONDS_DS)
            st.session_state.myrecording = False
            answer = f'Audio Recording saved successfully !!'
        else:
            answer = f'Folder "{folder}" already exist in audio dataset db.'
        delete_folder(folder)
    else:
        answer = f'No audio record found'
    return answer


def delete_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        answer = f'Temporal folder {folder} deleted.'
    else:
        answer = f'Temporal folder {folder} does not exist'

    return answer
