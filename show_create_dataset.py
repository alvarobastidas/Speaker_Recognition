import os
import streamlit as st
import numpy as np
from constants import Constants as cts
from Audio_treatment.audio_treatment import record_audio, play_audio
from Folder_management.folder_management import create_folder, move_temporal_to_audio_dataset_folder


# WEB PAGE TO CREATE A NEW RECORDING
def create_dataset():

    if "folder_name" not in st.session_state:
        st.session_state.folder_name = True
        st.session_state.last_folder = ""
        st.session_state.resp = None
        st.session_state.folder_created = None
        st.session_state.audio_bytes = b"\0"
        st.session_state.myrecording = None
        st.session_state.msg = ''
        st.session_state.user_input = ''
        st.session_state.recording = np.array([0])

    st.session_state.new_record = st.button("CREATE NEW RECORD")

    if st.session_state.new_record:
        st.session_state.folder_name = False
        st.session_state.folder_created = False
        st.session_state.msg = '### 1. Define a folder name for storing audio recordings.'

    st.write(st.session_state.msg)
    folder_name = st.text_input('')

    if folder_name != st.session_state.last_folder:
        st.session_state.folder_name = True
        st.session_state.audio_bytes = b"\0"
    else:
        if st.session_state.new_record:
            st.text('Please enter a new folder name')

    if st.session_state.folder_name and folder_name:
        if not st.session_state.folder_created:
            st.session_state.resp = create_folder(folder_name)
            st.session_state.folder_created = True
            st.session_state.last_folder = folder_name

        st.text(st.session_state.resp)
        st.session_state.ok = st.button("RECORD")

        if st.session_state.ok:
            st.warning("Recording will take about 25 minutes.")
            path_temp = folder_name + '/' + folder_name + '.wav'
            if os.path.exists(folder_name):
                st.session_state.recording = record_audio(path_temp, cts.RECORD_SECONDS_DS)
                st.session_state.audio_bytes = play_audio(path_temp)
                st.session_state.myrecording = True
            else:
                st.text('Please first CREATE NEW RECORD')
                st.session_state.audio_bytes = b"\0"

        st.audio(st.session_state.audio_bytes)

        st.session_state.save = st.button("SAVE AUDIO RECORD")
        if st.session_state.save:
            answ = move_temporal_to_audio_dataset_folder(folder_name, st.session_state.recording)
            st.text(answ)
            st.session_state.resp = ''




