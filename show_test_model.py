import os
import streamlit as st
import matplotlib.pyplot as plt
from constants import Constants as cts
from Audio_treatment.audio_treatment import save_audio_dataset, record_audio, play_audio, graph_audio
from Test_Trained_Model_SR import make_prediction, make_prediction_with_dataset, show_valid_dataset


def show_test_model():
    st.write("""### 1. RECORD AUDIO """)
    filename = 'audio_test.wav'
    temporal = os.path.join("/Users/fernando/Final_Project/temporal")
    audio_path = os.path.join(temporal, filename)

    if 'prediction' not in st.session_state:
        st.session_state.prediction = "No audio for prediction!!"
        st.session_state.audio_bytes = b"\0"
        st.session_state.fig = plt.figure()

    ok = st.button("RECORD AUDIO")
    # graph_audio(audio_path)

    if ok:
        myrecording = record_audio(audio_path, cts.RECORD_SECONDS)
        st.session_state.audio_bytes = play_audio(audio_path)
        st.session_state.fig = graph_audio(audio_path)
        save_audio_dataset(cts.DATASET_ROOT_TEST, myrecording, duration=cts.RECORD_SECONDS)

    st.pyplot(st.session_state.fig)
    st.audio(st.session_state.audio_bytes)

    pred_button = st.button("PREDICT")
    noise_check = st.checkbox("ADD NOISE")

    if pred_button:
        st.session_state.prediction = make_prediction(noise_check)

    st.write(st.session_state.prediction)

    st.write("""### 2. USING VALIDATION DATASET """)
    text = """
    This prediction will be made, using pre-record dataset, specifically for validation, 
    which not was used in the training stage.
    """
    st.text(text)

    info = show_valid_dataset()
    st.write(info)

    data_predict_button = st.button("PREDICTION")
    noise_check_2 = st.checkbox("ADDING NOISE")

    if data_predict_button:
        st.write(make_prediction_with_dataset(noise_check_2))






