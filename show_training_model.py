import streamlit as st
import os
import pandas as pd
import base64
from streamlit_tensorboard import st_tensorboard
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from constants import Constants as cts
from Model.model_definition import Model
from Speaker_Recognition import training_model, model_info


def show_training_model():
    # Show speakers table
    st.write("""### 1. DATASET DESCRIPTION""")
    records = []
    items = []
    class_names = os.listdir(cts.DATASET_AUDIO_PATH)
    for item, speaker in enumerate(class_names):
        path = cts.DATASET_AUDIO_PATH + "/" + speaker
        records.append(len(os.listdir(path)))
        items.append(item)

    table_data = {"SPEAKER": class_names, "RECORDINGS": records}
    df = pd.DataFrame(table_data)
    st.table(df)

    # Listen a sample of speakers set
    speaker = st.selectbox("Select Speaker", class_names)
    path = cts.DATASET_AUDIO_PATH + "/" + speaker
    for audio in os.listdir(path):
        audio_path = path + "/" + audio
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        audio_file.close()
        break

    # Machine Learning Model
    st.write("""### 2. ML MODEL DESCRIPTION""")
    # Show model layers
    output = st.empty()
    with st_capture(output.code):
        model_summary()

    show_pdf()

    # Training Model
    st.write("""### 3. TRAINING MODEL""")
    st.warning("This operation could take a long time. Are you sure that you want to run the training?")
    # Show model parameters.
    output = st.empty()
    with st_capture(output.code):
        model_info()

    ok, noise_check = st.button("TRAINING MODEL"), st.checkbox("ADDING NOISE")

    if ok:
        output = st.empty()
        with st_capture(output.code):
            training_model(noise_check)

    tensor_board = st.button("Start Tensorboard")
    if tensor_board:
        st_tensorboard(logdir="logs/", port=6006, width=800)


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


def model_summary():
    class_names = os.listdir(cts.DATASET_AUDIO_PATH)
    model = Model().build_model((cts.SAMPLING_RATE // 2, 1), len(class_names))
    model.summary()


def show_pdf():
    with open("Images/cnn.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
