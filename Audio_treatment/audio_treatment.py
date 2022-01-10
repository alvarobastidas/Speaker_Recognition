import os
import streamlit as st
import sounddevice as sd
from constants import Constants as cts
import tensorflow as tf
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt


# RECORD AUDIO
def record_audio(path, duration):
    st.text("RECORDING .....")
    myrecording = sd.rec(int(duration * cts.SAMPLING_RATE), samplerate=cts.SAMPLING_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    audio_record = tf.audio.encode_wav(myrecording, cts.SAMPLING_RATE)
    tf.io.write_file(path, audio_record)
    st.text("Done!! audio recorded")
    return myrecording


# PLAY AUDIO
def play_audio(audio_path):
    audio_file = open(audio_path, 'rb')
    audio_bytes = audio_file.read()
    audio_file.close()
    return audio_bytes


# SHOW AUDIO FORM
def graph_audio(audio_path):
    samplerate, data = read(audio_path)
    duration = len(data) / samplerate
    time = np.arange(0, duration, 1 / samplerate)  # time vector
    fig, ax = plt.subplots()
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(audio_path)
    return fig


# SAVE AUDIO
def save_audio_dataset(path, myrecording, duration):
    # Splitting the record into chunks of 1 second each.
    splitted_size = int(myrecording.shape[0] / duration)
    record_splited = [myrecording[x:x + splitted_size] for x in range(0, myrecording.shape[0], splitted_size)]
    n = 0
    for record in record_splited:
        filename = f'{n}.wav'
        dataset_audio_path_file = os.path.join(path, filename)
        audio = tf.audio.encode_wav(record, cts.SAMPLING_RATE)
        tf.io.write_file(dataset_audio_path_file, audio)
        n += 1
