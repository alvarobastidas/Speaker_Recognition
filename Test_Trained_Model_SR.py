import os
import numpy as np
import json
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from constants import Constants as cts
from Noise.noise import Noise
from Dataset.dataset import Dataset
from Dataset.dataset_generation import DatasetGeneration
from collections import Counter


def make_prediction(noise_check=False):
    # NOISE PREPATATION
    noise_paths = Noise().noise_preparation()
    noises = Noise().get_noise_tensor(noise_paths)

    # TEST DATASET GENERATION
    class_names_new = os.listdir(cts.DATASET_ROOT_TEST)
    batch_size = 10
    audio_paths = []
    labels = []

    for label, name in enumerate(class_names_new):
        if name.endswith(".wav"):
            dir_path = cts.DATASET_ROOT_TEST + "/" + name
            audio_paths.append(dir_path)
            labels.append(0)

    test_ds = Dataset().create_dataset_train(audio_paths, labels, batch=batch_size)

    if noise_check:
        test_ds = test_ds.map(lambda x, y: (DatasetGeneration().add_noise(x, noises, scale=cts.SCALE), y))
        model = keras.models.load_model("/Users/fernando/Desktop/google_sync/model_repository/model_noise.h5")
    else:
        model = keras.models.load_model("/Users/fernando/Desktop/google_sync/model_repository/model.h5")

    class_names = get_class_names(noise_check)

    # PREDICTION
    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = DatasetGeneration().audio_to_fft(audios)
        # Predict
        y_pred = model.predict(ffts)
        prob = y_pred[:]
        # Take random samples
        # rnd = np.random.randint(0, batch_size, cts.SAMPLE_TO_DISPLAY)
        rnd = [n for n in range(batch_size)]
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]
        speaker_index = y_pred[:]
        probabilities_table(prob, speaker_index)

        for index in range(cts.SAMPLE_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            prediction = f'Speaker - Predicted as: {class_names[y_pred[index]]}'
            st.text(prediction)

            # play = display(Audio(audios[index, :, :].squeeze(), rate=cts.SAMPLING_RATE))
            record_audio_temp(audios[index, :, :], index)
            filename = f'temporal/audio_temp_{index}.wav'
            play_audio_temp(filename)

        prediction = predicted_speaker(speaker_index, class_names)

    return f' #### Speaker predicted is {prediction} in overall ####'


def make_prediction_with_dataset(noise_check_2=False):
    # NOISE PREPATATION
    noise_paths = Noise().noise_preparation()
    noises = Noise().get_noise_tensor(noise_paths)

    # GET VALIDATION DATASET
    validation_dataset = get_validation_dataset(noise_check_2)
    valid_audio_paths = validation_dataset[0]
    valid_labels = validation_dataset[1]

    # DATASET PREPARATION
    test_ds = Dataset().create_dataset_train(valid_audio_paths, valid_labels)

    if noise_check_2:
        test_ds = test_ds.map(lambda x, y: (DatasetGeneration().add_noise(x, noises, scale=cts.SCALE), y))
        model = keras.models.load_model("/Users/fernando/Desktop/google_sync/model_repository/model_noise.h5")
    else:
        model = keras.models.load_model("/Users/fernando/Desktop/google_sync/model_repository/model.h5")

    class_names = get_class_names(noise_check_2)

    # PREDICTION
    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = DatasetGeneration().audio_to_fft(audios)
        # Predict
        y_pred = model.predict(ffts)
        accuracy = calculate_sr_accuracy(y_pred, labels)
        # Take random samples
        rnd = np.random.randint(0, cts.BATCH_SIZE, cts.SAMPLE_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

        for index in range(cts.SAMPLE_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            prediction = f'Speaker: {class_names[labels[index]]} - Predicted: {class_names[y_pred[index]]}'
            st.text(prediction)

            # play = display(Audio(audios[index, :, :].squeeze(), rate=cts.SAMPLING_RATE))
            record_audio_temp(audios[index, :, :], index)
            filename = f'temporal/audio_temp_{index}.wav'
            play_audio_temp(filename)

    return accuracy


def get_validation_dataset(check=False):
    if check:
        filename = '/model_valid_data_noise.json'
    else:
        filename = '/model_valid_data.json'

    with open(cts.MODEL_REPO + filename, 'r') as json_file:
        data = json.load(json_file)
    paths = []
    labels = []
    for value in data.values():
        labels.append(value[0])
        paths.append(value[1])

    return paths, labels


def get_class_names(check=False):
    if check:
        filename = '/model_class_names_noise.json'
    else:
        filename = '/model_class_names.json'

    with open(cts.MODEL_REPO + filename, 'r') as json_file:
        data = json.load(json_file)
    names = list(data.values())
    return names


def record_audio_temp(_audio, index):
    filename = f'temporal/audio_temp_{index}.wav'
    audio_record = tf.audio.encode_wav(_audio, cts.SAMPLING_RATE)
    tf.io.write_file(filename, audio_record)


def play_audio_temp(filename):
    audio_file = open(filename, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    audio_file.close()


def show_valid_dataset():
    paths = get_validation_dataset()[0]
    class_names = get_class_names()
    name_list = []
    for path in paths:
        name = path.split("/")[-2]
        name_list.append(name)

    validation_info = f"The validation dataset has {len(name_list)} files\n\n"

    for item in class_names:
        info = f'{item} has {name_list.count(item)} files in the validation dataset                                 \n'
        validation_info += info

    return validation_info


def probabilities_table(prob_list, index_list):
    samples = ["sample" for i in range(len(prob_list))]
    class_prob = [[prob_list[i][j] * 100 for i in range(len(prob_list))] for j in range(8)]
    table_data = {"SPEAKER": samples,
                  "JS(0)": class_prob[0],
                  "BN(1)": class_prob[1],
                  "MM(2)": class_prob[2],
                  "JG(3)": class_prob[3],
                  "MT(4)": class_prob[4],
                  "NM(5)": class_prob[5],
                  "AB(6)": class_prob[6],
                  "DT(7)": class_prob[7]}
    df = pd.DataFrame(table_data)
    st.table(df)


def predicted_speaker(speaker_index, class_names):
    b = Counter(speaker_index)
    speaker = b.most_common(1)[0][0]
    return class_names[speaker]


def calculate_sr_accuracy(y_predicted, labels):
    prediction = np.argmax(y_predicted, axis=-1)
    labels = labels.numpy()
    tnsv = 0
    tntv = len(prediction)
    for index in range(len(labels)):
        if labels[index] == prediction[index]:
            tnsv += 1
    accuracy = round(100 * tnsv / tntv, 4)
    return f'### SR Model Accuraccy = {accuracy} %'








