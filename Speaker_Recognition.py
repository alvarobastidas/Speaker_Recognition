import os
import numpy as np
import json
from tensorflow import keras
from Noise.noise import Noise
from constants import Constants as cts
from Dataset.dataset import Dataset
from Model.model_definition import Model
from tensorflow.keras.callbacks import TensorBoard


def training_model(noise_check=False):
    tensorboard = TensorBoard(log_dir="logs/{}".format(cts.NAME))

    # NOISE PREPATATION
    noise_paths = Noise().noise_preparation()
    noises = Noise().get_noise_tensor(noise_paths)

    # Get the list of audio file paths along with their corresponding labels
    class_names = os.listdir(cts.DATASET_AUDIO_PATH)
    audio_paths_and_labels = Dataset().get_audio_paths_and_labels(class_names)
    audio_paths = audio_paths_and_labels[0]
    labels = audio_paths_and_labels[1]

    # Shuffle
    rng = np.random.RandomState(cts.SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(cts.SHUFFLE_SEED)
    rng.shuffle(labels)

    # Split into training and validation
    training_audio_paths_and_labels = Dataset().get_training_and_validation_data(cts.VALID_SPLIT, audio_paths, labels)
    train_audio_paths = training_audio_paths_and_labels[0]
    train_labels = training_audio_paths_and_labels[1]
    valid_audio_paths = training_audio_paths_and_labels[2]
    valid_labels = training_audio_paths_and_labels[3]

    # Create 2 datasets, one for training and the other for validation
    train_ds = Dataset().create_dataset_train(train_audio_paths, train_labels)
    valid_ds = Dataset().create_dataset_valid(valid_audio_paths, valid_labels)

    # Add noise to the training set
    if noise_check:
        print("WITH NOISE")
        train_ds = Dataset().add_noise_training_set(train_ds, noises)
        model_save_filename = "/Users/fernando/Desktop/google_sync/model_repository/model_noise.h5"
        save_model_info(class_names, valid_audio_paths, valid_labels, filename="_noise")  # Save the dataset Validation
    else:
        print("WITHOUT NOISE")
        model_save_filename = "/Users/fernando/Desktop/google_sync/model_repository/model.h5"
        save_model_info(class_names, valid_audio_paths, valid_labels)  # Save the dataset for validation

    # Transform audio wave to the frequency domain using FFT
    train_ds = Dataset().transform_audio_to_frequency(train_ds)
    valid_ds = Dataset().transform_audio_to_frequency(valid_ds)

    # Model Definition
    model = Model().build_model((cts.SAMPLING_RATE // 2, 1), len(class_names))
    # model.summary()

    # Compile the model using Adam's default learning rate
    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Add callbacks:
    # 'EarlyStopping' to stop training when the model is not enhancing anymore
    # 'ModelCheckPoint' to always keep the model that has the best val_accuracy

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="val_accuracy", save_best_only=True
    )

    # TRAINING
    history = model.fit(
        train_ds,
        epochs=cts.EPOCHS,
        validation_data=valid_ds,
        callbacks=[earlystopping_cb, mdlcheckpoint_cb, tensorboard],
    )

    model.evaluate(valid_ds)


def save_model_info(_class_names, _valid_audio_paths, _valid_labels, filename=""):
    valid_path = cts.MODEL_REPO + "/model_valid_data" + filename + ".json"
    class_path = cts.MODEL_REPO + "/model_class_names" + filename + ".json"
    valid_dict = {}
    for i in range(len(_valid_labels)):
        valid_dict[i] = (_valid_labels[i], _valid_audio_paths[i])

    class_names_dict = {}
    for i, name in enumerate(_class_names):
        class_names_dict[i] = name

    with open(valid_path, 'w') as json_file:
        json.dump(valid_dict, json_file)

    with open(class_path, 'w') as json_file:
        json.dump(class_names_dict, json_file)


def model_info():
    # Get the list of audio file paths along with their corresponding labels
    class_names = os.listdir(cts.DATASET_AUDIO_PATH)
    print("Class Names: ", class_names)
    print()
    audio_paths_and_labels = Dataset().get_audio_paths_and_labels(class_names)
    audio_paths = audio_paths_and_labels[0]
    labels = audio_paths_and_labels[1]

    # Shuffle
    rng = np.random.RandomState(cts.SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(cts.SHUFFLE_SEED)
    rng.shuffle(labels)
    print()
    Dataset().get_training_and_validation_data(cts.VALID_SPLIT, audio_paths, labels)


# TRAINING###
# training_model()
