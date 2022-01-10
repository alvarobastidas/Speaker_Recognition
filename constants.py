# Constants Definition
import os
import time


class Constants:
    # Dataset got  from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
    # and save it to the following path:
    DATASET_ROOT = os.path.join(os.path.expanduser('~'), "/Users/fernando/Desktop/google_sync/16000_pcm_speeches")
    DATASET_ROOT_TEST = os.path.join(os.path.expanduser('~'), "/Users/fernando/Desktop/google_sync/test_16000_pcm_speeches")
    MODEL_REPO = os.path.join(os.path.expanduser('~'), "/Users/fernando/Desktop/google_sync/model_repository")

    # Folders to put samples of audio and noise respectively
    AUDIO_SUBFOLDER = 'audio'
    NOISE_SUBFOLDER = 'noise'
    DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
    DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)
    # Percentage of samples to use for validation
    VALID_SPLIT = 0.1
    # Seed to use when shuffling the dataset and the noise
    SHUFFLE_SEED = 43
    # The sampling rate to use for all audio samples
    # All noise file are resamples with this sampling rate
    SAMPLING_RATE = 16000
    # The factor to multiply the noise with accoirding to:
    # noise_sample = sample + noise * prop * scale
    # Where prop = sample_amplitude / noise_amplitude
    SCALE = 0.5
    BATCH_SIZE = 128
    EPOCHS = 100

    RECORD_SECONDS = 10  # Recording duration for testing
    RECORD_SECONDS_DS = 15  # 1500  # Recording duration for dataset

    SAMPLE_TO_DISPLAY = 10  # Batch Size for validation

    NAME = "speaker_recognition_cnn".format(int(time.time()))

    if '.DS_Store' in os.listdir(DATASET_AUDIO_PATH):
        os.remove(DATASET_AUDIO_PATH + '/.DS_Store')

    if '.DS_Store' in os.listdir(DATASET_ROOT_TEST):
        os.remove(DATASET_ROOT_TEST + '/.DS_Store')



