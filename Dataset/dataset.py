import os
import tensorflow as tf
from pathlib import Path
from constants import Constants as cts
from .dataset_generation import DatasetGeneration


class Dataset:
    def __init__(self):
        self.audio_paths = []
        self.labels = []

    def get_audio_paths_and_labels(self, class_names):
        for label, name in enumerate(class_names):
            print(f"Processing Speaker {name}")
            dir_path = Path(cts.DATASET_AUDIO_PATH) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            self.audio_paths += speaker_sample_paths
            self.labels += [label] * len(speaker_sample_paths)

        print(
            f"Found {len(self.audio_paths)} files belonging to {len(class_names)} classes"
        )

        return self.audio_paths, self.labels

    def get_training_and_validation_data(self, valid_split, audio_paths, labels):
        num_val_samples = int(valid_split * len(audio_paths))
        print(f"Using {len(audio_paths) - num_val_samples} files for training.")
        train_audio_paths = audio_paths[:-num_val_samples]
        train_labels = labels[:-num_val_samples]

        print(f"Using {num_val_samples} files for validation.")
        valid_audio_paths = audio_paths[-num_val_samples:]
        valid_labels = labels[-num_val_samples:]
        return train_audio_paths, train_labels, valid_audio_paths, valid_labels

    def create_dataset_train(self, train_audio_paths, train_labels, batch=cts.BATCH_SIZE):
        train_ds = DatasetGeneration().paths_and_labels_to_dataset(train_audio_paths, train_labels)
        train_ds = train_ds.shuffle(buffer_size=batch * 8, seed=cts.SHUFFLE_SEED).batch(batch)
        return train_ds

    def create_dataset_valid(self, valid_audio_paths, valid_labels):
        valid_ds = DatasetGeneration().paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=cts.SHUFFLE_SEED).batch(32)
        return valid_ds

    def add_noise_training_set(self, train_ds, noises):
        train_ds = train_ds.map(
            lambda x, y: (DatasetGeneration().add_noise(x, noises, scale=cts.SCALE), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return train_ds

    def transform_audio_to_frequency(self, train_ds):
        train_ds = train_ds.map(
            lambda x, y: (DatasetGeneration().audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds







