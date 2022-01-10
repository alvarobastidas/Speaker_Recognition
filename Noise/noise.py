import os
import tensorflow as tf
from pathlib import Path
from constants import Constants as cts


class Noise:
    def __init__(self):
        self.noise_paths = []
        self.noises = []

    def noise_preparation(self):
        # NOISE PREPATATION
        if '.DS_Store' in os.listdir(cts.DATASET_NOISE_PATH):
            os.remove(cts.DATASET_NOISE_PATH + '/.DS_Store')

        for subdir in os.listdir(cts.DATASET_NOISE_PATH):
            subdir_path = Path(cts.DATASET_NOISE_PATH) / subdir
            if os.path.isdir(subdir_path):
                self.noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
                ]
        return self.noise_paths

    def get_noise_tensor(self, noise_paths):
        # Resampling all noise samples to 16000 hz
        resampling_noise_to_16000hz()
        # Split noise into chunks of 16000 each
        self.noises = []
        for path in noise_paths:
            sample = load_noise_sample(path)
            if sample:
                self.noises.extend(sample)
        self.noises = tf.stack(self.noises)

        return self.noises


def resampling_noise_to_16000hz():
    command = (
            "for dir in `ls -1 " + cts.DATASET_NOISE_PATH + "`; do "
                                                            "for file in `ls -1 " + cts.DATASET_NOISE_PATH + "/$dir/*.wav`; do "
                                                                                                             "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
                                                                                                             "$file | grep sample_rate | cut -f2 -d=`; "
                                                                                                             "if [ $sample_rate -ne 16000 ]; then "
                                                                                                             "ffmpeg -hide_banner -loglevel panic -y "
                                                                                                             "-i $file -ar 16000 temp.wav; "
                                                                                                             "mv temp.wav $file; "
                                                                                                             "fi; done; done"
    )
    os.system(command)


def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == cts.SAMPLING_RATE:
        slices = int(sample.shape[0] / cts.SAMPLING_RATE)
        sample = tf.split(sample[: slices * cts.SAMPLING_RATE], slices)
        return sample
    else:
        print(f"Sampling rate {path} is incorrect. Ignorit it")
        return None






