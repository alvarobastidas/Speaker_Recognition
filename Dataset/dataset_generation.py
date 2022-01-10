import tensorflow as tf
from constants import Constants as cts


class DatasetGeneration:
    def __init__(self):
        pass

    def paths_and_labels_to_dataset(self, audio_paths, labels):
        """Constructs a dataset of audios and labels."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x: self.path_to_audio(x))
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds, label_ds))

    def path_to_audio(self, path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, cts.SAMPLING_RATE)
        return audio

    def add_noise(self, audio, noises=None, scale=0.5):
        if noises is not None:
            # Create a random tensor of the same size as audio ranging from
            # 0 to the numnber of noise stream samples that we have.
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
            noise = tf.gather(noises, tf_rnd, axis=0)

            # Get the amplitude proportion between the audio and the noise
            prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
            prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

            # Adding the rescaled noise to audio
            audio = audio + noise * prop * scale

        return audio

    def audio_to_fft(self, audio):
        # Since tf.signal.fft applies Fast Fourier Transformation on the innermost dimension,
        # we need to squeeze the dimensions and then expand them again
        # after FFT
        print(audio)
        audio = tf.squeeze(audio, axis=-1)
        # audio = tf.squeeze(audio)
        fft = tf.signal.fft(
            tf.cast(
                tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        ftt = tf.expand_dims(fft, axis=-1)

        # Return the absolute value of the first half of the FFT
        # which represents the positive frequencies
        return tf.math.abs(ftt[:, : (audio.shape[1] // 2), :])

