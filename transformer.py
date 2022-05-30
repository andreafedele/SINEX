import os
import wave
import glob
import time
import pickle
import random
import librosa
import numpy as np
import tensorflow as tf

# source and destination folders name path
src = 'audiomnist'
dst = 'logmel'

# source and target sample rate
source_sr = 48000
target_sr = 44100

# melspectrogram librosa parameters
n_fft = 4096
n_mels = 224
hop_length = 197

def get_db(y, sr):
    ''' Transforms to log-mel spectrogram '''
    mel_signal = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    spectrogram = np.abs(mel_signal)
    db = librosa.power_to_db(spectrogram, ref=1.0)

    # Add a `channels` dimension, so that the spectrogram can be used as image-like input data with convolution layers which expect shape (`batch_size`, `height`, `width`, `channels`)
    return db[..., tf.newaxis]

def zeropad(y):
    '''  Pads vector y by placing the signal at random position within the vector, filling it with zeros '''
    pad_y = y

    # zero padding
    if len(y) < target_sr:
        pad_y = np.zeros(target_sr)
        offset = random.randint(0, target_sr - len(y))
        pad_y[offset:offset+len(y)] = y

    return pad_y

def load_and_get_db(filepath):
    # Load audio
    y, sr = librosa.load(filepath, sr = source_sr)
    # Resample to 44,1 khz
    y = librosa.core.resample(y=y.astype(np.float32), orig_sr=sr, target_sr=target_sr, res_type="scipy")
    # Pad data
    pad_y = zeropad(y)
    # Get the spectrogram
    db = get_db(pad_y, target_sr)

    return db

def main():
    # setting random seed
    random.seed(94)

    # Saving starting time
    st = time.time()

    for idx, folder in enumerate(os.listdir(src)):
        # only process folders
        if not os.path.isdir(os.path.join(src, folder)):
            continue

        # print execution percentage
        percentage = round((idx * 100) / len(os.listdir(src)), 1)
        print("Execution progress: " + str(percentage) + "%")

        # create the speaker folder in the destination folder
        if not os.path.exists(dst + '/' + folder):
            os.makedirs(dst + '/' + folder)

        # get current speaker wav's paths
        paths = glob.glob(os.path.join(src, folder, "*wav"))

        for wavpath in os.listdir(os.path.join(src, folder)):
            destpath = os.path.join(dst, folder, wavpath.split('.')[0]) + '.pickle'
            db = load_and_get_db(os.path.join(src, folder, wavpath))

            # dump log-mel spect of the audio
            with open(destpath, 'wb') as f:
                pickle.dump(db, f)

    elapsed = round((time.time() - st) / 60, 2)
    print("Finished in " + str(elapsed) + " minutes.")

if __name__ == '__main__':
    main()
