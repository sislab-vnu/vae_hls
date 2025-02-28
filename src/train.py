from fxpmath import Fxp
import sys, os, random, shutil, warnings, glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import backend as K

from keras import layers, Model
from keras import Sequential
from vae_model import VAE, encoder, decoder
from data_processing import get_dataset
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def train(dataset_path):
    x_all = get_dataset(dataset_path)
    enc = encoder((64, 64,3), name="encoder")
    dec = decoder(32, name="decoder")
    vae = VAE(enc, dec, beta=0.001)
    vae.compile(optimizer="adam", loss="mse",metrics=['accuracy'] )
    vae.summary()

    MODEL_PATH="./model/capsule.keras"
    EPOCHS = 200

    # Callbacks để lưu mô hình tốt nhất trong quá trình huấn luyện
    callbacks_list = [
        ModelCheckpoint(filepath=MODEL_PATH,  # Dùng cùng một đường dẫn để lưu mô hình và trọng số
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max',
                        verbose=1)
    ]


    history = vae.fit(x=x_all, y=x_all, epochs=EPOCHS, callbacks=callbacks_list, validation_split=0.2, verbose=1)

    vae.encoder.save("vae_encoder_float.h5")
    vae.decoder.save("vae_decoder_float.h5")

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    sns.lineplot(data={k: history.history[k] for k in ('loss', 'val_loss')}, ax=ax[0])
    sns.lineplot(data={k: history.history[k] for k in history.history.keys() if k not in ('loss', 'val_loss')}, ax=ax[1])
    plt.savefig('accuracy_validation_loss_float_model.pdf')
    plt.show()

if __name__ == "__main__":
    import sys
    train(sys.argv[1])
