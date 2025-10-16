from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


model = load_model("cnn_mid.keras")

def evaluate_dir(dir_path):
    data_gen = ImageDataGenerator(rescale=1./255)
    val_gen = data_gen.flow_from_directory(
        "img",
        target_size=(32, 32),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )

    loss, acc = model.evaluate(val_gen, verbose=0)
    return loss, acc
