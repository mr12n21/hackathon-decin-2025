from model import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

m = Model("labels.csv")

RESOLUTION = 32 # Musí se změnit i v initu Modelu
EPOCHS = 25
BATCH_SIZE = 16
NEURON_COUNT = 32
DROPOUT = 0.1

train_gen, val_gen = m.get_data_generators("dataset_split", BATCH_SIZE, (RESOLUTION, RESOLUTION))
model = m.build_model(NEURON_COUNT, DROPOUT, RESOLUTION)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
)
model.save("cnn_mid.keras")