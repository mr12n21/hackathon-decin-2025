from model import *
from const import *

m = Model("labels.csv")
X_train, X_test, y_train, y_test = m.get_data_from_dir("dataset_split")



model = m.build_model(NEURON_COUNT, DROPOUT) # opět z const.py

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS, # EPOCHS a další jsou z const.py
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

#model.save("cnn_model.keras")
