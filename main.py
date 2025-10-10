from model import *
from const import *

m = Model("labels.csv")
#X_train, X_test, y_train, y_test = m.get_data_from_dir("dataset_split")

train_gen, val_gen = m.get_data_generators("dataset_split", BATCH_SIZE, (RESOLUTION, RESOLUTION))


model = m.build_model(NEURON_COUNT, DROPOUT) # opět z const.py

#history = model.fit(
#    X_train, y_train,
#    epochs=EPOCHS, # EPOCHS a další jsou z const.py
#    batch_size=BATCH_SIZE,
#    validation_data=(X_test, y_test)
#)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save("cnn_model.keras")
loss, acc = model.evaluate(val_gen)
print(f"Test accuracy: {acc * 100:.2f}")
