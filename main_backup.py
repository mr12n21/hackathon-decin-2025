from model import *
from const import *



#X_train, X_test, y_train, y_test = m.split_data() # PRO TRÉNOVÁNÍ Z TABULKY
#train_gen, val_gen = m.get_data_generators("dataset_split", BATCH_SIZE, (RESOLUTION, RESOLUTION)) # PRO TRÉNOVÁNÍ Z UŽ SPLITNUTÉHO DATASETU


#model = m.build_model(NEURON_COUNT, DROPOUT) # opět z const.py

#history = model.fit( # PRO TRÉNOVÁNÍ Z TABULKY
#    X_train, y_train,
#    epochs=EPOCHS, # EPOCHS a další jsou z const.py
#    batch_size=BATCH_SIZE,
#    validation_data=(X_test, y_test)
#)

#history = model.fit( # PRO TRÉNOVÁNÍ Z UŽ SPLITNUTÉHO DATASETU
#    train_gen,
#    validation_data=val_gen,
#    epochs=EPOCHS
#)

#model.save("cnn_david.keras")
#loss, acc = model.evaluate(val_gen)
#print(f"Test accuracy: {acc * 100:.2f}")

m = Model("labels.csv")

train_gen, val_gen = m.get_data_generators("dataset_split", BATCH_SIZE, (RESOLUTION, RESOLUTION))
model = m.build_model(256, 0.3, 224)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50
)
model.save("cnn_max.keras")
loss, acc = model.evaluate(val_gen)
print(f"TEST ACCURACY: {acc * 100:.2f}")

#for resolution in range(32, 224, 16):
#    for neuron_count in range(64, 256, 32):
#        for dropout in range(2, 5):
#            print("-------------------------- STARTING NEW TRAINING --------------------------")
#            print(f"Resolution: {resolution}, Neuron count: {neuron_count}, Dropout: 0.{dropout}")

#            dropout = dropout / 10
#            print(dropout)
#            train_gen, val_gen = m.get_data_generators("dataset_split", BATCH_SIZE, (resolution, resolution))
#            model = m.build_model(neuron_count, dropout, resolution)
#            model.fit(
#                train_gen,
#                validation_data=val_gen,
#                epochs=20
#            )
#            model.save(f"cnn_r{resolution}_nc{neuron_count}_d{dropout}.keras")
#            loss, acc = model.evaluate(val_gen)
#            print(f"TEST ACCURACY: {acc * 100:.2f}")
