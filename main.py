from model import *
from const import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

m = Model("labels.csv")

train_gen, val_gen = m.get_data_generators("dataset_split", BATCH_SIZE, (RESOLUTION, RESOLUTION))
model = m.build_model(NEURON_COUNT, DROPOUT, RESOLUTION)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)
model.save("cnn_minmax.keras")
loss, acc = model.evaluate(val_gen)
print(f"TEST ACCURACY: {acc * 100:.2f}")

y_true = val_gen.classes # Skutečné stavy
y_pred_probs = model.predict(val_gen) # Předpovědi na základě modelu
y_pred = np.argmax(y_pred_probs, axis=1) # Vybere nejpravděpodobnější stav ze tří na základě předpovědí
                                         # (ze tří polí tedy vznikne jedno lajdácky řečeno)

cm = confusion_matrix(y_true, y_pred) # Vytvoří confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_gen.class_indices.keys()))

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()