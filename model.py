import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Pokus načtení jednoho z obrázků na základě filename
# simple_path = labels.iloc[0]["filename"]
# img = cv2.imread(simple_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.title(labels.iloc[0]["mode"])
# plt.savefig("test.png")
# print("image saved")

# self-explanatory, k načtení také normalizace hodnot


class Model:
    def __init__(self, table): # Table: název souboru csv tabulky
        self.table = pd.read_csv(table) # Načtení tabulky
        self.filenames = tuple(self.table["filename"].values)
        self.encoder = LabelEncoder()
    

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64)) # Zmenšení obrázku pro vyšší efektivitu
        img = img / 255.0 # RGB hodnoty jsou převedeny na formát od 0 do 1
        return img


    def get_X(self): # X: vstupové hodnoty 
        X = []
        for i in range(8225, 8235): # později změnit na "for path in self.filenames", nyní jen pro test. účely
            path = self.filenames[i]
            X.append(self.load_image(path))
        X = np.array(X) # Vytvoření matice obsahující všechny obrázky
        return X


    def get_y(self): # Y: známky jednotlivých obrázků
        y_text = self.table["mode"].values
        y_text = y_text[8225:8235] # Pro vyfiltrování konkrétně deseti hodnot, které obsahují všechny znaky, jen pro test
        y_numeric = self.encoder.fit_transform(y_text)
        y = to_categorical(y_numeric)
        return y
