import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from const import *


class Model:
    def __init__(self, table): # Table: název souboru csv tabulky
        self.table = pd.read_csv(table) # Načtení tabulky
        self.filenames = tuple(self.table["filename"].values)
        self.encoder = LabelEncoder()
        self.res = RESOLUTION
    

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.res, self.res)) # Zmenšení obrázku pro vyšší efektivitu
        img = img / 255.0 # RGB hodnoty jsou převedeny na formát od 0 do 1
        return img


    def get_X(self): # X: vstupové hodnoty 
        X = []
        for i in range(8225, 10225): # později změnit na "for path in self.filenames", nyní jen pro test. účely
            path = self.filenames[i]
            X.append(self.load_image(path))
        X = np.array(X) # Vytvoření matice obsahující všechny obrázky
        return X


    def get_y(self): # Y: známky jednotlivých obrázků
        y_text = self.table["mode"].values
        y_text = y_text[8225:10225] # Později smazat tento řádek, nyní jen pro test. účely
        y_numeric = self.encoder.fit_transform(y_text)
        y = to_categorical(y_numeric)
        return y

    def split_data(self):
        X = self.get_X()
        y = self.get_y()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2, # 20 % dat je pro test
            random_state=42, # "seed" pro rozdělení dat
            stratify=y # Bude udržovat poměr výsledků v train a test množinách
        )

        return X_train, X_test, y_train, y_test

    # CNN!!!
    def build_model(self, m, n): # m: počet neuronů, n: dropout
        model = Sequential([
            # Konvulace
            Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            # Klasifikace
            Flatten(),
            Dense(m, activation="relu"), # 
            Dropout(n), # Vypínání neuronů pro prevenci předučení
            Dense(3, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
