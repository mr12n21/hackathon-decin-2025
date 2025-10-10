import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
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


    def get_paths_from_dir(self, data_dir): # Získání cest obrázků z UŽ SPLITNUTÉHO data setu
        base = Path(data_dir)
        image_paths = []
        labels = []

        for path in base.rglob("*"):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            image_paths.append(str(path))
            labels.append(path.parent.name) # label = název nadřazené složky, tedy příznaků

        return image_paths, labels

    
    def get_data_from_dir(self, data_dir): # Získání dat z UŽ SPLITNUTÉHO data setu 
        X_train_paths, y_train_text = self.get_paths_from_dir(f"{data_dir}/train")
        X_test_paths, y_test_text = self.get_paths_from_dir(f"{data_dir}/valid")
        X_train = [] # Tato funkce jen používá stejnou logiku, jako u jiných funkcí
        X_test = [] # Tudíž viz komentáře tam

        for path in X_train_paths:
            X_train.append(self.load_image(path))
        X_train = np.array(X_train)

        for path in X_test_paths:
            X_test.append(self.load_image(path))
        X_test = np.array(X_test)

        y_train_numeric = self.encoder.fit_transform(y_train_text)
        y_test_numeric = self.encoder.fit_transform(y_train_text)
        y_train = to_categorical(y_train_numeric)
        y_test = to_categorical(y_test_numeric)

        return X_train, X_test, y_train, y_test


    def get_data_generators(self, data_dir, batch_size, target_size): # target_size: norm. rozlišení
        datagen = ImageDataGenerator(rescale=1./255) # RGB hodnoty na rozsah 0 až 1

        train_generator = datagen.flow_from_directory(
            f"{data_dir}/train",
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical"
        )

        test_generator = datagen.flow_from_directory(
            f"{data_dir}/valid",
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical"
        )

        return train_generator, test_generator


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

