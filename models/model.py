import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from skimage.color import rgb2gray
from tensorflow.keras.callbacks import EarlyStopping
from const import *  # Assuming RESOLUTION, BATCH_SIZE are defined here

class Model:
    def __init__(self, table=None):
        self.table = pd.read_csv(table) if table else None
        self.filenames = tuple(self.table["filename"].values) if table else None
        self.encoder = LabelEncoder()
        self.res = RESOLUTION
        self.use_alpha = False  # Disable alpha channel for directory-based loading
        self.use_grayscale = True  # Enable grayscale augmentation
        self.channels = 3  # flow_from_directory yields 3-channel RGB images
        self.data_dir = os.path.join('data', 'img')  # Adjusted to your dataset structure
        # NOTE: If you want to support RGBA, you must use a custom generator and set self.channels=4

    def load_image(self, path):
        # Load image with alpha channel if PNG and use_alpha=True
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED if self.use_alpha and path.endswith('.png') else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (self.res, self.res))
        img = img / 255.0  # Normalize to [0, 1]
        if self.use_alpha and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)  # Convert to RGBA
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return img

    def get_X(self):
        X = []
        for i in range(8225, 10225):  # Temporary for testing
            path = self.filenames[i]
            img = self.load_image(path)
            X.append(img)
            # Add grayscale version if enabled
            if self.use_grayscale:
                gray_img = rgb2gray(img[..., :3])  # Convert to grayscale
                gray_img = np.expand_dims(gray_img, axis=-1)  # Add channel dimension
                if self.channels == 4:
                    alpha = img[..., 3:4] if img.shape[-1] == 4 else np.ones_like(gray_img)
                    gray_img = np.repeat(gray_img, 3, axis=-1)  # Repeat for 3 channels
                    gray_img = np.concatenate([gray_img, alpha], axis=-1)  # Add alpha
                else:
                    gray_img = np.repeat(gray_img, 3, axis=-1)  # Repeat for RGB
                X.append(gray_img)
        return np.array(X)

    def get_y(self):
        y_text = self.table["mode"].values
        y_text = y_text[8225:10225]  # Temporary for testing
        if self.use_grayscale:
            y_text = np.concatenate([y_text, y_text[8225:10225]])  # Duplicate labels for grayscale
        y_numeric = self.encoder.fit_transform(y_text)
        y = to_categorical(y_numeric, num_classes=3)
        return y

    def split_data(self):
        X = self.get_X()
        y = self.get_y()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def get_paths_from_dir(self, data_dir):
        base = Path(data_dir)
        image_paths = []
        labels = []
        for path in base.rglob("*"):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(str(path))
                labels.append(path.parent.name)
        return image_paths, labels

    def get_data_from_dir(self, data_dir):
        X_train_paths, y_train_text = self.get_paths_from_dir(os.path.join(data_dir, "L-mode") + 
                                                              os.path.join(data_dir, "H-mode") + 
                                                              os.path.join(data_dir, "ELM"))
        X_test_paths, y_test_text = self.get_paths_from_dir(os.path.join(data_dir, "L-mode") + 
                                                            os.path.join(data_dir, "H-mode") + 
                                                            os.path.join(data_dir, "ELM"))
        X_train, X_test = [], []
        for path in X_train_paths:
            X_train.append(self.load_image(path))
        X_train = np.array(X_train)
        for path in X_test_paths:
            X_test.append(self.load_image(path))
        X_test = np.array(X_test)
        y_train_numeric = self.encoder.fit_transform(y_train_text)
        y_test_numeric = self.encoder.transform(y_test_text)
        y_train = to_categorical(y_train_numeric, num_classes=3)
        y_test = to_categorical(y_test_numeric, num_classes=3)
        return X_train, X_test, y_train, y_test

    def get_data_generators(self, data_dir, batch_size, target_size):
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            channel_shift_range=0.2  # For channel robustness
        )
        train_generator = datagen.flow_from_directory(
            data_dir,  # Use data/img directly
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            classes=["L-mode", "H-mode", "ELM"]
        )
        test_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            classes=["L-mode", "H-mode", "ELM"],
            subset="validation"
        )
        return train_generator, test_generator

    def build_model(self, m=512, n=0.5):
        model = Sequential([
            Conv2D(64, (3, 3), activation="relu", input_shape=(self.res, self.res, self.channels)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(m, activation="relu"),
            Dropout(n),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def evaluate_robustness(self, model, X_test, y_test):
        # Standard evaluation
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Standard Test Accuracy: {acc*100:.2f}%")

        # Channel swapping (RGB -> BGR or RGBA -> BGRA)
        def swap_channels(data):
            if self.channels == 3:
                return data[..., [2, 1, 0]]
            elif self.channels == 4:
                return np.concatenate([data[..., [2, 1, 0]], data[..., 3:4]], axis=-1)
        X_test_swapped = swap_channels(X_test)
        loss_swapped, acc_swapped = model.evaluate(X_test_swapped, y_test)
        print(f"Swapped Channels Test Accuracy: {acc_swapped*100:.2f}%")

        # Grayscale conversion
        def convert_to_grayscale(data):
            gray_data = np.array([rgb2gray(img[..., :3]) for img in data])
            gray_data = np.expand_dims(gray_data, axis=-1)
            if self.channels == 3:
                gray_data = np.repeat(gray_data, 3, axis=-1)
            elif self.channels == 4:
                alpha = data[..., 3:4] if data.shape[-1] == 4 else np.ones_like(gray_data)
                gray_data = np.repeat(gray_data, 3, axis=-1)
                gray_data = np.concatenate([gray_data, alpha], axis=-1)
            return gray_data
        X_test_gray = convert_to_grayscale(X_test)
        loss_gray, acc_gray = model.evaluate(X_test_gray, y_test)
        print(f"Grayscale Test Accuracy: {acc_gray*100:.2f}%")

        # Alpha channel modification
        if self.use_alpha and self.channels == 4:
            def modify_alpha(data, alpha_value=0.5):
                modified = data.copy()
                modified[..., 3] = alpha_value
                return modified
            X_test_alpha_mod = modify_alpha(X_test, alpha_value=0.5)
            loss_alpha, acc_alpha = model.evaluate(X_test_alpha_mod, y_test)
            print(f"Modified Alpha (0.5) Test Accuracy: {acc_alpha*100:.2f}%")

# Example usage in main.py
if __name__ == "__main__":
    model_instance = Model(table="path/to/your/csv.csv")  # Replace with your CSV path
    X_train, X_test, y_train, y_test = model_instance.split_data()
    model = model_instance.build_model()
    
    # Training with data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        channel_shift_range=0.2
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    
    # Evaluate robustness
    model_instance.evaluate_robustness(model, X_test, y_test)
    
    # Save model
    model.save("complex_model.h5")