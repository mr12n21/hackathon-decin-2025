import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- Nastavení ---
IMG_SIZE = 64
PATCH_SIZE = 32
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
BATCH_SIZE = 100

# --- Funkce pro načtení obrázků ---
def load_images_from_folder(folder_path, label):
    images, labels = [], []
    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg')):
            img = load_img(os.path.join(folder_path, file), target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            for i in range(0, IMG_SIZE, PATCH_SIZE):
                for j in range(0, IMG_SIZE, PATCH_SIZE):
                    patch = img_array[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE, 3):
                        images.append(patch)
                        labels.append(label)
    return images, labels

# --- Načtení datasetu ---
X, y = [], []

folders = {
    "L-mode": 0,
    "H-mode": 1,
    "ELM": 2
}

for folder_name, label in folders.items():
    folder_path = os.path.join(DATA_DIR, folder_name)
    imgs, labels = load_images_from_folder(folder_path, label)
    X.extend(imgs)
    y.extend(labels)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)  # one-hot encoding

# --- Rozdělení datasetu ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- Model CNN ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(PATCH_SIZE, PATCH_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # nyní 3 třídy: L, H, ELM
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Trénování ---
model.fit(X_train, y_train, epochs=20, batch_size=BATCH_SIZE, validation_split=0.1, shuffle=True)

# --- Vyhodnocení ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
