import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Pro augmentace
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Nastavení ---
IMG_SIZE = 128
PATCH_SIZE = 32
DATA_DIR = 'dataset/'
CSV_PATH = os.path.join(DATA_DIR, 'labels.csv')
IMG_DIR = os.path.join(DATA_DIR, 'img/')
BATCH_SIZE = 32
EPOCHS = 20
USE_ALPHA = True

def load_and_patch_image(img_path, use_alpha=USE_ALPHA):
    img = Image.open(img_path).convert('RGB')
    img_array = img_to_array(img) / 255.0
    patches = []
    for i in range(0, IMG_SIZE, PATCH_SIZE):
        for j in range(0, IMG_SIZE, PATCH_SIZE):
            patch = img_array[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            if patch.shape == (PATCH_SIZE, PATCH_SIZE, 3):
                if use_alpha:
                    gray = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    _, alpha = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
                    alpha = alpha.astype(np.float32) / 255.0
                    alpha *= np.random.uniform(0.8, 1.2)
                    alpha = np.clip(alpha, 0, 1)
                    patch = np.concatenate([patch, alpha[..., np.newaxis]], axis=-1)
                patches.append(patch)
    return patches

df = pd.read_csv(CSV_PATH)
mode_map = {'L-mode': 0, 'H-mode': 1, 'ELM': 2}
df['label'] = df['mode'].map(mode_map)

print("Distribuce tříd:")
print(df['mode'].value_counts())

class_counts = df['label'].value_counts().sort_index()
class_weights = {i: max(class_counts) / count for i, count in enumerate(class_counts)}

X, y = [], []
for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row['filename'])
    if os.path.exists(img_path):
        patches = load_and_patch_image(img_path)
        X.extend(patches)
        y.extend([row['label']] * len(patches))

X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['discharge'].repeat((IMG_SIZE // PATCH_SIZE) ** 2)))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

input_shape = (PATCH_SIZE, PATCH_SIZE, 4 if USE_ALPHA else 3)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- Trénování s class weights ---
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# --- Vyhodnocení ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1), average='weighted')
print(f"F1-score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# --- Stabilita (příklad pro jeden discharge) ---
sample_discharge = df['discharge'].unique()[0]
seq_df = df[df['discharge'] == sample_discharge].sort_values('time')
seq_patches = []
for _, row in seq_df.iterrows():
    img_path = os.path.join(IMG_DIR, row['filename'])
    seq_patches.extend(load_and_patch_image(img_path))

seq_patches = np.array(seq_patches)
seq_preds = np.argmax(model.predict(seq_patches), axis=1)
print(f"Stabilita pro discharge {sample_discharge}: Predikce sekvence (po patchech): {seq_preds}")

# --- Export modelu ---
model.save('saved_model.h5')
print("Model exportován do saved_model.h5")

# --- Porovnání s/bez alpha kanálu ---
if USE_ALPHA:
    print("\nPorovnání bez alpha:")
    X_train_no_alpha = X_train[..., :3] if USE_ALPHA else X_train
    X_test_no_alpha = X_test[..., :3]
    model_no_alpha = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(PATCH_SIZE, PATCH_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model_no_alpha.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_no_alpha.fit(datagen.flow(X_train_no_alpha, y_train, batch_size=BATCH_SIZE), epochs=5, validation_data=(X_test_no_alpha, y_test))  # Kratší pro test
    loss_no, acc_no = model_no_alpha.evaluate(X_test_no_alpha, y_test)
    print(f"Accuracy bez alpha: {acc_no*100:.2f}% (vs. s alpha: {acc*100:.2f}%)")
