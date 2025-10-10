import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Nastavení ---
IMG_SIZE = 128
PATCH_SIZE = 32
DATA_DIR = 'data/'
CSV_PATH = os.path.join(DATA_DIR, 'labels.csv')
IMG_DIR = os.path.join(DATA_DIR, 'img/')
BATCH_SIZE = 32
EPOCHS = 20
USE_ALPHA = True

def load_and_patch_image(img_path, use_alpha=USE_ALPHA):
    try:
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
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return []

# --- Diagnostics ---
print("Current working directory:", os.getcwd())
print("CSV_PATH:", os.path.abspath(CSV_PATH))
print("IMG_DIR:", os.path.abspath(IMG_DIR))

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Sample from CSV:")
print(df.head())
print(f"Total rows in CSV: {len(df)}")
print("Sample filenames from CSV:", df['filename'].head().tolist())

mode_map = {'L-mode': 0, 'H-mode': 1, 'ELM': 2}
df['label'] = df['mode'].map(mode_map)

print("Distribuce tříd:")
print(df['mode'].value_counts())

class_counts = df['label'].value_counts().sort_index()
class_weights = {i: max(class_counts) / count for i, count in enumerate(class_counts)}

X, y, groups = [], [], []
num_missing = 0
for idx, row in df.iterrows():
    # Remove 'img/' prefix and map mode to subdirectory
    filename = row['filename'].lstrip('img/')
    subfolder = row['mode']  # Use mode as subfolder name (L-mode, H-mode, ELM)
    img_path = os.path.join(IMG_DIR, subfolder, filename)
    if os.path.exists(img_path):
        patches = load_and_patch_image(img_path)
        if patches:
            X.extend(patches)
            y.extend([row['label']] * len(patches))
            groups.extend([row['discharge']] * len(patches))
    else:
        num_missing += 1
        if num_missing <= 5:
            print(f"Missing: {img_path}")

print(f"Total missing images: {num_missing} out of {len(df)}")
print(f"Loaded patches: {len(X)}")

if len(X) == 0:
    print("No images loaded. Please ensure images are in 'data/img/<mode>/' (e.g., 'data/img/L-mode/', 'data/img/H-mode/', 'data/img/ELM/'). Contact the hackathon organizers if the dataset is incomplete.")
    raise ValueError("No images loaded. Cannot proceed with training. Please check data directory and file paths.")

X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)
groups = np.array(groups)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
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
    filename = row['filename'].lstrip('img/')
    subfolder = row['mode']
    img_path = os.path.join(IMG_DIR, subfolder, filename)
    if os.path.exists(img_path):
        seq_patches.extend(load_and_patch_image(img_path))
    else:
        print(f"Missing for stability check: {img_path}")

if seq_patches:
    seq_patches = np.array(seq_patches)
    seq_preds = np.argmax(model.predict(seq_patches), axis=1)
    print(f"Stabilita pro discharge {sample_discharge}: Predikce sekvence (po patchech): {seq_preds}")
else:
    print(f"No patches loaded for stability check of discharge {sample_discharge}")

# --- Export modelu ---
model.save('saved_model.h5')
print("Model exportován do saved_model.h5")

# --- Porovnání s/bez alpha kanálu ---
if USE_ALPHA:
    print("\nPorovnání bez alpha:")
    X_train_no_alpha = X_train[..., :3] if USE_ALPHA else X_train
    X_test_no_alpha = X_test[..., :3] if USE_ALPHA else X_test
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
    model_no_alpha.fit(datagen.flow(X_train_no_alpha, y_train, batch_size=BATCH_SIZE), epochs=5, validation_data=(X_test_no_alpha, y_test))
    loss_no, acc_no = model_no_alpha.evaluate(X_test_no_alpha, y_test)
    print(f"Accuracy bez alpha: {acc_no*100:.2f}% (vs. s alpha: {acc*100:.2f}%)")