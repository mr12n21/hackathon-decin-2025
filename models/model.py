import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from skimage.color import rgb2gray
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --- Nastavení ---
IMG_SIZE = 64
PATCH_SIZE = 32
# We have 3 classes (L-mode, H-mode, ELM)
NUM_CLASSES = 3
DATA_DIR = os.path.join('data', 'img')
BATCH_SIZE = 32
EPOCHS = 100  # Reduced from 200 to a more typical value; EarlyStopping will stop earlier if needed
USE_ALPHA = True  # Keep support for alpha channel if present
USE_GRAYSCALE = True  # Add grayscale augmented patches as extra samples

# Quick debug mode: set env var QUICK_DEBUG=1 to run a short smoke-test (small subset, 1 epoch)
QUICK_DEBUG = os.environ.get('QUICK_DEBUG') == '1'

# --- Funkce pro načtení obrázků s podporou alpha a grayscale ---
def load_images_from_folder(folder_path, label):
    images, labels = [], []
    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg')):
            # Načtení obrázku s alpha kanálem, pokud je PNG
            # Decide desired channels globally (4 if using alpha, else 3)
            desired_channels = 4 if USE_ALPHA else 3

            # Load as RGB or RGBA depending on file type; some formats (jpg) load as RGB
            color_mode = 'rgba' if file.endswith('.png') else 'rgb'
            img = load_img(os.path.join(folder_path, file), target_size=(IMG_SIZE, IMG_SIZE), color_mode=color_mode)
            img_array = img_to_array(img) / 255.0

            # If loaded as RGB but we want RGBA, add an alpha channel of ones
            if img_array.shape[-1] == 3 and desired_channels == 4:
                alpha = np.ones((IMG_SIZE, IMG_SIZE, 1), dtype=img_array.dtype)
                img_array = np.concatenate([img_array, alpha], axis=-1)

            # If loaded as RGBA but we want only RGB, drop alpha
            if img_array.shape[-1] == 4 and desired_channels == 3:
                img_array = img_array[..., :3]

            # Rozdělení na patche
            for i in range(0, IMG_SIZE, PATCH_SIZE):
                for j in range(0, IMG_SIZE, PATCH_SIZE):
                    patch = img_array[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE, desired_channels):
                        # Přidání grayscale verze pro augmentaci, pokud povoleno
                        if USE_GRAYSCALE:
                            gray_patch = rgb2gray(patch[..., :3])  # Grayscale z RGB části
                            gray_patch = np.expand_dims(gray_patch, axis=-1)  # 1 kanál
                            if desired_channels == 4:
                                # If model expects 4 channels, replicate grayscale to 3 channels and append alpha
                                gray_rgb = np.repeat(gray_patch, 3, axis=-1)
                                alpha_channel = patch[..., 3:4] if patch.shape[-1] == 4 else np.ones_like(gray_patch)
                                gray_patch_final = np.concatenate([gray_rgb, alpha_channel], axis=-1)
                            else:
                                # For 3-channel models, repeat grayscale to 3 channels
                                gray_patch_final = np.repeat(gray_patch, 3, axis=-1)
                            images.append(gray_patch_final)
                            labels.append(label)

                        # Přidání originálního patche (už má správný počet kanálů)
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

channels = 4 if USE_ALPHA else 3  # Globální počet kanálů pro model

for folder_name, label in folders.items():
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        print(f"Warning: class folder not found, skipping: {folder_path}")
        continue
    imgs, labels = load_images_from_folder(folder_path, label)
    X.extend(imgs)
    y.extend(labels)

print(f"Total patches loaded: {len(X)} from folders: {', '.join([f for f in folders.keys() if os.path.isdir(os.path.join(DATA_DIR, f))])}")

X = np.array(X)
y = to_categorical(np.array(y), num_classes=NUM_CLASSES)  # one-hot encoding

# --- Rozdělení datasetu ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- Data augmentace (včetně channel swapping pro testování robustness) ---
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    channel_shift_range=0.2  # Lehké prohazování intenzit kanálů pro robustnost
)

# --- Komplexnější CNN model ---
weight_decay = 1e-4

model = Sequential([
    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(weight_decay), input_shape=(PATCH_SIZE, PATCH_SIZE, channels)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(weight_decay)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(weight_decay)),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# --- Callbacky pro lepší trénování ---
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# --- Trénování s augmentací ---
# Compute class weights to balance the loss for imbalanced dataset
y_integers = np.argmax(y, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(class_weights))

# QUICK_DEBUG: use a small subset and fewer epochs for fast smoke testing
if QUICK_DEBUG:
    EPOCHS = 1
    # reduce dataset for quick run
    X_train = X_train[:1024]
    y_train = y_train[:1024]

model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weights
)

# --- Vyhodnocení na standardním test setu ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Standard Test Accuracy: {acc*100:.2f}%")

# --- Testování robustness: Prohazování kanálů (např. RGB -> BGR) ---
def swap_channels(data):
    if channels == 3:
        return data[..., [2,1,0]]  # RGB -> BGR
    elif channels == 4:
        return np.concatenate([data[..., [2,1,0]], data[..., 3:4]], axis=-1)  # RGBA -> BGRA

X_test_swapped = swap_channels(X_test)
loss_swapped, acc_swapped = model.evaluate(X_test_swapped, y_test)
print(f"Swapped Channels Test Accuracy: {acc_swapped*100:.2f}%")

# --- Testování na grayscale verzi test setu ---
def convert_to_grayscale(data):
    gray_data = np.array([rgb2gray(img[..., :3]) for img in data])
    gray_data = np.expand_dims(gray_data, axis=-1)
    if channels == 3:
        gray_data = np.repeat(gray_data, 3, axis=-1)  # Opakovat pro 3 kanály
    elif channels == 4:
        alpha = data[..., 3:4] if data.shape[-1] == 4 else np.ones_like(gray_data)
        gray_data = np.repeat(gray_data, 3, axis=-1)
        gray_data = np.concatenate([gray_data, alpha], axis=-1)
    return gray_data

X_test_gray = convert_to_grayscale(X_test)
loss_gray, acc_gray = model.evaluate(X_test_gray, y_test)
print(f"Grayscale Test Accuracy: {acc_gray*100:.2f}%")

# --- Testování s modifikovaným alpha kanálem (pokud USE_ALPHA=True) ---
if USE_ALPHA and channels == 4:
    def modify_alpha(data, alpha_value=0.5):
        modified = data.copy()
        modified[..., 3] = alpha_value  # Nastavit konstantní alpha
        return modified
    
    X_test_alpha_mod = modify_alpha(X_test, alpha_value=0.5)  # Poloprůhledné
    loss_alpha, acc_alpha = model.evaluate(X_test_alpha_mod, y_test)
    print(f"Modified Alpha (0.5) Test Accuracy: {acc_alpha*100:.2f}%")

# --- Uložení modelu ---
model.save("bramboracek.h5")
print("Model uložen jako 'bramboracek.h5'")