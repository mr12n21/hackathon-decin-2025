import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

IMG_SIZE = 64
PATCH_SIZE = 32
CLASS_NAMES = ["L-mode", "H-mode", "ELM"]
MODEL_PATH = "cnn_mid.keras"

try:
    model = load_model(MODEL_PATH)
    print(f"Model úspěšně načten z: {MODEL_PATH}")
except Exception as e:
    print(f"Chyba při načítání modelu: {e}")
    exit()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgba')
    img_array = img_to_array(img) / 255.0
    patches = []
    
    for i in range(0, IMG_SIZE, PATCH_SIZE):
        for j in range(0, IMG_SIZE, PATCH_SIZE):
            patch = img_array[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            if patch.shape == (PATCH_SIZE, PATCH_SIZE, 4):
                patches.append(patch)
    
    return np.array(patches)

def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Obrázek nebyl nalezen: {image_path}")
        return None, None, None

    patches = preprocess_image(image_path)
    if patches.size == 0:
        print(f"Obrázek {image_path} neobsahuje žádné validní patchy.")
        return None, None, None

    predictions = model.predict(patches)
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = float(avg_prediction[predicted_class])

    return predicted_label, confidence, avg_prediction

def get_true_label(image_path):
    # Předpoklad: Skutečná třída je v názvu podadresáře (např. data/test/L-mode/obrazek.png)
    folder = os.path.basename(os.path.dirname(image_path))
    if folder in CLASS_NAMES:
        return folder
    return None

def predict_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Složka nebyla nalezena: {folder_path}")
        return None, None
    
    true_labels = []
    predicted_labels = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                image_path = os.path.join(root, file)
                true_label = get_true_label(image_path)
                
                if true_label is None:
                    print(f"Skutečná třída nenalezena pro: {image_path}")
                    continue
                
                predicted_label, confidence, probabilities = predict_image(model, image_path)
                if predicted_label is None:
                    continue
                
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                
                print(f"Obrázek: {file}")
                print(f"Skutečná třída: {true_label}")
                print(f"Predikovaná třída: {predicted_label}")
                print(f"Jistota: {confidence*100:.2f}%")
                print(f"Pravděpodobnosti: {probabilities}")
                print("-" * 50)
    
    return true_labels, predicted_labels

def create_confusion_matrix(true_labels, predicted_labels):
    if not true_labels or not predicted_labels:
        print("Žádná data pro vytvoření matice záměn.")
        return None
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=CLASS_NAMES)
    return cm

# --- Hlavní část ---
if __name__ == "__main__":
    test_folder = "img"  # Nahraďte cestou k testovací složce (např. data/test)
    print("Predikce na složce obrázků:")
    true_labels, predicted_labels = predict_images_in_folder(test_folder)
    
    # Vytvoření matice záměn
    cm = create_confusion_matrix(true_labels, predicted_labels)
    if cm is not None:
        print("Matice záměn:")
        print(cm)