import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 64
PATCH_SIZE = 32
CLASS_NAMES = ["L-mode", "H-mode", "ELM"]
MODEL_PATH = "complex_model.h5"

try:
    model = load_model(MODEL_PATH)
    print(f"Model úspěšně načten z: {MODEL_PATH}")
except Exception as e:
    print(f"Chyba při načítání modelu: {e}")
    exit()

def preprocess_image(image_path):
    # Load image including alpha channel so patches match models trained with 4 channels.
    # If your model expects 3 channels, change color_mode to 'rgb'.
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgba')
    img_array = img_to_array(img) / 255.0  # Normalizace
    patches = []
    
    for i in range(0, IMG_SIZE, PATCH_SIZE):
        for j in range(0, IMG_SIZE, PATCH_SIZE):
            patch = img_array[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            # Expect 4 channels now (RGBA). If your model uses RGB, update accordingly.
            if patch.shape == (PATCH_SIZE, PATCH_SIZE, 4):
                patches.append(patch)
    
    return np.array(patches)

def predict_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Obrázek nebyl nalezen: {image_path}")
        return None, None, None

    patches = preprocess_image(image_path)
    if patches.size == 0:
        print(f"Obrázek {image_path} neobsahuje žádné validní patchy (neodpovídá velikosti).")
        return None, None, None

    predictions = model.predict(patches)

    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = float(avg_prediction[predicted_class])

    return predicted_label, confidence, avg_prediction

# --- Predikce na jednom obrázku ---
def predict_single_image(image_path):
    predicted_label, confidence, probabilities = predict_image(model, image_path)
    if predicted_label is None:
        # predict_image already printed a helpful message
        return

    print(f"Obrázek: {os.path.basename(image_path)}")
    print(f"Predikovaná třída: {predicted_label}")
    print(f"Jistota: {confidence*100:.2f}%")
    print(f"Pravděpodobnosti: {probabilities}")

# --- Predikce na složce obrázků ---
def predict_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Složka nebyla nalezena: {folder_path}")
        return
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(folder_path, file)
            predicted_label, confidence, probabilities = predict_image(model, image_path)
            if predicted_label is None:
                # error already printed inside predict_image
                continue
            print(f"Obrázek: {file}")
            print(f"Predikovaná třída: {predicted_label}")
            print(f"Jistota: {confidence*100:.2f}%")
            print(f"Pravděpodobnosti: {probabilities}")
            print("-" * 50)

# --- Hlavní část ---
if __name__ == "__main__":
    # Příklad: Predikce na jednom obrázku
    # Resolve example paths relative to this script so the defaults work when
    # running from the repository root or from the `models` directory.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    single_image_path = os.path.join(base_dir, "data", "test.jpg")  # Nahraďte cestou k vašemu obrázku
    print("Predikce na jednom obrázku:")
    predict_single_image(single_image_path)
    print("\n" + "="*50 + "\n")

    # Příklad: Predikce na složce
    test_folder = os.path.join(base_dir, "data", "img")  # Nahraďte cestou k testovací složce (např. data/test)
    print("Predikce na složce obrázků:")
    predict_images_in_folder(test_folder)