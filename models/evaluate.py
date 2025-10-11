import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from const import RESOLUTION, BATCH_SIZE


def get_val_generator(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb'
    )
    return val_gen


def tta_predict_image(model, img_array, tta_transforms=None):
    # img_array expected shape (H, W, C) normalized [0,1]
    if tta_transforms is None:
        tta_transforms = [lambda x: x,
                          lambda x: np.fliplr(x),
                          lambda x: np.flipud(x),
                          lambda x: np.rot90(x, k=1),
                          lambda x: np.rot90(x, k=3)]
    preds = []
    for fn in tta_transforms:
        aug = fn(img_array)
        x = np.expand_dims(aug, axis=0)
        p = model.predict(x, verbose=0)
        preds.append(p[0])
    avg = np.mean(preds, axis=0)
    return avg


def evaluate_model(model_path, data_dir, output_prefix='eval'):
    print('Loading model:', model_path)
    model = load_model(model_path)

    val_gen = get_val_generator(data_dir)
    filenames = val_gen.filenames
    y_true = val_gen.classes
    labels = list(val_gen.class_indices.keys())

    # Single-pass predictions
    val_gen.reset()
    preds_single = model.predict(val_gen, verbose=1)
    y_pred_single = np.argmax(preds_single, axis=1)

    # TTA predictions (loop per image)
    tta_preds = []
    for i, fname in enumerate(filenames):
        path = os.path.join(data_dir, fname)
        img = load_img(path, target_size=(RESOLUTION, RESOLUTION))
        arr = img_to_array(img) / 255.0
        avg = tta_predict_image(model, arr)
        tta_preds.append(avg)
    tta_preds = np.array(tta_preds)
    y_pred_tta = np.argmax(tta_preds, axis=1)

    # Reports
    print('\nSingle-pass Classification report:')
    print(classification_report(y_true, y_pred_single, target_names=labels))
    cm_single = confusion_matrix(y_true, y_pred_single)
    print('Confusion matrix (single):\n', cm_single)

    print('\nTTA Classification report:')
    print(classification_report(y_true, y_pred_tta, target_names=labels))
    cm_tta = confusion_matrix(y_true, y_pred_tta)
    print('Confusion matrix (TTA):\n', cm_tta)

    # Save CSV with predictions and probabilities
    rows = []
    for i, fname in enumerate(filenames):
        row = {
            'filename': fname,
            'true_label': labels[y_true[i]],
            'pred_single': labels[y_pred_single[i]],
            'pred_tta': labels[y_pred_tta[i]],
            'probs_single': json.dumps(preds_single[i].tolist()),
            'probs_tta': json.dumps(tta_preds[i].tolist())
        }
        rows.append(row)
    import csv
    csv_path = f'{output_prefix}_predictions.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved predictions to', csv_path)

    # Plot confusion matrices
    def plot_cm(cm, title, fname):
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap='Blues')
        plt.title(title)
        plt.colorbar()
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45)
        plt.yticks(ticks, labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(fname)

    plot_cm(cm_single, 'Confusion Matrix - Single', f'{output_prefix}_cm_single.png')
    plot_cm(cm_tta, 'Confusion Matrix - TTA', f'{output_prefix}_cm_tta.png')
    print('Saved confusion matrices')


if __name__ == '__main__':
    # Configure paths
    MODEL_PATH = 'best_model.keras' if os.path.exists('best_model.keras') else 'complex_model.keras'
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'img')
    evaluate_model(MODEL_PATH, DATA_DIR, output_prefix='eval')
