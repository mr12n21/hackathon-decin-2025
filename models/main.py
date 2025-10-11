



import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from const import RESOLUTION, BATCH_SIZE

if __name__ == "__main__":

    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data", "img")

    # Strong augmentations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7,1.3],
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        classes=["L-mode", "H-mode", "ELM"],
        subset="training",
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        classes=["L-mode", "H-mode", "ELM"],
        subset="validation",
        shuffle=False
    )

    # Transfer learning model
    # Build base model with MobileNetV2 (ImageNet weights)
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(RESOLUTION, RESOLUTION, 3))
    base_model.trainable = False  # Freeze base
    inputs = Input(shape=(RESOLUTION, RESOLUTION, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation="softmax")(x)
    model = KerasModel(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Compute class weights to address imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = train_gen.classes
    class_labels = np.unique(classes)
    class_weights = compute_class_weight('balanced', classes=class_labels, y=classes)
    class_weight_dict = {int(label): float(weight) for label, weight in zip(class_labels, class_weights)}
    print("Class weights:", class_weight_dict)

    # Stage 1: train top classifier only
    for layer in base_model.layers:
        layer.trainable = False
    history1 = model.fit(
        train_gen,
        epochs=6,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Stage 2: fine-tune some top layers of the base model
    for layer in base_model.layers[-40:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    history2 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Merge history objects for plotting
    history = {}
    for k in set(list(history1.history.keys()) + list(history2.history.keys())):
        history[k] = history1.history.get(k, []) + history2.history.get(k, [])

    # Save the trained model
    model.save("complex_model.keras")
    print("Model saved as 'complex_model.keras'")

    # Plot training & validation accuracy and loss (history is a dict of lists)
    train_acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    train_loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if train_acc:
        plt.plot(train_acc, label='Train Accuracy')
    if val_acc:
        plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    if train_loss:
        plt.plot(train_loss, label='Train Loss')
    if val_loss:
        plt.plot(val_loss, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Evaluate on test/validation set and print results
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"Validation/Test Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")

    # Produce confusion matrix and classification report
    from sklearn.metrics import classification_report, confusion_matrix
    # Predict on validation set
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    labels = list(val_gen.class_indices.keys())
    print("Classification report:\n", classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    # Save confusion matrix plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')