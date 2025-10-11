import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from models import config
from models import data_utils
from models.model_builder import build_model


def run(max_images_per_class=None):
    print('Loading dataset...')
    X, y, classes = data_utils.load_dataset(config.DATA_DIR, max_images_per_class=max_images_per_class)
    print(f'Loaded patches: {X.shape}, labels: {y.shape}, classes: {classes}')

    # Convert to categorical
    y_cat = to_categorical(y, num_classes=len(classes))

    # Train/test split - stratified per-image to avoid leakage
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    y_integers = y
    train_idx, test_idx = next(sss.split(X, y_integers))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_cat[train_idx], y_cat[test_idx]

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Build model
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, num_classes=len(classes))
    print(model.summary())

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    checkpoint = ModelCheckpoint(config.BEST_MODEL_PATH, monitor='val_loss', save_best_only=True)

    # Class weights
    y_integers = np.argmax(y_cat, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights = dict(enumerate(class_weights))
    print('Class weights:', class_weights)

    # QUICK_DEBUG: shorten training and sample fewer images
    epochs = config.EPOCHS
    if config.QUICK_DEBUG:
        epochs = 1
        X_train = X_train[:2048]
        y_train = y_train[:2048]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights
    )

    # Optionally fine-tune last layers of a pretrained base
    if config.FINE_TUNE and config.BASE_MODEL == 'EfficientNetB0':
        print('Starting fine-tuning: unfreezing top layers')
        # Unfreeze last blocks
        base_model = None
        for layer in model.layers:
            if isinstance(layer, Model):
                base_model = layer
                break
        if base_model is None:
            # try to locate by name
            for l in model.layers:
                if l.__class__.__name__.startswith('Functional') or l.__class__.__name__.startswith('Model'):
                    base_model = l
                    break

        if base_model is not None:
            # Unfreeze from FINE_TUNE_AT
            for i, layer in enumerate(base_model.layers):
                if i < config.FINE_TUNE_AT:
                    layer.trainable = False
                else:
                    layer.trainable = True

            # recompile with lower LR
            from tensorflow.keras.optimizers import Adam
            model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE/10), loss='categorical_crossentropy', metrics=['accuracy'])
            # short fine-tuning run
            model.fit(
                datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
                epochs=10,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), reduce_lr, checkpoint],
                class_weight=class_weights
            )

    loss, acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {acc*100:.2f}%')
    return model, classes
