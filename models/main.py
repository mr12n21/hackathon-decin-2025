



import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.clear_session()

from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
    Input, Conv2D, LayerNormalization, RandomRotation, RandomTranslation,
    RandomFlip, GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    BackupAndRestore, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from const import RESOLUTION, BATCH_SIZE

# Custom focal loss to handle class imbalance
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        fl = alpha * weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss

# MixUp augmentation
def mixup(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    return mixed_x, mixed_y

if __name__ == "__main__":

    RESOLUTION = 224  # Force 224x224 for EfficientNetV2
    data_dir = os.path.join(os.path.dirname(__file__), "data", "img")
    
    # Strong data augmentation
    augmentation_layers = tf.keras.Sequential([
        RandomRotation(factor=0.2),
        RandomTranslation(height_factor=0.1, width_factor=0.1),
        RandomFlip(mode="horizontal_and_vertical"),
        GaussianNoise(0.1),
    ])

    # Custom generator with MixUp
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=50.0,
        validation_split=0.2,
        preprocessing_function=lambda x: augmentation_layers(tf.expand_dims(x, 0), training=True)[0]
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

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

    print("Loading EfficientNetV2S with input shape:", (RESOLUTION, RESOLUTION, 3))
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(RESOLUTION, RESOLUTION, 3),
        include_preprocessing=True
    )
    
    # Complex head with residual connections
    inputs = Input(shape=(RESOLUTION, RESOLUTION, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    
    # First branch - direct
    direct = x
    direct = Dense(512, activation='swish')(direct)
    direct = Dropout(0.5)(direct)
    direct = BatchNormalization()(direct)
    
    # Second branch - deeper
    deep = x
    deep = Dense(1024, activation='swish')(deep)
    deep = Dropout(0.5)(deep)
    deep = BatchNormalization()(deep)
    deep = Dense(512, activation='swish')(deep)
    deep = Dropout(0.4)(deep)
    deep = BatchNormalization()(deep)
    
    # Combine branches
    combined = tf.keras.layers.Add()([direct, deep])
    combined = Dense(256, activation='swish')(combined)
    combined = Dropout(0.3)(combined)
    combined = LayerNormalization()(combined)
    
    # Output with label smoothing
    outputs = Dense(3, activation='softmax')(combined)
    model = Model(inputs, outputs)

    # Loss and optimizer
    loss = focal_loss(gamma=2.0, alpha=0.25)
    optimizer = AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'AUC']
    )

    # Class weights based on confusion matrix analysis
    class_weight = {
        0: 2.0,  # L-mode: increase weight due to confusion with ELM
        1: 1.5,  # H-mode: slight increase
        2: 0.7   # ELM: decrease weight to prevent over-prediction
    }

    # Callbacks with backup
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        BackupAndRestore(backup_dir='./backup'),
        CSVLogger('training_log.csv')
    ]

    # Compute class weights to address errors observed in confusion matrix
    # If you have a confusion matrix from previous runs, use it to bias training towards classes with low recall.
    # Provided confusion matrix (rows=true, cols=pred):
    # [[35  8 36]
    #  [ 4 31 45]
    #  [ 1 22 57]]
    conf_matrix = np.array([[35, 8, 36], [4, 31, 45], [1, 22, 57]])
    true_counts = conf_matrix.sum(axis=1)
    true_positive = np.diag(conf_matrix)
    recalls = true_positive / (true_counts + 1e-9)
    # Target recall: use max recall to boost weaker classes
    target_recall = np.max(recalls)
    raw_weights = target_recall / (recalls + 1e-9)
    # Normalize so mean weight == 1.0 to keep loss scale similar
    class_weight_dict = {i: float(w / np.mean(raw_weights)) for i, w in enumerate(raw_weights)}
    print("Derived class weights from confusion matrix:", class_weight_dict)

    # Implement focal loss to focus on hard examples
    import tensorflow as tf
    def focal_loss(gamma=2., alpha=None):
        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weights = tf.pow(1 - y_pred, gamma)
            if alpha is not None:
                alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
                loss = alpha_factor * weights * cross_entropy
            else:
                loss = weights * cross_entropy
            return tf.reduce_sum(loss, axis=1)
        return loss_fn
    loss_to_use = focal_loss(gamma=2.0, alpha=None)

    print("Stage 1: Training top layers...")
    base_model.trainable = False
    history1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight
    )

    print("Stage 2: Fine-tuning...")
    # Unfreeze and train with smaller learning rate
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=AdamW(learning_rate=5e-5, weight_decay=1e-6),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', 'AUC']
    )

    history2 = model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Combine histories for plotting
    history = {}
    for k in set(list(history1.history.keys()) + list(history2.history.keys())):
        history[k] = history1.history.get(k, []) + history2.history.get(k, [])

    # Save the trained model
    model.save("complex_model.keras")
    print("Model saved as 'complex_model.keras'")

    # Plot training & validation metrics
    metrics = ['accuracy', 'loss', 'auc']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        axes[i].plot(history.get(metric, []), label=f'Train {metric}')
        axes[i].plot(history.get(f'val_{metric}', []), label=f'Val {metric}')
        axes[i].set_title(f'{metric.capitalize()} over Epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Final evaluation
    val_metrics = model.evaluate(val_gen)
    print("\nFinal validation metrics:")
    for name, value in zip(model.metrics_names, val_metrics):
        print(f"{name}: {value:.4f}")

    # Confusion matrix and classification report
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Predict on validation set
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    labels = list(val_gen.class_indices.keys())
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Enhanced confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Per-class metrics
    print("\nPer-class Metrics:")
    for i, label in enumerate(labels):
        class_preds = (y_pred == i)
        class_true = (y_true == i)
        precision = np.sum(class_preds & class_true) / (np.sum(class_preds) + 1e-10)
        recall = np.sum(class_preds & class_true) / (np.sum(class_true) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        print(f"\n{label}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")