from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from models import config
from tensorflow.keras import applications
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def focal_loss(gamma=2., alpha=.25):
    def focal(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal


def build_model(input_shape, num_classes=config.NUM_CLASSES, weight_decay=config.WEIGHT_DECAY, dropout_rate=config.DROPOUT_RATE, lr=config.LEARNING_RATE):
    if config.BASE_MODEL == 'EfficientNetB0':
        # Use EfficientNetB0 as backbone. Pretrained ImageNet weights expect 3-channel input.
        # Force input tensor to 3 channels when using pretrained weights.
        img_h, img_w = input_shape[0], input_shape[1]
        inp = Input(shape=(img_h, img_w, 3))
        try:
            base = applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inp)
        except Exception as e:
            # Fallback: try without weights if mismatch
            base = applications.EfficientNetB0(include_top=False, weights=None, input_tensor=inp)
            print('Warning: EfficientNetB0 pretrained weights could not be loaded, using random init. Error:', e)
        base.trainable = False
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(dropout_rate/2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=outputs)
    else:
        model = Sequential([
            Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(weight_decay), input_shape=input_shape),
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
            Dropout(dropout_rate),
            Dense(256, activation='relu', kernel_regularizer=l2(weight_decay)),
            Dropout(dropout_rate/2),
            Dense(num_classes, activation='softmax')
        ])

    optimizer = Adam(learning_rate=lr)
    loss_fn = 'categorical_crossentropy'
    if config.USE_FOCAL_LOSS:
        loss_fn = focal_loss()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model
