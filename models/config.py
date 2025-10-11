import os

# Centralized configuration for training and preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'img')
IMG_SIZE = 128  # increase size to allow better feature extraction
PATCH_SIZE = 64
USE_ALPHA = True  # Keep alpha channel if present
USE_GRAYSCALE = True  # include grayscale patches to simplify analysis
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 10
QUICK_DEBUG = os.environ.get('QUICK_DEBUG') == '1'
SEED = 42

# Model and training hyperparams
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4

# New settings for the rewrite
USE_FULL_IMAGE = True  # if True, train on whole images resized to IMG_SIZE (no patching)
GRAYSCALE_ONLY = False  # if True, convert images to grayscale (replicated to 3 channels for pretrained nets)
BASE_MODEL = 'EfficientNetB0'  # options: EfficientNetB0, ResNet50
FINE_TUNE = True  # after training top layers, unfreeze last blocks for fine-tuning
FINE_TUNE_AT = 100  # layer at which to start fine-tuning (approx index)
USE_FOCAL_LOSS = True
USE_MIXUP = False

# Paths
SAVE_DIR = os.path.join(BASE_DIR, 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.h5')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
