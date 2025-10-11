import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray
from models import config


def list_class_folders(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return classes


def load_patches_from_image(path, img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE, use_alpha=config.USE_ALPHA, use_grayscale=config.USE_GRAYSCALE):
    """Load an image, resize and split into patches. Return list of patches (numpy arrays).
    Patches will have channel count = 4 if use_alpha True (RGB + alpha) otherwise 3.
    If grayscale augmentation is enabled, both grayscale and color patches are returned for each patch.
    """
    desired_channels = 4 if use_alpha else 3
    ext = os.path.splitext(path)[1].lower()
    color_mode = 'rgba' if ext == '.png' and use_alpha else 'rgb'
    try:
        img = load_img(path, target_size=(img_size, img_size), color_mode=color_mode)
    except Exception:
        # fallback to rgb
        img = load_img(path, target_size=(img_size, img_size), color_mode='rgb')
    img_array = img_to_array(img) / 255.0

    # If we loaded RGB but want RGBA, append alpha channel of ones
    if img_array.shape[-1] == 3 and desired_channels == 4:
        alpha = np.ones((img_size, img_size, 1), dtype=img_array.dtype)
        img_array = np.concatenate([img_array, alpha], axis=-1)
    # If we loaded RGBA but want RGB, drop alpha
    if img_array.shape[-1] == 4 and desired_channels == 3:
        img_array = img_array[..., :3]

    patches = []
    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            patch = img_array[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size, desired_channels):
                # add grayscale variant if requested
                if use_grayscale:
                    gray = rgb2gray(patch[..., :3])
                    gray = np.expand_dims(gray, axis=-1)
                    if desired_channels == 4:
                        gray_rgb = np.repeat(gray, 3, axis=-1)
                        alpha_channel = patch[..., 3:4] if patch.shape[-1] == 4 else np.ones_like(gray)
                        gray_patch = np.concatenate([gray_rgb, alpha_channel], axis=-1)
                    else:
                        gray_patch = np.repeat(gray, 3, axis=-1)
                    patches.append(gray_patch)
                patches.append(patch)
    return patches


def load_full_image(path, img_size=config.IMG_SIZE, use_alpha=config.USE_ALPHA, grayscale_only=config.GRAYSCALE_ONLY):
    ext = os.path.splitext(path)[1].lower()
    color_mode = 'rgba' if ext == '.png' and use_alpha else 'rgb'
    try:
        img = load_img(path, target_size=(img_size, img_size), color_mode=color_mode)
    except Exception:
        img = load_img(path, target_size=(img_size, img_size), color_mode='rgb')
    img_array = img_to_array(img) / 255.0

    # If using a pretrained backbone that expects 3 channels (EfficientNet/ResNet),
    # force desired_channels=3. Otherwise keep alpha if requested.
    if config.BASE_MODEL in ('EfficientNetB0', 'ResNet50'):
        desired_channels = 3
    else:
        desired_channels = 4 if use_alpha else 3
    if img_array.shape[-1] == 3 and desired_channels == 4:
        alpha = np.ones((img_size, img_size, 1), dtype=img_array.dtype)
        img_array = np.concatenate([img_array, alpha], axis=-1)
    if img_array.shape[-1] == 4 and desired_channels == 3:
        # drop alpha for pretrained backbone
        img_array = img_array[..., :3]

    if grayscale_only:
        gray = rgb2gray(img_array[..., :3])
        gray = np.expand_dims(gray, axis=-1)
        gray = np.repeat(gray, 3, axis=-1)
        if desired_channels == 4:
            alpha_channel = img_array[..., 3:4] if img_array.shape[-1] == 4 else np.ones_like(gray[..., :1])
            img_array = np.concatenate([gray, alpha_channel], axis=-1)
        else:
            img_array = gray

    return img_array


def load_dataset(data_dir=config.DATA_DIR, classes=None, max_images_per_class=None):
    X = []
    y = []
    img_paths = []
    if classes is None:
        classes = list_class_folders(data_dir)
    classes = sorted(classes)
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_images_per_class:
            files = files[:max_images_per_class]
        for f in files:
            path = os.path.join(folder, f)
            img_paths.append((path, idx))

    # If using full-image mode, load each image once (no patch-level leakage)
    if config.USE_FULL_IMAGE:
        for path, idx in img_paths:
            img = load_full_image(path)
            X.append(img)
            y.append(idx)
    else:
        for path, idx in img_paths:
            patches = load_patches_from_image(path)
            for p in patches:
                X.append(p)
                y.append(idx)

    X = np.array(X)
    y = np.array(y)
    return X, y, classes
