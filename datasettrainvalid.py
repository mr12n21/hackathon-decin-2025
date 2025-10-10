import os
import random
import shutil

# === CONFIGURATION ===
folders = [r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted\L-mode", r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted\ELM", r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted\H-mode"]  # Source folders
output_root = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\dataF"  # Root folder for train/valid
train_ratio = 0.8  # 80% train, 20% validation
samples_per_class = 10000  # Number of images to sample from each class
copy_files = True  # True = copy, False = move

# ======================

train_folder = os.path.join(output_root, "train")
valid_folder = os.path.join(output_root, "valid")

# Create folder structure
for mode in ["L-mode", "H-mode", "ELM"]:
    os.makedirs(os.path.join(train_folder, mode), exist_ok=True)
    os.makedirs(os.path.join(valid_folder, mode), exist_ok=True)

for folder in folders:
    mode = os.path.basename(folder)

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    all_images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if len(all_images) < samples_per_class:
        print(f"⚠️ Not enough images in {folder}. Found {len(all_images)}, required {samples_per_class}")
        continue

    sampled_images = random.sample(all_images, samples_per_class)

    train_count = int(train_ratio * samples_per_class)
    train_images = sampled_images[:train_count]
    valid_images = sampled_images[train_count:]

    # Copy/move train images
    for img in train_images:
        src = os.path.join(folder, img)
        dst = os.path.join(train_folder, mode, img)
        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    # Copy/move validation images
    for img in valid_images:
        src = os.path.join(folder, img)
        dst = os.path.join(valid_folder, mode, img)
        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

    print(f"✅ {mode}: {len(train_images)} train, {len(valid_images)} valid")

print(f"\n🎉 Dataset split complete! Check the '{output_root}' folder")