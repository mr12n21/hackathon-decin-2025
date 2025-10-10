import os
import random
import shutil

# === CONFIGURATION ===
folders = [r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted\L-mode", r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted\ELM", r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted\H-mode"]  # Source folders
output_root = "sampled_sets"      # Root folder to store all sets
num_sets = 5                       # How many sets to generate
min_samples = 10          # Min per folder per set
max_samples = 33                 # Max per folder per set
copy_files = True                  # True to copy, False to move

# ======================

os.makedirs(output_root, exist_ok=True)

for set_index in range(1, num_sets + 1):
    set_folder = os.path.join(output_root, f"set_{set_index}")
    os.makedirs(set_folder, exist_ok=True)

    print(f"\n🎯 Creating set {set_index}")

    for folder in folders:
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue

        all_images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        sample_count = min(len(all_images), random.randint(min_samples, max_samples))
        sampled_images = random.sample(all_images, sample_count)

        for img in sampled_images:
            src = os.path.join(folder, img)
            dst = os.path.join(set_folder, img)

            # Handle potential filename collisions
            if os.path.exists(dst):
                base, ext = os.path.splitext(img)
                i = 1
                while os.path.exists(dst):
                    dst = os.path.join(set_folder, f"{base}_{i}{ext}")
                    i += 1

            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

        print(f"✅ Sampled {sample_count} images from {folder} into {set_folder}")

print(f"\n🎉 All {num_sets} sets created in {output_root}")