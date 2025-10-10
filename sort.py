import csv
import os
import shutil

# === CONFIGURATION ===
csv_path = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\labels.csv"        # Path to your CSV file
images_root = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\img"       # Folder where the images currently are
output_root = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\sorted"  # Folder where the 3 folders will be created
move_files = False             # Set to True if you want to MOVE instead of COPY

# ======================

# Create output folders
modes = ["L-mode", "H-mode", "ELM"]
for mode in modes:
    os.makedirs(os.path.join(output_root, mode), exist_ok=True)

# Read the CSV
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 4:
            continue  # skip malformed lines

        img_path, *_, mode = row
        mode = mode.strip()

        if mode not in modes:
            print(f"Skipping unknown mode: {mode} ({img_path})")
            continue

        src = os.path.join(images_root, img_path)
        dst = os.path.join(output_root, mode, os.path.basename(img_path))

        if not os.path.exists(src):
            print(f"File not found: {src}")
            continue

        # Copy or move
        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

        print(f"{'Moved' if move_files else 'Copied'}: {src} → {dst}")

print("Sorting complete!")