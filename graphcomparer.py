import pandas as pd
import cv2
import os

# --- CONFIG ---
csv_file = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\labels.csv"  # path to your CSV
img_folder = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\img\img"  # folder containing images
output_video = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\output.mp4"
fps = 30  # frames per second

# --- LOAD CSV ---
df = pd.read_csv(csv_file)

# Filter rows where discharge == 0
df = df[df['discharge'] == 0]

# Sort by time (optional, ensures correct sequence)
df = df.sort_values('time')

# --- READ FIRST IMAGE TO GET DIMENSIONS ---
first_image_path = os.path.join(img_folder, os.path.basename(df.iloc[0]['filename']))
frame = cv2.imread(first_image_path)
if frame is None:
    raise FileNotFoundError(f"Cannot read image: {first_image_path}")
height, width, layers = frame.shape

# --- SETUP VIDEO WRITER ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# --- ADD FRAMES ---
for idx, row in df.iterrows():
    img_path = os.path.join(img_folder, os.path.basename(row['filename']))
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)
    else:
        print(f"Warning: Could not read {img_path}")

# --- FINISH ---
video.release()
print(f"Video saved as {output_video}")
