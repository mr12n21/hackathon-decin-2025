import cv2
import numpy as np
import onnxruntime as ort
import time

# ---- Load ONNX model ----
onnx_path = "runs/classify/train4/weights/best.onnx"
session = ort.InferenceSession(onnx_path)

# Your class names
classes = ["L-mode", "H-mode", "ELM"]

# ---- Open video / camera ----
cap = cv2.VideoCapture(r"C:\Users\v\Desktop\video_interpolated_jitter.mp4")  # or use 0 for webcam

last_state = None
log_file = "tokamakimageanalizer_log.txt"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # ---- Preprocess ----
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0       # normalize
    img = np.transpose(img, (2, 0, 1))         # HWC -> CHW
    img = np.expand_dims(img, axis=0)          # add batch dim

    # ---- Run inference ----
    outputs = session.run(None, {"images": img})
    probs = outputs[0][0]  # shape: [num_classes]

    # ---- Get top class ----
    top_idx = np.argmax(probs)
    top_label = classes[top_idx]
    top_conf = probs[top_idx]

    # ---- Log state changes ----
    if top_label != last_state:
        last_state = top_label
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp}: {top_label} ({top_conf:.2f})\n")
        print(f"{timestamp}: {top_label} ({top_conf:.2f})")

    # ---- Draw background rectangle for text ----
    margin = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_height = 25
    rect_width = 200
    rect_height = text_height * len(classes) + margin * 2
    rect_x = width - rect_width - margin
    rect_y = margin

    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height + 100), (0,0,0), -1)

    # ---- Display class probabilities ----
    bar_height = 20
    bar_spacing = 10
    max_bar_width = rect_width - 20

    for i, cls in enumerate(classes):
        prob = probs[i]
        text = f"{cls}: {prob:.2f}"
        y_text = rect_y + margin + (i+1)*text_height
        cv2.putText(frame, text, (rect_x + 5, y_text), font, font_scale, (0,255,0), thickness)

        # Draw probability bar
        bar_y = rect_y + rect_height + i*(bar_height + bar_spacing)
        bar_length = int(prob * max_bar_width)
        cv2.rectangle(frame, (rect_x + 10, bar_y), (rect_x + 10 + bar_length, bar_y + bar_height), (0,255,0), -1)
        cv2.rectangle(frame, (rect_x + 10, bar_y), (rect_x + 10 + max_bar_width, bar_y + bar_height), (255,255,255), 1)

    # ---- Show video ----
    cv2.imshow("Tokamak Monitor", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()