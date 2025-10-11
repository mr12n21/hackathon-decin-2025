import sys
import os
import cv2
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QMessageBox, QLineEdit, QSpinBox, QFormLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt

# Matplotlib for live graph
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------------------------------------
#                  CONFIG / MODEL
# ----------------------------------------------------------
CONFIG_FILE = "ui_presets.json"
onnx_path = "runs/classify/train4/weights/best.onnx"
session = ort.InferenceSession(onnx_path)

classes = ["ELM", "H-mode", "L-mode"]


# ----------------------------------------------------------
#             CSV → VIDEO GENERATION FUNCTION
# ----------------------------------------------------------
def generate_video_from_csv(csv_file, img_folder, output_video, discharge_value=0, fps=5):
    df = pd.read_csv(csv_file)
    df = df[df['discharge'] == discharge_value]
    if df.empty:
        raise ValueError(f"No rows found with discharge == {discharge_value}")

    df = df.sort_values('time')

    first_image_path = os.path.join(img_folder, os.path.basename(df.iloc[0]['filename']))
    frame = cv2.imread(first_image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {first_image_path}")
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    total = len(df)
    for idx, row in enumerate(df.itertuples(), start=1):
        img_path = os.path.join(img_folder, os.path.basename(row.filename))
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"⚠️ Warning: Could not read {img_path}")
        if idx % 100 == 0:
            print(f"Processed {idx}/{total} frames...")

    video.release()
    print(f"✅ Video saved as {output_video}")


# ----------------------------------------------------------
#                 LIVE GRAPH CLASS
# ----------------------------------------------------------
class LiveGraph(FigureCanvas):
    def __init__(self, max_points=150):
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Real-Time Class Probabilities")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Probability")
        self.ax.set_ylim(0, 2)
        self.ax.set_xlim(0, max_points)
        self.max_points = max_points

        self.data = {cls: deque(maxlen=max_points) for cls in classes}
        self.lines = {cls: self.ax.plot([], [], label=cls)[0] for cls in classes}
        self.ax.legend()

    def update_plot(self, probs):
        for cls, prob in zip(classes, probs):
            self.data[cls].append(float(prob))
        for cls, line in self.lines.items():
            y = list(self.data[cls])
            x = list(range(len(y)))
            line.set_data(x, y)
        self.ax.figure.canvas.draw_idle()

    def clear_plot(self):
        for cls in self.data:
            self.data[cls].clear()
        for line in self.lines.values():
            line.set_data([], [])
        self.ax.figure.canvas.draw_idle()


# ----------------------------------------------------------
#                 MAIN UI CLASS
# ----------------------------------------------------------
class TokamakUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tokamak Analyzer (with Presets)")
        self.resize(1300, 800)
        self.setAcceptDrops(True)

        # --- Left side (video feed) ---
        self.video_label = QLabel("Drop a Video or Image Here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px dashed #888; color: #888;")

        # --- Right side (analysis + graph) ---
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFixedSize(400, 200)
        self.graph = LiveGraph(max_points=150)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.analysis_text)
        right_layout.addWidget(self.graph)

        # --- Preset inputs ---
        self.csv_path = QLineEdit()
        self.img_folder = QLineEdit()
        self.output_path = QLineEdit()
        self.discharge_spin = QSpinBox()
        self.discharge_spin.setRange(0, 100000)
        self.discharge_spin.setValue(0)

        self.csv_browse = QPushButton("📄")
        self.csv_browse.clicked.connect(self.browse_csv)
        self.img_browse = QPushButton("🖼️")
        self.img_browse.clicked.connect(self.browse_img_folder)
        self.output_browse = QPushButton("💾")
        self.output_browse.clicked.connect(self.browse_output)

        form_layout = QFormLayout()
        h1 = QHBoxLayout()
        h1.addWidget(self.csv_path)
        h1.addWidget(self.csv_browse)
        form_layout.addRow("CSV File:", h1)
        h2 = QHBoxLayout()
        h2.addWidget(self.img_folder)
        h2.addWidget(self.img_browse)
        form_layout.addRow("Image Folder:", h2)
        h3 = QHBoxLayout()
        h3.addWidget(self.output_path)
        h3.addWidget(self.output_browse)
        form_layout.addRow("Output Video:", h3)
        form_layout.addRow("Discharge:", self.discharge_spin)

        # --- Buttons ---
        self.load_btn = QPushButton("Load Video/Image")
        self.load_btn.clicked.connect(self.load_file)
        self.csv_btn = QPushButton("Generate Video from CSV")
        self.csv_btn.clicked.connect(self.generate_video_ui)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.load_btn)
        buttons_layout.addWidget(self.csv_btn)

        # --- Combine layouts ---
        left_side = QVBoxLayout()
        left_side.addWidget(self.video_label)
        left_side.addLayout(form_layout)
        left_side.addLayout(buttons_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_side)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # --- Timer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.cap = None

        # --- Load saved presets if available ---
        self.load_presets()

    # ----------------------------------------------------------
    #              Drag & Drop Handlers
    # ----------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.video_label.setStyleSheet("border: 2px dashed #4CAF50; color: #4CAF50;")

    def dragLeaveEvent(self, event):
        self.video_label.setStyleSheet("border: 2px dashed #888; color: #888;")

    def dropEvent(self, event):
        self.video_label.setStyleSheet("border: 2px dashed #888; color: #888;")
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.handle_new_file(file_path)
            break

    # ----------------------------------------------------------
    #              File Loading
    # ----------------------------------------------------------
    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video or Image", "", "Media (*.mp4 *.avi *.mov *.png *.jpg *.jpeg)")
        if file_name:
            self.handle_new_file(file_name)

    def handle_new_file(self, file_path):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        self.graph.clear_plot()
        self.analysis_text.clear()

        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            self.show_image(file_path)
        elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)
        else:
            self.analysis_text.setText("Unsupported file type.")

    def show_image(self, path):
        frame = cv2.imread(path)
        if frame is not None:
            self.process_and_display(frame)
        else:
            QMessageBox.warning(self, "Error", "Failed to load image.")

    def next_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display(frame)
            else:
                self.cap.release()
                self.timer.stop()

    # ----------------------------------------------------------
    #              Inference & Display
    # ----------------------------------------------------------
    def process_and_display(self, frame):
        img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        outputs = session.run(None, {"images": img})
        probs = outputs[0][0]
        top_idx = int(np.argmax(probs))
        top_label = classes[top_idx]

        analysis_str = f"Top Prediction: {top_label}\n\n"
        for cls, prob in zip(classes, probs):
            analysis_str += f"{cls}: {prob:.4f}\n"
        self.analysis_text.setText(analysis_str)

        self.graph.update_plot(probs)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    # ----------------------------------------------------------
    #              CSV → Video UI Integration
    # ----------------------------------------------------------
    def generate_video_ui(self):
        csv_file = self.csv_path.text().strip()
        img_folder = self.img_folder.text().strip()
        output_video = self.output_path.text().strip()
        discharge_value = self.discharge_spin.value()

        if not all([csv_file, img_folder, output_video]):
            QMessageBox.warning(self, "Missing Info", "Please fill in all file paths first.")
            return

        self.save_presets(csv_file, img_folder, output_video, discharge_value)

        try:
            generate_video_from_csv(csv_file, img_folder, output_video, discharge_value, fps=5)
            QMessageBox.information(self, "Done", f"✅ Video generated:\n{output_video}")
            self.handle_new_file(output_video)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate video:\n{str(e)}")

    # ----------------------------------------------------------
    #              Browse Buttons
    # ----------------------------------------------------------
    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if path:
            self.csv_path.setText(path)

    def browse_img_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.img_folder.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Video (*.mp4)")
        if path:
            self.output_path.setText(path)

    # ----------------------------------------------------------
    #              Preset Save/Load
    # ----------------------------------------------------------
    def save_presets(self, csv, img, output, discharge):
        data = {
            "csv_path": csv,
            "img_folder": img,
            "output_path": output,
            "discharge": discharge
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load_presets(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            self.csv_path.setText(data.get("csv_path", ""))
            self.img_folder.setText(data.get("img_folder", ""))
            self.output_path.setText(data.get("output_path", ""))
            self.discharge_spin.setValue(data.get("discharge", 0))


# ----------------------------------------------------------
#                  MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TokamakUI()
    window.show()
    sys.exit(app.exec())