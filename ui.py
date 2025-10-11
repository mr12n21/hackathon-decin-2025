import sys
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit,
    QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt

# ---- Matplotlib for live graph ----
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---- Load ONNX model ----
onnx_path = "runs/classify/train4/weights/best.onnx"
session = ort.InferenceSession(onnx_path)

# class order — ensure this matches your model output order
classes = ["ELM", "H-mode", "L-mode"]


def get_model_output_length():
    """Detect model output length (number of classes)."""
    try:
        inp = session.get_inputs()[0]
        shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
        dummy = np.zeros(shape, dtype=np.float32)
        out = session.run(None, {inp.name: dummy})
        probs = out[0]
        if probs.ndim == 2:
            return probs.shape[1]
        elif probs.ndim == 1:
            return probs.shape[0]
        return len(classes)
    except Exception:
        return len(classes)


class LiveGraph(FigureCanvas):
    def __init__(self, max_points=150):
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Real-Time Class Probabilities")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Probability")
        self.ax.set_ylim(0, 2)  # y-range fixed 0–2
        self.max_points = max_points
        self.ax.set_xlim(0, max_points)

        # store history for each class
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


class TokamakUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tokamak Analyzer")
        self.resize(1300, 700)
        self.setAcceptDrops(True)

        # ---- Left side: video feed ----
        self.video_label = QLabel("Drop a Video or Image Here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px dashed #888; color: #888;")

        # ---- Right side: analysis text + graph ----
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFixedSize(400, 200)

        self.graph = LiveGraph(max_points=150)

        # Layouts
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.analysis_text)
        right_layout.addWidget(self.graph)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(right_layout)

        self.load_btn = QPushButton("Load Video/Image")
        self.load_btn.clicked.connect(self.load_file)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.load_btn)

        outer_layout = QVBoxLayout()
        outer_layout.addLayout(main_layout)
        outer_layout.addLayout(buttons_layout)

        self.setLayout(outer_layout)

        # ---- Timer for video playback ----
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.cap = None

    # ---- Drag & drop ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.video_label.setStyleSheet("border: 2px dashed #4CAF50; color: #4CAF50;")

    def dragLeaveEvent(self, event):
        self.video_label.setStyleSheet("border: 2px dashed #888; color: #888;")

    def dropEvent(self, event):
        self.video_label.setStyleSheet("border: 2px dashed #888; color: #888;")
        for url in event.mimeData().urls():
            self.handle_new_file(url.toLocalFile())
            break

    # ---- File selection ----
    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video or Image", "",
            "Media Files (*.mp4 *.avi *.mov *.png *.jpg *.jpeg)"
        )
        if file_name:
            self.handle_new_file(file_name)

    # ---- Handle file loading ----
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

    # ---- Show single image ----
    def show_image(self, path):
        frame = cv2.imread(path)
        if frame is not None:
            self.process_and_display(frame)
        else:
            QMessageBox.warning(self, "Error", "Failed to load image.")

    # ---- Handle next video frame ----
    def next_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display(frame)
            else:
                self.cap.release()
                self.timer.stop()

    # ---- Inference + visualization ----
    def process_and_display(self, frame):
        img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = session.run(None, {"images": img})
        probs = outputs[0][0]

        top_idx = int(np.argmax(probs))
        top_label = classes[top_idx] if 0 <= top_idx < len(classes) else f"idx {top_idx}"

        # update text
        text = f"Top Prediction: {top_label}\n\nClass Probabilities:\n"
        for cls, prob in zip(classes, probs):
            text += f"{cls}: {prob:.4f}\n"
        self.analysis_text.setText(text)

        # update graph
        self.graph.update_plot(probs)

        # display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TokamakUI()
    window.show()
    sys.exit(app.exec())