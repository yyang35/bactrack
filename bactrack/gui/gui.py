import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
import numpy as np
import cv2
from viz import Viz

from run import run_track

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Backtrack")
        self.layout = QVBoxLayout()

        self.label = QLabel()
        self.layout.addWidget(self.label)

        self.drag_label = QLabel("Drag Folder here")
        self.drag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.drag_label)

        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(self.layout)

        self.images = []
        self.current_image_index = 0

    def show_image(self):
        image = self.images[self.current_image_index]
        height, width = image.shape
        bytes_per_line =  width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        
        import matplotlib.pyplot as plt
        plt.imshow(self.images[self.current_image_index], cmap='gray')
        plt.show()

    def load_images_from_folder(self, folder_path):
        print("Loading images from folder", folder_path)

        import cellpose_omni.io as omni_io
        from bactrack import io

        self.images = io.load(folder_path, omni_io)

        if self.images:
            self.show_image()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            composer, G = run_track(folder_path)
            Viz(composer, G)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
