import os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
import numpy as np
import cv2

from run import run_track

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.layout = QVBoxLayout()

        self.label = QLabel()
        self.layout.addWidget(self.label)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        self.layout.addWidget(self.next_button)

        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(self.layout)

        self.images = []
        self.current_image_index = 0

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_image()

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
        print("Dropped")
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            run_track(folder_path)
            self.load_images_from_folder(folder_path)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
