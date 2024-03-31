import sys


from PyQt6.QtCore import Qt, QSize, QObject, QEvent, pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit,
                             QHBoxLayout, QComboBox, QPushButton, QTextEdit, QScrollBar, QLabel, QStackedWidget, QFileDialog, QSplitter)

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import threading
from run import run_track
from viz import Viz 
import logging
from lineage import Lineage
from visualizer import get_graph_stats_text
import logging
from raw_image import RawImage

        
class StreamRedirect(QObject):
    text_written = pyqtSignal(str)
    
    def write(self, text):
        self.text_written.emit(str(text))
    
    def flush(self):
        pass


class Worker(QObject):
    finished = pyqtSignal(object, object)  # Signal for when the task is finished
    error = pyqtSignal(Exception)  # Signal for emitting an error

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run_track(self):
        try:
            # Placeholder for the time-consuming task
            composer, G = run_track(self.folder_path)
            self.finished.emit(composer, G)  # Emit the results upon completion
        except Exception as e:
            self.error.emit(e)  # Emit any errors encountered during the task

class QSignalEmitter(QObject):
    # Define a signal to emit log messages
    emit_log = pyqtSignal(str)

class QTextEditLogger(logging.Handler, QSignalEmitter):
    def __init__(self):
        logging.Handler.__init__(self)
        QSignalEmitter.__init__(self)

    def emit(self, record):
        message = self.format(record)
        self.emit_log.emit(message)
    

class ToggleButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setText("Raw Image Off")
    

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.frame = 0 
        self.label_index = 0
        self.style_index = 0

        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout(self.main_widget)
        self.setGeometry(100, 100, 800, 600)

        # Redirect stdout
        self.stream_redirect = StreamRedirect()
        self.stream_redirect.text_written.connect(self.log_to_terminal)
        sys.stdout = self.stream_redirect

        # Redirect logging

        log_handler = QTextEditLogger()  # Create your custom handler
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(log_handler)


        log_handler.emit_log.connect(self.log_to_terminal)


        # Preparing all wigets
        self.type_dropdown = QComboBox()
        self.model_dropdown = QComboBox()
        self.solver_dropdown = QComboBox()
        
        self.run_button = QPushButton('Run Tracking')
        self.reset_button = QPushButton('Reset Zoom')
        self.save_button = QPushButton('Save Result')
        self.toggle_button = ToggleButton(self)
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)

        self.main_canvas = QStackedWidget()
        self.raw_image = RawImage(self)
        self.track_timelapse_canvas = Viz(self)
        self.toolbar = NavigationToolbar(self.track_timelapse_canvas, self)

        self.lineage = Lineage()


        #  ================== Left vertical section ================
        left_layout = QVBoxLayout()
        # Create the left section with the button and dropdowns
            # Type drop down 
        self.type_dropdown.addItems([".png", ".tif", ".jpg"])
        type_layout = QHBoxLayout()  
        type_layout.addWidget( QLabel('Type:') , 1)
        type_layout.addWidget(self.type_dropdown, 3)

            # Model drop down
        self.model_dropdown.addItems(["Model 1", "Model 2", "Model 3"])
        model_layout = QHBoxLayout()  
        model_layout.addWidget(QLabel('model:') , 1)
        model_layout.addWidget(self.model_dropdown, 3)

            # Solver drop down
        self.model_dropdown.addItems(["Scipy solver", "Mip solver"])
        model_layout = QHBoxLayout()  
        model_layout.addWidget(QLabel('model:') , 1)
        model_layout.addWidget(self.model_dropdown, 3)



            # Buttons
        self.toggle_button.toggled.connect(self.rawImageToggled)
        self.reset_button.clicked.connect(self.track_timelapse_canvas.reset_zoom)

        # Left layout construction
        left_layout.addLayout(type_layout)
        left_layout.addLayout(model_layout)
        left_layout.addWidget(self.toggle_button)
        left_layout.addWidget(self.reset_button)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.save_button)
        left_layout.addStretch(1) 

        #  ================== Right vertical section ================
        right_layout = QVBoxLayout()
        # Create the right section with the canvas and terminal
            # Main canvas section:
        #self.main_canvas.setFixedHeight(400)
                # Text placeholder
        self.text_widget = QLabel("Drop Your Folder Here")
        self.text_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_widget.setStyleSheet("border: 2px dashed #000000;")
        self.main_canvas.addWidget(self.text_widget)
                # Raw image placeholder
        self.main_canvas.addWidget(self.raw_image)
                # Video canvas placeholder
        self.main_canvas.addWidget(self.track_timelapse_canvas)
        self.main_canvas.setCurrentIndex(0)
        self.main_canvas.setFocus()

                # Scrollbar section:
        self.scrollbar.valueChanged.connect(self.updateFrameView)

                # Terminal section:
        self.terminal = QPlainTextEdit(self)
        self.terminal.setReadOnly(True)

        
        # Right layout construction
        right_layout.addWidget(self.toolbar, 1)
        right_layout.addWidget(self.main_canvas, 5)
        right_layout.addWidget(self.scrollbar, 1) # Placeholder for the scrollbar
        right_layout.addWidget(self.terminal, 1)


        #  ================== Option section: Linage ================
        self.linage_stat = QLabel("Linage")
        self.linage_stat.setFixedHeight(100)
        extra_layout = QVBoxLayout()
        extra_layout.addWidget(self.linage_stat)
        extra_layout.addWidget(self.lineage)


        # Create the splitter and add the two layouts
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        extra_widget = QWidget()
        extra_widget.setLayout(extra_layout)

        # Finalize everything 
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.addWidget(extra_widget)

        extra_widget.setVisible(False)
        self.extra_widget = extra_widget

        splitter.setSizes([200, 600, 300])
        
       
        # ================== Put everthing together main section setup ================
        # Set the splitter as the central widget
        self.main_layout.addWidget(splitter)
        self.setCentralWidget(self.main_widget)

        # Set window title and show the application
        self.setWindowTitle("Bactrack")
        self.show()

    def log_to_terminal(self, message):
        self.terminal.insertPlainText(message + '\n') 
        self.terminal.ensureCursorVisible()
    

    def rawImageToggled(self, checked):
        if checked:
            self.toggle_button.setText("Raw image On")
            self.main_canvas.setCurrentIndex(1)
        else:
            self.toggle_button.setText("Raw image Off")
            self.main_canvas.setCurrentIndex(2)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        logging.info(f"{urls} dropped")
        if urls:
            folder_path = urls[0].toLocalFile()

            from bactrack import io
            from cellpose_omni import io as omni_io
            images = io.load(folder_path, omni_io)

            self.raw_image.show(images)
            self.main_canvas.setCurrentIndex(1)

            # Setup the worker and thread
            self.thread = QThread()  # Create a QThread object
            self.worker = Worker(folder_path)  # Create a Worker object
            self.worker.moveToThread(self.thread)  # Move the Worker to the thread

            # Connect signals
            self.worker.finished.connect(self.on_track_finished)  # Connect the finished signal to a slot
            self.worker.finished.connect(self.thread.quit)  # Ensure the thread quits when the task is done
            self.worker.error.connect(self.on_track_error)  # Connect the error signal to a slot
            self.thread.started.connect(self.worker.run_track)  # Start the worker's task when the thread starts

            # Start the thread
            self.thread.start()

            # Optional: Change the cursor to indicate processing
            QApplication.setOverrideCursor(Qt.BusyCursor)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def on_track_finished(self, composer, G):
        QApplication.restoreOverrideCursor()  # Restore the cursor
        self.track_timelapse_canvas.run(composer, G)
        self.main_canvas.setCurrentIndex(2)
        self.scrollbar.setMaximum(self.raw_image.max_frame)
        self.lineage.show(G)
        self.linage_stat.setText(get_graph_stats_text(G))
        self.setGeometry(100, 100, 1100, 600)
        self.extra_widget.setVisible(True)

    def on_track_error(self, error):
        QApplication.restoreOverrideCursor()  # Restore the cursor
        logging.info(f"Error: {error}")  # Handle the error (e.g., show a message box)

    def updateFrameView(self, value):
        self.frame = value
        self.update_plot()

    def update_plot(self):
        if self.main_canvas.currentIndex() == 1:
            self.raw_image.update_plot(self)
        elif self.main_canvas.currentIndex() == 2:
            self.track_timelapse_canvas.update_plot(self)
    
    def keyPressEvent(self, event):
        if self.main_canvas.currentIndex() != 1:
            event.ignore()
            return
        if event.key() == Qt.Key_BracketRight:
            self.frame = min(self.frame + 1, self.track_timelapse_canvas.max_frame) 
        elif event.key() == Qt.Key_BracketLeft:
            self.frame = max(self.frame - 1, 0)
        elif event.key() == Qt.Key_C:
            self.style_index = (self.style_index + 1) % len(self.track_timelapse_canvas.label_styles)
        elif event.key() == Qt.Key_L:
            self.label_index = (self.label_index + 1) % len(self.track_timelapse_canvas.labels)
        else:
            event.ignore() 
            return  
        
        self.update_plot()

    def save_result(self):
        options = QFileDialog.Options()
        # Set the default file type filter
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "","CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            print("File path:", fileName)
            # Example DataFrame
            df = pd.DataFrame({'Name': ['John', 'Anna'], 'Age': [28, 22]})
            # Save DataFrame to the selected file path
        df.to_csv(fileName, index=False)


def main():
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.log_to_terminal("Bactrack start.")
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
