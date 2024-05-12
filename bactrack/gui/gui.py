import sys
import concurrent.futures
from functools import partial
from PyQt6.QtCore import Qt, QSize, QObject, QEvent, pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit,
                             QHBoxLayout, QComboBox, QPushButton, QTextEdit, QScrollBar, QLabel, QStackedWidget, QFileDialog, QSplitter, QRadioButton,)
from PyQt6.QtGui import QIcon, QPalette, QCursor
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import threading
from run import run_track
from viz import Viz 
import logging

from bactrack.gui.lineage import Lineage
from bactrack.gui.visualizer import get_graph_stats_text
from bactrack.gui.viz import ImageEnum
from bactrack import core
from tqdm.auto import tqdm

        
class StreamRedirect(QObject):
    text_written = pyqtSignal(str)
    
    def write(self, text):
        self.text_written.emit(str(text))
    
    def flush(self):
        pass

class TqdmLoggingHandler(logging.StreamHandler):

    def __init__(self, level=logging.NOTSET):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
        # from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634
        self.flush()

class Worker(QObject):
    finished = pyqtSignal(object, object)  # Signal for when the task is finished
    error = pyqtSignal(Exception)  # Signal for emitting an error

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run_track(self, **kwargs):
        try:
            composer, G = run_track(self.folder_path, **kwargs)
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

class MyMainWindow(QMainWindow):
    def __init__(self, bg_color=None):
        super().__init__()

        self.frame = 0 
        self.label_index = 0
        self.style_index = 0
        self.folder_path = None
        self.bg_color = bg_color

        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout(self.main_widget)
        self.setGeometry(100, 100, 1000, 700)
        self.terminal = QPlainTextEdit(self)
        self.terminal.setReadOnly(True)

        # Redirect stdout
        self.stream_redirect = StreamRedirect()
        self.stream_redirect.text_written.connect(self.log_to_terminal)
        sys.stdout = self.stream_redirect

        # Redirect logging
        log_handler = QTextEditLogger()
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
        self.weight_dropdown = QComboBox()
        
        self.run_button = QPushButton(' Run Tracking')
        self.save_button = QPushButton('Save Result')
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)

        self.raw_image_choice = QRadioButton('Raw Image')
        self.link_result_choice = QRadioButton('Link Result')
        self.flow_field_choice = QRadioButton('Flow Field')

        self.main_canvas = QStackedWidget()
        self.track_timelapse_canvas = Viz(self)
        self.toolbar = NavigationToolbar(self.track_timelapse_canvas, self)
        self.lineage = Lineage(self)
        
         # All constants
        self.solvers_mapping = {
            "mip solver": "mip_solver",
            "scipy solver": "scipy_solver",
        }

        self.weights_mapping = {
            "overlap": "overlap_weight",
            "IOU": "iou_weight",
            "distance": "distance_weight",
        }

        omnipose_models, cellpose_models = core.load_models()
        omni_mapping = {f"OMNI: {model}": [core.ModelEnum.OMNIPOSE, model] for model in omnipose_models}
        cp_mapping = {f"CP: {model}": [core.ModelEnum.CELLPOSE, model] for model in cellpose_models}
        combined_mapping = {**omni_mapping, **cp_mapping}
        self.models_mapping = combined_mapping

        #  ================== Left vertical section ================
        left_layout = QVBoxLayout()
        # Create the left section with the button and dropdowns
            # Type drop down 
        self.type_dropdown.addItems([".tif", ".png",".jpg"])
        type_layout = QHBoxLayout()  
        type_layout.addWidget( QLabel('Image type:') , 1)
        type_layout.addWidget(self.type_dropdown, 3)

            # Model drop down
        self.model_dropdown.addItems(self.models_mapping.keys())
        model_layout = QHBoxLayout()  
        model_layout.addWidget(QLabel('Models:') , 1)
        model_layout.addWidget(self.model_dropdown, 3)

            # Solver drop down
        self.solver_dropdown.addItems(self.solvers_mapping.keys())
        solver_layout = QHBoxLayout()  
        solver_layout.addWidget(QLabel('Solvers:') , 1)
        solver_layout.addWidget(self.solver_dropdown, 3)

            # Weight drop down
        self.weight_dropdown.addItems(self.weights_mapping.keys())
        weight_layout = QHBoxLayout()  
        weight_layout.addWidget(QLabel('Weight:') , 1)
        weight_layout.addWidget(self.weight_dropdown, 3)

            # Radio buttons
        self.raw_image_choice.setChecked(True)
        choice_layout = QHBoxLayout()
        choice_layout.addWidget(self.raw_image_choice)
        choice_layout.addWidget(self.link_result_choice)
        choice_layout.addWidget(self.flow_field_choice)
        self.raw_image_choice.clicked.connect(lambda: self.changeImageType(ImageEnum.RAW))
        self.link_result_choice.clicked.connect(lambda: self.changeImageType(ImageEnum.LINK))
        self.flow_field_choice.clicked.connect(lambda: self.changeImageType(ImageEnum.FLOW))

            # Buttons
        self.run_button.setCheckable(True)
        self.run_button.clicked.connect(self.runEvent)
        self.save_button.clicked.connect(self.save_result)
        self.run_button.setIcon(QIcon(("/Users/sherryyang/Projects/bactrack/bactrack/gui/images/run.ico")))

        self.run_button.setStyleSheet('''
            QPushButton {
                border-radius: 5px;
                color: white;
                padding: 6px 12px;
            }

            QPushButton:!checked {
                background-color: #007BFF;
            }

            QPushButton:checked {
                background-color: #6c757d; 
            }
        ''')

        def set_button_cursor(enabled):
            if enabled:
                self.run_button.setCursor(QCursor(Qt.PointingHandCursor))  # Change cursor to pointing hand (pointer)
            else:
                self.run_button.setCursor(QCursor(Qt.ArrowCursor))  # Use default arrow cursor

        # Connect hover events to cursor change function
        self.run_button.enterEvent = lambda event: set_button_cursor(self.run_button.isEnabled())
        self.run_button.leaveEvent = lambda event: set_button_cursor(self.run_button.isEnabled())


        # Left layout construction
        left_layout.addLayout(model_layout)
        left_layout.addLayout(solver_layout)
        left_layout.addLayout(weight_layout)
        left_layout.addLayout(type_layout)
        left_layout.addLayout(choice_layout)
        left_layout.addWidget(self.run_button)
        left_layout.addStretch(1) 
        left_layout.addWidget(self.save_button)

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
                # Video canvas placeholder
        self.main_canvas.addWidget(self.track_timelapse_canvas)
        self.main_canvas.setCurrentIndex(0)
        self.main_canvas.setFocus()

                # Scrollbar section:
        self.scrollbar.valueChanged.connect(self.updateFrameView)
     
        # Right layout construction
        right_layout.addWidget(self.toolbar, 1)
        right_layout.addWidget(self.main_canvas, 5)
        right_layout.addWidget(self.scrollbar, 1) # Placeholder for the scrollbar
        right_layout.addWidget(self.terminal, 1)

        #  ================== Option section: Linage ================
        self.linage_stat = QLabel("Linage")
        extra_layout = QVBoxLayout()
        extra_layout.addWidget(self.lineage, 7)
        extra_layout.addWidget(self.linage_stat, 1)

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

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        logging.info(f"{urls} dropped")
        if urls:
            folder_path = urls[0].toLocalFile()

            from bactrack import io
            from cellpose_omni import io as omni_io
            images = io.load(folder_path, omni_io)

            self.track_timelapse_canvas.choice = ImageEnum.RAW
            self.track_timelapse_canvas.show_raw(images)
            self.changeImageType(ImageEnum.RAW)
            self.scrollbar.setMaximum(self.track_timelapse_canvas.max_frame -1 )
            self.main_canvas.setCurrentIndex(1)
            self.worker = Worker(folder_path)

    def runEvent(self, event):
        # Setup the worker and thread
        hypermodel = self.models_mapping[self.model_dropdown.currentText()][0]
        submodel = self.models_mapping[self.model_dropdown.currentText()][1]
        solver_name = self.solvers_mapping[self.solver_dropdown.currentText()]
        weight_name = self.weights_mapping[self.weight_dropdown.currentText()]
        file_extension = "*" + self.type_dropdown.currentText()

        self.thread = QThread() 
        self.worker.moveToThread(self.thread) 

        # Connect signals
        self.worker.finished.connect(self.on_track_finished)  # Connect the finished signal to a slot
        self.worker.finished.connect(self.thread.quit)  # Ensure the thread quits when the task is done
        self.worker.error.connect(self.on_track_error)  # Connect the error signal to a slot
        self.thread.started.connect(lambda: self.worker.run_track(
                hypermodel=hypermodel, 
                submodel=submodel, 
                solver_name=solver_name, 
                weight_name=weight_name, 
                file_extension=file_extension,
            )
        )  
        self.thread.start()


    def changeImageType(self, type: ImageEnum):
        self.track_timelapse_canvas.choice = type
        if type == ImageEnum.LINK:
            self.link_result_choice.setChecked(True)
        elif type == ImageEnum.RAW:
            self.raw_image_choice.setChecked(True)
        elif type == ImageEnum.FLOW:
            self.flow_field_choice.setChecked(True)
        self.track_timelapse_canvas.update_plot()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def on_track_finished(self, composer, G):
        QApplication.restoreOverrideCursor()  # Restore the cursor
        self.track_timelapse_canvas.choice = ImageEnum.LINK
        self.track_timelapse_canvas.run(composer, G)
        self.scrollbar.setMaximum(self.track_timelapse_canvas.max_frame -1 )
        self.lineage.show(G)
        self.linage_stat.setText(get_graph_stats_text(G))
        self.changeImageType(ImageEnum.LINK)
        self.setGeometry(100, 100, 1400, 700)
        self.extra_widget.setVisible(True)

    def on_track_error(self, error):
        QApplication.restoreOverrideCursor()  # Restore the cursor
        logging.info(f"Error: {error}")  # Handle the error (e.g., show a message box)

    def updateFrameView(self, value):
        self.frame = value
        self.track_timelapse_canvas.update_plot()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_BracketRight:
            self.changeFrame(min(self.frame + 1, self.track_timelapse_canvas.max_frame))
        elif event.key() == Qt.Key_BracketLeft:
            self.changeFrame(max(self.frame - 1, 0))

        if self.main_canvas.currentIndex() != 1:
            event.ignore()
            return
        
        elif event.key() == Qt.Key_C:
            self.style_index = (self.style_index + 1) % len(self.track_timelapse_canvas.label_styles)
            self.track_timelapse_canvas.update_plot()
        elif event.key() == Qt.Key_L:
            self.label_index = (self.label_index + 1) % len(self.track_timelapse_canvas.labels)
            self.track_timelapse_canvas.update_plot()
        else:
            event.ignore() 
            return  
        
    def changeFrame(self, frame):
        self.frame = frame
        self.scrollbar.setValue(self.frame)

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
    app.setWindowIcon(QIcon('/Users/sherryyang/Projects/bactrack/bactrack/gui/images/logo.png'))
    palette = app.palette()
    window_color = palette.color(QPalette.Window)
    r, g, b, _ = window_color.getRgbF()
    mainWindow = MyMainWindow(bg_color= (r, g, b))
    mainWindow.log_to_terminal("Bactrack start.")

    mainWindow.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
