from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QPushButton, QTreeWidget, QSplitter, QVBoxLayout

from pyHegel.gui import ScientificSpinBox
import pyqtgraph as pg

# TODO:
# - start/pause button
# - averaging

class VideoModeWindow(QMainWindow):
    
    def __init__(self, vm, dim=1):
        super().__init__()
        self.vm = vm
        self.setWindowTitle("Video mode")
        
        splitter = QSplitter()
        self.graph = pg.PlotWidget()
        
        self.curve = self.graph.plot()
        self.image = pg.ImageItem()
        self.graph.addItem(self.image)
        
        self.tree = QTreeWidget()
        
        splitter.addWidget(self.tree)
        splitter.addWidget(self.graph)
        
        self.setCentralWidget(splitter)

    def plot(self, data_1d):
        self.curve.setData(data_1d)