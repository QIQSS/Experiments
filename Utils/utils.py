
def ensureList(obj):
    if isinstance(obj, (tuple, list)):
        return obj
    return [obj]
    
def mergeDict(dict1, dict2):
    """Merge dict2 into dict1, with dict2 values overriding dict1 values."""
    result = dict1.copy()
    result.update(dict2)
    return result


class customDict(dict):
    """ custom version of a dictionnary
    
    new method:
        rget: recursive get, return first val with key in child dicts
        
    """
    
    def __init__(self, *arg, **kw):
        super(customDict, self).__init__(*arg, **kw)

    def rget(self, key, default=None):
        """ recursive get:
            search inside values that are dict.
            return the first one find
            or default
        """    
        def _searchKey(dic, key_to_find):
            for key, val in dic.items():
                if isinstance(val, (dict, customDict)):
                    return _searchKey(val, key_to_find)
                if key == key_to_find:
                    return val
            return default
        return _searchKey(self, key)

import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QToolBar

class mplqt(QMainWindow):
    """ to force a fig to be interactive even if %matplotlib is set to inline
    require: %gui qt
    """
    def __init__(self, fig):
        super().__init__()
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        layout = QVBoxLayout()
        self.widget.setLayout(layout)
        self.canvas = FigureCanvas(fig)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setWindowTitle('interactive pls')
        
        self.resize(800, 600)
        self.show()
    
    def tight_layout(self):
        self.canvas.figure.tight_layout()
        self.canvas.draw()