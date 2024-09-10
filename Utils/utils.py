import numpy as np

def ensureList(obj):
    if isinstance(obj, (tuple, list)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return obj
    return [obj]
    
def mergeDict(dict1, dict2):
    """Merge dict2 into dict1, with dict2 values overriding dict1 values."""
    result = dict1.copy()
    result.update(dict2)
    return result

def try_(function_no_arg, fallback):
    try:
        return function_no_arg()
    except Exception as e:
        print(f"{e}")
        return fallback

def fname():
    """ return the name of the function from which called """
    import inspect
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    function_name = caller_frame.f_code.co_name
    return str(f"<{function_name}>")

class customDict(dict):
    """ custom version of a dictionnary
    
    new things:
        method rget: recursive get, return first val with key in child dicts
        
        keys as attributes: customDict.key -> customDict['key']
    """
    
    def __init__(self, *arg, **kw):
        super(customDict, self).__init__(*arg, **kw)
        
    def __getattr__(self, key):
        return self[key]
    
    def __setattr__(self, key, value):
        self[key] = value

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
    def __init__(self, obj):
        super().__init__()
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        layout = QVBoxLayout()
        self.widget.setLayout(layout)
        
        if isinstance(obj, list):
            fig = obj[0].figure
        else:
            fig = obj
            
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