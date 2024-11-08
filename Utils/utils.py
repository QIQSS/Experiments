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
    """ return the name of the function from which called
    def f():
        print(fname()) -> <f>
        
    I don't remember why this is useful but it exists.
    """
    import inspect
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    function_name = caller_frame.f_code.co_name
    return str(f"<{function_name}>")

def qdict(*args):
    """ quick dict, give only the vals, keys are str(vals)
    return dict(str(arg)=arg)
    all args must be  string-able else its undefined behavior
    """
    return {str(arg):arg for arg in locals()['args']}

class customDict(dict):
    """ custom version of a dictionnary
    
    new things:
        can get and set with attributes.
        method rget: recursive get, return first val with key in child dicts
        
        keys as attributes: customDict.key -> customDict['key']
    """
    
    def __init__(self, *arg, **kw):
        super(customDict, self).__init__(*arg, **kw)
        #self._convert_nested_dicts()
    
    def _convert_nested_dicts(self):
        """ Convert all nested dictionaries to customDict instances. """
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = customDict(value)
            
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
    

class ModuloList(list):
    """ a list with circular indexes 
    inherit from list so it has all the normal methods
    """
    def __getitem__(self, index):
        if isinstance(index, int):
            index %= len(self)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            index %= len(self)
        super().__setitem__(index, value)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout

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
        
from PyQt5.QtCore import QThread
import time as timemodule
class delayExec(QThread):
    """ run 'fn' in a thread after some 'time' """
    def __init__(self, time, fn):
        super().__init__()
        self.fn = fn
        self.time = time

    def run(self):
        timemodule.sleep(self.time)
        self.fn()
        
        
def enumtq(*args):
    """ faster tqdm enumerate, and no need for import """
    from tqdm import tqdm
    for item in enumerate(tqdm(*args)):
        yield item

def copy(arg):
    from pyperclip import copy
    copy(arg)
    
def paste():
    from pyperclip import paste
    return paste()