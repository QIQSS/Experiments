"""
my ipython startup config
"""
#
T = True
F = False


# Packages:
import numpy as np

# Matplotlib
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

# PyHegel:
try:
    from pyHegel.scipy_fortran_fix import fix_problem_new
    from pyHegel.commands import *
    _init_pyHegel_globals()
except ImportError:
    print('no pyHegel.')

# Pyperclip
try:
    from pyperclip import copy, paste
except ImportError:
    pass

# Garbage
import gc
ramasse_miette = gc.collect

# Notify-run
try:
    from notify_run import Notify
    notify = Notify(endpoint="https://notify.run/WvLaaa2BYb9iSBM3tM5R")
except ImportError:
    pass

# Detect OS
def currentOS():
    import platform
    os_name = platform.system().lower()
    if 'windows' in os_name:
        return 'windows'
    elif 'linux' in os_name:
        return 'linux'
    else:
        return 'unknown'  # In case it's not Linux or Windows


# ipython:
from IPython import get_ipython
if (_ipy := get_ipython()):
    _ipy.run_line_magic('load_ext', 'autoreload')
    _ipy.run_line_magic('autoreload', '2')
    _ipy.run_line_magic('autocall', '1')
    _ipy.run_line_magic('gui', 'qt')
    mplqt = lambda: _ipy.run_line_magic('matplotlib', 'qt')
    mplil = lambda: _ipy.run_line_magic('matplotlib', 'inline')
else:
    print('no ipython?')