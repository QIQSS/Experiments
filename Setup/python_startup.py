"""
my ipython startup config
"""
#
T = True
F = False

# ipython:
from IPython import get_ipython
if (_ipy := get_ipython()):
    _ipy.run_line_magic('load_ext', 'autoreload')
    _ipy.run_line_magic('autoreload', '2')
    _ipy.run_line_magic('autocall', '1')
    _ipy.run_line_magic('gui', 'qt')
else:
    print('no ipython?')

# Packages:
import numpy as np
from matplotlib import pyplot as plt

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
