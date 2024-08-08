"""
my ipython startup config
"""

# ipython:
from IPython import get_ipython
if (ipy := get_ipython()):
    ipy.run_line_magic('load_ext', 'autoreload')
    ipy.run_line_magic('autoreload', '2')
    ipy.run_line_magic('autocall', '1')
    ipy.run_line_magic('gui', 'qt')

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
