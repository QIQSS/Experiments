"""
my ipython startup config
"""

# iPython:
if (ipy := get_ipython()):
    ipy.run_line_magic('load_ext', 'autoreload')
    ipy.run_line_magic('autoreload', '2')
    ipy.run_line_magic('autocall', '1')

# Packages:
import numpy as np
from matplotlib import pyplot as plt

# PyHegel:
from pyHegel.scipy_fortran_fix import fix_problem_new
from PyHegel.commands import *