"""
my ipython startup config
"""
#
T = True
F = False

def tryimport(command):
    try:
        exec(command, globals())
    except Exception as e:
        print(e)
        pass
    
# Packages:
import numpy as np
#tryimport('from icecream import ic')

# Matplotlib
import matplotlib
from matplotlib import pyplot as plt
def mpldpi(val):
    matplotlib.rcParams['figure.dpi'] = val
mpldpi(100)

# Garbage
import gc
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
OS = currentOS()

# lab-script
LS_PATH, LOG_BOB_PATH = \
            {'windows':['C:/Codes/Lab-Scripts', 
                  '//bob.physique.usherbrooke.ca/recherche/Dupont-Ferrier/Projets/IMEC_DD_reflecto/QBB16_SD11b_3/Spin/'],
             'linux':['/home/local/USHERBROOKE/mora0502/Codes/Lab-Scripts',
                  '/run/user/1338691803/gvfs/smb-share:server=bob.physique.usherbrooke.ca,share=recherche/Dupont-Ferrier/Projets/IMEC_DD_reflecto/QBB16_SD11b_3/Spin/']}\
            [OS]

# PyHegel:
try:
    from pyHegel.scipy_fortran_fix import fix_problem_new
    from pyHegel.commands import *
    _init_pyHegel_globals()
except ImportError as e:
    print('no pyHegel.')


# ipython:
from IPython import get_ipython
if (IPY := get_ipython()):
    IPY.run_line_magic('load_ext', 'autoreload')
    IPY.run_line_magic('autoreload', '2')
    IPY.run_line_magic('autocall', '2')
    IPY.run_line_magic('gui', 'qt')
    mplqt = lambda: IPY.run_line_magic('matplotlib', 'qt')
    mplil = lambda: IPY.run_line_magic('matplotlib', 'inline')
else:
    print('no ipython?')
