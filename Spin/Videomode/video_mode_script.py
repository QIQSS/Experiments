# %% resources
#https://qtt.readthedocs.io/en/latest/notebooks/measurements/example_videomode.html
# %% ipython setup and imports
%load_ext autoreload
%autoreload 2
%gui qt


from pyHegel.commands import *
_init_pyHegel_globals()


%cd /Codes/Lab-Scripts/Spin/Videomode

from video_mode_core import VM1d

# %% Load pyHegel instrument

bilt_8 = instruments.iTest_be214x("TCPIP::192.168.150.112::5025::SOCKET", slot=8)
P1 = custom_dev((bilt_8.ramp, {'ch': 1}))
B1 = custom_dev((bilt_8.ramp, {'ch': 2}))
P2 = custom_dev((bilt_8.ramp, {'ch': 3}))
B2 = custom_dev((bilt_8.ramp, {'ch': 4}))

ats = instruments.ATSBoard(systemId=1, cardId=1)
awg = instruments.tektronix.tektronix_AWG('USB0::0x0699::0x0503::B030793::0')

# %% 1d video mode
dm = instruments.dummy()

vm = VM1d(lambda: np.random.rand(10), span=0.002, nbstep=100)
vm.start()
 
# %% 2d video mode
vm = VM2d(lambda: np.random.rand(10), span=0.002, nbstep=100)
vm.start()