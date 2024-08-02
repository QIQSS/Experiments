# %%
import sys
sys.path.append('C:\Codes\QIQSS-CODE\experiments')
from Pulses.Builder import *
from utils import *

from pyHegel.commands import *

# VARIABLES
path = "C:\\Codes\\QIQSS-CODE\\experiments\\logs\\tunnel_rate\\" # path where to save the data
AWG_SR = 32e4
gain = 1/(0.02512)*0.4

# %% INSTRUMENTS
awg = instruments.tektronix.tektronix_AWG('USB0::0x0699::0x0503::B030793::0')
ats = instruments.ATSBoard(systemId=1, cardId=1)
zi = instruments.zurich_UHF('dev2949')
get_zi = lambda: get((zi.readval, dict(vals=['r'], ch=0)))[0] # return the demod 0 r value
bilt_9 = instruments.iTest_be214x("TCPIP::192.168.150.112::5025::SOCKET", slot=9)
P1 = instruments.LimitDevice((bilt_9.ramp, {'ch': 1}), min=-3.0, max=3.0)
B1 = instruments.LimitDevice((bilt_9.ramp, {'ch': 2}), min=-3.0, max=3.0)
P2 = instruments.LimitDevice((bilt_9.ramp, {'ch': 3}), min=-3.0, max=3.0)
B2 = instruments.LimitDevice((bilt_9.ramp, {'ch': 4}), min=-3.0, max=3.0)


# %% sweep P1 around the transition
p1_list = np.linspace(0.5, 1.5, 701)
out_vals = manualSweep(P1, p1_list, get_zi, plot=True)

# %% find best transition in given window
p1_trans = autoFindTransition(p1_list, out_vals, sigma=2, plot=True)
P1.set(p1_trans)

#filename = saveNpz(path, "transition", out_vals, x_axis=p1_list, metadata={'bilt':bilt_9, 'detected_transition':p1_trans})

# %% compute treshold
threshold = autoFindThreshold(ats, awg, gain, plot=True)

# %% compute tunnel rate
tunnel_rate = autoFindTunnelRate(ats, awg, gain, threshold, opposite_offset=True, fit_skip_firsts_point=9, plot=True)


# %% AUTO ALL
p1_list = np.linspace(0.65, 0.9, 301)
b1_list = np.linspace(-0.7, 0.7, 101)

tr_list = []
for b1_lvl in b1_list:
    B1.set(b1_lvl)
    
    p1_trans = autoFindTransition(p1_list, out_vals, sigma=2, plot=False)  
    P1.set(p1_trans)
    
    threshold = autoFindThreshold(ats, awg, gain, plot=False)
    
    tunnel_rate = autoFindTunnelRate(ats, awg, gain, threshold, opposite_offset=True, fit_skip_firsts_point=9, plot=False)
    tr_list.append(tr_list)
    
