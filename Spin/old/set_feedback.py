# %%
import sys
sys.path.append('C:\Codes\QIQSS-CODE')

from pyHegel.commands import *

# VARIABLES
path = "C:\\Codes\\QIQSS-CODE\\experiments\\logs\\" # path where to save the data

# %% INSTRUMENTS
zurich_UHF = instruments.zurich_UHF("dev2949")
demod0 = (zurich_UHF.readval, {'ch': 0, 'vals':'r'})

bilt_9 = instruments.iTest_be214x("TCPIP::192.168.150.112::5025::SOCKET", slot=9)
P1 = instruments.LimitDevice((bilt_9.ramp, {'ch': 1}), min=-3.0, max=3.0)
B1 = instruments.LimitDevice((bilt_9.ramp, {'ch': 2}), min=-3.0, max=3.0)
P2 = instruments.LimitDevice((bilt_9.ramp, {'ch': 3}), min=-3.0, max=3.0)
B2 = instruments.LimitDevice((bilt_9.ramp, {'ch': 4}), min=-3.0, max=3.0)
P1.name, B1.name, P2.name, B2.name = 'P1', 'B1', 'P2', 'B2'

bilt_7 = instruments.iTest_be214x("TCPIP::192.168.150.112::5025::SOCKET", slot=7)
ST = instruments.LimitDevice((bilt_7.ramp, {'ch': 1}), min=-4.0, max=4.0)
ST.name = 'ST'


gates = lambda: {'ST':ST.get(), 'P1':P1.get(), 'B1':B1.get(), 'P2':P2.get(), 'B2':B2.get()}

sweep.path.set("R:/Projets/IMEC_DD_reflecto/QBB16_SD11b_2/20240627/")
# %%
sweep(ST, 3.1, 3.15, 151, out=demod0, filename='%t_ST.txt')

# %% RETRO

def genRetroFunction(st_first, p1_first, p2_first, coeff_p1, coeff_p2):
    
    def retroFunction(datas):

        p1_val = P1.getcache()
        delta_p1 = p1_first-p1_val
        
        p2_val = P2.getcache()
        delta_p2 = p2_first-p2_val
        
        st_val = st_first - coeff_p1 * delta_p1 - coeff_p2 * delta_p2
        ST.set(st_val)
        print(st_val)

        
    return retroFunction

# config for best set signal:
st_first = 3.118
p1_first = 0.950
p2_first = 0.950

ST.set(st_first)
P1.set(p1_first)
P2.set(p2_first)
wait(0.1)


sweep(P1, 0.8, 1.4, 151, out=demod0, exec_before=genRetroFunction(st_first, p1_first, p2_first, 
                                                                       coeff_p1=14e-3, coeff_p2=7e-3))

sweep(P2, 0.5, 1, 151, out=demod0, exec_before=genRetroFunction(st_first, p1_first, p2_first, coeff_p1=14e-3, coeff_p2=7e-3))

# %% P1 ST / P2 ST
sweep_multi([P1, ST], [0.8, 3.1], [1.5, 3.3], [6, 151], out=demod0, filename='%t_P1_ST.txt')


# %% STAB

sweep_multi([P1, P2], [0.8, 0.9], [1.2, 1.3], [81, 81], out=demod0,
            exec_before=genRetroFunction(st_first, p1_first, p2_first, coeff_p1=14e-3, coeff_p2=7e-3),
            #updown='alternate',
            filename='%t_P1_P2_retroST.txt')



