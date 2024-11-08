from Pulses.Builder import Pulse, equalizeTime

def genABCDE(abcde_pos, abcde_time, abcde_ramp, tinit=0.05, plot=False):
    A, B, C, D, E = abcde_pos
    tA, tB, tC, tD, tE = abcde_time
    trampAB, trampBC, trampCD, trampDE = abcde_ramp
    
    pulseP1 = Pulse(name='P1', shape_comp=True)
    
    pulseP1.add(tinit)
    
    pulseP1.add(tA, offset=A[0], mark=True)
    pulseP1.addRamp(trampAB, A[0], B[0],mark=True)
    
    pulseP1.add(tB, offset=B[0], mark=True)
    pulseP1.addRamp(trampBC, B[0], C[0],mark=True)
    
    pulseP1.add(tC, offset=C[0], mark=True)
    pulseP1.addRamp(trampCD, C[0], D[0],mark=True)
    
    pulseP1.add(tD, offset=D[0], mark=True)
    pulseP1.addRamp(trampDE, D[0], E[0],mark=True)
    
    pulseP1.add(tE, offset=E[0], mark=True)
    
    #pulseP1.addCompensationZeroMean(value=0.035)
    
    
    pulseP2 = Pulse(name='P2', shape_comp=True)
    pulseP2.add(tinit)
    
    pulseP2.add(tA, offset=A[1], mark=False)
    pulseP2.addRamp(trampAB, A[1], B[1] ,mark=False)
    
    pulseP2.add(tB, B[1])
    pulseP2.addRamp(trampBC, B[1], C[1], mark=False)
    
    pulseP2.add(tC, offset=C[1])
    pulseP2.addRamp(trampCD, C[1], D[1], mark=False)
    
    pulseP2.add(tD, offset=D[1])
    pulseP2.addRamp(trampDE, D[1], E[1], mark=False)
    
    pulseP2.add(tE, offset=E[1])

    
    equalizeTime(pulseP1, pulseP2)
    if plot: pulseP2.plot(pulseP1)
    
    return pulseP1, pulseP2

def genDrive(abcde_pos, abcde_time, abcde_ramp, tdrive, drive_on='B', gain=15.923566878980893, tinit=0.05, plot=False):
    """ only on B for now """
    A, B, C, D, E = abcde_pos
    tA, tB, tC, tD, tE = abcde_time
    trampAB, trampBC, trampCD, trampDE = abcde_ramp
    
    pulseSR = Pulse(name='SR', shape_comp=False)
    pulseSR.add(tinit)
    pulseSR.add(tA)
    pulseSR.add(trampAB)
    pulseSR.add(tB/2-tdrive/2)
    pulseSR.addRamp(tdrive, -1.75/gain, +1.75/gain, mark=True)
    pulseSR.add(tB/2-tdrive/2)
    pulseSR.add(trampBC)
    pulseSR.add(tC)
    pulseSR.add(trampCD)
    pulseSR.add(tD)
    pulseSR.add(trampDE)
    pulseSR.add(tE)
    
    
    return pulseSR

def genDriveMarkerPulse(abcde_pos, abcde_time, abcde_ramp, tdrive, drive_on='B', gain=15.923566878980893, tinit=0.05, plot=False):
    """ only on B for now """
    A, B, C, D, E = abcde_pos
    tA, tB, tC, tD, tE = abcde_time
    trampAB, trampBC, trampCD, trampDE = abcde_ramp
    
    pulseSR = Pulse(name='SR', shape_comp=False)
    pulseSR.add(tinit)
    pulseSR.add(tA)
    pulseSR.add(trampAB)
    pulseSR.add(tB/2-tdrive/2)
    pulseSR.add(tdrive, mark=True)
    pulseSR.add(tB/2-tdrive/2, mark=True)
    pulseSR.add(trampBC, mark=True)
    pulseSR.add(tC, mark=True)
    pulseSR.add(trampCD, mark=True)
    pulseSR.add(tD, mark=True)
    pulseSR.add(trampDE, mark=True)
    pulseSR.add(tE, mark=True)
    
    
    return pulseSR