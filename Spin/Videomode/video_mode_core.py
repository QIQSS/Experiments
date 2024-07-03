
import os.path, sys
if (parent := os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) not in sys.path:
    sys.path.append(parent) # import parent folder
del parent

import time

from Pulses.Builder import *
from MyUtils import *

from VideoModeWindow import VideoModeWindow

from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np

class VideoThread(QThread):
    sig_frameDone = pyqtSignal(np.ndarray)
    
    def __init__(self, fn_get_data):
        super().__init__()
        self.get = fn_get_data
        
        self.stop = False
        
    def run(self):
        while True:
            if not self.stop:
                data = self.get()
                self.sig_frameDone.emit(data)
                time.sleep(0.001)
            else:
                time.sleep(1)
            

class VM1d:
    
    def __init__(self, fn_get, span, nbstep, step_duration=0.0001, gain=1/(0.02512)*0.4, ats=None, awg=None):
        """ video mode 1d
        gen a pulse from -span to +span
        nbpts: will be force to an even number
        """
        self.span = span
        self.nbstep = nbstep if nbstep%2==0 else nbstep+1
        
        self.step_duration = step_duration
        
        self.gain = gain
        
        if ats is not None:
            self.setupAts(ats)
        if awg is not None:
            self.setupAwg(awg)
        
        self.win = VideoModeWindow(self)
        
        self.thread = None
        self.thread = VideoThread(fn_get)
        self.thread.sig_frameDone.connect(self.onNewData)
    
    def setupAts(self, ats):
        configureAts(ats)    
        #ats.trigger_slope_1.set('ascend')
        ats.nbwindows.set(1)
    
    def setupAwg(self, awg):
        steps = []
        steps.append(Segment(duration = self.step_duration, offset=0, mark=(0.0,1.0)))
        
        for i, step in enumerate(np.linspace(-self.span, +self.span, self.nbstep)):
            ramp_val_end = i/self.nbstep * 0.00015
            steps.append(Segment(duration=self.step_duration, waveform=Ramp(val_start=0.00, val_end=ramp_val_end), offset=step))
        
        self.pulse = Pulse(steps)
        
        sendSeqToAWG(awg, pulse, self.gain, channel=1, run_after=True)

        
    def start(self):
        self.thread.start()
        self.thread.stop = False
        self.win.show()
    
    def pause(self):
        self.thread.stop = True
        
    def onNewData(self, data):
        #print(data)
        self.win.plot(data)


class VM2d:
    
    def __init__(self, fn_get, spanx, spany, nbstepx, nbstepy, step_duration=0.0001, gain=1/(0.02512)*0.4, ats=None, awg=None):
        """ video mode 1d
        gen a pulse from -span to +span
        nbpts: will be force to an even number
        """
        self.spanx, self.spany = spanx, spany
        self.nbstepx = nbstepx if nbstepx%2==0 else nbstepx+1
        self.nbstepy = nbstepy if nbstepy%2==0 else nbstepy+1
        self.step_duration = step_duration
        
        self.gain = gain
        
        if ats is not None:
            self.setupAts(ats)
        if awg is not None:
            self.setupAwg(awg)
        
        self.win = VideoModeWindow(self)
        
        self.thread = None
        self.thread = VideoThread(fn_get)
        self.thread.sig_frameDone.connect(self.onNewData)
    
    def setupAts(self, ats):
        configureAts(ats)
        ats.nbwindows.set(self.nbstepy)
    
    def setupAwg(self, awg):
        steps_x = []
        steps_x.append(Segment(duration = self.step_duration, offset=0, mark=(0.0,1.0)))
        
        # gen pulse x
        for i, step in enumerate(np.linspace(-self.spanx, +self.spanx, self.nbstepx)):
            ramp_val_end = i/self.nbstepx * 0.00015 * np.sign(step)
            steps.append(Segment(duration=self.step_duration, waveform=Ramp(val_start=0.00, val_end=ramp_val_end), offset=step))
        pulse_x = Pulse(steps_x)
        
        
        # gen pulse y
        steps_y = []
        for i, step in enumerate(np.linspace(-self.spany, +self.spany, self.nbstepy)):
            ramp_val_end = i/self.nbstepy * 0.0002 * np.sign(step)
            steps.append(Segment(duration=pulse_x.duration, waveform=Ramp(val_start=0.00, val_end=ramp_val_end), offset=step))
        pulse_y = Pulse(steps_y)
        
        
        sendSeqToAWG(awg, pulse_x, self.gain, channel=1, run_after=True)
        sendSeqToAWG(awg, pulse_y, self.gain, channel=2, run_after=True)

        
    def start(self):
        self.thread.start()
        self.thread.stop = False
        self.win.show()
    
    def pause(self):
        self.thread.stop = True
        
    def onNewData(self, data):
        #print(data)
        self.win.plot(data)
