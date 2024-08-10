from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QMainWindow, QPushButton, \
    QTreeWidget, QSplitter, QVBoxLayout, QHBoxLayout, \
    QGridLayout, QSpinBox, QWidget, QProgressBar, QLabel, QApplication

from pyHegel.gui import ScientificSpinBox
from pyHegel.commands import wait
import pyqtgraph as pg
import pyqtgraph.exporters

import time
import traceback
import numpy as np


class VideoModeWindow(QMainWindow):
    
    def __init__(self, fn_get=None, dim=1, show=True, play=False, take_focus=False,
                 sec_between_frame=0.01,
                 axes_dict={'x': [0, 1], 'y': [0, 1]},
                 fn_xshift=None, fn_yshift=None,
                 xlabel=None, ylabel=None,
                 #fix_xaxis=False, fix_yaxis=False,
                 wrap_at=0, wrap_direction='h',
                 pause_after_one=False,
                 ysweep=None, xsweep=None,
                 window_size=False):
        """
        Opens a window and start a thread that exec and show `fn_get`.

        Parameters
        ----------
        fn_get : FUNCTION
        dim : dim of the array returned by fn_get
        show : BOOL
            show the window by default
        play : BOOL
            pause by default
            if True and take_focus: will not return until pause is pressed
        sec_between_frame : float
            if fn_get is too fast, this should be >0 to not overload the display thread.
        axes_dict: dict
            of the form {'x': [start, stop]} for 1d
            {'x': [start, stop], 'y':[start, stop] for 2d
             OR
            of the form {'x': stop} and it will be interpreted as [0, stop]
        fn_xshift: function
            called when pressing x shift buttons, with arg step
        wrap_at: int
            when dim=1, you can choose to still display an image. 'wrap_at' is the second dimension of this image.
            the shape of get_fn() must be constant.
        wrap_direction: 'v' | 'h'
        pause_after_one: bool, pause after a map has been completed. (== after an image is in the buffer)
        ysweep: give a SweepAxis object and it will set the right ylabel, yaxis, fn_yshift and wrap_at (overridding kw args).
        xsweep: same as ysweep but wrap_direction is set to 'v'. Can only use one or the other.
        window_size: False:default | 'wide' | 'wider'
        Returns
        -------
        None.

        """
        super().__init__()
        self.frame_count = 0
        self._wrap_mode = False
        self.pause_after_one = pause_after_one
                        
        # sweep object
        if ysweep:
            wrap_at = len(ysweep)
            ylabel = ysweep.label
            fn_yshift = ysweep.shift
            axes_dict['y'] = ysweep.axis
        elif xsweep:
            wrap_at = len(xsweep)
            xlabel = xsweep.label
            fn_xshift = xsweep.shift
            axes_dict['x'] = xsweep.axis 
            wrap_direction = 'v'
        
        # VM
        # init a dummy vm
        self.navg = 1
        self.data_buffer = []
        self.avg_data = None # store the current total avg image
        self.x = axes_dict.get('x', [0, 1])
        if isinstance(self.x, (int, float)): self.x = [0, self.x]
        self.y = axes_dict.get('y', [0, 1] if wrap_at == 0 else [0, wrap_at])
        if isinstance(self.y, (int, float)): self.y = [0, self.y]
        
        self.fn_xshift = fn_xshift
        self.fn_yshift = fn_yshift

        
        # for wrapping mode
        if dim==1 and wrap_at > 0:
            self._wrap_single_img = None # buffer to store a single full image
            self._wrap_counter = wrap_at
            self._wrap_direction = wrap_direction
            self._wrap_mode = True

        def get_fn_1d_example(): return np.random.rand(100)
        def get_fn_2d_example(): return np.random.rand(10,10)
        if fn_get is None:
            fn_get = [get_fn_1d_example, get_fn_2d_example][dim-1]
        self.continousGet(fn_get, dim=dim, sec_between_frame=sec_between_frame)
        self.vthread.start() # start thread, but vm is paused.
        # setting
        self.pause_at_max_avg = False
        
        # UI
        self.setWindowTitle("Video mode")
        
        splitter = QSplitter()
        self.graph = pg.PlotWidget()
        self.graph.plotItem.setLabel(axis='bottom', text=xlabel)
        self.graph.plotItem.setLabel(axis='left', text=ylabel)
        
        
        self.curve = self.graph.plot()
        self.image = pg.ImageItem()
        self.cm = pg.colormap.get('viridis')
        self.image.setColorMap(self.cm)
        self.graph.addItem(self.image)
        
        self.left = QWidget()
        self.commands = QGridLayout()
        self.btnPlay = QPushButton('Play')
        self.btnPlay.clicked.connect(self.togglePlay)
        self.btnCopy = QPushButton('Copy to clipboard')
        self.btnCopy.clicked.connect(self.copyToClipboard)
        self.spinAvg = QSpinBox()
        self.progress = QProgressBar()
        self.progress.setMaximum(self.navg)
        self.spinAvg.setMinimum(1)
        self.spinAvg.setMaximum(1000)
        self.spinAvg.setValue(1)
        self.spinAvg.valueChanged.connect(self.setNavg)
        self.lblFps = QLabel('. fps')
        self.commands.addWidget(self.btnPlay, 0, 0)
        self.commands.addWidget(self.btnCopy, 1, 0)
        self.commands.addWidget(self.spinAvg, 2, 0)
        self.commands.addWidget(self.progress, 3, 0)
        self.btnYminus = QPushButton('y-')
        self.btnYplus = QPushButton('y+')
        self.btnYminus.clicked.connect(lambda: self.yShift(direction=-1))
        self.btnYplus.clicked.connect(lambda: self.yShift(direction=+1))
        self.spinYstep = ScientificSpinBox.PyScientificSpinBox()
        self.spinYstep.setValue(0.001)
        if (dim ==2 or self._wrap_mode) and fn_yshift is not None:
            self.commands.addWidget(self.btnYplus, 5, 0)
            self.commands.addWidget(self.spinYstep, 6, 0)
            self.commands.addWidget(self.btnYminus, 7, 0)
        
        self.commands.addWidget(self.lblFps, 8, 0)
        self.left.setLayout(self.commands)

        self.right = QWidget()
        self.right_layout = QGridLayout()
        self.right_layout.addWidget(self.graph, 0, 0, 1, 3)
        self.btnXminus = QPushButton('x-')
        self.btnXplus = QPushButton('x+')
        self.btnXminus.clicked.connect(lambda: self.xShift(direction=-1))
        self.btnXplus.clicked.connect(lambda: self.xShift(direction=+1))
        self.spinXstep = ScientificSpinBox.PyScientificSpinBox()
        self.spinXstep.setValue(0.001)
        if fn_xshift is not None:
            self.right_layout.addWidget(self.btnXminus, 1, 0)
            self.right_layout.addWidget(self.spinXstep, 1, 1)
            self.right_layout.addWidget(self.btnXplus, 1, 2)
        self.right.setLayout(self.right_layout)

        splitter.addWidget(self.left)
        splitter.addWidget(self.right)
        
        self.setCentralWidget(splitter)
        
        if show: self.show()
        if play: self.play()
        
        
        if window_size == 'wide':
            self.resize(1000, 500)
        elif window_size == 'wider':
            self.resize(1000, 200)
            
        if take_focus:
            while not self.vthread.pause:
                wait(2)
            return
        
    def closeEvent(self, event):
        self.stop()
        print("closed")
        self.close()
        event.accept()
    
    def _doAvg(self, data, store_in_buffer=True):
        # this is called by plot and imgplot.
        # so we average but also do some general stuff

        # buffer
        if store_in_buffer:
            # we get here after each completed frame.
            if self.pause_after_one:
                self.pause()
                        
            # fps
            self.frame_count += 1
            if self.frame_count == 1: # first frame
                self.t0 = time.time()
            else:
                self.lblFps.setText(str(round(self.frame_count / (time.time()-self.t0), 1))+' fps')
            
            
            self.data_buffer.append(data)
            if len(self.data_buffer) > self.navg:
                self.data_buffer = self.data_buffer[1:]
            self.progress.setValue(len(self.data_buffer))
        
            avg_data = np.nanmean(self.data_buffer, axis=0)
        
        else:
            #if len(self.data_buffer) < 1:
            #    avg_data = data
            #else:
            if len(self.data_buffer) < 1: # first map
                avg_data = data
                
            elif len(self.data_buffer) == self.navg:
                self.data_buffer[0] = np.where(np.isnan(data), self.data_buffer[0], data)
                avg_data = np.nanmean(self.data_buffer, axis=0)
                
            else:
                
                avg_data = np.nanmean(np.concatenate((self.data_buffer, [data])), axis=0)
        
        if len(self.data_buffer) == self.navg and self.pause_at_max_avg:
            self.stop()
        return avg_data


    def plot(self, data_1d):
        self.curve.setData(np.linspace(*self.x, len(data_1d)), self._doAvg(data_1d))
    
    def plotToImg(self, data_1d):
        """ called in wrapping_mode instead of 'plot' """
        
        # first call ever
        if self._wrap_single_img is None:
            self._wrap_single_img = np.empty((self._wrap_counter, len(data_1d)))
            self._wrap_single_img[:] = np.nan
        
        nb_line = len(self._wrap_single_img)
        i = self._wrap_counter % nb_line
        is_last_line = i == (nb_line-1) 
        self._wrap_counter = i+1
        
        self._wrap_single_img[i] = data_1d
        if is_last_line:
            img = self._wrap_single_img.copy().T if self._wrap_direction == 'h' else self._wrap_single_img.copy()
            self.imgplot(img, store_in_buffer=True)
            self._wrap_single_img = np.empty((self._wrap_counter, len(data_1d)))
            self._wrap_single_img[:] = np.nan
        else:
            img = self._wrap_single_img.T if self._wrap_direction == 'h' else self._wrap_single_img
            self.imgplot(img, store_in_buffer=False)
            
    def imgplot(self, data_2d, store_in_buffer=True):
        data_2d = self._doAvg(data_2d, store_in_buffer)
        self.image.setImage(data_2d)
        self.image.setRect(self.x[0],self.y[0],self.x[1]-self.x[0],self.y[1]-self.y[0]) # x,y,w,h

    def continousGet(self, get_fn, dim, sec_between_frame=1):
        """
        run a thread that periodically exec `get_fn`.
        use self.pause to pause.

        Parameters
        ----------
        get_fn : function
            a function that returns a 1d or 2d array, specified in `dim`.

        Returns
        -------
        None.

        """
        self.vthread = VideoThread(get_fn, wait_time=sec_between_frame)
        
        if self._wrap_mode:
            self.vthread.sig_frameDone.connect(self.plotToImg)
        elif dim == 1:
            self.vthread.sig_frameDone.connect(self.plot)
        elif dim == 2:
            self.vthread.sig_frameDone.connect(self.imgplot)
        self.vthread.start()
    
    def xShift(self, direction: int):
        # direction: +1|-1
        shift = direction*self.spinXstep.value()
        self.x[0] = self.x[0] + shift
        self.x[1] = self.x[1] + shift
        if self.fn_xshift is not None:
            self.fn_xshift(shift)
    
    def yShift(self, direction: int):
        # direction: +1|-1
        shift = direction*self.spinYstep.value()
        self.y[0] = self.y[0] + shift
        self.y[1] = self.y[1] + shift
        if self.fn_yshift is not None:
            self.fn_yshift(shift)
    
    
    def setNavg(self, val):
        self.data_buffer = self.data_buffer[len(self.data_buffer)-val:]
        self.navg = val
        self.progress.setMaximum(val)
    
    def play(self): 
        self.vthread.pause = False
        self.btnPlay.setText('Pause')
    def pause(self): 
        self.vthread.pause = True
        self.btnPlay.setText('Play')
    def togglePlay(self):
        self.play() if self.vthread.pause else self.pause()
        
    def stop(self): 
        self.vthread.terminate()
        self.btnPlay.setText('Stopped')
        self.btnPlay.setDisabled(True)
    
    def copyToClipboard(self):
        clipboard = QApplication([]).clipboard()
        exp = pg.exporters.ImageExporter(self.graph.scene())
        exp.export(copy=True)
        clipboard.setImage(exp.png)

class VideoThread(QThread):
    sig_frameDone = pyqtSignal(np.ndarray)
    
    def __init__(self, fn_get_data, wait_time=1):
        super().__init__()
        self.get = fn_get_data
        self.wait_time = wait_time
        
        self.pause = True
        
    def run(self):
        import time
        while True:
            if not self.pause:
                try:
                    data = self.get()
                    if data is not None:
                        self.sig_frameDone.emit(data)
                except Exception as exc:
                    print(traceback.format_exc())
                    print(exc)
                time.sleep(self.wait_time)
            else:
                time.sleep(1)
    
class SweepAxis:
    """ we usually want to sweep an axis in video mode 1d wrapping.
    This class is to make it less verbose.
    args:
        val_list: list of values to sweep
        fn_sweep: function called with the 'val' arg from the val_list at each iteration
    
    example:
        
    ```

    p2_sweep = SweepAxis(np.linspace(1.052, 1.058, 101), fn_next=rhP2.set, label='rhP2 level')
    
    def vmget():
        p2_sweep.next()
        data = ... meas ...
        return data
    
    vm = VideoModeWindow(fn_get=vmget, dim=1, wrap_at=len(y_read_dict['list']),
                         axes_dict={'y':p2_sweep.axis},
                         ylabel=p2_sweep.label,
                         fn_yshift=p2_sweep.shift)

    ```
    """
    
    def __init__(self, val_list, fn_next = lambda val: print(val), label='sweep', enable=True):
        self.current_index = 0
        self.val_list = val_list
        self.axis = [min(val_list), max(val_list)]
        self.label = label
        self.fn_next = fn_next
        self.enable = enable
        if not enable:
            self.axis = [0, len(val_list)]
            self.label = "count"

    def next(self):
        if not self.enable: return
        self.fn_next(self.val_list[self.current_index])
        self.current_index = (self.current_index + 1) % len(self.val_list)
    
    def shift(self, step):
        if not self.enable: return
        self.val_list = np.array(self.val_list) + step
    
    def __len__(self):
        return len(self.val_list)