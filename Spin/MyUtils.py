import time as timemodule
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from pyHegel import commands
from pyHegel import fitting, fit_functions
from pyHegel.types import dict_improved
from pyHegel.instruments_base import BaseInstrument

from Pulses.Builder import *
from Analyse.analyse import *

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import io

import threading
#### INSTRUMENTS ####

    
#ats = instruments.ATSBoard(systemId=1, cardId=1)
#ats_conf = dict(sample_rate=10e5,
#                input_range=4000) # vertical resolution in mV
def configureATS(ats, input_range=4000, sample_rate=10e5, verbose=False):
    """ apply the default configuration
    """
    if not ats: pass
    ats.active_channels.set(['A','B']) # read on A and B
    ats.buffer_count.set(4) # usually 4
    ats.clock_type.set('INT')
    ats.trigger_delay.set(0) # waiting time after trigger
    ats.trigger_channel_1.set('ext') # channel A trigger on external channel
    ats.trigger_channel_2.set('ext') # channel B trigger on external channel
    ats.trigger_slope_1.set('ascend')
    ats.trigger_slope_2.set('ascend')
    ats.sample_rate.set(sample_rate)
    
    ats.input_range.set(input_range) # vertical resolution in mV [20, 40, 50, 80, 100, 125, 200, 250, 400, 500, 800, 1000, 2000, 2500, 4000, 5000, 8000, 1e4, 2e4, 1.6e4, 2e4, 4e4]
    ats.sample_rate.set(sample_rate)
    
    ats.ConfigureBoard() # apply changes

def getATSImage(ats, with_time=False):
    """ returns an array of shape (nbwindows, samples_per_record)
        ret = [[trace0], [trace1], ..., [tracen]]
    If with_time is True:
        return is [[trace0], [trace1], ..., [tracen]], [t0, t1, ..., tm]
        with t0, ..., tm the timestamp for ONE trace.
    """
    indexes, times, traces = ats.fetch_all.get() #2D array of size (nbwindows+1, samples_per_record)
    nb_windows = ats.nbwindows.get()
    ret = traces.reshape((nb_windows, -1)) # from [trace0, ..., tracen] to [[trace0], ..., [tracen]]
    if with_time:
        timelist = times[:len(ret[0])]
        return ret, timelist
    return ret

#awg = instruments.tektronix.tektronix_AWG('USB0::0x0699::0x0503::B030793::0')
def sendSeqToAWG(awg, sequence, gain=None, channel=1, awg_sr=32e4, 
                 wv_name='waveform', plot=False, run_after=True, close_channel=None,
                 round_nbpts_to_mod64=None):
    """ Stop the awg then send the sequence (object from Pulse code) to the awg.
    gain can be None and will be set to awg.gain if it exists or this value if not: 1/(0.02512)*0.4
    If run_after: it play the wave after sending it.
    If close_channel = 1 (2), close the channel 1 (2) before sending the wave.
    nbpts_mod64: 'last' | 'zeros' | num, pad wave to be a multiple of 64.
    """
    wv_name += '_' + str(channel)
    wave = sequence.getWaveNormalized(awg_sr)
    wave_max_val = max(abs(sequence.getWave(awg_sr)))
    marks = sequence.getMarks(awg_sr, val_low=1, val_high=-1)
    
    if round_nbpts_to_mod64:
        padding_val = {'zeros':0, 'last':wave[-1]}.get(round_nbpts_to_mod64, round_nbpts_to_mod64)
        
        nb_padding_points = 64 - (len(wave) % 64)
        wave = np.concatenate((wave, np.ones(nb_padding_points)*padding_val))
        marks = np.concatenate((marks, np.ones(nb_padding_points)*marks[-1]))
    
    if gain is None:
        gain = getattr(awg, 'gain', None)
    if gain is None:
        gain = 1/(0.02512)*0.4
    
    awg.run(False)
    awg.waveform_create(wave, wv_name, sample_rate=awg_sr, amplitude=2*wave_max_val*gain, force=True)
    awg.waveform_marker_data.set(marks, wfname=wv_name)
    awg.channel_waveform.set(wv_name,ch=channel)
    amp = 2*wave_max_val*gain
    if amp > 0.750:
        print(f"Warning: need a volt amplitude with gain above 750mV: {amp}V")
    awg.volt_ampl.set(amp, ch=channel)

        
    awg.sample_rate.set(awg_sr)
    awg.current_channel.set(channel)
    awg.current_wfname.set(wv_name)

    
    if run_after: awg.run(True)
    
    if close_channel:
        awg.current_channel.set(close_channel)
        awg.output_en.set(False)
        
    if plot:
        plt.figure()
        plt.plot(wave)
        plt.plot(marks)
        plt.title(wv_name)


#### DOT ####

def manualSweep(dev_sw, points, out_function, plot=False):
    """ does a manual sweep. Sets a point, call out_function(), append result in a list, returns the list. """
    out = []
    for point in points:
        dev_sw.set(point)
        out.append(out_function())
    if plot:
        plt.plot(points, out)
    return np.array(out)

def _exp(x, tau, a=1., b=0., c=0.):
    return a*np.exp(-(x+b)/tau)+c

def autoFindTunnelRate(ats, awg, gain, threshold, opposite_offset=False, plot=False, verbose=True, fit_skip_firsts_point=0):
    """ TESTING
    assuming P1 is set exactly on the transition:
    sends a load/empty pulse, digitalize and average the result, fit an exponential on the unload time.
    opposite_offset: tells the pulse to be up/down or down/up
    """
    offsets = np.array([0.002, -0.002])
    offsets = offsets*-1 if opposite_offset else offsets
    load = Segment(duration=0.0002, waveform=Ramp(val_start=0.00, val_end=0.000), offset=offsets[0])
    empty = Segment(duration=0.0002, waveform=Ramp(val_start=0.000, val_end=0.000), offset=offsets[1], mark=(0.0, 1.0))
    pulse = Pulse(load, empty)
    
    configureATS(ats)
    ats.acquisition_length_sec.set(empty.duration)
    ats.nbwindows.set(1000)
    
    sendSeqToAWG(awg, pulse, gain, channel=1, run_after=True)
    trace, timelist = acquire(ats, return_average_trace=True, digitize=True, threshold=threshold, show_plot=plot)
    awg.run(False)
    
    timelist, trace = timelist[fit_skip_firsts_point:], trace[fit_skip_firsts_point:]
    
    if plot:
        plt.plot(timelist, trace)
        fitting.fitplot(_exp, timelist, trace, p0=[0.0001, 1., 0., 0.])
        
    fit_result = fitting.fitcurve(_exp, timelist, trace, p0=[0.0001])
    tunnel_rate = 1/fit_result[0][0]
    if verbose:
        print(fit_result)
        print('tunnel rate: ' + str(tunnel_rate))
    return tunnel_rate

#### SIGNAL FILTERING ####

def _doubleGaussian(x, sigma1, sigma2, mu1=0., mu2=0., A1=3.5, A2=3.5):
    """ use for fitting state repartition
    sigma: curvature =1:sharp, =15:very flatten
    mu: center
    A: height
    """
    g1 = fit_functions.gaussian(x, sigma1, mu1, A1)
    g2 = fit_functions.gaussian(x, sigma2, mu2, A2)
    return g1+g2

def estimateDigitThreshold(image, p0=[7, 10, 25, 62, 3.5, 3.5], bins=100, show_plot=True, verbose=True):
    # 1 prepare data
    samples = image.flatten()
    hist, bins = np.histogram(samples, bins=bins, density=True)
    x = np.linspace(0, len(hist)-1, len(hist))
    
    # 2 do the fit
    fit_result = fitting.fitcurve(_doubleGaussian, x, hist, p0)
    fit_curve = _doubleGaussian(x, *fit_result[0])
    
    # 3 find threshold index and value
    A1, A2 = fit_result[0][2], fit_result[0][3]
    threshold_ind = np.argmin(fit_curve[int(A1):int(A2)]) # find the min between the two peaks
    threshold_ind += int(A1) # "recenter" the treshold
    threshold_val = bins[threshold_ind]
    
    # 4 show print return
    if show_plot:
        plt.figure()
        plt.plot(bins[:len(hist)], hist)
        plt.plot(bins[:len(fit_curve)], fit_curve)
        plt.axvline(x=threshold_val, color='r', linestyle=':', label='threshold: '+str(threshold_val))
        plt.legend()
    if verbose:
        print('Threshold found at x='+str(threshold_val))
    return threshold_val

def classifiyArray(image, threshold):
    """ return the image with values 0 for below TH and 1 for above TH
    """
    bool_image = image>threshold
    int_image = np.array(bool_image, dtype=int)
    return int_image





#### SIGNAL ANALYSIS ####


def countEvents(data_digit, time, one_out_only=False):
    """ count the number of event in a given a 2d array:
        data_digit = [[trace0], [trace1], .... ] with traces: list of 0/1
    trace_time: timestamp for one trace; same length as a trace.
    return of the form event_out_avg, event_in_avg, count_exclude
    event_out_avg is the average number of event out per trace
    event_in_avg is the average number of event in per trace
    if one_out_only:
        we exclude the traces with more than one event out.
    """
    # 1. count events
    count_event_out = 0
    count_event_in = 0
    count_exclude = 0
    
    base_value = round(np.mean(data_digit[:,5])) # find the value for the "loaded" state
    
    for trace in data_digit: # trace by trace
        events = np.diff(trace)
        downs = np.where(events == -1)[0] # array of all the down events positions in the trace
        ups = np.where(events == 1)[0] # array of all the up events potisions in the trace
        nb_down = len(downs)
        nb_up = len(ups)

        if base_value==0:   # first point is 0 -> event_out==up and event_in==down
            events_out = ups
            events_in = downs
        elif base_value==1: # first point is 1 -> event_out==down and event_in==up
            events_out = downs
            events_in = ups
        
        if one_out_only and len(events_out) > 1:
            count_exclude += 1
            continue # stop current for loop iteration; go to the next trace
        else:
            count_event_out += len(events_out)
            count_event_in += len(events_in)

    # 2. make stats
    event_out_avg = count_event_out/int(data_digit.shape[0]-count_exclude)
    event_in_avg = count_event_in/int(data_digit.shape[0]-count_exclude)

    return event_out_avg, event_in_avg, count_exclude


def classifyTraces(data_digit, time, return_stats=True):
    """ return the number of trace with and without a blip
    almost the same as countEvents
    
    decision making:
        00011100000 -> blip: keep
        00000000000 -> no blip: keep
        00011111111 -> exclude
        00011100011 -> exclude
        
    """
    base_value = round(np.mean(data_digit[:,5])) # infer the value for the "loaded" (first) state
    if base_value == 0:
        data_digit = np.where(data_digit==0, 1, 0) # invert the map so the loaded state is 1
    
    exclude_traces = []
    blip_traces = []
    no_blip_traces = []
    
    for trace in data_digit:
        events = np.diff(trace)
        out_list = np.where(events == -1)[0] # array of all the out events positions in the trace
        in_list = np.where(events == 1)[0] # array of all the out events positions in the trace
        
        if trace[0] == 0: # empty
            exclude_traces.append(trace)
        
        elif len(out_list)==1 and len(in_list)==1:
            blip_traces.append(trace)
        
        elif len(out_list)==0 and len(in_list)==0:
            no_blip_traces.append(trace)
        
        else:
            exclude_traces.append(trace)

    if not return_stats:
        return blip_traces, no_blip_traces, exclude_traces
    
    # stats:
    avg_blip_trace = len(blip_traces) / (len(data_digit)-len(exclude_traces))
    avg_no_blip_trace = len(no_blip_traces) / (len(data_digit)-len(exclude_traces))
    
    return {'avg_spin_up': avg_blip_trace, 'avg_spin_down': avg_no_blip_trace, 'nb_exclude':len(exclude_traces)}

def histogram(arr, bins=100, get_bins=False, **kwargs):
    hist, bin_list = np.histogram(arr, bins, **kwargs)
    return hist if not get_bins else hist, bin_list

def histogramOnEachColumns(arr, bins=100, get_bins=False):
    im = []
    _, bin_list = np.histogram(arr.flatten(), bins=bins)
    
    for i in range(arr.shape[1]):
        col = arr[::,i]
        hist, _ = np.histogram(col, bins=bin_list, density=True)
        im.append(hist)
    if get_bins:
        return np.array(im).T, bin_list
    return np.array(im).T


### USEFUL THINGS ###

def threadWaitThenRun(my_function, wait_time=.1):
    """ return: a thread object. Call the method .start() on it to start the wait_time. At the end of wait_time, my_function is run."""
    def threadFn(wtime=.1):
        commands.wait(wtime)
        fn()
    thread = threading.Thread(target=my_function)
    return thread


#### FILE SAVING/LOADING ####

def read2fileAndPlotDiff(file1, file2, filtre=lambda arr: arr):
    """ for 1 dimensional sweep, assuming the swept values are the same. """
    rf1 = commands.readfile(file1)
    rf2 = commands.readfile(file2)
    sw1, data1 = rf1[0], rf1[1]
    sw2, data2 = rf2[0], rf2[1]
    
    data1, data2 = filtre(data1), filtre(data2)
    
    delta = np.abs(data1 - data2)
    delta_max = np.argmax(delta)
    value_max = sw1[delta_max]
    print(f"Max delta at: {value_max}")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.plot(sw1, data1, label=f"file1")
    ax1.plot(sw1, data2, label=f"file2")
    ax2.plot(sw1, delta)
    ax1.axvline(x=value_max, color='r', linestyle=':', label='Delta max at '+str(value_max))
    ax2.axvline(x=value_max, color='r', linestyle=':', label='Delta max at '+str(value_max))
    ax1.legend()
    return value_max

def readfileNdim(file):
    """ a memo of pyHegel.readfile for sweep with N dimensions 
    for i in range(10):
        imshow(data[3][i].T, x_axis=data[2,1,0], y_axis=data[1,1][::,1], x_label='P2', y_label='P1', title=f"B1={round(data[0][i][0][0], 3)}")
    """
    
    data, titles, headers = commands.readfile(file, getheaders=True, multi_sweep='force', multi_force_def=np.nan)
    return data, titles, headers

def showfile2dim(data, x_label='', y_label='', title='', cbar=False,
                 is_alternate=False, transpose=False, deinterlace=False,
                 out_id=2):
    """
    data is the result of readfileNdim[0]
    titles is the result of readfileNdim[1]
    transpose: False | True (data and axes)
    """
    if is_alternate:
        img = alternate(data[out_id]).T
    else:
        img = data[out_id].T
        
    if transpose != False:
        x_label, y_label = y_label, x_label
        img = img.T
    
    imshow_kw = dict(x_axis=data[0][::,1], y_axis=data[1,0], 
                     x_label=x_label, y_label=y_label, title=title, cbar=cbar)

    if deinterlace:
        img1 = img.T[0::2, :].T
        img2 = img.T[1::2, :].T
        imshow(img1, **imshow_kw)
        imshow(img2, **imshow_kw)
        return        
    imshow(img, **imshow_kw)

def getFirstAxisList(data3d):
    return [(i, round(image_i[0][0], 4)) for i, image_i in enumerate(data3d[0])]

def showfile3dim(data, first_axis_label='', x_label='', y_label='', cbar=False,
                 is_alternate=False, transpose=False, deinterlace=False,
                 first_axis_ids=[]):
    """ take the output of a readfile data for a 3d sweep, plot an image for each first axis values
    handles 'alternate' sweep.
    first_axis_ids is a list of ids instead of plotting for each first axis values.
    deinterlace: plot two figure per 2d sweep,
    """
    first_axis_list = getFirstAxisList(data)
        
    iterator = range(len(data[0])) if len(first_axis_ids) == 0 else first_axis_ids
    for i in iterator:
        
        
        if is_alternate:
            y_axis = flip(data[1,1][::,1])
            img = alternate(data[3][i])
            if i%2==1:
                img = flip(img.T, axis=-1).T
        else:
            y_axis = data[1,1][::,1]
            img = data[3][i]
        
        imshow_kw = dict(x_axis=data[2,1,0] if transpose else y_axis,
                         y_axis=y_axis if transpose else data[2,1,0], 
                         x_label=x_label if not transpose else y_label, 
                         y_label=y_label if not transpose else x_label,
                         cbar=cbar)

        if deinterlace:
            img1 = img[0::2, :]
            img2 = img[1::2, :]
            imshow(img1 if transpose else img1.T, title=f"{first_axis_label}={first_axis_list[i][1]}, paires", **imshow_kw)
            imshow(img2 if transpose else img2.T, title=f"{first_axis_label}={first_axis_list[i][1]}, impaires", **imshow_kw)
            continue
            
        imshow(img if transpose else img.T, title=f"B1=0.35, B2={first_axis_list[i][1]}", **imshow_kw )

def saveToNpz(path, filename, array, metadata={}):
    """ Save array to an npz file.
    metadata is a dictionnary, it can have pyHegel instruments as values: the iprint will be saved.
    """
    if not path.endswith(('/', '\\')):
        path += '/'
    timestamp = timemodule.strftime('%Y%m%d-%H%M%S-')
    fullname = path + timestamp + filename
    
    # formating metadata
    for key, val in metadata.items():
        if isinstance(val, BaseInstrument): # is pyHegel instrument
            metadata[key] = val.iprint()
    metadata['_filename'] = timestamp+filename
    
    # saving zip
    np.savez(fullname, array=array, metadata=metadata)
    
    print('Saved file to: ' + fullname)
    return fullname+'.npz'
    

def loadNpz(name):
    """ Returns a dictionnary build from the npzfile.
    if saveNpz was used, the return should be:
        {'array': array(),
         'metadata': {}}
    """
    if not name.endswith('.npz'):
        name += '.npz'
    npzfile = np.load(name, allow_pickle=True)
    ret =  {}
    for key in npzfile:
        obj = npzfile[key]
        try:
            python_obj = obj.item()
            ret[key] = python_obj
        except ValueError:
            ret[key] = obj
    return ret

def imshowFromNpz(filename, return_dict=False, **kwargs):
    """ if the Npz was saved with imshow(*args, **kwargs, save=True),
    it will load the file and call imshow(array, **kwargs)
    """
    npzdict = loadNpz(filename)
    array = npzdict.get('array')
    imshow(array, **npzdict.get('metadata', {}).get('imshow_kwargs', {}))
    if return_dict:
        return npzdict
    
def imshow(array, **kwargs):
    """ my custom imshow function.
    with easier axis extent: x_axis=, y_axis=.
    and saving to npz with all kwargs
    """
    kwargs['interpolation'] = 'none'
    kwargs['aspect'] = 'auto'
    kwargs['origin'] = 'lower'
    
    # save array and kwargs
    path = kwargs.pop('path', './')
    filename = kwargs.pop('filename', '')
    metadata = kwargs.pop('metadata', {})
    metadata['imshow_kwargs'] = kwargs
    if kwargs.pop('save', False):
        saveToNpz(path, filename, array, metadata=metadata)
        
        
    # AXES: [start, stop] or just 'stop' (will be converted to [0, stop])
    def _prepAxis(lbl):
        axis = kwargs.pop(lbl, [None])
        if isinstance(axis, (int, float)):
            axis = [0, axis] if axis > 0 else [axis, 0]
        return axis
    x_axis = _prepAxis('x_axis')
    y_axis = _prepAxis('y_axis')
    if None not in x_axis and None in y_axis:
        y_axis = [0, len(array)]
        
    extent = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
    if None not in extent:  
        kwargs['extent'] = extent
    
    x_axis2 = _prepAxis('x_axis2')
    x_label2 = kwargs.pop('x_label2', None)
        
    x_label = kwargs.pop('x_label', None)
    y_label = kwargs.pop('y_label', None)
    title = kwargs.pop('title', '')
    cbar = kwargs.pop('cbar', True)
    
    # PLOT
    fig, ax = plt.subplots()
    im = ax.imshow(array, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if cbar: fig.colorbar(im, ax=ax)
        
    # Create secondary x-axis
    if x_axis2 != [None]:
        ax2 = ax.twiny()
        ax2.set_xlim(*x_axis2)
        ax2.set_xlabel(x_label2)

    # TODO save this to metadata when save=True
    def figToClipboard():
        with io.BytesIO() as buffer:
             fig.savefig(buffer)
             QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))
    def onKeyPress(event):
        if event.key == "ctrl+c":
            figToClipboard()

    fig.canvas.mpl_connect('key_press_event', onKeyPress)
    fig.tight_layout() 
    fig.show()

def qplot(x, y=None, x_label='', y_label='', title='', same_fig=False):
    """ quick 1d plot """
    if not same_fig:
        plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)




def plotColumns(array, interval, x_axis=None, y_axis=None, x_label='', y_label='', title='', 
                z_label='', reverse=False, cbar=False):
    """chatgpt
    Plots every 'interval'-th column of a 2D array with a color gradient.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    cmap = cm.get_cmap('viridis')
    num_columns = (array.shape[1] - 1) // interval + 1
    
    if reverse:
        column_indices = range(array.shape[1] - 1, -1, -interval)
    else:
        column_indices = range(0, array.shape[1], interval)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a ScalarMappable object for the colorbar
    norm = mcolors.Normalize(vmin=0, vmax=num_columns - 1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    
    colors = []

    # Plot each selected column with a color from the color map
    for i, idx in enumerate(column_indices):
        color = cmap(i / num_columns)  # Normalize index for color map
        colors.append(color)
        if x_axis is None:
            x_values = range(array.shape[0])
        else:
            x_values = x_axis
        
        if y_axis is None:
            y_values = array[:, idx]
        else:
            y_values = y_axis[:, idx]  # Assuming y_axis has the same shape as array
        
        ax.plot(x_values, y_values, color=color, label=f'Column {idx}')

    # Add colorbar if required
    if cbar:
        fig.colorbar(sm, ax=ax, label=z_label)

    # Set axis labels and title
    ax.set_xlabel(x_label if x_label else 'Index')
    ax.set_ylabel(y_label if y_label else 'Value')
    ax.set_title(title)
    #ax.legend()

    plt.show()
