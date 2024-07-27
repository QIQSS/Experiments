import time as timemodule
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from pyHegel import commands
from pyHegel import fitting, fit_functions
from pyHegel.types import dict_improved
from pyHegel.instruments_base import BaseInstrument

from Pulses.Builder import *

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
    awg.volt_ampl.set(2*wave_max_val*gain, ch=channel)

        
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

def acquire(ats, digitize=False, threshold=0., sigma=1, return_average_trace=False, show_plot=False):
    """ do the measurment sequence:
        ats get image and timelist
        filter and digitize if necessary
        return either full image or average trace
    """

    image, timelist = getATSImage(ats, with_time=True) # [[trace0], [trace1], ..., [traceNWindow]], [t0, t1, ..., t_SR*AcquisitionLength]    

    if digitize: 
        image = gaussianLineByLine(image, sigma=sigma)
        image = digitizeArray(image, threshold)
    
    if return_average_trace:
        average_trace = averageLines(image)
        if show_plot:
            plt.plot(timelist, average_trace)
        return average_trace, timelist
    if show_plot:
        imshow(image, x_axis=timelist)
    return image, timelist


#### DOT ####

def manualSweep(dev_sw, points, out_function, plot=False):
    """ does a manual sweep. Sets a point, call out_function(), append result in a list. Returns the list. """
    out = []
    for point in points:
        dev_sw.set(point)
        out.append(out_function())
    if plot:
        plt.plot(points, out)
    return np.array(out)

def autoFindBestST(ST, st_points, P1, p1_trans_val, zi_lambda, offset=0.005, show_plot=True):
    """ Does a sweep of ST for P1=p1_trans_val+offset and -offset.
    The ST sweep values are: ST.get()-offset ->ST.get()+offset
    Then returns the optimal value for ST
    # %% Find best ST example
    st_points = np.linspace(3.07, 3.09, 101)
    p1_trans_val = 0.826
    st_val = autoFindBestST(ST, st_points, P1, p1_trans_val, zi_get, offset=0.005, show_plot=True)

    """
    # my things
    # 1 sweep P1 pour trouver la transition
    p1_1, p1_2 = p1_trans_val + offset, p1_trans_val - offset
    P1.set(p1_1)
    print(f"sweeping ST for P1={p1_1}")
    ST_1 = manualSweep(ST, st_points, zi_lambda)
    P1.set(p1_2)
    print(f"sweeping ST for P1={p1_2}")
    ST_2 = manualSweep(ST, st_points, zi_lambda)
    delta_ST = np.abs(ST_1 - ST_2)
    delta_max = np.argmax(delta_ST)
    ST_max = st_points[delta_max]
    print(f"Found the optimal ST to be: {ST_max}")
    
    if show_plot:
        plt.figure()
        plt.plot(st_points, ST_1)
        plt.plot(st_points, ST_2)
        plt.plot(st_points, delta_ST)
        plt.axvline(x=ST_max, color='r', linestyle=':', label='ST optimal: '+str(ST_max))
        plt.legend()
    return ST_max

def _exp(x, tau, a=1., b=0., c=0.):
    return a*np.exp(-(x+b)/tau)+c

def autoFindTunnelRate(ats, awg, gain, threshold, opposite_offset=False, plot=False, verbose=True, fit_skip_firsts_point=0):
    """ assuming P1 is set exactly on the transition:
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

def gaussianLineByLine(image, sigma=20, **kwargs): 
    try:
        dim2 = len(image[0])
        return np.array([ndimage.gaussian_filter1d(line, sigma, **kwargs) for line in image])
    except:
        return ndimage.gaussian_filter1d(image, sigma, **kwargs)

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

def digitizeArray(image, threshold):
    """ return the image with values 0 for below TH and 1 for above TH
    """
    bool_image = image>threshold
    int_image = np.array(bool_image, dtype=int)
    return int_image

def averageLines(image):
    """ from [[trace1], ..., [tracen]] to [trace_mean].
        from 2d to 1d.
    example:
        arr = np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
        
        arr.mean(axis=0)
        Out[]: array([2., 2., 2., 2., 2.]) <-- GOOD
        
        arr.mean(axis=1)
        Out[]: array([1., 2., 3.])
    """
    image = np.array(image)
    return image.mean(axis=0)







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


#### Fast sweep ####
    
def fastSweepPulseInit(x_step_time, y_range, awg=None, gain=None):   
    """ awg gets y wave on channel 1, x wave on channel 2.
    triggers are on both channels.
    y wave is a ramp of magnitude +-(y_range/2)
    x wave is the rf part of the staircase
    
    The bilt command is set to the start value, and the awg is started
    """

    trig_pulse = Pulse(Segment(duration=x_step_time, offset=0, mark=(0,0.5)))
    ramp_pulse = Pulse(Segment(duration=x_step_time/2, waveform=Ramp(0, +y_range/2,)),
                       Segment(duration=x_step_time/2, waveform=Ramp(-y_range/2, 0), mark=(0,0.5)))
    sendSeqToAWG(awg, ramp_pulse, gain, channel=1, run_after=False)
    sendSeqToAWG(awg, ramp_pulse, gain, channel=2, run_after=False)

    awg.run(True)

def stairTo(val, bi=None, nb_step=100):
    """ assuming the bilt is in STEP mode
    step the bi.current_channel to val and
    set the amplitude so that it will take nb_step to reach val
    """
    #nb_trig = abs(bi.level.get()-val)/bi.step_amplitude.get()
    step_amplitude = abs(bi.level.getcache()-val) / nb_step
    if step_amplitude != 0:
        bi.step_amplitude.set(step_amplitude)
    print(f"step amp set to: {step_amplitude}. Reaching {val} will take {nb_step} steps.")
    bi.level.set(val, trig=False)



def fastSweepStart(x_stop, x_step_time, y_range, x_nbpts=100, bi=None, ats=None, awg=None, run_fn=None, plot=False):
    """ 
    sweep from bilt current value to x_stop with x_nbpts points.
    
    It's made that way since ats.nbwindows must be a squared for some reason, so we adjust step_amplitude so the sweep is always nb_trig long.
    
    # TODO: the awg cannot be run after the command `getATSImage` so it must be start manually for now until a solution is found.
    """
    current_val = bi.level.get()
    step_amplitude = abs(current_val - x_stop)/x_nbpts
    if step_amplitude == 0:
        print('Already at val')
        return
    print(f"amplitude for {x_nbpts} points: ", step_amplitude)
    bi.step_amplitude.set(step_amplitude)
    ats.nbwindows.set(x_nbpts)

    ats.acquisition_length_sec.set(x_step_time*0.98)
    
    awg.run(False)
    bi.level.set(x_stop, trig=False)
    
    def startAWGAfter(awg, wtime=.1):
        commands.wait(wtime)
        awg.run(True)
    awgThreadRun = threading.Thread(target=startAWGAfter, args=(awg,))
    awgThreadRun.start()

    t0 = timemodule.time()
    out, time = getATSImage(ats, with_time=True)
    print(timemodule.time() - t0)
    out=out.T
    stair_axis = np.linspace(current_val, x_stop, x_nbpts)
    
    if plot:
        y_axis = np.linspace(-y_range/2, y_range/2, len(out[0])) if y_range!=0 else [None]
        imshow(out, x_axis=stair_axis, y_axis=y_axis, x_label=f"P2 (step: {x_step_time}s/{round(step_amplitude,6)}V)", y_label='delta P1')
    
    return out, stair_axis

def threadWaitThenRun(my_function, wait_time=.1):
    """ return: a thread object. Call the method .start() on it to start the wait_time. At the end of wait_time, my_function is run."""
    def threadFn(wtime=.1):
        commands.wait(wtime)
        fn()
    thread = threading.Thread(target=my_function)
    return thread

#### FILE SAVING/LOADING ####
def saveNpz(*arg, **kwargs):
    saveToNpz(*arg, **kwargs)
    
def saveToNpz(path, filename, array, x_axis=None, y_axis=None, metadata={}):
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
    np.savez(fullname, array=array, x_axis=x_axis, y_axis=y_axis, metadata=metadata)
    
    print('Saved file to: ' + fullname)
    return fullname+'.npz'
    

def loadNpz(name):
    """ Returns a dictionnary build from the npzfile.
    if saveNpz was used, the return should be:
        {'array': array(),
         'x_axis': array() or None,
         'y_axis': array() or None,
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
    npzdict = loadNpz(filename)
    array = npzdict.get('array')
    x_axis = npzdict.get('x_axis', [None])
    y_axis = npzdict.get('y_axis', [None])
    imshow(array, x_axis=x_axis, y_axis=y_axis, **kwargs)
    if return_dict: return npzdict
    
def imshow(*args, **kwargs):
    """ my custom imshow function.
    with easier axis extent: x_axis=, y_axis=.
    and side by side mode with args= im1, im2
    """
    kwargs['interpolation'] = 'none'
    kwargs['aspect'] = 'auto'
    kwargs['origin'] = 'lower'
    
    # axes
    x_axis = kwargs.pop('x_axis', [None])
    if x_axis is None: x_axis = [None]
    y_axis = kwargs.pop('y_axis', [None])
    if y_axis is None: y_axis = [None]
    if None not in x_axis and None in y_axis:
        y_axis = [0, len(args[0])]
    extent = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
    if None not in extent:  
        kwargs['extent'] = extent
    x_label = kwargs.pop('x_label', None)
    y_label = kwargs.pop('y_label', None)
    
    # multi-image
    if len(args) == 2:
        f, axarr = plt.subplots(1,2) 
        axarr[0].imshow(args[0], **kwargs)
        axarr[1].imshow(args[1], **kwargs)
    
    else:
        fig = plt.figure()
        im = plt.imshow(*args, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        fig.colorbar(im)
        fig.show()

def qplot(x, y=None, xlabel='', ylabel='', title='', same_fig=False):
    """ quick plot """
    if not same_fig:
        plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    