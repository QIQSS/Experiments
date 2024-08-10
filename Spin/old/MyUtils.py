import numpy as np
from matplotlib import pyplot as plt
from Pulses.Builder import *
from Analyse.analyse import *



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

