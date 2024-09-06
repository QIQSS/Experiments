import numpy as np
from matplotlib import pyplot as plt


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
    if len(wave) < 2048:
        print(f"Probably not enough points: {len(wave)}")
    awg.waveform_create(wave, wv_name, sample_rate=awg_sr, amplitude=2*wave_max_val*gain, force=True)
    awg.waveform_marker_data.set(marks, wfname=wv_name)
    awg.channel_waveform.set(wv_name,ch=channel)
    amp = 2*wave_max_val*gain
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


    if amp > 1.5:
        print(f"Warning: volt amplitude with gain above 1.5V: {amp}V")
        return True

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

