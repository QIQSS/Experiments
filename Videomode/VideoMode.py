"""
collection of common use case of VideoModeWindow for faster use
"""



def VMRampSet(dev_ramp, val_dev_ramp, delta_ramp, time,
              dev_set, val_dev_set, delta_set, nbpts
              awg, awg_ch_ramp,
              ats,
              get_fn,
              dev_ramp_name='Ramp', dev_set_name='Set',
              ramp_direction: str = 'up',
              vm_direction: str = 'v',
              sync_time_after_ramp=0.015
              )
    """ make and send a pulse to init a video mode with set/ramp
    ramp_direction: str = 'up' | 'down'
    vm_direction: str = 'v' | 'h'
    """

    # pulse making
    ramp_sign = {'up':+1, 'down':-1}[ramp_direction]
    ramp_delta = float(delta_ramp)/2.
    p_ramp = Pulse(dev_ramp_name)
    p_ramp.add(time)
    p_ramp.add(time, waveform=Ramp(val_dev_ramp-(-ramp_sign*ramp_delta),
                                   val_dev_ramp+(ramp_sign*ramp_delta)), mark=True)
    p_ramp.add(sync_time_after_ramp)
    p_set = p_ramp.genMarksOnly(name=dev_set_name)

    # send to awg
    sendSeqToAWG(awg, p_ramp, channel=awg_ch_ramp, run_after=False, round_nbpts_to_mod64='last')
    sendSeqToAWG(awg, p_set, channel={1:2, 2:1}[awg_ch_ramp], run_after=False, round_nbpts_to_mod64='last')
    sendSeqToAWG(awg, p_ramp, channel=3, run_after=True, round_nbpts_to_mod64='last')

    # config ats
    ats.nbwindow.set(1)
    ats.acquisition_length_sec.set(time)

    # sweep axis
    set_values = np.linspace(val_dev_set-(delta_set/2.), val_dev_set+(delta_set/2.), nbpts_set)
    sweep_set = SweepAxis(set_values, fn_next=dev_set._vm_sweep_fn_next, label=dev_set_name, enable=True)

    # set dev
    dev_ramp.set(val_dev_ramp)
    dev_set.set(set_values[0])

    # vm
    def get():
        sweep_set.next()
        data = get_fn(time)
        if ramp_direction =='down': data = np.flip(data)
        return data

    if vm_direction == 'v':
        vmkw = dict(ylabel = f"{dev_ramp_name} ramp",
                    xlabel = f"{dev_set_name}",
                    xsweep = sweep_set,
                    fn_yshift = dev_ramp._vm_shift_fn,
                    axes_dict={'y':[val_dev_ramp-ramp_delta, val_dev_ramp+ramp_delta]})
    elif vm_direction == 'h':
        vmkw = dict(xlabel = f"{dev_ramp_name} ramp",
                    ylabel = f"{dev_set_name}",
                    ysweep = sweep_set,
                    fn_xshift = dev_ramp._vm_shift_fn,
                    axes_dict={'x':[val_dev_ramp-ramp_delta, val_dev_ramp+ramp_delta]})
    vm = VideoModeWindow(fn_get=get, dim=2, wrap_direction=vm_direction, **vmkw)
    return vm, [p_ramp, p_set]
