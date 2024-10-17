import os

import scipy
from scipy import ndimage
from scipy.fft import fftfreq, fftshift
from scipy.optimize import minimize_scalar

from pyHegel import fitting, fit_functions
from scipy.optimize import curve_fit

import numpy as np
from matplotlib import pyplot as plt

from typing import Literal

from . import plot as up
from . import utils as uu
#### new array


def linlen(arr, end=None):
    """ linspace "ids of arr" """
    if end is None:
        end = len(arr)
    return np.linspace(0, end, len(arr))

#### array handling

def alternate(arr, enable=True):
    """ returns a copied array with odd rows flipped """
    if not enable: return arr
    ret = arr.copy()
    ret[1::2, :] = ret[1::2, ::-1]
    return ret

def deinterlace(img):
    img1 = img[0::2, :]
    img2 = img[1::2, :]
    return img1, img2

def flip(arr, axis=-1, enable=True):
    """ returns a copied flipped array allong axis """
    if not enable: return arr
    ret = arr.copy()
    ret = np.flip(ret, axis=axis)
    return ret

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

def head(arr, x=10): 
    """ extract the x first lines of array """
    return arr[:x]
def tail(arr, x=10):    
    """ extract the x last lines of array """
    return arr[len(arr)-x:]
def chead(arr2d, x=10): 
    """ extract the x first columns of 2d array """
    return arr2d[:, :x]
def ctail(arr2d, x=10): 
    """ extract the x last columns of 2d array """
    return arr2d[:, len(arr2d[0])-x:]

def sliceColumns(arr, start=None, stop=None, slice_by_val=False):
    """ return arr[:, start:stop]
    """ 
    if slice_by_val:
        start = findNearest(arr, start, return_type='id')
        stop = findNearest(arr, stop, return_type='id')

    return arr[:, start:stop]


def meandiff(a):
    return np.mean(np.diff(a))

def multiget(arr, list_of_indexes):
    return [arr[i] for i in list_of_indexes]

#### filters

def fft(arr):
    """ returns a list of frequence and vals """
    vals = fftshift(scipy.fft.fft(arr))
    freq = fftshift(fftfreq(len(arr)))
    return freq, vals

def gaussian(arr, sigma=20, **kwargs):
    """ returns a copied array with gaussian filter """
    if sigma == 0: return arr
    return ndimage.gaussian_filter1d(arr, sigma, **kwargs)
 
def gaussianlbl(image, sigma=20, **kwargs):
    """ gaussian line by line """
    if sigma == 0: return image
    return ndimage.gaussian_filter1d(image, sigma, axis=1, **kwargs)

def gaussian2d(image, sigma=20, **kwargs):
    if sigma == 0: return image
    return ndimage.gaussian_filter(image, sigma, **kwargs)

#### compute things

def findNearest(array, value, 
                return_type: Literal['val', 'id'] = 'val'):
    """ find the nearest element of value in array.
    return the value or the index """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_type=='id': return idx
    return array[idx]


def rSquared(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def histogram(arr, bins=100, 
              return_type: Literal['all', 'hist'] = 'hist',
              show_plot=False,
              **kwargs):
    """
    return_type: hist or all: x, hist
    """
    arr = np.asarray(arr).flatten()
    hist, bin_list = np.histogram(arr, bins, **kwargs)
    
    x = (bin_list[:-1] + bin_list[1:]) / 2
    if show_plot:
        up.qplot(hist, x)
    if return_type == 'all':
        return x, hist
    else:
        return hist

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

def classify(image, threshold, inverse=False):
    """ return the image with values 0 for below TH and 1 for above TH
    """
    bool_image = image>threshold
    if inverse: bool_image = ~bool_image
    int_image = np.array(bool_image, dtype=int)
    return int_image

def allequal(arr, val):
    return np.all(arr == val) 

def atleastoneequal(arr, val):
    return np.sum(arr == val) == 1

def countHighLow(arr1d, high=1, low=0):
    """ count and return the proportion of high and low value """
    h_count = np.sum(arr1d == high)
    l_count = np.sum(arr1d == low)
    h_prop = h_count / arr1d.size
    l_prop = l_count / arr1d.size
    return dict(high=h_prop, low=l_prop, high_count=h_count, low_count=l_count)

def removeSmallEvents(trace, tolerance, verbose=False, show_plot=False):
    """ repllace event with less points than tolerance 
    by zeros (ones) if event is ones (zeros).
    trace is a 1d array with only zeros and ones.
    """

    event_indexes = np.where(np.diff(trace) != 0)[0]+1
    event_indexes = np.concatenate(([0], event_indexes, [len(trace)]))

    event_lengths = np.diff(event_indexes)
    valid_events = event_lengths >= tolerance
    
    
    cleaned_trace = np.copy(trace)
    
    for start, end, valid in zip(event_indexes[:-1], event_indexes[1:], valid_events):
        if not valid:
            cleaned_trace[start:end] = 0 if cleaned_trace[start]==1 else 1
    
    if verbose:
        total_events = len(event_lengths)
        num_valid_events = np.sum(valid_events)
        num_removed_events = total_events - num_valid_events
        print(f"Total events: {total_events}")
        print(f"Valid events: {num_valid_events}")
        print(f"Events removed: {num_removed_events}")
        print(f"Tolerance: {tolerance}")
       
    if show_plot:
        # Plot the original and cleaned traces
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(trace, label='Original Trace', color='blue')
        plt.title('Original Trace');plt.xlabel('Index');plt.ylabel('Value')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(cleaned_trace, label='Cleaned Trace', color='green')
        plt.title('Cleaned Trace');plt.xlabel('Index');plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return np.array(cleaned_trace)

def classTraces(arr2d, timelist, blip_tolerance=0):
    """ wip
    timelist same size as arr2d.shape[1]
    """
    d = {'low':0, 'high':0, 'exclude':0, 'low_ids':[], 'high_ids':[], 'exclude_ids':[],
         'high_fall_time':[]}
    
    for i, trace in enumerate(arr2d):
        trace = removeSmallEvents(trace, blip_tolerance)
        
        if not np.any(trace): # 0000000
            d['low'] += 1
            d['low_ids'].append(i)
            continue
                    
        if np.all(trace): # 1111111
            d['high'] += 1
            d['high_ids'].append(i)
            d['high_fall_time'].append(timelist[-1])
            continue
        
        event_index = np.where(np.diff(trace) == -1)[0]
        if len(event_index) == 1:
            event_index = event_index[0]
            if allequal(trace[:event_index + 1], 1) and \
                allequal(trace[event_index + 1:], 0): # all ones then all zeros
                    
                d['high'] += 1
                d['high_ids'].append(i)
                d['high_fall_time'].append(timelist[event_index])
                continue

    
        #if len(event_index) != 1: # more than one event
        d['exclude'] += 1
        d['exclude_ids'].append(i)
        continue
    return d

def findClassifyingThreshold(double_gaussian_parameters,
                             method: Literal['min', 'mid'] = 'min'):
    """ estimate the treshold for classifying a double gaussian.
    use the results from fitDoubleGaussian """
    if double_gaussian_parameters is None: return False
    
    sigma1, sigma2, mu1, mu2, A1, A2 = double_gaussian_parameters
    match method:
        case 'mid':
            th = (mu1 + mu2) / 2
        case 'min':
            def f(x): return f_doubleGaussian(x, *double_gaussian_parameters)
            result = minimize_scalar(f, bounds=(min(mu1, mu2), max(mu1, mu2)), method='bounded')
            th = result.x
    return th

def findPeaks(points, show_plot=False, 
              prominence=1,
              **scipy_kwargs):
    import scipy
    peaks, properties = scipy.signal.find_peaks(points, prominence=prominence, **scipy_kwargs)
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(points)
        plt.plot(peaks, points[peaks], 'rx')
        plt.vlines(peaks, ymin=0, ymax=max(points), color='r', linestyle='--')
        for peak in peaks:
            plt.text(peak, points[peak], f'({peak}, {points[peak]:.2f})', 
                     ha='center', va='bottom', color='red', fontsize=9)
        plt.grid()
        plt.show()
    return peaks, properties

def autoClassify(array, filter_sigma=2, width_tolerance=0, prominence_factor=0.03, verbose=0):
    """ automated histogram, peaks analysis, gaussian fit then classify """
    array = gaussianlbl(array, sigma=filter_sigma)
    bins, hist = histogram(array, return_type='all')
    peaks, prop = findPeaks(hist, show_plot=verbose>1, prominence=max(hist)*prominence_factor)

    if len(peaks) < 1:
        return False
    if len(peaks) > 2 and len(peaks) < 10:
        print("More than 2 peaks found, trying with a sigma = 2*sigma")
        print(filter_sigma)
        return autoClassify(array, filter_sigma*2, width_tolerance, prominence_factor, verbose)
    elif len(peaks)>=10:
        print(f"{uu.fname()}: nb peaks != 2, can't classify")
        
        return False

    p0 = [0.4, 0.4, bins[peaks[0]], bins[peaks[1]], hist[peaks[0]], hist[peaks[1]]]
    dg_params = ajustementDeCourbe(f_doubleGaussian, bins, hist, p0=p0, show_plot=verbose>1)
    th = findClassifyingThreshold(dg_params, 'min')

    clas = classify(array, th)
    #up.imshow(clas)
    clas_clean = np.apply_along_axis(removeSmallEvents, arr=clas, axis=1, tolerance=width_tolerance)
    #up.imshow(clas_clean-clas)
    
    return clas

def blockade_probability(read1, read2, tolerance=20):
    """ take read1 and read2 maps.
    exclude from read2 all the traces that are singlet in read1
    count the number of singlet / triplet in read2
    returns the blockade probability
    """
    read1clas = autoClassify(read1, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
    read2clas = autoClassify(read2, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
    

    ids_triplet_read1 = [id_ for id_, trace in enumerate(read1clas) if np.all(trace == 0)]
    
    old_triplet_read2 = multiget(read2clas, ids_triplet_read1)
    
    ids_triplet_read2 = [id_ for id_, trace in enumerate(old_triplet_read2) if np.all(trace == 0)]
    ids_singlet_read2 = [id_ for id_, trace in enumerate(old_triplet_read2) if np.all(trace == 1)]
    
    nb_triplet = len(ids_triplet_read2)
    nb_singlet = len(ids_singlet_read2)
    
    p_blockade = nb_singlet / (nb_triplet + nb_singlet)

    return p_blockade, nb_singlet, nb_triplet

def flip_probability(read1, read2, tolerance=20):
    """ take read1 and read2 maps.
    class from read1 all singlets and all triplets
    count the number of flips
    returns the flip probability
    """
    read1clas = autoClassify(read1, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
    read2clas = autoClassify(read2, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)

    ids_T_read1 = [id_ for id_, trace in enumerate(read1clas) if np.all(trace == 0)]
    ids_S_read1 = [id_ for id_, trace in enumerate(read1clas) if np.all(trace == 1)]
    
    ids_T_read2 = [id_ for id_, trace in enumerate(read2clas) if np.all(trace == 0)]
    ids_S_read2 = [id_ for id_, trace in enumerate(read2clas) if np.all(trace == 1)]
    
    ids_T_to_S = [id_ for id_ in ids_S_read2 if id_ in ids_T_read1]
    ids_S_to_T = [id_ for id_ in ids_T_read2 if id_ in ids_S_read1]
    
    ids_T_to_T = [id_ for id_ in ids_T_read2 if id_ in ids_T_read1]
    ids_S_to_S = [id_ for id_ in ids_S_read2 if id_ in ids_S_read1]
    
    nb_flip = len(ids_T_to_S) + len(ids_S_to_T)
    nb_no_flip = len(ids_T_to_T) + len(ids_S_to_S)
    
    p_flip = nb_flip / (nb_flip + nb_no_flip)
    
    return p_flip, nb_flip, nb_no_flip

def onCol(function, arr, col=0):
    arr = np.asarray(arr)
    return function(arr[:,col])

#### fit functions
def f_gaussian(x, sigma, mu, A):
    return fit_functions.gaussian(x, sigma, mu, A)

def f_doubleGaussian(x, sigma1, sigma2, mu1=0., mu2=0., A1=3.5, A2=3.5):
    """ use for fitting state repartition
    sigma: curvature =1:sharp, =15:very flatten
    mu: center
    A: height
    """
    g1 = fit_functions.gaussian(x, sigma1, mu1, A1)
    g2 = fit_functions.gaussian(x, sigma2, mu2, A2)
    return g1+g2

def f_expDecay(x, tau, a=1., c=0.):
    return a*np.exp(-x/tau)+c

def ajustementDeCourbe(function, x, y, p0=[], threshold=0,
                       verbose=False, show_plot=False,
                       plot_title='',
                       inspect=False):
    """ do a fit, optimize: y = function(x, *p0)
    can give a list of p0 to try them all. it will take the best one.
    
    inspect: bool, will not do the fit, just plot the p0. (p0[0] if it's a p0_list)
    """
    error = False
    best_params = None
    best_error = float('inf')
    
    p0_list = []
    if len(p0) == 0:
        print('give a p0.')
    elif isinstance(p0[0], (list, tuple)):
        p0_list = p0
    else:
        p0_list = [p0]
        
    for i, p0 in enumerate(p0_list):
        if inspect:
            show_plot = True
            best_params = p0
            break
        
        try:
            params, _  = curve_fit(function, x, y, p0=p0)
            y_pred = function(x, *params)
            error = rSquared(y, y_pred)

            if verbose:
                print(f"Iteration {i}: Parameters = {params}, Error = {error}")
            
            if error < best_error:
                best_params = params
                best_error = error
            
            if error < threshold:
                if verbose:
                    print(f"Error below threshold ({threshold}). Done.")
                break
            
        except Exception as e:
            if verbose: print(f"Error with p0={p0}: {e}")
            continue

    if show_plot and best_params is not None:
        plt.figure()
        plt.plot(x, function(x, *best_params), color='red', lw=3, label='Fit')
        plt.plot(x, y, color='blue', marker='o', linestyle='None', label='Data')
        
        # extract parameters names
        import inspect
        sig = inspect.signature(function)
        param_names = list(sig.parameters.keys())[1:]  # skip the first parameter (x)

        param_text = ', '.join(f'{name}={value:.3f}' for name, value in zip(param_names, best_params))
        plt.text(0.6, 0.7, f'Fit parameters:\n{param_text}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.title(plot_title)
        plt.legend()
        plt.show()
        
    return best_params

def fitExpDecayLinear(x, y, verbose=False, show_plot=False, text=''):
    """ fit y = Ae^(-t/tau)
    but with ln(y) = len(A) - 1/tau * t
             lny   =   c     + m * t
    """
    lny = np.log(y)
    m, c = np.polyfit(x, lny, 1)
    A = np.exp(c)
    tau = -1/m

    if verbose:
        print(f"Linear fit results:")
        print(f"  m: {m}")
        print(f"  c: {c}")
        print(f"  A: {A}")
        print(f"  tau: {tau}")
    
    # Plotting if show_plot is enabled
    if show_plot:
        plt.scatter(x, y, label='Data', color='blue')

        fitted_y = A * np.exp(-x / tau)
        plt.plot(x, fitted_y, label=f'Fitted Curve: A*e^(-x/tau)\nA={A:.2f}, tau={tau:.2f}', color='red')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(text)
        plt.grid(True)
        plt.show()
    
    return A, tau
    
#### mesures

def arange(start, stop, step=1, endpoint=True):
    arr = np.arange(start, stop, step)

    if endpoint and arr[-1]+step==stop:
        arr = np.concatenate([arr,[stop]])

    return arr

def reorder_list(lst, n):
    length = len(lst)
    reordered = []
    
    # Add elements in the order [0, n, 2n, ...]
    for i in range(0, length, n):
        reordered.append(lst[i])
    
    # Add elements in the order [1, 1+n, 1+2n, ...]
    for i in range(1, n):
        for j in range(0, length, n):
            index = i + j
            if index < length:
                reordered.append(lst[index])
    return reordered

def gen2dTraceSweep(x_start, x_stop, y_start, y_stop, nbpts):
    x_list = np.linspace(x_start, x_stop, nbpts)
    y_list = np.linspace(y_start, y_stop, nbpts)
    return [(x, y) for x, y in zip(x_list, y_list)]

def gen2dTraceSweepDiag(x_start1, x_stop1, y_start1, y_stop1, nbpts1,
                       x_start2, y_start2, nbpts2):
    """
    (x_start1, y_start1) -> (x_stop1, y_stop1)
    -> (x_start2, y_start1+nbpts2*x_step) -> (x_stop2, y_stop1+nbpts2*x_step)
    """
    sw1 = list(zip(np.linspace(x_start1, x_stop1, nbpts1), np.linspace(y_start1, y_stop1, nbpts1)))
    points = sw1[:]
    
    x_step = abs(x_start1-x_start2)/float(nbpts2)
    y_step = abs(y_start1-y_start2)/float(nbpts2)
    for i in range(nbpts2):
        swi = [(p[0]+i*x_step, p[1]+i*y_step) for p in sw1]
        points += list(swi)
    return list(points)

def genTrapezoidSweep(x_start, x_stop, x_nb, y_start0, y_stop0, y_startn, y_stopn, y_nb):
    """ return a list of tuple """
    result = []    
    x_values = [x_start + (x_stop - x_start) * i / (x_nb - 1) for i in range(x_nb)]

    for x in x_values:
        # Interpolate the y-values for the current x
        y_start = y_start0 + (y_startn - y_start0) * (x - x_start) / (x_stop - x_start)
        y_stop = y_stop0 + (y_stopn - y_stop0) * (x - x_start) / (x_stop - x_start)
        
        # Generate y-values
        y_values = [y_start + (y_stop - y_start) * i / (y_nb - 1) for i in range(y_nb)]
        
        for y in y_values:
            result.append((x, y))
    
    return np.array(result)
