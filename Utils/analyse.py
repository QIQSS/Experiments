import os

import scipy
from scipy import ndimage
from scipy.fft import fftfreq, fftshift
from scipy.fftpack import fft, ifft
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

def removeId(array, id_):
    arr = np.concatenate((array[:id_],array[id_+1:]))
    return arr

def padTo(array, length, value=np.nan):
    padding = np.full(length-len(array),value)
    arr = np.concatenate((array,padding))
    return arr

def removeNans(x):
    x = x[~np.isnan(x)]
    return x

def meandiff(a):
    return np.mean(np.diff(a))

def multiget(arr, list_of_indexes, fallback=np.nan):
    def isValidIndice(i):
        return 0 <= i < len(arr) and arr[i] is not None and not np.isnan(arr[i])
    return [arr[i] if isValidIndice(i) else fallback for i in list_of_indexes]

def firstNonNanValue(array):
    for value in array:
        if not np.isnan(value):
            return value
    return None


def downsampleColumns(array, step):
    return array[:,::step]

#### filters

# def fft(arr):
#     """ returns a list of frequence and vals """
#     vals = fftshift(scipy.fft.fft(arr))
#     freq = fftshift(fftfreq(len(arr)))
#     return freq, vals

def filter_frequencies(y, x, filter_freqs,
                       show_plot=False):
    """
    Filter specified frequencies from a 1D signal using FFT.
    
    Parameters:
    - y: 1D array of the signal to be filtered.
    - x: 1D array representing the axis
    - filter_freqs: List of frequencies to filter out from the signal.
    """
    fft_y = fft(y)

    sampling_interval = x[1] - x[0]
    frequencies = np.fft.fftfreq(len(y), d=sampling_interval)

    for ff in filter_freqs:
        idx_positive = np.abs(frequencies - ff).argmin()
        idx_negative = np.abs(frequencies + ff).argmin()
        fft_y[idx_positive] = 0
        fft_y[idx_negative] = 0

    # Perform IFFT to get the filtered signal
    filtered_y = ifft(fft_y)

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x, y, label="Original Signal")
        plt.title("Original Signal")
        plt.subplot(2, 1, 2)
        plt.plot(x, filtered_y.real, label="Filtered Signal", color='orange')
        plt.title(f"Filtered Signal (Frequencies {filter_freqs} units removed)")
        plt.tight_layout()
        plt.show()
    
    return filtered_y

def gaussian(arr, sigma=20, **kwargs):
    """ returns a copied array with gaussian filter """
    if sigma == 0: return arr
    return ndimage.gaussian_filter1d(arr, sigma, **kwargs)
 
def gaussianlbl(image, sigma=20, **kwargs):
    """ gaussian line by line """
    if sigma == 0: return image
    if image.ndim == 1:
        return ndimage.gaussian_filter1d(image, sigma, **kwargs)
    return ndimage.gaussian_filter1d(image, sigma, axis=1, **kwargs)

def gaussian2d(image, sigma=20, **kwargs):
    if sigma == 0: return image
    return ndimage.gaussian_filter(image, sigma, **kwargs)

def lfilter(trace, n):
    from scipy.signal import lfilter
    if n == 0: return trace.copy()
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, trace)
    return yy

def filtfilt(data, order, cutoff_frequency=0.1):
    from scipy.signal import butter, filtfilt
    if order == 0:
        return data
    # Define cutoff frequency (adjustable)
    cutoff_frequency = 0.1  # This is normalized between 0 and 1 (Nyquist frequency)
    
    # Create Butterworth filter coefficients
    b, a = butter(order, cutoff_frequency, btype='low')
    
    # Apply filter to the data using filtfilt
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data
    
def derivative2d(array, axis='xy'):
    # Calculate the derivatives along the x (columns) and y (rows) axes
    if 'x' in axis:
        return np.diff(array, axis=1)  # Derivative along the x-axis
    if 'y' in axis:
        return np.diff(array, axis=0)  # Derivative along the y-axis
    
#### compute things

def findNearest(array, value, 
                return_type: Literal['val', 'id'] = 'val'):
    """ find the nearest element of value in array.
    return the value or the index """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_type=='id': return idx
    return array[idx]

def getValue(x_array, y_array, x_to_find):
    # Interpolate to find the y value of `y_array` at `x_to_find`
    y_interpolated = np.interp(x_to_find, x_array, y_array)
    return y_interpolated


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

def countHighLowTrace(arr2d, high=1, low=0):
    """ count and return the proportion of traces that are only high or only low."""
    
    highs = np.all(arr2d == high, axis=1)
    lows = np.all(arr2d == low, axis=1)
    
    high_count = np.sum(highs)
    low_count = np.sum(lows)
    exclude_count = len(arr2d) - high_count - low_count
    
    high_trace_prop = high_count / len(arr2d) *100
    low_trace_prop = low_count / len(arr2d) *100
    exclude_trace_prop = exclude_count / len(arr2d) *100
    
    return {
        "high_trace_count": high_count,
        "low_trace_count": low_count,
        "exclude_trace_count": exclude_count,
        "high_trace_prop": high_trace_prop,
        "low_trace_prop": low_trace_prop,
        "exclude_trace_prop": exclude_trace_prop
    }


def removeSmallSegments_lbl(boolean_array, tolerance, skip_first_seg=False):
    """Remove small segments from a boolean array (supports both 1D and 2D arrays)
       and repeat until no more small segments (blips) exist, without modifying the input in place."""
    
    # Create a copy of the input array to avoid in-place modification
    processed_array = boolean_array.copy()

    # Define a helper function to remove small segments from a single line (1D)
    def process_line(line):
        # Work with a copy of the line to avoid in-place changes
        line_copy = line.copy()

        while True:
            # Make a copy of the line to track changes
            prev_line = line_copy.copy()

            # Identify where the transitions occur (changes between 0 and 1)
            transitions = np.diff(line_copy.astype(int))

            # Find the indices of the transitions
            change_indices = np.where(transitions != 0)[0] + 1

            # If there are no transitions or only one type of value, return the line as is
            if len(change_indices) == 0:
                return line_copy

            # Get the start and end indices of segments
            segment_indices = np.concatenate(([0], change_indices, [len(line_copy)]))

            # Filter out segments smaller than tolerance
            for j in range(len(segment_indices) - 1):
                start, end = segment_indices[j], segment_indices[j + 1]
                if start == 0 and skip_first_seg: continue
                segment = line_copy[start:end]
                if len(segment) < tolerance:
                    # Determine the value to replace with
                    if start > 0:
                        replacement_value = line_copy[start - 1]  # Use the value before the segment
                    elif end < len(line_copy):
                        replacement_value = line_copy[end]  # Use the value after the segment
                    else:
                        replacement_value = line_copy[0]  # Fallback for edge case (1st element)

                    # Replace the small segment with the chosen value
                    line_copy[start:end] = replacement_value

            # If no further changes occurred, stop the loop
            if np.array_equal(line_copy, prev_line):
                break

        return line_copy

    # Check if the array is 1D
    if processed_array.ndim == 1:
        return process_line(processed_array)
    
    # If it's 2D, iterate over each row and process each row without modifying the input array
    for i in range(processed_array.shape[0]):  # Iterate over each row
        processed_array[i] = process_line(processed_array[i])

    return processed_array

def classTracesT1(arr2d, timelist, low_val=0, blip_tolerance=0):
    """
    low_val 0 or 1
    Classifies binary traces from a 2D array based on whether they are always low (low_val),
    always high (1), or transition from high to low at a specific time. Traces that 
    don't fit these categories are excluded.

    Parameters:
    - arr2d (2D numpy array): Each row is a binary trace (1s and low_val).
    - timelist (list): A list of time points corresponding to the columns of arr2d.
    - low_val (int): The value that represents the 'low' state (either 0 or 1).
    - blip_tolerance (int): Minimum segment length tolerance to remove small segments (blips).
    
    Returns:
    - dict: Contains the counts and ids of traces classified as 'low', 'high', or 'exclude',
            as well as the fall times for high-to-low transitions.
    """
    high_val = 1 if low_val == 0 else 0
    
    d = {'low': 0, 'high': 0, 'exclude': 0, 'low_ids': [], 'high_ids': [], 'exclude_ids': [],
         'high_fall_time': []}
    
    for i, trace in enumerate(arr2d):
        if blip_tolerance != 0:
            trace = removeSmallSegments_lbl(trace, blip_tolerance)
        
        if np.all(trace == low_val):  # all low
            d['low'] += 1
            d['low_ids'].append(i)
            continue
                    
        if np.all(trace == high_val):  # all high
            d['high'] += 1
            d['high_ids'].append(i)
            d['high_fall_time'].append(timelist[-1])
            continue
        
        event_indexes = np.where(np.diff(trace) != 0)[0]  # find where it falls from 1 to low_val
        
        # If exactly one transition
        if len(event_indexes) == 1:
            event_index = event_indexes[0]
            if allequal(trace[:event_index + 1], high_val) and \
                allequal(trace[event_index + 1:], low_val): 
                # all hihg before the fall and all low after the fall
                d['high'] += 1
                d['high_ids'].append(i)
                d['high_fall_time'].append(timelist[event_index])
            else:
                d['exclude'] += 1
                d['exclude_ids'].append(i)
            continue

        # If zero or more than one event, exclude
        d['exclude'] += 1
        d['exclude_ids'].append(i)
    
    return uu.customDict(d)

def computeT1(classTracesT1dict, timelist):
    fall_times = np.array(classTracesT1dict.get('high_fall_time'))
    points = [np.sum(np.where(fall_times > timebin, True, False)) for timebin in timelist]
    # fit:
    #ua.ajustementDeCourbe(ua.f_expDecay, x_axis, t1, p0=[0.02,600], show_plot=True)
    return points

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

def findPeaks(points, 
              prominence=1,
              show_plot=False, 
              **scipy_kwargs):

    peaks, properties = scipy.signal.find_peaks(points, prominence=prominence, **scipy_kwargs)
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(points)
        plt.plot(peaks, points[peaks], 'rx')
        plt.vlines(peaks, ymin=np.nanmin(points), ymax=np.nanmax(points), color='r', linestyle='--')
        for peak in peaks:
            plt.text(peak, points[peak], f'({peak}, {points[peak]:.2f})', 
                     ha='center', va='bottom', color='grey', fontsize=9)
        plt.grid()
        plt.show()
    return peaks, properties

def autoClassify(array, filter_sigma=2, prominence_factor=0.03, verbose=0, on_fail_value=0):
    """Automated histogram, peaks analysis, gaussian fit then classify on a smoothed array.
    
    Args:
        array (numpy array): The input data array to classify.
        filter_sigma (int, optional): The sigma value for the Gaussian smoothing filter. Defaults to 2.
        prominence_factor (float, optional): The prominence factor for peak detection. Defaults to 0.03.
        verbose (int, optional): Verbose level for debug messages. Defaults to 0.
        on_fail_value (int, optional): The value to return in case of failure. Defaults to 0.
    
    Returns:
        numpy array: The classified array or an array filled with `on_fail_value` if classification fails.
    """
    
    array_smooth = gaussianlbl(array, sigma=filter_sigma)
    bins, hist = histogram(array_smooth, return_type='all')
    hist_smooth = gaussian(hist, 1)
    peaks, prop = findPeaks(hist_smooth, show_plot=verbose>1, prominence=max(hist) * prominence_factor)

    # Define the on_fail case here
    on_fail = np.full_like(array, on_fail_value)

    if len(peaks) == 0:
        return on_fail  # No peaks found
    elif len(peaks) > 2 and len(peaks) < 10 and filter_sigma < 20:
        if verbose:
            print(f"More than 2 peaks found with sigma={filter_sigma}, retrying with sigma={2 * filter_sigma}")
        return autoClassify(array, filter_sigma + 1, prominence_factor + 0.01, verbose, on_fail_value)
    elif len(peaks) != 2:
        if verbose:
            print(f"Unexpected number of peaks ({len(peaks)}), can't classify.")
        return on_fail  # Unexpected number of peaks
    
    p0 = [0.4, 0.4, bins[peaks[0]], bins[peaks[1]], hist[peaks[0]], hist[peaks[1]]]
    # fit the curve with double Gaussian
    dg_params = ajustementDeCourbe(f_doubleGaussian, bins, hist, p0=p0, show_plot=verbose > 1)
    # Find the threshold for classification
    th = findClassifyingThreshold(dg_params, 'min')
    
    classified = classify(array_smooth, th)
    
    return classified


def autoClassifyAndRemoveBlips(array, 
                               filter_function=gaussianlbl,
                               filter_sigma=2, 
                               width_tolerance=0, skip_first_seg=False,
                               prominence_factor=0.03, 
                               verbose=0, on_fail=False):
    """ automated histogram, peaks analysis, gaussian fit then classify ON RAW ARRAY"""
    array_smooth = filter_function(array, filter_sigma)
    bins, hist = histogram(array_smooth, return_type='all')
    hist_smooth = gaussian(hist, 1)
    peaks, prop = findPeaks(hist_smooth, show_plot=verbose>1, prominence=max(hist)*prominence_factor)
    
    if len(peaks) == 0:
        return on_fail
    elif len(peaks) > 2 and len(peaks) < 10 and filter_sigma < 20:
        if verbose:
            print(f"More than 2 peaks found with sigma={filter_sigma}, retrying with sigma={2 * filter_sigma}")
        return autoClassifyAndRemoveBlips(array, 
                                          filter_function=filter_function,
                                          filter_sigma=filter_sigma + 1,
                                          width_tolerance=width_tolerance, 
                                          prominence_factor=prominence_factor + 0.01, 
                                          verbose=verbose, on_fail=on_fail)
    elif len(peaks) != 2:
        if verbose:
            print(f"Unexpected number of peaks ({len(peaks)}), can't classify.")
        return on_fail
    
    p0 = [0.4, 0.4, bins[peaks[0]], bins[peaks[1]], hist[peaks[0]], hist[peaks[1]]]
    dg_params = ajustementDeCourbe(f_doubleGaussian, bins, hist, p0=p0, show_plot=verbose>1)
    th = findClassifyingThreshold(dg_params, 'min')
    classified = classify(array, th)
    classified_cleaned = removeSmallSegments_lbl(classified, tolerance=width_tolerance,
                                                 skip_first_seg=skip_first_seg)
    
    return classified_cleaned


def blockade_probability(read1, read2, threshold=None, tolerance=20):
    """ take read1 and read2 maps.
    exclude from read2 all the traces that are singlet in read1
    count the number of singlet / triplet in read2
    returns the blockade probability
    """
    if threshold is None:
        read1clas = autoClassifyAndRemoveBlips(read1, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
        read2clas = autoClassifyAndRemoveBlips(read2, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
        if isinstance(read1clas, bool) or isinstance(read2clas, bool):
            return np.nan, 0, 0
    else:
        read1clas_ = classify(read1, threshold)
        read2clas_ = classify(read2, threshold)
        
        read1clas = np.apply_along_axis(removeSmallSegments_lbl, arr=read1clas_, axis=1, tolerance=tolerance)
        read2clas = np.apply_along_axis(removeSmallSegments_lbl, arr=read2clas_, axis=1, tolerance=tolerance)

    ids_triplet_read1 = [id_ for id_, trace in enumerate(read1clas) if np.all(trace == 0)]
    
    old_triplet_read2 = multiget(read2clas, ids_triplet_read1)
    
    ids_triplet_read2 = [id_ for id_, trace in enumerate(old_triplet_read2) if np.all(trace == 0)]
    ids_singlet_read2 = [id_ for id_, trace in enumerate(old_triplet_read2) if np.all(trace == 1)]
    
    nb_triplet = len(ids_triplet_read2)
    nb_singlet = len(ids_singlet_read2)
    
    p_blockade = nb_singlet / (nb_triplet + nb_singlet)

    return p_blockade, nb_singlet, nb_triplet

def flip_probability(read1, read2, threshold=None, tolerance=20):
    """ take read1 and read2 maps.
    class from read1 all singlets and all triplets
    count the number of flips
    returns the flip probability
    """
    if threshold is None:
        read1clas = autoClassifyAndRemoveBlips(read1, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
        read2clas = autoClassifyAndRemoveBlips(read2, width_tolerance=tolerance, prominence_factor=0.04, verbose=0)
        if isinstance(read1clas, bool) or isinstance(read2clas, bool):
            return np.nan, 0, 0
    else:
        read1clas_ = classify(read1, threshold)
        read2clas_ = classify(read2, threshold)
        
        read1clas = np.apply_along_axis(removeSmallSegments_lbl, arr=read1clas_, axis=1, tolerance=tolerance)
        read2clas = np.apply_along_axis(removeSmallSegments_lbl, arr=read2clas_, axis=1, tolerance=tolerance)
        
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

def f_expDecay0(x, tau, a=1):
    return f_expDecay(x, tau, a, 0)

def ajustementDeCourbe(function, x, y, p0=[], threshold=0,
                       verbose=False, show_plot=False,
                       plot_title='',
                       inspect=False,
                       get_errors=False):
    """ do a fit, optimize: y = function(x, *p0)
    can give a list of p0 to try them all. it will take the best one.
    
    inspect: bool, will not do the fit, just plot the p0. (p0[0] if it's a p0_list)
    """
    error = False
    best_params = None
    best_error = float('inf')
    fitting_errors = []
    
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
            params, covariance  = curve_fit(function, x, y, p0=p0)
            y_pred = function(x, *params)
            error = rSquared(y, y_pred)

            if verbose:
                print(f"Iteration {i}: Parameters = {params}, Error = {error}")
            
            param_errors = np.sqrt(np.diag(covariance))
            fitting_errors.append(param_errors)
            
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
        plt.plot(x, y, color='blue', marker='o', linestyle='None', label='Data')
        plt.plot(x, function(x, *best_params), color='red', lw=3, label='Fit')
        
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
    
    if get_errors: return best_params, fitting_errors
    return best_params

def fitExpDecayLinear(x, y, verbose=False, show_plot=False, text=''):
    """ fit y = Ae^(-t/tau)
    but with ln(y) = len(A) - 1/tau * t
             lny   =   c     + m * t
    """
    x,y = np.asarray(x), np.asarray(y)
    mask = y > 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    lny = np.log(y_filtered)
    m, c = np.polyfit(x_filtered, lny, 1)
    A = np.exp(c)
    tau = -1 / m

    if verbose:
        print("Linear fit results:")
        print(f"  m: {m}")
        print(f"  c: {c}")
        print(f"  A: {A}")
        print(f"  tau: {tau}")
    
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
    
def ffit(y, x=None, n=1):
    if x is None:
        x = linlen(y)
    coefficients = np.polyfit(x, y, n)
    # Create a polynomial from the coefficients
    polynomial = np.poly1d(coefficients)
    return polynomial(x)
    #return polynomial, polynomial
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
