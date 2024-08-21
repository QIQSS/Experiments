import os

import scipy
from scipy import ndimage
from scipy.fft import fftfreq, fftshift

from pyHegel import fitting, fit_functions
from scipy.optimize import curve_fit

import numpy as np
from matplotlib import pyplot as plt

#### array handling

def alternate(arr, enable=True):
    """ returns a copied array with odd rows flipped """
    if not enable: return arr
    ret = arr.copy()
    ret[1::2, :] = ret[1::2, ::-1]
    return ret

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

def meandiff(a):
    return np.mean(np.diff(a))

#### filters

def fft(arr):
    """ returns a list of frequence and vals """
    vals = fftshift(scipy.fft.fft(arr))
    freq = fftshift(fftfreq(len(arr)))
    return freq, vals

def gaussian(arr, sigma=2):
    """ returns a copied array with gaussian filter """
    return ndimage.gaussian_filter1d(arr, sigma)

def gaussianLineByLine(image, sigma=20, **kwargs): 
    image = np.asarray(image)
    return ndimage.gaussian_filter1d(image, sigma, axis=1, **kwargs)

#### compute things

def findNearest(array, value, return_id=False):
    """ find the nearest element of value in array.
    return the value or the index """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_id: return idx
    return array[idx]


def rSquared(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def histogram(arr, bins=100, 
              get_x_axis=False, 
              **kwargs):
    arr = np.asarray(arr).flatten()
    hist, bin_list = np.histogram(arr, bins, **kwargs)
    if get_x_axis:
        x = (bin_list[:-1] + bin_list[1:]) / 2
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

def countHighLow(arr1d, high=1, low=0):
    """ count and return the proportion of high and low value """
    h_count = np.sum(arr1d == high)
    l_count = np.sum(arr1d == low)
    h_prop = h_count / arr1d.size
    l_prop = l_count / arr1d.size
    return dict(high=h_prop, low=l_prop, high_count=h_count, low_count=l_count)
    

def classTraces(arr2d, timelist):
    """ wip
    timelist same size as arr2d.shape[1]
    """
    d = {'low':0, 'high':0, 'exclude':0, 'low_ids':[], 'high_ids':[], 'exclude_ids':[],
         'high_fall_time':[]}
    
    for i, trace in enumerate(arr2d):
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

def findClassifyingThreshold(double_gaussian_parameters):
    """ estimate the treshold for classifying a double gaussian.
    use the results from fitDoubleGaussian """
    sigma1, sigma2, mu1, mu2, A1, A2 = double_gaussian_parameters
    midpoint_threshold = (mu1 + mu2) / 2
    return midpoint_threshold

def findPeaks(points, show_plot=False, **kwargs):
    import scipy
    peaks, properties = scipy.signal.find_peaks(points, **kwargs)
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(points)
        plt.plot(peaks, points[peaks], 'rx')
        plt.vlines(peaks, ymin=0, ymax=max(points), color='r', linestyle='--')
        for peak in peaks:
            plt.text(peak, points[peak], f'({peak}, {points[peak]:.2f})', 
                     ha='center', va='bottom', color='red', fontsize=9)
        plt.show()
    return peaks, properties


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

def f_expDecay(x, tau, a=1., b=0., c=0.):
    return a*np.exp(-(x+b)/tau)+c

def ajustementDeCourbe(function, x, y, p0=[], threshold=0, 
                       verbose=False, show_plot=False,
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
        plt.plot(x, function(x, *best_params), color='red', lw=3, label='Fit')
        plt.plot(x, y, color='blue', marker='o', linestyle='None', label='Data')
        
        # extract parameters names
        import inspect
        sig = inspect.signature(function)
        param_names = list(sig.parameters.keys())[1:]  # skip the first parameter (x)

        param_text = ', '.join(f'{name}={value:.3f}' for name, value in zip(param_names, best_params))
        plt.text(0.05, 0.95, f'Fit parameters:\n{param_text}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.legend()
        plt.show()
        
    return best_params
    
def fitDoubleGaussian(points, x, p0=None, p0_list=[], **kwargs):
    delta = meandiff(x)
    p0_list = p0_list[:]
    p0_list += [
        [10, 10, x[0], x[-1], np.max(points),  np.max(points) / 2, 1000],
        [10, 10, np.mean(x)+10*delta, np.mean(x)-10*delta, np.max(points),  np.max(points) / 2, 1000],
        [1, 1, x[20], x[-20], np.max(points),  np.max(points), 1000],
        [1, 1.1, x[20], x[-20], np.max(points),  np.max(points)/4, 1000],
    ]
    if p0 is not None: p0_list = [p0]
    
    params = ajustementDeCourbe(f_doubleGaussian, x, points, p0_list, **kwargs)

    return params

#### mesures

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
    
    return result
