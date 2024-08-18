import os

import scipy
from scipy import ndimage
from scipy.fft import fftfreq, fftshift

from pyHegel import fitting, fit_functions

import numpy as np
from matplotlib import pyplot as plt

#### file
def fileIn(path, full_path=False):
    """ List all files in the directory.    """
    # Get the list of files
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if full_path: return [os.path.join(path, f) for f in files]
    else: return files

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

def histogram(arr, bins=100, get_bins=False, **kwargs):
    hist, bin_list = np.histogram(arr, bins, **kwargs)
    if get_bins:
       return hist, bin_list
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


def findClassifyingThreshold(image, p0=[7, 10, 25, 62, 3.5, 3.5], bins=100, show_plot=True, verbose=False):
    # 1 prepare data
    samples = image.flatten()
    hist, bins = np.histogram(samples, bins=bins, density=True)
    x = np.linspace(0, len(hist)-1, len(hist))
    
    # 2 do the fit
    fit_result = fitting.fitcurve(f_doubleGaussian, x, hist, p0)
    fit_curve = f_doubleGaussian(x, *fit_result[0])

    # 3 find threshold index and value
    A1, A2 = fit_result[0][2], fit_result[0][3]
    peak1, peak2 = int(min(A1, A2)), int(max(A1, A2))
    if peak1 == peak2: peak2 += 2 # not good
    threshold_ind = np.argmin(fit_curve[peak1:peak2]) # find the min between the two peaks
    threshold_ind += peak1 # "recenter" the treshold
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

#### fit functions
def f_doubleGaussian(x, sigma1, sigma2, mu1=0., mu2=0., A1=3.5, A2=3.5):
    """ use for fitting state repartition
    sigma: curvature =1:sharp, =15:very flatten
    mu: center
    A: height
    """
    g1 = fit_functions.gaussian(x, sigma1, mu1, A1)
    g2 = fit_functions.gaussian(x, sigma2, mu2, A2)
    return g1+g2

def f_exp(x, tau, a=1., b=0., c=0.):
    return a*np.exp(-(x+b)/tau)+c


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
