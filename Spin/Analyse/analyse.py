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
    try:
        dim2 = len(image[0])
        return np.array([ndimage.gaussian_filter1d(line, sigma, **kwargs) for line in image])
    except:
        return ndimage.gaussian_filter1d(image, sigma, **kwargs)


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


def findClassifingThreshold(image, p0=[7, 10, 25, 62, 3.5, 3.5], bins=100, show_plot=True, verbose=True):
    # 1 prepare data
    samples = image.flatten()
    hist, bins = np.histogram(samples, bins=bins, density=True)
    x = np.linspace(0, len(hist)-1, len(hist))
    
    # 2 do the fit
    fit_result = fitting.fitcurve(f_doubleGaussian, x, hist, p0)
    fit_curve = f_doubleGaussian(x, *fit_result[0])
    
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


