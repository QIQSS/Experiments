
import scipy
from scipy import ndimage
from scipy.fft import fftfreq, fftshift

import numpy as np

### array handling

def gaussian(arr, sigma=2):
    """ returns a copied array with gaussian filter """
    return ndimage.gaussian_filter1d(arr, sigma)

def gaussianLineByLine(image, sigma=20, **kwargs): 
    try:
        dim2 = len(image[0])
        return np.array([ndimage.gaussian_filter1d(line, sigma, **kwargs) for line in image])
    except:
        return ndimage.gaussian_filter1d(image, sigma, **kwargs)


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

def head(arr, x): return arr[:x]
def tail(arr, x): return arr[x:]

### filters

def fft(arr):
    """ returns a list of frequence and vals """
    vals = fftshift(scipy.fft.fft(arr))
    freq = fftshift(fftfreq(len(arr)))
    return freq, vals
