import time as timemodule
import numpy as np
from matplotlib import pyplot as plt

from pyHegel import commands
from pyHegel.instruments_base import BaseInstrument


from . import analyse as a
from . import plot as p


#### FILE SAVING/LOADING


#### loading/saving from/to npz format

def loadNpz(name):
    """ Returns a dictionnary build from the npzfile.
    if saveNpz was used, the return should be:
        {'array': array(),
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


def saveToNpz(path, filename, array, metadata={}):
    """ Save array to an npz file.
    metadata is a dictionnary, it can have pyHegel instruments as values: the iprint will be saved.
    """
    if not path.endswith(('/', '\\')):
        path += '/'
    timestamp = timemodule.strftime('%Y%m%d-%H%M%S-')
    if filename == '': timestamp = timestamp[:-1]
    fullname = path + timestamp + filename
    
    # formating metadata
    for key, val in metadata.items():
        if isinstance(val, BaseInstrument): # is pyHegel instrument
            metadata[key] = val.iprint()
    metadata['_filename'] = timestamp+filename
    
    # saving zip
    np.savez_compressed(fullname, array=array, metadata=metadata)
    
    print('Saved file to: ' + fullname)
    return fullname+'.npz'



#### pyHegel files

def readfileNdim(file):
    """ a memo of pyHegel.readfile for sweep with N dimensions 
    for i in range(10):
        imshow(data[3][i].T, x_axis=data[2,1,0], y_axis=data[1,1][::,1], x_label='P2', y_label='P1', title=f"B1={round(data[0][i][0][0], 3)}")
    """
    
    data, titles, headers = commands.readfile(file, getheaders=True, multi_sweep='force', multi_force_def=np.nan)
    return data, titles, headers

def showfile2dim(data, x_label='', y_label='', title='', cbar=False,
                 is_alternate=False, transpose=False, deinterlace=False,
                 out_id=2):
    """
    data is the result of readfileNdim[0]
    titles is the result of readfileNdim[1]
    transpose: False | True (data and axes)
    """
    if is_alternate:
        img = a.alternate(data[out_id]).T
    else:
        img = data[out_id].T
        
    if transpose != False:
        x_label, y_label = y_label, x_label
        img = img.T
    
    imshow_kw = dict(x_axis=data[0][::,1], y_axis=data[1,0], 
                     x_label=x_label, y_label=y_label, title=title, cbar=cbar)

    if deinterlace:
        img1 = img.T[0::2, :].T
        img2 = img.T[1::2, :].T
        p.imshow(img1, **imshow_kw)
        p.imshow(img2, **imshow_kw)
        return        
    p.imshow(img, **imshow_kw)

def read2fileAndPlotDiff(file1, file2, filtre=lambda arr: arr):
    """ for 1 dimensional sweep, assuming the swept values are the same. """
    rf1 = commands.readfile(file1)
    rf2 = commands.readfile(file2)
    sw1, data1 = rf1[0], rf1[1]
    sw2, data2 = rf2[0], rf2[1]
    
    data1, data2 = filtre(data1), filtre(data2)
    
    delta = np.abs(data1 - data2)
    delta_max = np.argmax(delta)
    value_max = sw1[delta_max]
    print(f"Max delta at: {value_max}")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.plot(sw1, data1, label=f"file1")
    ax1.plot(sw1, data2, label=f"file2")
    ax2.plot(sw1, delta)
    ax1.axvline(x=value_max, color='r', linestyle=':', label='Delta max at '+str(value_max))
    ax2.axvline(x=value_max, color='r', linestyle=':', label='Delta max at '+str(value_max))
    ax1.legend()
    return value_max

def getFirstAxisList(data3d):
    return [(i, round(image_i[0][0], 4)) for i, image_i in enumerate(data3d[0])]

def showfile3dim(data, first_axis_label='', x_label='', y_label='', cbar=False,
                 is_alternate=False, transpose=False, deinterlace=False,
                 first_axis_ids=[]):
    """ take the output of a readfile data for a 3d sweep, plot an image for each first axis values
    handles 'alternate' sweep.
    first_axis_ids is a list of ids instead of plotting for each first axis values.
    deinterlace: plot two figure per 2d sweep,
    """
    first_axis_list = getFirstAxisList(data)
        
    iterator = range(len(data[0])) if len(first_axis_ids) == 0 else first_axis_ids
    for i in iterator:
        
        
        if is_alternate:
            y_axis = a.flip(data[1,1][::,1])
            img = a.alternate(data[3][i])
            if i%2==1:
                img = a.flip(img.T, axis=-1).T
        else:
            y_axis = data[1,1][::,1]
            img = data[3][i]
        
        imshow_kw = dict(x_axis=data[2,1,0] if transpose else y_axis,
                         y_axis=y_axis if transpose else data[2,1,0], 
                         x_label=x_label if not transpose else y_label, 
                         y_label=y_label if not transpose else x_label,
                         cbar=cbar)

        if deinterlace:
            img1 = img[0::2, :]
            img2 = img[1::2, :]
            p.imshow(img1 if transpose else img1.T, title=f"{first_axis_label}={first_axis_list[i][1]}, paires", **imshow_kw)
            p.imshow(img2 if transpose else img2.T, title=f"{first_axis_label}={first_axis_list[i][1]}, impaires", **imshow_kw)
            continue
            
        p.imshow(img if transpose else img.T, title=f"B1=0.35, B2={first_axis_list[i][1]}", **imshow_kw )

