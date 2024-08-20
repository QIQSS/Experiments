import time as timemodule
import os
import numpy as np
from matplotlib import pyplot as plt

from pyHegel import commands
from pyHegel.instruments_base import BaseInstrument


from . import analyse as a
from . import plot as p

#### file
def fileIn(paths, full_path=True):
    """ List all files in the directory(/ies).    """
    files = []
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        ls = os.listdir(path)
        ls_full = [os.path.join(path, f) for f in ls]
        files_in_p = [f_or_d for f_or_d in ls_full if os.path.isfile(f_or_d)]
        if full_path:
            files_in_p = [os.path.join(path, f) for f in files_in_p]
        files += files_in_p
    return files


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

def _makeDateFolder(path):
    date = timemodule.strftime("%Y%m%d")
    path += date
    if not os.path.exists(path): os.mkdir(path)
    return path + '/'

def saveToNpz(path, filename, array, metadata={}, make_date_folder=True):
    """ Save array to an npz file.
    metadata is a dictionnary, it can have pyHegel instruments as values: the iprint will be saved.
    """
    if not path.endswith(('/', '\\')):
        path += '/'
    if make_date_folder: path = _makeDateFolder(path)
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

def readfileNdim(file, return_raw=False):
    """ a memo of pyHegel.readfile for sweep with N dimensions 
    for i in range(10):
        imshow(data[3][i].T, x_axis=data[2,1,0], y_axis=data[1,1][::,1], x_label='P2', y_label='P1', title=f"B1={round(data[0][i][0][0], 3)}")
        
    return a dictionnary: {'dev_1':data, 'dev_2':data, ....}
    """
    
    data, titles, headers = commands.readfile(file, getheaders=True, multi_sweep='force', multi_force_def=np.nan)
    if return_raw: return data, titles, headers
    dictionnary = {key:val for key, val in zip(titles, data)}
    dictionnary['_headers'] = headers
    return dictionnary


def _completeAxis(incomplete_axis):
    """ try to build axis from incomplete pyHegel sweep """
    nbpts = len(incomplete_axis)
    nonan = incomplete_axis[~np.isnan(incomplete_axis)]
    #step = (nonan[-1]-nonan[0]) / len(nonan)
    step = round(nonan[1]-nonan[0], 10)
    return np.linspace(nonan[0], nonan[0]+step*(nbpts-1), nbpts)

def showfile2dim(data, x_label='', y_label='', title='', cbar=False,
                 is_alternate=False, transpose=False, deinterlace=False,
                 out_id=2,
                 **kwargs_for_imshow):
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
    
    y_axis = data[1,0]
    x_axis = data[0][::,1]
    x_axis = _completeAxis(x_axis)
    
    imshow_kw = dict(x_axis=[x_axis[0], x_axis[-1]], y_axis=[y_axis[0], y_axis[-1]], 
                     x_label=x_label, y_label=y_label, title=title, cbar=cbar,
                     **kwargs_for_imshow)

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

def showfile3dim(data, first_axis_label='{val}', x_label='', y_label='', cbar=False,
                 is_alternate=False, transpose=False, deinterlace=False,
                 first_axis_ids=[], **imshowkw_user):
    """ take the output of a readfile data for a 3d sweep, plot an image for each first axis values
    handles 'alternate' sweep.
    first_axis_ids is a list of ids instead of plotting for each first axis values.
    deinterlace: plot two figure per 2d sweep,
    title: a string with '{val}' in it. {val} will be replaced by the ith value of the first axis
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
                         cbar=cbar,
                         title=first_axis_label.format(val=first_axis_list[i][1]),
                         **imshowkw_user)

        if deinterlace:
            img1 = img[0::2, :]
            img2 = img[1::2, :]
            base_title = imshow_kw['title']
            imshow_kw['title'] = base_title + ', paire'
            p.imshow(img1 if transpose else img1.T, **imshow_kw)
            imshow_kw['title'] = base_title + ', impaire'
            p.imshow(img2 if transpose else img2.T, **imshow_kw)
            continue
            
        p.imshow(img if transpose else img.T, 
                 **imshow_kw )

