# -*- coding: utf-8 -*-
import time as timemodule
import os

from . import files as uf
from . import utils as uu
import numpy as np

class Measure:
    """ a measure object
    creates a directory for the measure and save a file inside, containing metadata.
    ``` MEASURE
    meas = um.Measure(LOG_PATH, 'measure1')
    for ...:
        array = ...
        meas.save(array, metadata=dict())
    ```
    
    ``` LOADING
    f_res, f_points = um.Measure.load(PATH+'/', '20240918-133611-measure1')
    
    arr_res = np.array([analyse(p) for p in points])
    
    npz_res = uf.loadNpz(f_res)
    uf.saveToNpz('', f_res, arr_res, metadata=npz_res.metadata, make_date_folder=False, prepend_date=False)
    
    ```

    
    
    """
    def __init__(self, path, name,
                 metadata={},
                 prepend_date=True):
        timestamp = timemodule.strftime('%Y%m%d-%H%M%S-') if prepend_date else ''
        
        self.name = name
        
        self.path = path + '/' + timestamp + self.name + '/'
        if not os.path.exists(self.path): os.mkdir(self.path)
        
        default_metadata = dict(_measure = name,)
        self.metadata = uu.mergeDict(default_metadata, metadata)
        

        uf.saveToNpz(self.path, '_'+self.name, np.array([]), metadata=self.metadata,
                     make_date_folder=False, prepend_date=False)

        self.npts = 0
    
    def saveArray(self, array, metadata):
        fname = uf.saveToNpz(self.path, f"{self.name}_{self.npts}", array, make_date_folder=False, prepend_date=True,
                             metadata=metadata)
        self.npts += 1
        print(fname)
    
    @staticmethod
    def load(path, name):
        files = uf.fileIn(path+name)
        return files[0], files[1:]

        