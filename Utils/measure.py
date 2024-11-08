# -*- coding: utf-8 -*-
import time as timemodule
import os

from . import files as uf
from . import utils as uu
import numpy as np

class Measure:
    """ class to handle multi points measures

    creates a directory for the measure and save a file inside containing metadata.
    then use .saveArray(array, metadata) to save a file in the measure.
    
    > DATE-measure_name:
        > _measure_name.npz (meas_file)
        
        > DATE_measure_name_0.npz (points file)
        > ...
        > DATE_measure_name_n.npz
        
    
    ``` MEASURE
    meas = um.Measure(LOG_PATH, 'measure1')
    for ...:
        array = ...
        meas.saveArray(array, metadata=dict())
    ```
    

    
    
    """
    def __init__(self, path, name,
                 metadata={},
                 prepend_date=True):
        timestamp = timemodule.strftime('%Y%m%d-%H%M%S-') if prepend_date else ''
        
        self.name = name
        
        self.path = path + '/' + timestamp + self.name + '/'
        if not os.path.exists(self.path): os.mkdir(self.path)
        
        default_metadata = dict(_measure = name)
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
    def getFiles(path, name):
        """ find the directory that containes 'name'
        returns the meas_file, followed by the list of points files
        """
        # in path, find directories that contains name:
        matches = []
        for dir_name in os.listdir(path):
            #if name in dir_name:
            if dir_name.endswith(name):
                matches.append(os.path.join(path, dir_name))
        if len(matches) > 1:
            raise Exception(f"More than one directory found for measure {name}: {matches}. Use <date>-name to distiguish, or delete the wrong one.")

        elif len(matches) == 0:
            raise Exception(f"Measure {name}: No directory found")
        
        directory = matches[0]    
        files = uf.fileIn(directory, full_path=False)
        
        measure_file = [(i,f) for i,f in enumerate(files) if f"_{name}" in f] 
        files.pop(measure_file[0][0]) # remove measure file
        
        files = [files[i] for i in range(len(files)) if name in files[i]] # remove files if {name} not in it.
        files = [files[i] for i in range(len(files)) if 'exclude' not in files[i]] # remove files if 'exclude' in it.
        
        files = [os.path.join(directory, f) for f in files] # prepend full path
        meas = os.path.join(directory, measure_file[0][1])
        
        
        
        return meas, files

    