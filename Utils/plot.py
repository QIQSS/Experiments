from matplotlib import pyplot as plt

# for clipboard and png
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import io
import numpy as np
# show png
import matplotlib.image as mplimg


from . import files
from . import analyse

import matplotlib.colors as mcolors

def imshow(array, show=True,
           save=False, save_fig=False, save_png=False, path='./', filename='', metadata={},
           
           x_axis=None, y_axis=None, x_label='', y_label='',
           x_axis2=None, x_label2='', y_axis2=None, y_label2='',
           x_slice=(None,None), y_slice=(None, None), slice_by_val=False,
           
           title='', text='', text_pos='dr', text_color='grey',
           cbar=True, cbar_label='', cbar_title='',
           grid=False,
           random_cmap=False, randomize_cmap=False,
           use_latex=False,
           figsize=None,
           **plot_kwargs):
    """ my custom imshow function.
    with easier axis extent: x_axis=, y_axis=.
    and saving to npz with all kwargs.
    Custom plot window: ctrl-c copy the plot to clipboard
    
    # saving:
        save = False
        path = './'
        filename = f"YYYMMDD-HHMMSS-{}"
        metadata = {} # any dict, + this function kwargs are appended to it
        show = False, for wanting to save only
        save_fig = False, save the figure object to metadata['_figure']
        save_png = False, save the figure as a png to metadata['_png']

    # axes:
        x_axis: list, will be from min to max.
                int, will be from 0 to value or value to 0 if value is negative
                None, will go from 0 to len(array[0])
        x_label: ''
        y_axis: same as x_axis
        y_label: ''
        x/y_axis2: same as x_axis for the axis on top
        x/y_label2: same as x_label for the axis on top
        
        x_slice: (s1, s2) cut array to array[s1:s2]. s1 and s2 can be None and/or negative
        y_slice: (s1, s2) cut array to array[:,s1:s2]. s1 and s2 can be None and/or negative
                 `int` is interpreted as (0, int)
        slice_by_val: bool, treat slicing tuple not as indexes but values.
                            Will slice at the index of the axis value closest to slice value.
                            (slice only for values on axis"1". axis2 will follow.)
    
        title: ''
        cbar: bool = True
        cbar_label
        cbar_title
        grid: bool = False
    
    # text:    
        call _writeText(ax, text, text_pos, text_color)
        
    # plot kwargs:
        cmap: str, mpl colormap. some idea: viridis, PiYG, seismic, cividis, RdBu, Purples, Blues
        random_cmap: False, use a random matplotlib cmap
        randomize_cmap: False, randomize the points of the cmap
    
    # other
        use_latex = False
    """
    # TODO: add overriding of argument for Npz and NpzDict
    if type(array) == dict: return imshowFromNpzDict(array)
    if type(array) == str: return imshowFromNpz(array)
    
    plot_kwargs['interpolation'] = 'none'
    plot_kwargs['aspect'] = 'auto'
    plot_kwargs['origin'] = 'lower'
    plot_kwargs['cmap'] = plot_kwargs.pop('cmap', 'viridis')
    if random_cmap:
        plot_kwargs['cmap'] = np.random.choice(plt.colormaps())
        print(f"cmap used: {plot_kwargs['cmap']}")
    if randomize_cmap:
        plot_kwargs['cmap'] = _randomizeColormap(plot_kwargs['cmap'])

    # save all kwargs:
    all_kwargs = locals() # !! no new vars before this line
    #print(all_kwargs)
    
    # latex
    plt.rcParams.update({'text.usetex': use_latex})
    
    # AXES:
    x_axis = _prepAxis(x_axis)
    y_axis = _prepAxis(y_axis)
    if None not in x_axis and None in y_axis:
        y_axis = [0, len(array)]
    if None not in y_axis and None in x_axis:
        x_axis = [0, len(array[0])]

    x_axis2 = _prepAxis(x_axis2)
    y_axis2 = _prepAxis(y_axis2)

    # slicing:
    if x_slice != (None, None):
        if isinstance(x_slice, int): x_slice = (0, x_slice)
        x_axis, x_slice = _sliceAxis(x_axis, array.shape[1], x_slice, slice_by_val)
        x_axis2, x_slice = _sliceAxis(x_axis2, array.shape[1], x_slice)
        array = array[:, x_slice[0]:x_slice[1]]
    if y_slice != (None, None):
        if isinstance(y_slice, int): y_slice = (0, y_slice)
        y_axis, y_slice = _sliceAxis(y_axis, array.shape[0], y_slice, slice_by_val)
        y_axis2, y_slice = _sliceAxis(y_axis2, array.shape[0], y_slice)
        array = array[y_slice[0]:y_slice[1]]
            
    extent = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
    if None not in extent:  
        plot_kwargs['extent'] = extent
    
    
    # PLOT
    if cbar:
        fig, [ax, cax] = plt.subplots(1,2, gridspec_kw=dict(width_ratios=[25,1]), figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(array, **plot_kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    #ax.autoscale(False)

    # secondary x/y -axis
    if len(x_axis2) > 1:
        ax2 = ax.twiny()
        ax2.set_xlim(*x_axis2)
        ax2.set_xlabel(x_label2)
    if len(y_axis2) > 1:
        axy2 = ax.twinx()
        axy2.set_ylim(*y_axis2)
        axy2.set_ylabel(y_label2)


    if cbar:
        fig.colorbar(im, label=cbar_label, ax=ax, cax=cax)
        cax.set_title(cbar_title)
    if grid: 
        ax.grid()

    # modded fig
    fig = _modFig(fig)

    # saving
    if save:
        all_kwargs.pop('save')
        all_kwargs.pop('array')
        all_kwargs.pop('show')
        metadata = all_kwargs.pop('metadata')
        metadata['imshow_kwargs'] = all_kwargs
        _saveDataAndFig(path, filename, array, fig, metadata, save_fig, save_png)
        
    if text:
        _writeText(ax, text, text_pos=text_pos, text_color=text_color)
    
    fig.tight_layout()
    if not show: 
        plt.close(fig)
        return
    #fig.show()
    #return fig

def _randomizeColormap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap) 
    colors = cmap(np.linspace(0, 1, 256))
    np.random.shuffle(colors)
    return mcolors.LinearSegmentedColormap.from_list('randomized', colors)

def _prepAxis(axis):
    """ convert axis = int | float | [start, ..., stop]
        to axis = [start, stop]
    for int | float, the axis will be [0, value] or [value, 0]
    """
    if axis is None: return [None]
    if isinstance(axis, (int, float)):
        axis = [0, axis] if axis > 0 else [axis, 0]
    return [axis[0], axis[-1]]
   
def _sliceAxis(axis, nbpts, slice_, slice_by_val=False):
    """ from axis = [start, stop] | [start, ...., stop]
    return the sliced axis and the slice indexes (for consistency with slice_by_val)
    if slice_by_val:
        find the slices index closest to slice_[0] and slice_[1]
        return the axis and the indexes
    """
    full_axis = axis
    if None in full_axis: # => axis not in use
        return full_axis, slice_
    if len(full_axis) == 2:
        full_axis = np.linspace(axis[0], axis[-1], nbpts)
        
    if slice_by_val:
        slice_ = [analyse.findNearest(full_axis, sli, True) if sli is not None else None for sli in slice_]
        slice_[1] = slice_[1]+1 if slice_[1] is not None else None
    sliced_axis = full_axis[slice_[0]:slice_[1]]

    return [sliced_axis[0], sliced_axis[-1]], slice_

def _writeText(ax, text, text_pos='dr', text_color='grey', text_size=12):
    """ wrtie text on ax
    text: str, text to write
    text_pos: str, position of text, two letter, up/down, left/right: ul ur dl dr
    """
    pos_dict = {'u': 0.95, 'r': 0.95, 'd':0.05, 'l':0.05 }
    pos = (pos_dict[text_pos[1]], pos_dict[text_pos[0]])
    va = {'u':'top', 'd':'bottom'}[text_pos[0]]
    ha = {'r':'right', 'l':'left'}[text_pos[1]]
    ax.text(*pos, f"{text}", color=text_color, fontsize=text_size,
        ha=ha, va=va, transform=ax.transAxes)
    
def _saveDataAndFig(path, filename, array, fig=None, metadata={}, save_fig=False, save_png=False):
    if fig and save_fig:
        metadata['_figure'] = fig
    if fig and save_png:
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format='png')
        png_buffer.seek(0)
        metadata['_png'] = png_buffer.getvalue()
    files.saveToNpz(path, filename, array, metadata=metadata)

def _modFig(fig):
    """ mod and return the plt fig with custom stuff """
    # ctrl+c to copy to clipboard
    def figToClipboard():
        with io.BytesIO() as buffer:
            fig.savefig(buffer)
            QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))
    def onKeyPress(event):
        if event.key == "ctrl+c":
            figToClipboard()
    fig.canvas.mpl_connect('key_press_event', onKeyPress)
    return fig            
    
def showPng(binary):
    """ display a png inside a mpl figure.
    binary: dictionnary from an imshow saving with save=True, save_png=True
            or directly a png in binary
            or a path to an npz file"""
    if type(binary) == str:
        npz = files.loadNpz(binary)
        return showPng(npz)
    if type(binary) == dict:
        binary = binary.get('metadata',None).get('_png',None)
        if not binary:
            print('no png in dict')
            return
    buffer = io.BytesIO(binary)
    img = mplimg.imread(buffer)
    plt.imshow(img)
    plt.axis('off')
    
def imshowFromNpzDict(npzdict):
    """ imshow the npzdict returned from a loadNpz:
        {'array':.., 'metadata':{'imshow_kwargs':{}, ...}} """
        
    imshow_kwargs = npzdict.get('metadata', {}).get('imshow_kwargs', {})
    array = npzdict.get('array')
    plot_kwargs = imshow_kwargs.pop('plot_kwargs', {})
    imshow(array, **imshow_kwargs, **plot_kwargs)
    
def imshowFromNpz(filename, return_dict=False, **kwargs):
    """ if the Npz was saved with imshow(array, save=True),
    it will load the file and call imshow with the right kwargs
    """
    npzdict = files.loadNpz(filename)
    imshowFromNpzDict(npzdict)
    if return_dict: return npzdict
 
    
    

def qplot(x, y=None, x_label='', y_label='', title='', same_fig=False,
          legend=False,
          
          text='', text_pos='dr', text_color='grey',
          
          **plot_kwargs):
    """ quick 1d plot """
    if not same_fig:
        plt.figure()
    ax = plt.gca()
    
    if y is None:
        plt.plot(x, **plot_kwargs)
    else:
        plt.plot(x, y, **plot_kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    if legend: plt.legend()
    if text: _writeText(ax, text, text_pos=text_pos, text_color=text_color)





def plotColumns(array, interval, x_axis=None, y_axis=None, x_label='', y_label='', title='', 
                z_label='', reverse=False, cbar=False):
    """chatgpt
    Plots every 'interval'-th column of a 2D array with a color gradient.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap('viridis')
    num_columns = (array.shape[1] - 1) // interval + 1
    
    if reverse:
        column_indices = range(array.shape[1] - 1, -1, -interval)
    else:
        column_indices = range(0, array.shape[1], interval)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a ScalarMappable object for the colorbar
    norm = mcolors.Normalize(vmin=0, vmax=num_columns - 1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    
    colors = []

    # Plot each selected column with a color from the color map
    for i, idx in enumerate(column_indices):
        color = cmap(i / num_columns)  # Normalize index for color map
        colors.append(color)
        if x_axis is None:
            x_values = range(array.shape[0])
        else:
            x_values = x_axis
        
        if y_axis is None:
            y_values = array[:, idx]
        else:
            y_values = y_axis[:, idx]  # Assuming y_axis has the same shape as array
        
        ax.plot(x_values, y_values, color=color, label=f'Column {idx}')

    # Add colorbar if required
    if cbar:
        fig.colorbar(sm, ax=ax, label=z_label)

    # Set axis labels and title
    ax.set_xlabel(x_label if x_label else 'Index')
    ax.set_ylabel(y_label if y_label else 'Value')
    ax.set_title(title)
    #ax.legend()

    plt.show()