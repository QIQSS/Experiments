from matplotlib import pyplot as plt

# for clipboard and png
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import io
import numpy as np
# show png
import matplotlib.image as mplimg


from . import files as uf
from . import analyse as ua
from . import utils as uu

import matplotlib.colors as mcolors

COLORS = ['#029ac3', '#ffba1e', '#59a694', '#fe66ca', '#cd93f9']
COLORS_D = ['#5b1d2c', '#c6e1ea', '#505c76', '#162067', '#070c20']


#### BASIC PLOTS

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
           return_fig=False,
           figsize=None, ax=None,
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
    # TODO: better handling of given ax
    if not ax:
        if cbar:
            fig, [ax, cax] = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[25, 1]), figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        if cbar:
            cax = fig.add_axes([1.0, 0.12, 0.02, 0.8])
        else:
            cax = None
        
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


    if cbar and cax:
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
    if return_fig:
        return fig

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
        slice_ = [ua.findNearest(full_axis, sli, 'id') if sli is not None else None for sli in slice_]
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
    uf.saveToNpz(path, filename, array, metadata=metadata)

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
        npz = uf.loadNpz(binary)
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
    npzdict = uf.loadNpz(filename)
    imshowFromNpzDict(npzdict)
    if return_dict: return npzdict

def qplot(array, show=True,
          save=False, save_fig=False, save_png=False, path='./', filename='', metadata={},
          
          x_axis=None, x_label='', y_label='',
          x_slice=(None, None), slice_by_val=False,
          
          title='', text='', text_pos='dr', text_color='grey',
          grid=True,
          log_y=False, log_x=False,
          
          vline = None,
          
          use_latex=False,
          return_fig=False,
          figsize=None, ax=None,
          **plot_kwargs):
    """ Custom 1D plot function for quick plotting.
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
                None, will go from 0 to len(array)
        x_label: ''
        y_label: ''
        
        x_slice: (s1, s2) cut array to array[s1:s2]. s1 and s2 can be None and/or negative
        slice_by_val: bool, treat slicing tuple not as indexes but values.
                            Will slice at the index of the axis value closest to slice value.
    
        title: ''
        grid: bool = False
    
    # text:    
        call _writeText(ax, text, text_pos, text_color)
        
    # plot kwargs:
        color: str, line color
        linestyle: str, line style (e.g., '-', '--', '-.', ':')
        marker: str, marker style (e.g., 'o', '^', 's')
    
    # other
        use_latex = False
    """
    # Ensure array is a numpy array
    array = np.asarray(array)
    
    # save all kwargs:
    all_kwargs = locals() # !! no new vars before this line
    
    # latex
    plt.rcParams.update({'text.usetex': use_latex})
    
    # AXES:
    if x_axis is None:
        x_axis = np.arange(len(array))
    elif isinstance(x_axis, (int, float)):
        x_axis = np.linspace(0, x_axis, len(array))
    elif len(x_axis) != len(array):
        x_axis = np.linspace(x_axis[0], x_axis[-1], len(array))
    else:
        x_axis = np.asarray(x_axis)
    
    if x_slice != (None, None):
        if isinstance(x_slice, int): x_slice = (0, x_slice)
        if slice_by_val:
            x_slice = (np.searchsorted(x_axis, x_slice[0]), np.searchsorted(x_axis, x_slice[1]))
        array = array[x_slice[0]:x_slice[1]]
        x_axis = x_axis[x_slice[0]:x_slice[1]]
    
    # PLOT
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.plot(x_axis, array, **plot_kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if grid: 
        ax.grid()
    if log_y:
        ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
    if vline:
        for vl in uu.ensureList(vline):
            ax.axvline(x=vl, linestyle=':', label=str(vline))
            
    # modded fig
    fig = _modFig(fig)
    
    # saving
    if save:
        all_kwargs.pop('save')
        all_kwargs.pop('array')
        all_kwargs.pop('show')
        metadata = all_kwargs.pop('metadata')
        metadata['qplot_kwargs'] = all_kwargs
        _saveDataAndFig(path, filename, array, fig, metadata, save_fig, save_png)
    
    if text:
        _writeText(ax, text, text_pos=text_pos, text_color=text_color)
    
    fig.tight_layout()
    if not show: 
        plt.close(fig)
        return
    if return_fig:
        return fig

def scatter(tuplelist):
    """ tuplelist: [(x,y,val), ...]
    """
    x, y = np.asarray(tuplelist)[:,0], np.asarray(tuplelist)[:,1]
    c = [i for i in range(len(tuplelist))]
    plt.scatter(x, y, c=c, cmap='inferno', s=150)

def scatterOverImg(sc_x, sc_y, sc_pts, im_x, im_y, im_pts,
                   sc_cbar_label=''):
    fig, ax = plt.subplots()
    img = imshow(im_pts, ax=ax, x_axis=im_x, y_axis=im_y,
                 return_fig=True, cbar=False)
    sc = ax.scatter(sc_x, sc_y, c=sc_pts, cmap='plasma', edgecolor='black', s=100, alpha=0.6)
    cbar_sc = plt.colorbar(sc, ax=ax)
    cbar_sc.set_label(sc_cbar_label)
    plt.show()


#### SPECIAL CASE PLOTS

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

    
def imshowSideBySide(*imgs, **kwargs):
    """ call imshow for imgs in a side by side figure.
    cbar is not supported.
    """
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    if len(imgs) == 1: axes = [axes]
    for ax, img in zip(axes, imgs):
        imshow(img, ax=ax, cbar=False, **kwargs)
        #ax.set_title(title)
        #ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plotDoubleGaussian(x, sigma1, sigma2, mu1, mu2, A1, A2, points=None,
                       vline=None, title=''):
    
    fig, ax = plt.subplots()
    gauss1 = ua.f_gaussian(x, sigma1, mu1, A1)
    gauss2 = ua.f_gaussian(x, sigma2, mu2, A2)
    c1, c2, c3, c4 = COLORS[0], COLORS[1], COLORS[3], COLORS[4]
    ax.plot(x, gauss1, c1)
    ax.fill_between(x, gauss1.min(), gauss1, facecolor=c1, alpha=0.5)
    ax.plot(x, gauss2, c2)
    ax.fill_between(x, gauss2.min(), gauss2, facecolor=c2, alpha=0.5)
    if points is not None:
        ax.plot(x, points, c3)
    if vline:
        ax.axvline(x=vline, color=c4, linestyle=':', label=str(vline))
        fig.legend()
    ax.set_title(title)
    
    
def plotSideBySide(*args, inline=False, link_all=False):
    """ plot arrays in args side by side.
    can take 1d or 2d arrays
    """
    import pyqtgraph as pg
    from pyqtgraph import colormap
    
    w = pg.GraphicsLayoutWidget()
    plots = [] # plots and images
    trace_plots = []
    trace_lines = []
    images = [] # images only
    
    def update_trace(line, index):
        pos = line.value()
        if 0 <= pos < images[index].shape[1]:
            trace_data = images[index][int(pos)]
            trace_plots[index].clear()
            trace_plots[index].plot(trace_data)
        else:
            trace_plots[index].clear()

    def sync_lines(line):
        pos = line.value()
        for l in trace_lines:
            l.setValue(pos)

    def line_moved(l, idx):
        update_trace(l, idx)
        if link_all:
            sync_lines(l)

    print(len(args))
    for i, arr in enumerate(args):
        if arr.ndim == 2:
            p = w.addPlot(row=0, col=i)
            img = pg.ImageItem(image=arr.T)
            img.setLookupTable(colormap.get('viridis').getLookupTable())
            p.addItem(img)
            images.append(arr)

            # draggable region
            line = pg.InfiniteLine(angle=0, movable=True)
            line.setZValue(10)
            p.addItem(line)
            trace_lines.append(line)
            
            # Add a plot below the 2D plot for the trace
            trace_plot = w.addPlot(row=1, col=i)
            trace_plots.append(trace_plot)
            trace_plot.setFixedHeight(100)
            line.sigPositionChanged.connect(lambda l, idx=i: line_moved(l, idx))

        elif np.asarray(arr).ndim == 1:
            p = w.addPlot(row=0, col=i)
            p.plot(arr)
        else:
            raise ValueError(f"Array at index {i} is not 1D or 2D")

        plots.append(p)
    

    if link_all:
        first_plot = plots[0]
        for p in plots:
            if p != first_plot:
                p.setXLink(first_plot)
                p.setYLink(first_plot)

    if inline:         
        from IPython.display import display, Image
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
        exporter = pg.exporters.ImageExporter(w.scene())
        exporter.export(temp_filename)
        display(Image(filename=temp_filename))
        os.remove(temp_filename)
    else:
         w.show()
         
    return w