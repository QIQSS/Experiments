from matplotlib import pyplot as plt

# for clipboard and png
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import io
import numpy as np
# show png
import matplotlib.image as mplimg


from . import load

import matplotlib.colors as mcolors

def randomize_colormap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap) 
    colors = cmap(np.linspace(0, 1, 256))
    np.random.shuffle(colors)
    return mcolors.LinearSegmentedColormap.from_list('randomized', colors)

def imshow(array, 
           save=False, save_fig=False, save_png=False, path='./', filename='', metadata={},
           x_axis=[None], y_axis=[None], x_label='', y_label='',
           x_axis2=[None], x_label2='', y_axis2=[None], y_label2='',
           title='', text_on_graph='',
           cbar=True, cbar_label='', cbar_title='',
           grid=False,
           random_cmap=False, randomize_cmap=False,
           use_latex=False,
           **plot_kwargs):
    """ my custom imshow function.
    with easier axis extent: x_axis=, y_axis=.
    and saving to npz with all kwargs.
    Custom plot window: ctrl-c copy the plot to clipboard
    
    # saving:
        save = False
        path = './'
        filename = f"YYYMMDD-HHMMSS-{}"
        metadata = {} # this function kwargs are automatically appended to it
        show = False, for wanting to save only
        save_fig = False, save the figure object to metadata['_figure']
        save_png = False, save the figure as a png to metadata['_png']

    # axes:
        x_axis: list, will be from min to max.
                int, will be from 0 to value or value to 0 if value is negative
                None,
        x_label: ''
        y_axis: same as x_axis
        y_label: ''
        
        title: ''
        cbar: bool = True
        cbar_label
        cbar_title
        grid: bool = False
        
        x_axis2: same as x_axis for the axis on top
        x_label2: same as x_label for the axis on top
    
    # plot kwargs:
        cmap: str, 
            some idea: viridis, PiYG, seismic, cividis, RdBu, Purples, Blues
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
        plot_kwargs['cmap'] = randomize_colormap(plot_kwargs['cmap'])

    # save all kwargs:
    all_kwargs = locals()
    #print(all_kwargs)
    
    # latex
    plt.rcParams.update({'text.usetex': use_latex})
    
    # AXES: [start, stop] or just 'stop' (will be converted to [0, stop])
    def _prepAxis(axis):
        if isinstance(axis, (int, float)):
            axis = [0, axis] if axis > 0 else [axis, 0]
        return axis
    x_axis = _prepAxis(x_axis)
    y_axis = _prepAxis(y_axis)
    if None not in x_axis and None in y_axis:
        y_axis = [0, len(array)]
        
    extent = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
    if None not in extent:  
        plot_kwargs['extent'] = extent
    
    x_axis2 = _prepAxis(x_axis2)
    y_axis2 = _prepAxis(y_axis2)
    
    # PLOT
    if cbar:
        fig, [ax, cax] = plt.subplots(1,2, gridspec_kw=dict(width_ratios=[25,1]),)
    else:
        fig, ax = plt.subplots()
    im = ax.imshow(array, **plot_kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    #ax.autoscale(False)

    # secondary x/y -axis
    if x_axis2 != [None]:
        ax2 = ax.twiny()
        ax2.set_xlim(*x_axis2)
        ax2.set_xlabel(x_label2)
    if y_axis2 != [None]:
        axy2 = ax.twinx()
        axy2.set_ylim(*y_axis2)
        axy2.set_ylabel(y_label2)


    if cbar:
        fig.colorbar(im, label=cbar_label, ax=ax, cax=cax)
        cax.set_title(cbar_title)
    if grid: 
        ax.grid()

    # modded fig
    def figToClipboard():
        with io.BytesIO() as buffer:
             fig.savefig(buffer)
             QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))
    def onKeyPress(event):
        if event.key == "ctrl+c":
            figToClipboard()
    fig.canvas.mpl_connect('key_press_event', onKeyPress)

    # saving
    if save:
        all_kwargs.pop('save')
        all_kwargs.pop('array')
        metadata = all_kwargs.pop('metadata')
        metadata['imshow_kwargs'] = all_kwargs
        if save_fig:
            metadata['_figure'] = fig
        if save_png:
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format='png')
            png_buffer.seek(0)
            metadata['_png'] = png_buffer.getvalue()
        text_on_graph = load.saveToNpz(path, filename, array, metadata=metadata)
                
    if text_on_graph:
        ax.text(0.95, 0.05, f"{text_on_graph}", color='grey', fontsize=12,
            ha='right', va='bottom', transform=ax.transAxes)
    
    fig.tight_layout()
    #fig.show()
    #return fig
    
def showPng(binary):
    """ display a png inside a mpl figure.
    binary: dictionnary from an imshow saving
            or directly a png in binary
            or a path to an npz file"""
    if type(binary) == str:
        npz = load.loadNpz(binary)
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
    imshow_kwargs = npzdict.get('metadata', {}).get('imshow_kwargs', {})
    array = npzdict.get('array')
    plot_kwargs = imshow_kwargs.pop('plot_kwargs', {})
    imshow(array, **imshow_kwargs, **plot_kwargs)
    
def imshowFromNpz(filename, return_dict=False, **kwargs):
    """ if the Npz was saved with imshow(array, **kwargs, save=True),
    it will load the file and call imshow(array, **kwargs)
    """
    npzdict = load.loadNpz(filename)
    imshowFromNpzDict(npzdict)
    if return_dict: return npzdict
 
    
    

def qplot(x, y=None, x_label='', y_label='', title='', same_fig=False):
    """ quick 1d plot """
    if not same_fig:
        plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


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
