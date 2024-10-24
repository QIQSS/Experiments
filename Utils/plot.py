from matplotlib import pyplot as plt

from typing import Literal

# for clipboard and png
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QLabel
import io
import numpy as np
# show png
import matplotlib.image as mplimg


from . import files as uf
from . import analyse as ua
from . import utils as uu

import matplotlib.colors as mcolors

COLORS = uu.ModuloList(['#029ac3', '#ffba1e', '#59a694', '#fe66ca', '#cd93f9'])
COLORS = uu.ModuloList([
    '#019CDE',  # Finn's shirt, blue1
    '#FEB825',  # Jake's fur, yellow1
    '#BC9DC9',  # LSP, purple
    '#23B1A5',  # BMO's body, greenish blue
    '#F53E51', # Flame princess accent, reddish pink
    
    '#006DB5',  # Finn's shirt, blue2 
    '#ECE966',  # Tree trunk, yellow2
    '#F171AA',  # Princess Bubblegum, pink   
    '#A6D39A', # Finn's backpack, green1
    '#EE4622', # Flame princess, orange red
    
    #'#00205B', # Marcelin's hair2, dark blue
    '#4DB2CD', # BMO's arm, blue
    '#FECF4D', # Fiona's hair, yellow
    '#F898A3', # Wildberry princess, pink
    '#6CB249', # Finn's backpack, green2
    '#5A1D2D', # Marcelin's boots, purple
    
    #'#620661', # Marcelin's mouth, dark purple
    
])
COLORS_D = uu.ModuloList(['#5b1d2c', '#c6e1ea', '#505c76', '#162067', '#070c20'])

PASTELS = uu.ModuloList(['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF'])

#### IMSHOW
def _imshow_make_kwargs(
        array,
        show=True,
           
           save=False, save_fig=False, save_png=False, path='./', filename='', metadata={}, show_filename=False,
           
           x_axis=None, y_axis=None, x_label='', y_label='',
           x_axis2=None, x_label2='', y_axis2=None, y_label2='',
           x_slice=(None,None), y_slice=(None, None), slice_by_val=False,
           
           transpose=False,
           
           title='', text='', text_pos='dr', text_color='white',
           cbar=True, cbar_label='', cbar_title='',
           cmap: Literal['<mpl_cmap>', 'random'] = 'viridis',
           grid=False,
           randomize_cmap=False,
           
           scatter_points=None, # [(val, x, y), ... , ] 
           scatter_points_label: Literal['none', 'id', 'val', 'both'] = 'none',
           scatter_size: int = 50,
           scatter_cbar: bool = False, 
           scatter_cmap: Literal['<mpl_cmap>'] = 'inferno',
           scatter_cbar_label: str = '',
           scatter_alpha: int = 1,
           scatter_x_id = 2,
           scatter_y_id = 1,
           scatter_c_id = 0,
           
           use_latex: bool = False,
           figsize: tuple = None,
           return_type: Literal['none', 'fig', 'png', 'qt'] = 'none',
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
        
        scatter_points: list of tuples (c, x, y), where c is the color or value, x and y are the coordinates.
        scatter_cmap: colormap for scatter points.
        scatter_cbar_label: colorbar label for scatter points.
        scatter_cbar: boolean to enable colorbar for scatter points.
    
    # text:    
        call _writeText(ax, text, text_pos, text_color)
        
        cmap: str, mpl colormap. some idea: viridis, PiYG, seismic, cividis, RdBu, Purples, Blues, random
        randomize_cmap: False, randomize the points of the cmap
    
    # other
        use_latex = False
    """
    return locals()

def imshow(array, **kwargs):

    # prepare kwargs
    called_kwargs = kwargs
    full_kwargs = _imshow_make_kwargs(array, **called_kwargs)
    kw = full_kwargs
    
    # if array is not an array:
    if isinstance(array, (dict, uu.customDict, uf.NpzDict)): 
        return imshowFromNpzDict(array, **called_kwargs)
    if type(array) == str:
        return imshowFromNpz(array, **called_kwargs)
    
    # default kwargs for plt.imshow
    plot_kwargs = {'interpolation': 'none', 'aspect': 'auto', 'origin': 'lower'}
    
    # prepare colormaps
    cmap = kw['cmap']
    if cmap == 'random':
        cmap = np.random.choice(plt.colormaps())
        print(f"cmap used: {cmap}")
    if kw['randomize_cmap']:
        plot_kwargs['cmap'] = _randomizeColormap(plot_kwargs['cmap'])
    plot_kwargs['cmap'] = cmap

    # latex
    plt.rcParams.update({'text.usetex': kw['use_latex']})
    
    # AXES:
    x_axis = _prepAxis(kw['x_axis'])
    y_axis = _prepAxis(kw['y_axis'])
    if None not in x_axis and None in y_axis:
        y_axis = [0, len(array)]
    if None not in y_axis and None in x_axis:
        x_axis = [0, len(array[0])]

    x_axis2 = _prepAxis(kw['x_axis2'])
    y_axis2 = _prepAxis(kw['y_axis2'])

    x_slice, y_slice = kw['x_slice'], kw['y_slice']
    x_label, y_label = kw['x_label'], kw['y_label']

    if kw['transpose']:
        x_axis, y_axis = y_axis, x_axis
        x_slice, y_slice = y_slice, x_slice
        x_label, y_label = y_label, x_label
        array = array.T

    # slicing:
    slice_by_val = kw['slice_by_val']
    if x_slice != (None, None):
        if isinstance(x_slice, int): x_slice = (0, x_slice)
        x_axis, x_slice = _sliceAxis(x_axis, array.shape[1], x_slice, slice_by_val)
        x_axis2, x_slice = _sliceAxis(kw['x_axis2'], array.shape[1], x_slice)
        array = array[:, x_slice[0]:x_slice[1]]
    if y_slice != (None, None):
        if isinstance(y_slice, int): y_slice = (0, y_slice)
        y_axis, y_slice = _sliceAxis(y_axis, array.shape[0], y_slice, slice_by_val)
        y_axis2, y_slice = _sliceAxis(kw['y_axis2'], array.shape[0], y_slice)
        array = array[y_slice[0]:y_slice[1]]
        
    extent = (x_axis[0], x_axis[-1], y_axis[0], y_axis[-1])
    if None not in extent:  
        plot_kwargs['extent'] = extent
    
    # PLOT
    # TODO: cbar position / size
    cbar, scatter_cbar = kw['cbar'], kw['scatter_cbar']
    figsize= kw['figsize']
    if cbar and scatter_cbar:
        fig, [ax, cbax, scbax] = plt.subplots(1, 3, gridspec_kw=dict(width_ratios=[25, 1.5, 1.5]), figsize=figsize)
    elif cbar and not scatter_cbar:
        fig, [ax, cbax] = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[25, 1.5]), figsize=figsize)
    elif not cbar and scatter_cbar:
        fig, [ax, scbax] = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[25, 1.5]), figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, gridspec_kw=dict(width_ratios=[25]), figsize=figsize)

    im = ax.imshow(array, **plot_kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(kw['title'])

    fig.cbar = None
    if cbar:
        fig.cbar = fig.colorbar(im, label=kw['cbar_label'], ax=ax, cax=cbax)
        cbax.set_title(kw['cbar_title'])
    
    if kw['grid']: ax.grid()

    # secondary x/y -axis
    if len(x_axis2) > 1:
        ax2 = ax.twiny()
        ax2.set_xlim(*x_axis2)
        ax2.set_xlabel(kw['x_label2'])
    if len(y_axis2) > 1:
        axy2 = ax.twinx()
        axy2.set_ylim(*y_axis2)
        axy2.set_ylabel(kw['y_label2'])

    sc_pts = kw['scatter_points']
    if sc_pts is not None:
        x_id, y_id, c_id = kw['scatter_x_id'], kw['scatter_y_id'], kw['scatter_c_id']
        scatter_x = [pt[x_id] for pt in sc_pts]
        scatter_y = [pt[y_id] for pt in sc_pts]
        scatter_c = [pt[c_id] for pt in sc_pts]
        #scatter_c = [i for i in range(len(sc_pts))]
        scatter = ax.scatter(scatter_x, scatter_y, c=scatter_c, s=kw['scatter_size'], cmap=kw['scatter_cmap'],
                             alpha=kw['scatter_alpha'])
        
        scatter_points_label = kw.get('scatter_points_label', 'none')
        if scatter_points_label in ['id', 'both']:
            for idx, (x, y) in enumerate(zip(scatter_x, scatter_y)):
                ax.text(x, y, str(idx), fontsize=11, ha='right')
        
        if scatter_points_label in ['val', 'both']:
            for (x, y, c) in zip(scatter_x, scatter_y, scatter_c):
                ax.text(x, y, f'{c:.2f}', fontsize=11, ha='right')

        if scatter_cbar:
            fig.colorbar(scatter, label=kw['scatter_cbar_label'], ax=ax, cax=scbax)
            #cbax.set_title(kw['scatter_cbar_title'])
            

    fig.tight_layout()
    modFig2d(fig, ax)
    
    # saving
    if kw['save']:
        called_kwargs.pop('save')
        called_kwargs.pop('show', None)
        metadata = kw.pop('metadata')
        metadata['imshow_kwargs'] = called_kwargs
        path, filename, save_fig, save_png = kw['path'], kw['filename'], kw['save_fig'], kw['save_png']
        
        #print(path, filename, metadata)
        _saveDataAndFig(path, filename, array, fig, metadata, save_fig, save_png)
    
        
    if kw['text']:
        _writeText(ax, kw['text'], text_pos=kw['text_pos'], text_color=kw['text_color'])
    
    #fig.tight_layout()
    if not kw['show']: 
        plt.close(fig)
        
    rt = kw['return_type']
    if rt == 'fig':
            return fig
    elif 'png':
         _figToPng(fig)
    elif 'qt':
        return uu.mplqt(fig)
    elif 'none':
            return

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
    full_axis = uu.ensureList(axis)
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
    
def _figToPng(fig):
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format='png')
    png_buffer.seek(0)
    return png_buffer.getvalue()
    
def _saveDataAndFig(path, filename, array, fig=None, metadata={}, save_fig=False, save_png=False):
    if fig and save_fig:
        metadata['_figure'] = fig
    if fig and save_png:
        metadata['_png'] = _figToPng(fig)
    uf.saveToNpz(path, filename, array, metadata=metadata)


def _figToClipboard(fig):
    with io.BytesIO() as buffer:
        fig.savefig(buffer)
        QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))    
        #print('Custom fig: image copied')

def _modFig(fig, ax):
    """ mod the plt fig with custom stuff (clipboard and vi bindings)"""
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    fig.status = QLabel('')   
    if toolbar := fig.canvas.manager.toolbar:
        toolbar.addSeparator()
        toolbar.addWidget(fig.status)
    def write(t=''):
        comment = fig.mode_comment.get(fig.mode, '')
        mode = 'n' if fig.mode == 'normal' else fig.mode
        if comment:
            mode += f" {comment}"
        fig.status.setText(mode+'>'+f"[{fig.key_mode}]{t}")
    fig.write = write
    
    fig.key_mode = '' # A key modifier.
    fig.mode = 'normal' # A mode is like a key_mode but persistent, with its own bindings
    fig.mode_comment = {'normal':'', 'cbar':'', 'gaussian':'', 'legend':'', 'histogram':'',
                        'markers':'', 'traces':''}
    
    fig.default_lims = [ax.get_xlim(), ax.get_ylim()]
    
    def changeKeyMode(mode='', text=''):
        fig.key_mode = mode
        write(text)
    changeKeyMode()

    fig.onModeChange_functions = {'normal': lambda boo, **kargs: None}
    def changeMode(new):
        fig.onModeChange_functions.get(fig.mode, lambda boo, **kwargs: None)(False, close=False)
        fig.onModeChange_functions.get(new, lambda boo, **kwargs: None)(True)
        fig.mode = new
        
    def zoomReset(xy='xy'):
        if 'x' in xy: ax.set_xlim(*fig.default_lims[0])
        if 'y' in xy: ax.set_ylim(*fig.default_lims[1])
    
    def move(xy, p=0.1):
        lim, setlim = {'x':(ax.get_xlim(),ax.set_xlim), 'y':(ax.get_ylim(),ax.set_ylim)}[xy]
        setlim(lim[0] + p * (lim[1] - lim[0]), lim[1] + p * (lim[1] - lim[0]))

    def zoom(xy, p=0.1):
        lim, setlim = {'x':(ax.get_xlim(),ax.set_xlim), 'y':(ax.get_ylim(),ax.set_ylim)}[xy]
        setlim(lim[0] + p * (lim[1] - lim[0]), lim[1] - p * (lim[1] - lim[0]))

    def onKeyPress(event):
        k = event.key
        
        mode_keys = {
                     'm':'ode: normal, cbar, legend, traces',
                     'q':'uit?', 
                     'z':'oom: in, out, reset', 
                     'zr':'eset: x, y, both',
                     't':'oggle: leg, fscrn, grid, tigh',
                     'f':'ilter: gaussian, histogram'
                    }

        if fig.key_mode == '':
            
            if k in ['escape', 'ctrl+c']:
                changeMode('normal')
            
            elif k in mode_keys.keys():
                changeKeyMode(k, text=mode_keys[k])
                return 'break'
            
            # vi basics
            mvmt = {'h': -0.1, 'l': +0.1, 'j': -0.1, 'k': +0.1}
            mvmt_precise = {'ctrl+'+k : v/10 for k, v in mvmt.items()}
            span = {k.upper() : v for k, v in mvmt.items()}
            span_precise = {'ctrl+'+k : v/10 for k, v in span.items()}
            
            if k in mvmt.keys() and fig.mode == 'normal':
                move('x' if k in ['h', 'l'] else 'y', mvmt[k])
                
            elif k in mvmt_precise.keys() and fig.mode == 'normal':
                move('x' if k in ['ctrl+h', 'ctrl+l'] else 'y', mvmt_precise[k])
                
            elif k in span and fig.mode == 'normal':
                zoom('x' if k in ['H', 'L'] else 'y', span[k])
                
            elif k in span_precise and fig.mode == 'normal':
                zoom('x' if k in ['ctrl+H', 'ctrl+L'] else 'y', span_precise[k])
            
            # copy
            elif k == "y":
                _figToClipboard(fig)
                write(" Yanked!")
                return 'break'
            
        elif fig.key_mode == 'm':
            modes = {'l':'legend', 'c':'colorbar', 'n': 'normal', 't':'traces'}
            if k in modes.keys():
                changeMode(modes[k])

        elif fig.key_mode == 't':
            if k == 'l':
                if ax.get_legend(): ax.get_legend().set_visible(not ax.get_legend().get_visible())
                else: ax.legend([])
            elif k == 'f':
                fig.canvas.manager.full_screen_toggle()
            elif k == 'g':
                ax.grid()
            elif k == 't':
                fig.tight_layout()
        
        # filter
        if fig.key_mode == 'f':
            if k == 'g':
                changeMode('gaussian')
                
            if k == 'h':
                changeMode('histogram')
        
        # zoom mode:  
        elif fig.key_mode == 'z':
            zoomio = {'i': 0.1, 'o': -0.1}
            
            if k == 'r':
                changeKeyMode('zr', mode_keys['zr'])
                return 'break'
            
            
            elif k in 'z':
                zoomReset()
            elif k in zoomio.keys():
                zoom('y', zoomio[k])
                zoom('x', zoomio[k])

        # zoom reset
        elif fig.key_mode == 'zr':
            if k in 'xy':
                zoomReset(k)
            if k in 'b': 
                zoomReset()
        
        
        
        elif fig.key_mode == 'q':
            if k == 'q':
                fig.canvas.manager.destroy()
                fig.canvas.window().close()
                plt.close(fig)
                [fn(False, close=False) for fn in fig.onModeChange_functions.values()]

            elif k == 'a':
                fig.canvas.manager.destroy()
                fig.canvas.window().close()
                plt.close(fig)
                [fn(False, close=True) for fn in fig.onModeChange_functions.values()]

        changeKeyMode()
    
        fig.canvas.draw_idle()

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
    
def imshowFromNpzDict(npzdict, **new_kwargs):
    """ imshow the npzdict returned from a loadNpz:
        {'array':.., 'metadata':{'imshow_kwargs':{}, ...}} """
        
    imshow_kwargs = npzdict.get('metadata', {}).get('imshow_kwargs', {})
    array = npzdict.get('array')
    imshow_kwargs = uu.mergeDict(imshow_kwargs, new_kwargs)
    return imshow(array, **imshow_kwargs)
    
def imshowFromNpz(filename, **new_kwargs):
    """ if the Npz was saved with imshow(array, save=True),
    it will load the file and call imshow with the right kwargs
    """
    npzdict = uf.loadNpz(filename)
    return imshowFromNpzDict(npzdict, **new_kwargs)

#### QPLOT
def _qplot_make_kwargs(
        array,
        x_axis=None,
        multi: bool = False, # if array = [trace1, ..., tracen] -> colormap
        x_label='', y_label='',
        x_axis2=None, x_label2='', y_axis2=None, y_label2='',
        x_slice=(None,None), slice_by_val=False,
        x_log=False, x_log2=False, y_log=False, y_log2=False,
        
        grid=True,
        title='', 
        text='', text_pos='dr', text_color='white',
        
        vline: list[int] = None, vline_style={},
        
        show = True,
        save = False, save_fig=False, save_png=False, 
        path='./', filename='', metadata={},
        
        cbar=True, cbar_label='', cbar_title='',
        cmap: Literal['<mpl_cmap>', 'random'] = 'viridis',
        randomize_cmap=False,
        
        use_latex: bool = False,
        figsize: tuple = None,
        return_type: Literal['none', 'fig', 'png', 'win'] = 'none',
        **plot_kwargs):
    return locals()


def qplot(array, x_axis=None, show=True, multi=False,
          save=False, save_fig=False, save_png=False, path='./', filename='', metadata={},
          
          x_label='', y_label='',
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
    if isinstance(array, str):
        filename = array
        npzdict = uf.loadNpz(filename)
        text = filename.split('/')[-1].split('\\')[-1]
        array = npzdict.array

    elif isinstance(array, (dict, uu.customDict, uf.NpzDict)):
        array = array.array


    # Ensure array is a numpy array
    #array = np.asarray(array)
    
    # save all kwargs:
    all_kwargs = locals() # !! no new vars before this line
    
    # latex
    plt.rcParams.update({'text.usetex': use_latex})
    
    # AXES:
    if multi:
        arrays = array
        array = arrays[0]
        
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
        
    if not multi:
        ax.plot(x_axis, array, **plot_kwargs)
    else:
        [ax.plot(x_axis, arr, **plot_kwargs) for arr in arrays]
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if grid: 
        ax.grid()
    if log_y:
        ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
    if vline is not None:
        for vl in uu.ensureList(vline):
            ax.axvline(x=vl, linestyle=':', label=str(vline))
            
    # modded fig
    fig = _modFig(fig, ax)
    
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

def scatter(tuplelist, x_id=0, y_id=1, val_id=2):
    """ tuplelist: [(val, x, y), ...]
    """
    x, y = np.asarray(tuplelist)[:,x_id], np.asarray(tuplelist)[:,y_id]
    c = np.asanyarray(tuplelist)[:,val_id]
    #c = [i for i in range(len(tuplelist))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=c, cmap='inferno', s=150)
    ax.grid(True)
    cbar = plt.colorbar(sc, ax=ax, label='T1 (s)')
    plt.show()

#### MOD FIG

def legend_lines_toggle(fig, ax):
    
    fig.legend_current_index = 0
    
    lines = list(ax.lines)
    if len(lines)>0:
        leg = ax.legend()
    else:
        leg = ax.legend([])
        
    leg_to_plot = {}
    for i, (legline, line) in enumerate(zip(leg.get_lines(), lines)):
        legline.set_picker(True)
        textline = leg.get_texts()[i]
        textline.set_picker(True)
        leg_to_plot[legline] = (line, legline, textline)
        leg_to_plot[textline] = (line, legline, textline)
        
    def toggle(artist):
        if artist in leg_to_plot:
            line, legline, text = leg_to_plot[artist]
            
            visible = not line.get_visible()
            line.set_visible(visible)
            text.set_alpha(1.0 if visible else 0.2)
            legline.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()
            
    def on_pick(event):
        artist = event.artist
        toggle(artist)
    
    def on_key(event):
        num_entries = len(leg.get_lines())
        k = event.key
        direction = {'j':+1, 'k':-1}
        if k in direction.keys() and fig.mode == 'legend':
            current_index = (fig.legend_current_index + direction[k]) % num_entries
            fig.legend_current_index = current_index
            highlight_current_entry()
            
        elif k == ' ' and fig.mode=='legend':
            textline = leg.get_texts()[fig.legend_current_index]
            toggle(textline)

        elif k == 'c' and fig.mode=='legend':
            textline = leg.get_texts()[fig.legend_current_index]
            line = leg_to_plot[textline][0]
            fig.cursor.visible = not fig.cursor.visible
            fig.cursor.enabled = not fig.cursor.enabled
            # simulate a mouse move on first point. Only way i found to trigger cursor
            from matplotlib.backend_bases import MouseEvent
            first_point = line.get_xydata()[0]
            t = ax.transData
            mouse_event = MouseEvent("motion_notify_event", ax.figure.canvas, *t.transform(first_point))
            ax.figure.canvas.callbacks.process('motion_notify_event', mouse_event)
            

    def highlight_current_entry(boo=True):
        # Reset background for all legend texts
        for i in range(len(leg.get_lines())):
            text = leg.get_texts()[i]
            text.set_bbox(dict(facecolor='none', edgecolor='none'))  # Clear the background
        if not boo: return
        # Highlight the current entry with a colored background
        selected_text = leg.get_texts()[fig.legend_current_index]
        selected_text.set_bbox(dict(facecolor='lightgray', edgecolor='none'))  # Change background color
        
        fig.canvas.draw()
    
    def on_legend_focus(boo, **kwargs):
        highlight_current_entry(boo)
        
    fig.onModeChange_functions['legend'] = on_legend_focus
        

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_key)

def sigma_filter(fig, ax):
    
    fig.gaussian_sigma = 0
        
    def apply_filter():
        fig.test = ax.lines
        for l in ax.lines:
            if not getattr(l, 'is_data', True): continue
            new_ydata = ua.gaussian(l.original_ydata, fig.gaussian_sigma)
            l.set_ydata(new_ydata)
        fig.mode_comment['gaussian'] = f"{fig.gaussian_sigma}"
        fig.write()
        fig.canvas.draw_idle()
        
    def on_key(event):
        direction = {'j':-1, 'k':+1}
        k=event.key
        if k in direction.keys() and fig.mode == 'gaussian':
            fig.gaussian_sigma = fig.gaussian_sigma + direction[k]
            if fig.gaussian_sigma < 0: fig.gaussian_sigma = 0
            apply_filter()
            
        elif k in 'xr' and fig.mode == 'gaussian':
            fig.gaussian_sigma = 0
            apply_filter()
        
        elif k in '0123456789':
            fig.gaussian_sigma = int(k)
            apply_filter()

    fig.canvas.mpl_connect('key_press_event', on_key)

def histogram_window(fig, ax):

    fig.histogram_bins = 100
    fig.histogram_density = False
    fig.histogram_ever_open = False
    
    def plotHist():
        figH, axH = fig.histogram_plot
        axH.clear()
        axH.grid()
        for l in ax.lines:
            x, hist = ua.histogram(np.array(l.get_ydata()).flatten(), bins=fig.histogram_bins, 
                                   return_type='all', density=fig.histogram_density)
            line_hist = axH.plot(x, hist, color=l.get_color(), label=l.get_label())[0]
            line_hist.original_ydata = line_hist.get_ydata()
            line_hist.original_xdata = line_hist.get_xdata()
        figH.canvas.draw_idle()
        figH.default_lims = [axH.get_xlim(), axH.get_ylim()] # reset default lims
        
        fig.mode_comment['histogram'] = f"bins:{fig.histogram_bins}, density:{fig.histogram_density}"
        fig.write()
        fig.canvas.draw_idle()
    
    def on_key(event):
        direction = {'j':-10, 'k':+10, 'J':-100, 'K':100, 'down':-1, 'up':+1}
        k=event.key
        if k in direction.keys() and fig.mode == 'histogram':
            fig.histogram_bins = fig.histogram_bins + direction[k]
            if fig.histogram_bins < 1: fig.histogram_bins = 1
            plotHist()
        elif k == 'd':
            fig.histogram_density = not fig.histogram_density
            plotHist()
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def on_change(boo, close=True):
        if boo:  
            if not fig.histogram_ever_open:
                plt.ioff()
                fig.histogram_plot = plt.subplots()
                modFig1d(*fig.histogram_plot)
                plt.ion()
                fig.histogram_ever_open = True

            figH = fig.histogram_plot[0]
        
            plotHist()
            figH.show()
        elif fig.histogram_ever_open:
            if close:
                figH = fig.histogram_plot[0]
                figH.canvas.window().hide()
                plt.close(figH)
        #fig.canvas.window().setFocus(True)
        
    
    fig.onModeChange_functions['histogram'] = on_change

def markers(fig, ax):
    # DEPRECATED
    fig.markers_position = list(fig.default_lims[0]) + list(fig.default_lims[1]) # v1, v2, h1, h2
    fig.markers_selected = 0    

    from matplotlib.lines import Line2D
    v_lines = [Line2D([pos, pos], [ax.get_ylim()[0], ax.get_ylim()[1]], color='grey', linestyle='--', linewidth=1) for pos in fig.markers_position[:2]]
    h_lines = [Line2D([ax.get_xlim()[0], ax.get_xlim()[1]], [pos, pos], color='grey', linestyle='--', linewidth=1) for pos in fig.markers_position[2:]]

    fig.markers_lines = v_lines + h_lines    
    for l in fig.markers_lines:
        l.is_data = False
        ax.add_artist(l)

    def toggle():
        visible = not v_lines[0].get_visible()  # Check the visibility of the first vertical line
        for line in v_lines + h_lines:
            line.set_visible(visible)
    toggle() # off by default
            
    def move_markers():
        for i, line in enumerate(v_lines + h_lines):
            if i == fig.markers_selected:
                line.set_color('red')
            else:
                line.set_color('grey')
                
        for i, line in enumerate(v_lines):
            line.set_xdata(fig.markers_position[i])  # Update vertical line positions
        for i, line in enumerate(h_lines):
            line.set_ydata(fig.markers_position[i + 2])  # Update horizontal line positions

        fig.mode_comment['markers'] = f"{fig.markers_position}"
        fig.write()
        fig.canvas.draw_idle()
        
    def on_key(event):
        v_direction = {'h':-1, 'l':+1, 'H':-0.1, 'L':+0.1}
        h_direction = {'j':-1, 'k':+1, 'J':-0.1, 'K':+0.1}
        k=event.key
        
        direction = v_direction if fig.markers_selected in [0,1] else h_direction
        if direction is not None and k in direction.keys() and fig.mode == 'markers':
            fig.markers_position[fig.markers_selected] += direction[k]
            move_markers()
        elif k == 's':
            fig.markers_selected = (fig.markers_selected+1) % 4
            move_markers()
        
    fig.canvas.mpl_connect('key_press_event', on_key)


    def on_change(boo, **kwargs):
        toggle()
        if boo: move_markers()
    
    fig.onModeChange_functions['markers'] = on_change


def slider(fig, ax, function, range_, name='param'):
    fig.subplots_adjust(right=0.8)
    ax_slide = fig.add_axes([0.85, 0.2, 0.03, 0.5])
   
    values = np.linspace(range_[0], range_[1], range_[2])
    
    from matplotlib.widgets import Slider
    
    slider = Slider(
        ax_slide, name, range_[0], range_[1],
        valinit=range_[0], valstep=values,
        color=COLORS_D[1],
        orientation="vertical"
    )
    fig.slider = slider # keep globally
    ax.original_lines = list(ax.lines)
    
    for l in ax.lines:
        l.original_data = l.get_ydata()
    
    def update(val):
        for l in ax.lines:
            new_ydata = function(l.original_data, val)
            l.set_ydata(new_ydata)
        fig.canvas.draw_idle()

    slider.on_changed(update)

def cursor_index(fig, ax):
    import mplcursors
    bindings = dict(toggle_enabled = None, toggle_visible = {'key':None}, left='left', right='right')
    c = mplcursors.cursor(ax, hover=True, bindings=bindings)
    
    def set_text(sel):
        if isinstance(sel.index, tuple):
            index = (int(sel.index[0], int(sel.index[1])))
            lbl = "index: "
        else:
            index = int(sel.index)
            xy = sel.artist.get_xydata()[index]
            lbl = f"i: {index}\nx: {xy[0]}\ny: {xy[1]}"
        sel.annotation.set_text(lbl)
    c.connect("add", set_text)
    
    fig.cursor = c # keep globally
    c.enabled = False
    c.visible = False
        

def modFig1d(fig, ax):
    fig = _modFig(fig, ax)

    for l in ax.lines: 
        l.original_ydata = l.get_ydata()
        l.original_xdata = l.get_xdata()
        
    cursor_index(fig, ax)
    legend_lines_toggle(fig, ax)
    sigma_filter(fig, ax)
    histogram_window(fig, ax)
    #markers(fig, ax)
    
def modFig2d(fig, ax):
    fig = _modFig(fig, ax)

    image = ax.images[0]
    
    ## general attr    
    fig.original_data = image.get_array()
    fig.default_clim = [image.get_clim(), image.norm.vmin, image.norm.vmax]

    #### TRACES
    im_shape = fig.original_data.shape
    delta_x = abs(np.diff(fig.default_lims[0])[0])/im_shape[1]
    delta_y = abs(np.diff(fig.default_lims[1])[0])/im_shape[0]
    id_to_xpos = lambda id_: id_ * delta_x + fig.default_lims[0][0] + delta_x / 2
    id_to_ypos = lambda id_: id_ * delta_y + fig.default_lims[1][0] + delta_y / 2
    
    fig.traces_position = 0
    fig.traces_orientation = 'horizontal'
    fig.traces_ever_open = False
    
    vline = ax.axvline(x=id_to_xpos(0), color='red', linestyle='--', linewidth=2, alpha=0.7)
    hline = ax.axhline(y=id_to_ypos(0), color='red', linestyle='--', linewidth=2, alpha=0.7)
    min_marker, = ax.plot([], [], 'o', color='g', markersize=5)
    max_marker, = ax.plot([], [], 'o', color='r', markersize=5)
    min_marker.set_visible(False)
    max_marker.set_visible(False)
    vline.set_visible(False)
    hline.set_visible(False)

    def plotTrace():
        figT, axT = fig.traces_plot
        axT.clear()
        axT.grid()
        if fig.traces_orientation == 'horizontal':
            trace = ax.images[0].get_array()[fig.traces_position]
        elif fig.traces_orientation == 'vertical':
            trace = ax.images[0].get_array()[:,fig.traces_position]
        #plt.pause(0.1)
        l = axT.plot(trace)[0]
        l.original_ydata = l.get_ydata()
        l.original_xdata = l.get_xdata()
        figT.default_lims = [axT.get_xlim(), axT.get_ylim()]
        
        figT.canvas.draw_idle()
    
    def update_trace_line():
        vline.set_visible(fig.traces_orientation == 'vertical')
        hline.set_visible(fig.traces_orientation == 'horizontal')
        min_marker.set_visible(True)
        max_marker.set_visible(True)
        
        if fig.traces_orientation == 'vertical':
            hline.set_visible(False)
            vline.set_visible(True)
            if fig.traces_position >= im_shape[1]:
                fig.traces_position = im_shape[1]-1
            pos_value = id_to_xpos(fig.traces_position)
            vline.set_xdata([pos_value]*2)
            trace = ax.images[0].get_array()[:,fig.traces_position]
            
            min_idx = np.nanargmin(trace)
            max_idx = np.nanargmax(trace)
            min_marker.set_data(pos_value, id_to_ypos(min_idx))
            max_marker.set_data(pos_value, id_to_ypos(max_idx))

            
        elif fig.traces_orientation == 'horizontal':
            hline.set_visible(True)
            vline.set_visible(False)

            if fig.traces_position >= im_shape[0]:
                fig.traces_position = im_shape[0]-1
            pos_value = id_to_ypos(fig.traces_position)
            hline.set_ydata([pos_value]*2)
            trace = ax.images[0].get_array()[fig.traces_position]
            
            min_idx = np.nanargmin(trace)
            max_idx = np.nanargmax(trace)
            
            min_marker.set_data(id_to_xpos(min_idx), pos_value)
            max_marker.set_data(id_to_xpos(max_idx), pos_value)

            
        min_value = round(float(np.nanmin(trace)), 4)
        max_value = round(float(np.nanmax(trace)), 4)
        pos_value = round(pos_value, 4)
        fig.mode_comment['traces'] = f"{fig.traces_orientation[0]}{fig.traces_position}: {pos_value} " \
                              f"<span style='color: red;'>min</span>{min_idx}: {min_value}, " \
                              f"<span style='color: green;'>max</span>{max_idx}: {max_value}"
        fig.write()
        fig.canvas.draw_idle()

    def on_change_traces(boo, close=False):
        if boo:
            if not fig.traces_ever_open:
                plt.ioff()
                fig.traces_plot = plt.subplots()
                #fig.traces_plot[0].set_title('test')
                modFig1d(*fig.traces_plot)
                plt.ion()
                fig.traces_ever_open = True
            
        elif not boo: 
            hline.set_visible(False)
            vline.set_visible(False)
            min_marker.set_visible(False)
            max_marker.set_visible(False)
            if close:
                fig.traces_plot[0].canvas.window().hide()
                plt.close(fig.traces_plot[0])
        update_trace_line()
    fig.onModeChange_functions['traces'] = on_change_traces
    
    #### sigma
    fig.gaussian_sigma = 0
    fig.gaussian_mode = 'lbl' # lbl, 2d
        
    def apply_filter():
        fn = {'lbl': ua.gaussianlbl, '2d': ua.gaussian2d}
        filtered = fn[fig.gaussian_mode](fig.original_data, fig.gaussian_sigma)
        image.set_data(filtered)
        fig.mode_comment['gaussian'] = f"alg:{fig.gaussian_mode}, sigma:{fig.gaussian_sigma}"
        fig.write()
        fig.canvas.draw_idle()
        
    def on_key_gaussian(event):
        direction = {'j':-1, 'k':+1}
        k=event.key
        if k in direction.keys() and fig.mode == 'gaussian':
            fig.gaussian_sigma = fig.gaussian_sigma + direction[k]
            if fig.gaussian_sigma < 0: fig.gaussian_sigma = 0
            apply_filter()
        elif k in 'xr' and fig.mode == 'gaussian':
            fig.gaussian_sigma = 0
            apply_filter()
        elif k in 'ca' and fig.mode == 'gaussian':
            fig.gaussian_mode = {'lbl':'2d', '2d':'lbl'}[fig.gaussian_mode]
            apply_filter()
        elif k in '0123456789' and fig.mode == 'gaussian':
            fig.gaussian_sigma = int(k)
            apply_filter()
        
    def on_change_gaussian(boo, **kwargs):
        if boo: apply_filter()
    fig.onModeChange_functions['gaussian'] = on_change_gaussian
    
    fig.canvas.mpl_connect('key_press_event', on_key_gaussian)
    
    #### hist
    fig.histogram_bins = 100
    fig.histogram_density = False
    fig.histogram_ever_open = False

    def plotHist():
        figH, axH = fig.histogram_plot
        axH.clear()
        x, hist = ua.histogram(image.get_array().flatten(), bins=fig.histogram_bins, 
                               return_type='all', density=fig.histogram_density)
        l = axH.plot(x,hist, label='histogram')[0]
        l.original_ydata = l.get_ydata()
        l.original_xdata = l.get_xdata()
        figH.default_lims = [axH.get_xlim(), axH.get_ylim()]
        axH.grid()
        figH.canvas.draw_idle()
        figH.default_lims = [axH.get_xlim(), axH.get_ylim()] # reset default lims
        fig.mode_comment['histogram'] = f"bins:{fig.histogram_bins}, density:{fig.histogram_density}"
        fig.write()
        fig.canvas.draw_idle()
    
    def on_key_hist(event):
        direction = {'j':-10, 'k':+10, 'J':-100, 'K':100, 'down':-1, 'up':+1}
        k=event.key
        if k in direction.keys() and fig.mode == 'histogram':
            fig.histogram_bins = fig.histogram_bins + direction[k]
            if fig.histogram_bins < 1: fig.histogram_bins = 1
            plotHist()
        elif k == 'd':
            fig.histogram_density = not fig.histogram_density
            plotHist()
    fig.canvas.mpl_connect('key_press_event', on_key_hist)
    
    def on_change_hist(boo, close=False):
        if boo:
            if not fig.histogram_ever_open:
                plt.ioff()
                fig.histogram_plot = plt.subplots()
                modFig1d(*fig.histogram_plot)
                plt.ion()
                
            plotHist()
            figH = fig.histogram_plot[0]
            figH.show()
            
        elif not boo:
            if close and fig.histogram_ever_open:
                figH.canvas.window().hide()
                plt.close(figH)
        
    fig.onModeChange_functions['histogram'] = on_change_hist

    
    #### binds
    def on_key(event):
        key = event.key
        if fig.mode == 'colorbar':
            
            clim = image.get_clim()
            
            delta = 0.1 * (clim[1] - clim[0])
            directions = {'j':[-1,-1], 'k':[1,1], 'J':[1,-1], 'K':[-1,1]}
            if key in directions.keys():
                new_clim = (clim[0] + directions[key][0] * delta, 
                            clim[1] + directions[key][1] * delta)
                image.set_clim(new_clim)
                image.norm.vmin = image.norm.vmin + (new_clim[0] - clim[0])
                image.norm.vmax = image.norm.vmax + (new_clim[1] - clim[1])

            elif key == 'r':
                image.set_clim(fig.default_clim[0])
                image.norm.vmin = fig.default_clim[1]
                image.norm.vmax = fig.default_clim[2]
            elif key in '1234567890':
                cmaps = {'1':'viridis', '2':'RdBu_r', 
                         '3':'inferno', '4':'cividis',
                         '5':'PiYG', '6':'seismic', 
                         '7':'Blues', '8':'Purples', '9':'Greens'}
                if key == '0': 
                    cmap = np.random.choice(plt.colormaps())
                else:
                    cmap = cmaps[key]
                image.set_cmap(cmap)
            
        elif key in 'jkJK' and fig.mode == 'traces':
            fig.traces_position += {'j':-1, 'k':+1, 'J':-20, 'K':+20}[key]
            if fig.traces_position < 0: fig.traces_position = 0
            update_trace_line()
        
        elif key in 'r' and fig.mode == 'traces':
            fig.traces_orientation = {'vertical':'horizontal', 'horizontal':'vertical'}[fig.traces_orientation]
            update_trace_line()

        elif key in 'g' and fig.mode == 'traces':
            fig.traces_position = 0
            update_trace_line()
        
        elif key in 'G' and fig.mode == 'traces':
            fig.traces_position = np.inf
            update_trace_line()
        
        elif key in 'p' and fig.mode == 'traces':
            fig.traces_plot[0].show()
            plotTrace()

    fig.canvas.mpl_connect('key_press_event', on_key)

    
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
    plt.show()
    
    
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
