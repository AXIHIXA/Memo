# `matplotlib` Tutorial Notes

- Notes of reading [`matplotlib` Tutorials](https://matplotlib.org/tutorials/index.html)

## Introductory

### 🌱 [Usage Guide](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py)

This tutorial covers some basic usage patterns and best-practices to help you get started with Matplotlib. 
```
import matplotlib.pyplot as plt
import numpy as np
```

#### 📌 A Simple Example

- Matplotlib graphs your data on [`Figure`](https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure)s. (The top level container for all the plot elements, i.e., windows, Jupyter widgets, etc.)
    - Some useful constructor *keyword* parameters: 
        - `figsize`: 2-tuple of floats, default: `rcParams["figure.figsize"]` (default: `[6.4, 4.8]`)
        - `dpi`: float, default: `rcParams["figure.dpi"]` (default: `100.0`)
- Each of which can contain one or more [`Axes`](https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes) (i.e., an area where points can be specified in terms of x-y coordinates (or theta-r in a polar plot, or x-y-z in a 3D plot, etc.)
    - The most simple way of creating a figure with an axes is using [`pyplot.subplots`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots): 
    ```
    matplotlib.pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)
    Create a figure and a set of subplots.
    
    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.
    
    Parameters
    ----------
    nrows, ncols : int, optional, default: 1
        Number of rows/columns of the subplot grid.
    
    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (`sharex`) or y (`sharey`)
        axes:
    
            - True or 'all': x- or y-axis will be shared among all
              subplots.
            - False or 'none': each subplot x- or y-axis will be
              independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.
    
        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are created. Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are created. To later turn other subplots' ticklabels
        on, use `~matplotlib.axes.Axes.tick_params`.
    
    squeeze : bool, optional, default: True
        - If True, extra dimensions are squeezed out from the returned
          array of `~matplotlib.axes.Axes`:
    
            - if only one subplot is constructed (nrows=ncols=1), the
              resulting single Axes object is returned as a scalar.
            - for Nx1 or 1xM subplots, the returned object is a 1D numpy
              object array of Axes objects.
            - for NxM, subplots with N>1 and M>1 are returned as a 2D array.
    
        - If False, no squeezing at all is done: the returned Axes object is
          always a 2D array containing Axes instances, even if it ends up
          being 1x1.
    
    num : integer or string, optional, default: None
        A `.pyplot.figure` keyword that sets the figure number or label.
    
    subplot_kw : dict, optional
        Dict with keywords passed to the
        `~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.
    
    gridspec_kw : dict, optional
        Dict with keywords passed to the `~matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.
    
    **fig_kw
        All additional keyword arguments are passed to the
        `.pyplot.figure` call.
    
    Returns
    -------
    fig : `~.figure.Figure`
    
    ax : `.axes.Axes` object or array of Axes objects.
        *ax* can be either a single `~matplotlib.axes.Axes` object or an
        array of Axes objects if more than one subplot was created.  The
        dimensions of the resulting array can be controlled with the squeeze
        keyword, see above.
    ```
    - We can then use Axes.plot to draw some data on the axes. 






















