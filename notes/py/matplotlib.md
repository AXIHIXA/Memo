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
- Each `Figure` can contain one or more [`Axes`](https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes) (i.e., an area where points can be specified in terms of x-y coordinates (or theta-r in a polar plot, or x-y-z in a 3D plot, etc.)
    - The most simple way of creating a figure with an axes is using [`matplotlib.pyplot.subplots`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots): 
    ```
    matplotlib.pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, 
    **fig_kw)
    
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
    - We can then use [`Axes.plot`](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot) to draw some data on the axes. 
    ```
    plot(self, *args, scalex=True, scaley=True, data=None, **kwargs)
    
    Plot y versus x as lines and/or markers.
    
    Call signatures::
    
        plot([x], y, [fmt], *, data=None, **kwargs)
        plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
    
    The coordinates of the points or line nodes are given by *x*, *y*.
    
    The optional parameter *fmt* is a convenient way for defining basic
    formatting like color, marker and linestyle. It's a shortcut string
    notation described in the *Notes* section below.
    
    >>> plot(x, y)        # plot x and y using default line style and color
    >>> plot(x, y, 'bo')  # plot x and y using blue circle markers
    >>> plot(y)           # plot y using x as index array 0..N-1
    >>> plot(y, 'r+')     # ditto, but with red plusses
    
    You can use `.Line2D` properties as keyword arguments for more
    control on the appearance. Line properties and *fmt* can be mixed.
    The following two calls yield identical results:
    
    >>> plot(x, y, 'go--', linewidth=2, markersize=12)
    >>> plot(x, y, color='green', marker='o', linestyle='dashed',
    ...      linewidth=2, markersize=12)
    
    When conflicting with *fmt*, keyword arguments take precedence.
    
    
    **Plotting labelled data**
    
    There's a convenient way for plotting objects with labelled data (i.e.
    data that can be accessed by index ``obj['y']``). Instead of giving
    the data in *x* and *y*, you can provide the object in the *data*
    parameter and just give the labels for *x* and *y*::
    
    >>> plot('xlabel', 'ylabel', data=obj)
    
    All indexable objects are supported. This could e.g. be a `dict`, a
    `pandas.DataFame` or a structured numpy array.
    
    
    **Plotting multiple sets of data**
    
    There are various ways to plot multiple sets of data.
    
    - The most straight forward way is just to call `plot` multiple times.
      Example:
    
      >>> plot(x1, y1, 'bo')
      >>> plot(x2, y2, 'go')
    
    - Alternatively, if your data is already a 2d array, you can pass it
      directly to *x*, *y*. A separate data set will be drawn for every
      column.
    
      Example: an array ``a`` where the first column represents the *x*
      values and the other columns are the *y* columns::
    
      >>> plot(a[0], a[1:])
    
    - The third way is to specify multiple sets of *[x]*, *y*, *[fmt]*
      groups::
    
      >>> plot(x1, y1, 'g^', x2, y2, 'g-')
    
      In this case, any additional keyword argument applies to all
      datasets. Also this syntax cannot be combined with the *data*
      parameter.
    
    By default, each line is assigned a different style specified by a
    'style cycle'. The *fmt* and line property parameters are only
    necessary if you want explicit deviations from these defaults.
    Alternatively, you can also change the style cycle using the
    'axes.prop_cycle' rcParam.
    
    
    Parameters
    ----------
    x, y : array-like or scalar
        The horizontal / vertical coordinates of the data points.
        *x* values are optional and default to `range(len(y))`.
    
        Commonly, these parameters are 1D arrays.
    
        They can also be scalars, or two-dimensional (in that case, the
        columns represent separate data sets).
    
        These arguments cannot be passed as keywords.
    
    fmt : str, optional
        A format string, e.g. 'ro' for red circles. See the *Notes*
        section for a full description of the format strings.
    
        Format strings are just an abbreviation for quickly setting
        basic line properties. All of these and more can also be
        controlled by keyword arguments.
    
        This argument cannot be passed as keyword.
    
    data : indexable object, optional
        An object with labelled data. If given, provide the label names to
        plot in *x* and *y*.
    
        .. note::
            Technically there's a slight ambiguity in calls where the
            second label is a valid *fmt*. `plot('n', 'o', data=obj)`
            could be `plt(x, y)` or `plt(y, fmt)`. In such cases,
            the former interpretation is chosen, but a warning is issued.
            You may suppress the warning by adding an empty format string
            `plot('n', 'o', '', data=obj)`.
    
    Other Parameters
    ----------------
    scalex, scaley : bool, optional, default: True
        These parameters determined if the view limits are adapted to
        the data limits. The values are passed on to `autoscale_view`.
    
    **kwargs : `.Line2D` properties, optional
        *kwargs* are used to specify properties like a line label (for
        auto legends), linewidth, antialiasing, marker face color.
        Example::
    
        >>> plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
        >>> plot([1,2,3], [1,4,9], 'rs',  label='line 2')
    
        If you make multiple lines with one plot command, the kwargs
        apply to all those lines.
    
        Here is a list of available `.Line2D` properties:
    
      agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array
      alpha: float
      animated: bool
      antialiased or aa: bool
      clip_box: `.Bbox`
      clip_on: bool
      clip_path: [(`~matplotlib.path.Path`, `.Transform`) | `.Patch` | None]
      color or c: color
      contains: callable
      dash_capstyle: {'butt', 'round', 'projecting'}
      dash_joinstyle: {'miter', 'round', 'bevel'}
      dashes: sequence of floats (on/off ink in points) or (None, None)
      drawstyle or ds: {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'
      figure: `.Figure`
      fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}
      gid: str
      in_layout: bool
      label: object
      linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
      linewidth or lw: float
      marker: marker style
      markeredgecolor or mec: color
      markeredgewidth or mew: float
      markerfacecolor or mfc: color
      markerfacecoloralt or mfcalt: color
      markersize or ms: float
      markevery: None or int or (int, int) or slice or List[int] or float or (float, float)
      path_effects: `.AbstractPathEffect`
      picker: float or callable[[Artist, Event], Tuple[bool, dict]]
      pickradius: float
      rasterized: bool or None
      sketch_params: (scale: float, length: float, randomness: float)
      snap: bool or None
      solid_capstyle: {'butt', 'round', 'projecting'}
      solid_joinstyle: {'miter', 'round', 'bevel'}
      transform: `matplotlib.transforms.Transform`
      url: str
      visible: bool
      xdata: 1D array
      ydata: 1D array
      zorder: float
    
    Returns
    -------
    lines
        A list of `.Line2D` objects representing the plotted data.
    
    See Also
    --------
    scatter : XY scatter plot with markers of varying size and/or color (
        sometimes also called bubble chart).
    
    Notes
    -----
    **Format Strings**
    
    A format string consists of a part for color, marker and line::
    
        fmt = '[marker][line][color]'
    
    Each of them is optional. If not provided, the value from the style
    cycle is used. Exception: If ``line`` is given, but no ``marker``,
    the data will be a line without markers.
    
    Other combinations such as ``[color][marker][line]`` are also
    supported, but note that their parsing may be ambiguous.
    
    **Markers**
    
    =============    ===============================
    character        description
    =============    ===============================
    ``'.'``          point marker
    ``','``          pixel marker
    ``'o'``          circle marker
    ``'v'``          triangle_down marker
    ``'^'``          triangle_up marker
    ``'<'``          triangle_left marker
    ``'>'``          triangle_right marker
    ``'1'``          tri_down marker
    ``'2'``          tri_up marker
    ``'3'``          tri_left marker
    ``'4'``          tri_right marker
    ``'s'``          square marker
    ``'p'``          pentagon marker
    ``'*'``          star marker
    ``'h'``          hexagon1 marker
    ``'H'``          hexagon2 marker
    ``'+'``          plus marker
    ``'x'``          x marker
    ``'D'``          diamond marker
    ``'d'``          thin_diamond marker
    ``'|'``          vline marker
    ``'_'``          hline marker
    =============    ===============================
    
    **Line Styles**
    
    =============    ===============================
    character        description
    =============    ===============================
    ``'-'``          solid line style
    ``'--'``         dashed line style
    ``'-.'``         dash-dot line style
    ``':'``          dotted line style
    =============    ===============================
    
    Example format strings::
    
        'b'    # blue markers with default shape
        'or'   # red circles
        '-g'   # green solid line
        '--'   # dashed line with default color
        '^k:'  # black triangle_up markers connected by a dotted line
    
    **Colors**
    
    The supported color abbreviations are the single letter codes
    
    =============    ===============================
    character        color
    =============    ===============================
    ``'b'``          blue
    ``'g'``          green
    ``'r'``          red
    ``'c'``          cyan
    ``'m'``          magenta
    ``'y'``          yellow
    ``'k'``          black
    ``'w'``          white
    =============    ===============================
    
    and the ``'CN'`` colors that index into the default property cycle.
    
    If the color is the only part of the format string, you can
    additionally use any  `matplotlib.colors` spec, e.g. full names
    (``'green'``) or hex strings (``'#008000'``).
    ```
    - Example: 
    ```
    fig, ax = plt.subplots()             # Create a figure containing a single axes.
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
    ```
- For each `Axes` graphing method, there is a corresponding function in the `matplotlib.pyplot` module that performs that plot on the *"current"* axes, creating that axes (and its parent figure) if they don't exist yet. So the previous example can be written more shortly as: 
```
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Matplotlib plot.
```
![](https://matplotlib.org/_images/sphx_glr_usage_002.png)

#### 📌 Parts of a Figure

![Components of a Matplotlib figure](https://matplotlib.org/_images/anatomy.png)
- **Figure**
    - The **whole** figure. 
    - The figure keeps track of all the:
        - Child Axes
            - A figure can contain any number of Axes, but will typically have at least one. 
        - Smattering of 'special' artists 
            - titles
            - figure legends, etc
        - Canvas 
            - Don't worry too much about the canvas, it is crucial as it is the object that actually does the drawing to get you your plot, but as the user it is more-or-less invisible to you)
- **Axes**
- **Axis**
- **Artist**















