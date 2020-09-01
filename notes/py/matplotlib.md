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
    matplotlib.axes.Axes.plot(self, *args, scalex=True, scaley=True, data=None, **kwargs)
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

- `Figure`
    - The **whole** figure. 
    - The figure keeps track of all the: 
        - Child `Axes`; 
            - A figure can contain any number of `Axes`, but will typically have at least one. 
        - Smattering of 'special' artists; 
            - titles; 
            - figure legends, etc. 
        - Canvas. 
            - Don't worry too much about the canvas, it is crucial as it is the object that actually does the drawing to get you your plot, but as the user it is more-or-less invisible to you. 
    - The easiest way to create a new figure is with `pyplot` (It's convenient to create the axes together with the figure, but you can also add axes later on, allowing for more complex axes layouts.): 
    ```
    fig = plt.figure()             # an empty figure with no Axes
    fig, ax = plt.subplots()       # a figure with a single Axes
    fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
    ```
- `Axes`
    - What you think of as *'a plot'*, it is the region of the image with the data space. 
    - A given `Figure` can contain many `Axes`, but a given `Axes` object can only be in one `Figure`. 
    - The `Axes` contains two (or three in the case of 3D) `Axis` objects (be aware of the difference between **Axes** and **Axis**) which take care of the data limits (the data limits can also be controlled via the `axes.Axes.set_xlim()` and `axes.Axes.set_ylim()` methods). 
    - Each `Axes` has a title (set via `set_title()`), an x-label (set via `set_xlabel()`), and a y-label set via `set_ylabel()`). 
    - The Axes class and its member functions are the primary entry point to working with the OO interface. 
- `Axis`
    - These are the number-line-like objects. They take care of setting the graph limits and generating the ticks (the marks on the axis) and ticklabels (strings labeling the ticks). The location of the ticks is determined by a `Locator` object and the ticklabel strings are formatted by a `Formatter`. The combination of the correct `Locator` and `Formatter` gives very fine control over the tick locations and labels. 
- `Artist`
    - Basically everything you can see on the figure is an artist (even the `Figure`, `Axes`, and `Axis` objects). This includes `Text` objects, `Line2D` objects, `collections` objects, `Patch` objects ... (you get the idea). When the figure is rendered, all of the artists are drawn to the canvas. Most Artists are tied to an Axes; such an Artist cannot be shared by multiple Axes, or moved from one to another.

#### 📌 Types of inputs to plotting functions

- All of plotting functions expect `numpy.array` or `numpy.ma.masked_array` as input.  
- Classes that are 'array-like' such as `pandas` data objects and `numpy.matrix` may or may not work as intended. It is best to convert these to `numpy.array` objects prior to plotting. 
```
a = pandas.DataFrame(np.random.rand(4, 5), columns = list('abcde'))
a_asarray = a.values

b = np.matrix([[1, 2], [3, 4]])
b_asarray = np.asarray(b)
```

#### 📌 The object-oriented interface and the pyplot interface

- As noted above, there are essentially two ways to use Matplotlib: 
    - Explicitly create figures and axes, and call methods on them (the "object-oriented (OO) style"). 
    - Rely on `pyplot` to automatically create and manage the figures and axes, and use pyplot functions for plotting. 
- So one can do (OO-style)
```
x = np.linspace(0, 2, 100)

# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig, ax = plt.subplots()              # Create a figure and an axes.
ax.plot(x, x, label='linear')         # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic')   # Plot more data on the axes...
ax.plot(x, x**3, label='cubic')       # ... and some more.
ax.set_xlabel('x label')              # Add an x-label to the axes.
ax.set_ylabel('y label')              # Add a y-label to the axes.
ax.set_title("Simple Plot")           # Add a title to the axes.
ax.legend()                           # Add a legend.
```
- or (pyplot-style)
```
x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')        # Plot some data on the (implicit) axes.
plt.plot(x, x**2, label='quadratic')  # etc.
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
```
![](https://matplotlib.org/_images/sphx_glr_usage_003.png)
- Matplotlib's documentation and examples use both the OO and the pyplot approaches (which are equally powerful), and you should feel free to use either (however, it is preferable pick one of them and stick to it, instead of mixing them). In general, we suggest to restrict pyplot to interactive plotting (e.g., in a Jupyter notebook), and to prefer the OO-style for non-interactive plotting (in functions and scripts that are intended to be reused as part of a larger project). 

#### 📌 Backends

- Three ways to configure your backend: 
    1. The `rcParams["backend"]` (default: `'agg'`) parameter in your matplotlibrc file; 
    2. The `MPLBACKEND` environment variable; 
    3. The function `matplotlib.use()`. 
        - This function **overrides** setting in your matplotlibrc. 
        - This should be done before any figure is created; otherwise Matplotlib may fail to switch the backend and raise an ImportError.
        - Using use will require changes in your code if users want to use a different backend. Therefore, you should avoid explicitly calling use unless absolutely necessary.
- [Built-in Backends](https://matplotlib.org/tutorials/introductory/usage.html#the-builtin-backends): 
    - non-interactive backends, capable of writing to a file. To save plots using the non-interactive backends, use the `matplotlib.pyplot.savefig('filename')` method: 
        - `agg`: `.png` file. raster graphics. 
        - `pdf`: `.pdf` file. vector graphics. 
        - `ps`: `.ps`, `.eps` file. postscript output. vector graphics. 
        - `svg`: `.svg` file. vector graphics. 
        - `pgf`: `.pgf`, `.pdf` file. using the `pgf` package. vector graphics. 
        - `Cairo`: `.png`, `.pdf`, `.ps` or `.svg` file. using the `Cairo` library. raster or vector graphics. 
    - interactive backends. 

### 🌱 [Pyplot Tutorial](https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py)

#### 📌 Intro to pyplot

[`matplotlib.pyplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot) is a collection of functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.

In `matplotlib.pyplot` various *states* are preserved across function calls, so that it keeps track of things like the current figure and plotting area, and the plotting functions are directed to the current axes (please note that "axes" here and in most places in the documentation refers to the axes part of a figure and not the strict mathematical term for more than one axis). 

**Note**: the `pyplot` API is generally less-flexible than the object-oriented API. Most of the function calls you see here can also be called as methods from an `Axes` object. 

```
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])  # this list regarded as y-values and x is set to range(len(y))
plt.ylabel('some numbers')
```

#### 📌 Formatting the style of your plot

For every x, y pair of arguments, there is an optional third argument which is the format string that indicates the color and line type of the plot. The letters and symbols of the format string are from MATLAB, and you concatenate a color string with a line style string. The default format string is 'b-', which is a solid blue line. For example, to plot the above with red circles, you would issue

```
import numpy as np
 
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

plt.axis([0, 5, 0, 120])

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
```

![](https://matplotlib.org/_images/sphx_glr_pyplot_004.png)

Help on [`matplotlib.pyplot.axis`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib.pyplot.axis):
 
```
matplotlib.pyplot.axis(*args, **kwargs)
    Convenience method to get or set some axis properties.
    
    Call signatures::
    
        xmin, xmax, ymin, ymax = axis()
        xmin, xmax, ymin, ymax = axis([xmin, xmax, ymin, ymax])
        xmin, xmax, ymin, ymax = axis(option)
        xmin, xmax, ymin, ymax = axis(**kwargs)
    
    Parameters
    ----------
    xmin, xmax, ymin, ymax : float, optional
        The axis limits to be set. Either none or all of the limits must
        be given. This can also be achieved using ::
    
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
    option : bool or str
        If a bool, turns axis lines and labels on or off. If a string,
        possible values are:
    
        ======== ==========================================================
        Value    Description
        ======== ==========================================================
        'on'     Turn on axis lines and labels. Same as ``True``.
        'off'    Turn off axis lines and labels. Same as ``False``.
        'equal'  Set equal scaling (i.e., make circles circular) by
                 changing axis limits.
        'scaled' Set equal scaling (i.e., make circles circular) by
                 changing dimensions of the plot box.
        'tight'  Set limits just large enough to show all data.
        'auto'   Automatic scaling (fill plot box with data).
        'normal' Same as 'auto'; deprecated.
        'image'  'scaled' with axis limits equal to data limits.
        'square' Square plot; similar to 'scaled', but initially forcing
                 ``xmax-xmin = ymax-ymin``.
        ======== ==========================================================
    
    emit : bool, optional, default *True*
        Whether observers are notified of the axis limit change.
        This option is passed on to `~.Axes.set_xlim` and
        `~.Axes.set_ylim`.
    
    Returns
    -------
    xmin, xmax, ymin, ymax : float
        The axis limits.
```

#### 📌 Plotting with keyword strings

```
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
```

![](https://matplotlib.org/_images/sphx_glr_pyplot_005.png)

#### 📌 Plotting with categorical variables

- `pyplot` way
```
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
```
- oop way
```
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
fig.suptitle('Categorical Plotting')
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.show()
```

![](https://matplotlib.org/_images/sphx_glr_pyplot_006.png)

Help on function `matplotlib.pyplot.subplot`:

```
matplotlib.pyplot.subplot(*args, **kwargs)
    Add a subplot to the current figure.
    
    Wrapper of `.Figure.add_subplot` with a difference in behavior
    explained in the notes section.
    
    Call signatures::
    
        subplot(nrows, ncols, index, **kwargs)
        subplot(pos, **kwargs)
        subplot(ax)
    
    Parameters
    ----------
    *args
        Either a 3-digit integer or three separate integers
        describing the position of the subplot. If the three
        integers are *nrows*, *ncols*, and *index* in order, the
        subplot will take the *index* position on a grid with *nrows*
        rows and *ncols* columns. *index* starts at 1 in the upper left
        corner and increases to the right.
    
        *pos* is a three digit integer, where the first digit is the
        number of rows, the second the number of columns, and the third
        the index of the subplot. i.e. fig.add_subplot(235) is the same as
        fig.add_subplot(2, 3, 5). Note that all integers must be less than
        10 for this form to work.
    
    projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
        The projection type of the subplot (`~.axes.Axes`). *str* is the name
        of a custom projection, see `~matplotlib.projections`. The default
        None results in a 'rectilinear' projection.
    
    polar : boolean, optional
        If True, equivalent to projection='polar'.
    
    sharex, sharey : `~.axes.Axes`, optional
        Share the x or y `~matplotlib.axis` with sharex and/or sharey. The
        axis will have the same limits, ticks, and scale as the axis of the
        shared axes.
    
    label : str
        A label for the returned axes.
    
    Other Parameters
    ----------------
    **kwargs
        This method also takes the keyword arguments for
        the returned axes base class. The keyword arguments for the
        rectilinear base class `~.axes.Axes` can be found in
        the following table but there might also be other keyword
        arguments if another projection is used.
            adjustable: {'box', 'datalim'}
            agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array
            alpha: float
            anchor: 2-tuple of floats or {'C', 'SW', 'S', 'SE', ...}
            animated: bool
            aspect: {'auto', 'equal'} or num
            autoscale_on: bool
            autoscalex_on: bool
            autoscaley_on: bool
            axes_locator: Callable[[Axes, Renderer], Bbox]
            axisbelow: bool or 'line'
            clip_box: `.Bbox`
            clip_on: bool
            clip_path: [(`~matplotlib.path.Path`, `.Transform`) | `.Patch` | None]
            contains: callable
            facecolor: color
            fc: color
            figure: `.Figure`
            frame_on: bool
            gid: str
            in_layout: bool
            label: object
            navigate: bool
            navigate_mode: unknown
            path_effects: `.AbstractPathEffect`
            picker: None or bool or float or callable
            position: [left, bottom, width, height] or `~matplotlib.transforms.Bbox`
            rasterization_zorder: float or None
            rasterized: bool or None
            sketch_params: (scale: float, length: float, randomness: float)
            snap: bool or None
            title: str
            transform: `.Transform`
            url: str
            visible: bool
            xbound: unknown
            xlabel: str
            xlim: (left: float, right: float)
            xmargin: float greater than -0.5
            xscale: {"linear", "log", "symlog", "logit", ...}
            xticklabels: List[str]
            xticks: list
            ybound: unknown
            ylabel: str
            ylim: (bottom: float, top: float)
            ymargin: float greater than -0.5
            yscale: {"linear", "log", "symlog", "logit", ...}
            yticklabels: List[str]
            yticks: list
            zorder: float
    
    Returns
    -------
    axes : an `.axes.SubplotBase` subclass of `~.axes.Axes` (or a subclass     of `~.axes.Axes`)
    
        The axes of the subplot. The returned axes base class depends on
        the projection used. It is `~.axes.Axes` if rectilinear projection
        are used and `.projections.polar.PolarAxes` if polar projection
        are used. The returned axes is then a subplot subclass of the
        base class.
    
    Notes
    -----
    Creating a subplot will delete any pre-existing subplot that overlaps
    with it beyond sharing a boundary::
    
        import matplotlib.pyplot as plt
        # plot a line, implicitly creating a subplot(111)
        plt.plot([1,2,3])
        # now create a subplot which represents the top plot of a grid
        # with 2 rows and 1 column. Since this subplot will overlap the
        # first, the plot (and its axes) previously created, will be removed
        plt.subplot(211)
    
    If you do not want this behavior, use the `.Figure.add_subplot` method
    or the `.pyplot.axes` function instead.
    
    If the figure already has a subplot with key (*args*,
    *kwargs*) then it will simply make that subplot current and
    return it.  This behavior is deprecated. Meanwhile, if you do
    not want this behavior (i.e., you want to force the creation of a
    new subplot), you must use a unique set of args and kwargs.  The axes
    *label* attribute has been exposed for this purpose: if you want
    two subplots that are otherwise identical to be added to the figure,
    make sure you give them unique labels.
    
    In rare circumstances, `.add_subplot` may be called with a single
    argument, a subplot axes instance already created in the
    present figure but not in the figure's list of axes.
```

#### 📌 Controlling line properties

- Lines have many attributes that you can set: linewidth, dash style, antialiased, etc; see [`matplotlib.lines.Line2D`](https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D). There are several ways to set line properties: 
    - Use keyword args: 
    ```
    plt.plot(x, y, linewidth=2.0)
    ```
    - Use the setter methods of a `Line2D` instance. plot returns a list of Line2D objects; e.g., `line1, line2 = plot(x1, y1, x2, y2)`. In the code below we will suppose that we have only one line so that the list returned is of length 1. We use tuple unpacking with `line`, to get the first element of that list: 
    ```
    line, = plt.plot(x, y, '-')
    line.set_antialiased(False) # turn off antialiasing
    ```
    - Use [`setp`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.setp.html#matplotlib.pyplot.setp): The example below uses a MATLAB-style function to set multiple properties on a list of lines. setp works transparently with a list of objects or a single object. You can either use python keyword arguments or MATLAB-style string/value pairs: 
    ```
    lines = plt.plot(x1, y1, x2, y2)
    # use keyword args
    plt.setp(lines, color='r', linewidth=2.0)
    # or MATLAB style string value pairs
    plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
    ```

#### 📌 Working with multiple figures and axes

```
import matplotlib.pyplot as plt
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
```

#### 📌 Working with text

```
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)


plt.xlabel('Smarts', fontsize=14, color='red')
plt.ylabel('Probability')
plt.title(r'$\sigma_i=15$')  # raw str literal representing latex formula
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
```

![](https://matplotlib.org/_images/sphx_glr_pyplot_008.png)

**Annotating text**

The uses of the basic [`text`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.text) function above place text at an arbitrary position on the Axes. A common use for text is to annotate some feature of the plot, and the [`annotate`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html#matplotlib.pyplot.annotate) method provides helper functionality to make annotations easy. In an annotation, there are two points to consider: the location being annotated represented by the argument xy and the location of the text xytext. Both of these arguments are (x, y) tuples. 

```
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()
```

![](https://matplotlib.org/_images/sphx_glr_pyplot_009.png)

Help on function `matplotlib.pyplot.text`:

```
matplotlib.pyplot.text(x, y, s, fontdict=None, withdash=<deprecated parameter>, **kwargs)
    Add text to the axes.
    
    Add the text *s* to the axes at location *x*, *y* in data coordinates.
    
    Parameters
    ----------
    x, y : scalars
        The position to place the text. By default, this is in data
        coordinates. The coordinate system can be changed using the
        *transform* parameter.
    
    s : str
        The text.
    
    fontdict : dictionary, optional, default: None
        A dictionary to override the default text properties. If fontdict
        is None, the defaults are determined by your rc parameters.
    
    withdash : boolean, optional, default: False
        Creates a `~matplotlib.text.TextWithDash` instance instead of a
        `~matplotlib.text.Text` instance.
    
    Returns
    -------
    text : `.Text`
        The created `.Text` instance.
    
    Other Parameters
    ----------------
    **kwargs : `~matplotlib.text.Text` properties.
        Other miscellaneous text parameters.
```

Help on function `matplotlib.pyplot.annotate`:

```
matplotlib.pyplot.annotate(s, xy, *args, **kwargs)
    Annotate the point *xy* with text *text*.
    
    In the simplest form, the text is placed at *xy*.
    
    Optionally, the text can be displayed in another position *xytext*.
    An arrow pointing from the text to the annotated point *xy* can then
    be added by defining *arrowprops*.
    
    Parameters
    ----------
    text : str
        The text of the annotation.  *s* is a deprecated synonym for this
        parameter.
    
    xy : (float, float)
        The point *(x,y)* to annotate.
    
    xytext : (float, float), optional
        The position *(x,y)* to place the text at.
        If *None*, defaults to *xy*.
    
    xycoords : str, `.Artist`, `.Transform`, callable or tuple, optional
    
        The coordinate system that *xy* is given in. The following types
        of values are supported:
    
        - One of the following strings:
    
          =================   =============================================
          Value               Description
          =================   =============================================
          'figure points'     Points from the lower left of the figure
          'figure pixels'     Pixels from the lower left of the figure
          'figure fraction'   Fraction of figure from lower left
          'axes points'       Points from lower left corner of axes
          'axes pixels'       Pixels from lower left corner of axes
          'axes fraction'     Fraction of axes from lower left
          'data'              Use the coordinate system of the object being
                              annotated (default)
          'polar'             *(theta,r)* if not native 'data' coordinates
          =================   =============================================
    
        - An `.Artist`: *xy* is interpreted as a fraction of the artists
          `~matplotlib.transforms.Bbox`. E.g. *(0, 0)* would be the lower
          left corner of the bounding box and *(0.5, 1)* would be the
          center top of the bounding box.
    
        - A `.Transform` to transform *xy* to screen coordinates.
    
        - A function with one of the following signatures::
    
            def transform(renderer) -> Bbox
            def transform(renderer) -> Transform
    
          where *renderer* is a `.RendererBase` subclass.
    
          The result of the function is interpreted like the `.Artist` and
          `.Transform` cases above.
    
        - A tuple *(xcoords, ycoords)* specifying separate coordinate
          systems for *x* and *y*. *xcoords* and *ycoords* must each be
          of one of the above described types.
    
        See :ref:`plotting-guide-annotation` for more details.
    
        Defaults to 'data'.
    
    textcoords : str, `.Artist`, `.Transform`, callable or tuple, optional
        The coordinate system that *xytext* is given in.
    
        All *xycoords* values are valid as well as the following
        strings:
    
        =================   =========================================
        Value               Description
        =================   =========================================
        'offset points'     Offset (in points) from the *xy* value
        'offset pixels'     Offset (in pixels) from the *xy* value
        =================   =========================================
    
        Defaults to the value of *xycoords*, i.e. use the same coordinate
        system for annotation point and text position.
    
    arrowprops : dict, optional
        The properties used to draw a
        `~matplotlib.patches.FancyArrowPatch` arrow between the
        positions *xy* and *xytext*.
    
        If *arrowprops* does not contain the key 'arrowstyle' the
        allowed keys are:
    
        ==========   ======================================================
        Key          Description
        ==========   ======================================================
        width        The width of the arrow in points
        headwidth    The width of the base of the arrow head in points
        headlength   The length of the arrow head in points
        shrink       Fraction of total length to shrink from both ends
        ?            Any key to :class:`matplotlib.patches.FancyArrowPatch`
        ==========   ======================================================
    
        If *arrowprops* contains the key 'arrowstyle' the
        above keys are forbidden.  The allowed values of
        ``'arrowstyle'`` are:
    
        ============   =============================================
        Name           Attrs
        ============   =============================================
        ``'-'``        None
        ``'->'``       head_length=0.4,head_width=0.2
        ``'-['``       widthB=1.0,lengthB=0.2,angleB=None
        ``'|-|'``      widthA=1.0,widthB=1.0
        ``'-|>'``      head_length=0.4,head_width=0.2
        ``'<-'``       head_length=0.4,head_width=0.2
        ``'<->'``      head_length=0.4,head_width=0.2
        ``'<|-'``      head_length=0.4,head_width=0.2
        ``'<|-|>'``    head_length=0.4,head_width=0.2
        ``'fancy'``    head_length=0.4,head_width=0.4,tail_width=0.4
        ``'simple'``   head_length=0.5,head_width=0.5,tail_width=0.2
        ``'wedge'``    tail_width=0.3,shrink_factor=0.5
        ============   =============================================
    
        Valid keys for `~matplotlib.patches.FancyArrowPatch` are:
    
        ===============  ==================================================
        Key              Description
        ===============  ==================================================
        arrowstyle       the arrow style
        connectionstyle  the connection style
        relpos           default is (0.5, 0.5)
        patchA           default is bounding box of the text
        patchB           default is None
        shrinkA          default is 2 points
        shrinkB          default is 2 points
        mutation_scale   default is text size (in points)
        mutation_aspect  default is 1.
        ?                any key for :class:`matplotlib.patches.PathPatch`
        ===============  ==================================================
    
        Defaults to None, i.e. no arrow is drawn.
    
    annotation_clip : bool or None, optional
        Whether to draw the annotation when the annotation point *xy* is
        outside the axes area.
    
        - If *True*, the annotation will only be drawn when *xy* is
          within the axes.
        - If *False*, the annotation will always be drawn.
        - If *None*, the annotation will only be drawn when *xy* is
          within the axes and *xycoords* is 'data'.
    
        Defaults to *None*.
    
    **kwargs
        Additional kwargs are passed to `~matplotlib.text.Text`.
    
    Returns
    -------
    annotation : `.Annotation`
```
​
#### 📌 Logarithmic and other nonlinear axes

`matplotlib.pyplot` supports not only linear axis scales, but also logarithmic and logit scales. This is commonly used if data spans many orders of magnitude. Changing the scale of an axis is easy:
```
plt.xscale('log')
```
An example of four plots with the same data and different scales for the y axis is shown below.

```
# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()
```

![](https://matplotlib.org/_images/sphx_glr_pyplot_010.png)












### 🌱

#### 📌




