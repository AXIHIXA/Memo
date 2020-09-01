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
    - The most simple way of creating a figure with an axes is using [`pyplot.subplots`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots). We can then use Axes.plot to draw some data on the axes. 
    - `matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)`























