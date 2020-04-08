#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import numpy as np

matplotlib.use("agg")
plt.style.use("seaborn-whitegrid")


def gradient_fill(x, y, line_color, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    line_color : tuple of three continues values
        between 0 and 1 representing the RGB color
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
        https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    # Setting axes
    if ax is None:
        ax = plt.gca()
        ax.set_facecolor((0, 0, 0))

    # Defining line and fill colors
    (line,) = ax.plot(x, y, color=line_color, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    # Set alpha (transparency) values of fills
    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    # make shaded fill
    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    # set x and y limits
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(
        z,
        aspect="auto",
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        zorder=zorder,
    )

    # plot fill
    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor="none", edgecolor="none", closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im
