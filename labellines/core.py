from math import atan2, degrees
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.dates import date2num, DateConverter, num2date
from matplotlib.container import ErrorbarContainer
from datetime import datetime


# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):
    '''Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    kwargs : dict, optional
       Optional arguments passed to ax.text
    '''
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # Find first segment of xdata containing x
    for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
        if min(xa, xb) <= x <= max(xa, xb):
            break
    else:
        raise Exception('x label location is outside data range!')

    def x_to_float(x):
        """Make sure datetime values are properly converted to floats."""
        return date2num(x) if isinstance(x, datetime) else x

    xfa = x_to_float(xa)
    xfb = x_to_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]
    y = ya + (yb - ya) * (x_to_float(x) - xfa) / (xfb - xfa)

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform((xfa, ya)) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=rotation, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):
    '''Label all lines with their respective legends.

    Parameters
    ----------
    lines : list of matplotlib lines
       The lines to label
    align : boolean, optional
       If True, the label will be aligned with the slope of the line
       at the location of the label. If False, they will be horizontal.
    xvals : (xfirst, xlast) or array of float, optional
       The location of the labels. If a tuple, the labels will be
       evenly spaced between xfirst and xlast (in the axis units).
    kwargs : dict, optional
       Optional arguments passed to ax.text
    '''
    ax = lines[0].axes

    labLines, labels = [], []
    handles, allLabels = ax.get_legend_handles_labels()

    all_lines = []
    for h in handles:
        if isinstance(h, ErrorbarContainer):
            all_lines.append(h.lines[0])
        else:
            all_lines.append(h)

    # Take only the lines which have labels other than the default ones
    for line in lines:
        if line in all_lines:
            label = allLabels[all_lines.index(line)]
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xvals = ax.get_xlim()  # set axis limits as annotation limits, xvals now a tuple
    if type(xvals) == tuple:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(labLines)+2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

        if isinstance(ax.xaxis.converter, DateConverter):
            # Convert float values back to datetime in case of datetime axis
            xvals = [num2date(x).replace(tzinfo=ax.xaxis.get_units()) 
                     for x in xvals]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)