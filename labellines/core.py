import warnings
from datetime import datetime
from math import atan2, degrees

import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.dates import DateConverter, date2num, num2date


# Label line with line2D label data
def labelLine(line, x, label=None, align=True, drop_label=False, yoffset=0, **kwargs):
    """Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    yoffset : double, optional
        Space to add to label's y position
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """

    def ensure_float(value):
        """Make sure datetime values are properly converted to floats."""
        try:
            # the last 3 boolean checks are for arrays with datetime64 and with
            # a timezone, see these SO posts:
            # https://stackoverflow.com/q/60714568/4549682
            # https://stackoverflow.com/q/23063362/4549682
            # somewhere, the datetime64 with timezone is getting converted to 'O' dtype
            if (
                isinstance(value, datetime)
                or isinstance(value, np.datetime64)
                or np.issubdtype(value.dtype, np.datetime64)
                or str(value.dtype).startswith("datetime64")
                or value.dtype == "O"
            ):
                return date2num(value)
            else:  # another numpy dtype like float64
                return value
        except AttributeError:  # possibly int or other float/int dtype
            return value

    ax = line.axes
    xdata = ensure_float(line.get_xdata())
    ydata = line.get_ydata()

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception(f"The line {line} only contains nan!")

    # Find first segment of xdata containing x
    if len(xdata) == 2:
        i = 0
        xa = min(xdata)
        xb = max(xdata)
    else:
        for imatch, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= ensure_float(x) <= max(xa, xb):
                i = imatch
                break
        else:
            raise Exception("x label location is outside data range!")

    xfa = ensure_float(xa)
    xfb = ensure_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]

    # Handle vertical case
    if xfb == xfa:
        fraction = 0.5
    else:
        fraction = (ensure_float(x) - xfa) / (xfb - xfa)
    y = ya + (yb - ya) * fraction + yoffset

    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(
            (
                "%s could not be annotated due to `nans` values. "
                "Consider using another location via the `x` argument."
            )
            % line,
            UserWarning,
        )
        return

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform(
            (xfa, ya)
        ) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = 0

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    ax.text(x, y, label, rotation=rotation, **kwargs)


def labelLines(
    lines,
    align=True,
    xvals=None,
    drop_label=False,
    shrink_factor=0.05,
    yoffsets=0,
    **kwargs,
):
    """Label all lines with their respective legends.

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
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    shrink_factor : double, optional
       Relative distance from the edges to place closest labels. Defaults to 0.05.
    yoffsets : number or list, optional.
        Distance relative to the line when positioning the labels. If given a number,
        the same value is used for all lines.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
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
        xvals_rng = xvals[1] - xvals[0]
        shrinkage = xvals_rng * shrink_factor
        xvals = (xvals[0] + shrinkage, xvals[1] - shrinkage)
    if type(xvals) == tuple:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(labLines) + 2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(labLines) + 2)[1:-1]

        if isinstance(ax.xaxis.converter, DateConverter):
            # Convert float values back to datetime in case of datetime axis
            xvals = [num2date(x).replace(tzinfo=ax.xaxis.get_units()) for x in xvals]

    txts = []
    try:
        yoffsets = [float(yoffsets)] * len(labLines)
    except TypeError:
        pass
    for line, x, yoffset, label in zip(labLines, xvals, yoffsets, labels):
        txts.append(labelLine(line, x, label, align, drop_label, yoffset, **kwargs))

    return txts
