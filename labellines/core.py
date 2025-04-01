import warnings
from typing import Optional, Union
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.container import ErrorbarContainer
from matplotlib.dates import (
    _SwitchableDateConverter,
    ConciseDateConverter,
    DateConverter,
    num2date,
)
from matplotlib.lines import Line2D
from more_itertools import always_iterable

from .line_label import LineLabel
from .utils import maximum_bipartite_matching, normalize_xydata


# Label line with line2D label data
def labelLine(
    line: Line2D,
    x,
    label: Optional[str] = None,
    align: Optional[bool] = None,
    drop_label: bool = False,
    xoffset: float = 0,
    xoffset_logspace: bool = False,
    yoffset: float = 0,
    yoffset_logspace: bool = False,
    outline_color: str = "auto",
    outline_width: float = 8,
    rotation: Optional[float] = None,
    **kwargs,
):
    """
    Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    align : boolean, optional
       If True, the label will be aligned with the slope of the line
       at the location of the label. If False, they will be horizontal.
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    xoffset : double, optional
        Space to add to label's x position
    xoffset_logspace : bool, optional
        If True, then xoffset will be added to the label's x position in
        log10 space
    yoffset : double, optional
        Space to add to label's y position
    yoffset_logspace : bool, optional
        If True, then yoffset will be added to the label's y position in
        log10 space
    outline_color : None | "auto" | color
        Colour of the outline. If set to "auto", use the background color.
        If set to None, do not draw an outline.
    outline_width : number
        Width of the outline
    rotation: float, optional
            If set and align = False, controls the angle of the label
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """

    try:
        txt = LineLabel(
            line,
            x,
            label=label,
            align=align,
            xoffset=xoffset,
            xoffset_logspace=xoffset_logspace,
            yoffset=yoffset,
            yoffset_logspace=yoffset_logspace,
            outline_color=outline_color,
            outline_width=outline_width,
            rotation=rotation,
            **kwargs,
        )
    except ValueError as err:
        if "does not have a well defined value" in str(err):
            warnings.warn(
                (
                    f"{line} could not be annotated due to `nans` values. "
                    "Consider using another location via the `x` argument."
                ),
                UserWarning,
                stacklevel=1,
            )
            return
        raise err

    if drop_label:
        line.set_label(None)

    return txt


def labelLines(
    lines: Optional[list[Line2D]] = None,
    align: Optional[bool] = None,
    xvals: Optional[Union[tuple[float, float], list[float]]] = None,
    drop_label: bool = False,
    shrink_factor: float = 0.05,
    xoffsets: Union[float, list[float]] = 0,
    yoffsets: Union[float, list[float]] = 0,
    outline_color: str = "auto",
    outline_width: float = 5,
    rotation: Optional[bool] = None,
    **kwargs,
):
    """Label all lines with their respective legends.

    Parameters
    ----------
    lines : list of matplotlib lines, optional.
       Lines to label. If empty, label all lines that have a label.
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
    xoffsets : number or list, optional.
        Distance relative to the line when positioning the labels. If given a number,
        the same value is used for all lines.
    yoffsets : number or list, optional.
        Distance relative to the line when positioning the labels. If given a number,
        the same value is used for all lines.
    outline_color : None | "auto" | color
        Colour of the outline. If set to "auto", use the background color.
        If set to None, do not draw an outline.
    outline_width : number
        Width of the outline
    rotation: float, optional
        If set and align = False, controls the angle of the label
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
    if lines:
        ax = lines[0].axes
    else:
        ax = plt.gca()

    handles, labels_of_handles = ax.get_legend_handles_labels()

    all_lines, all_labels = [], []
    for h, label in zip(handles, labels_of_handles):
        if isinstance(h, ErrorbarContainer):
            line = h.lines[0]
        else:
            line = h

        # If the user provided a list of lines to label, only label those
        if (lines is not None) and (line not in lines):
            continue
        all_lines.append(line)
        all_labels.append(label)

    # Check that the lines passed to the function have all a label
    if lines is not None:
        for line in lines:
            if line in all_lines:
                continue

            warnings.warn(
                f"Tried to label line {line}, but could not find a label for it.",
                UserWarning,
                stacklevel=1,
            )

    # In case no x location was provided, we need to use some heuristics
    # to generate them.
    if xvals is None:
        xvals = ax.get_xlim()
        xscale = ax.get_xscale()
        if xscale == "log":
            log10_xvals = np.log10(xvals)
            xvals_rng = log10_xvals[1] - log10_xvals[0]
            shrinkage = xvals_rng * shrink_factor
            xvals = (
                10 ** (log10_xvals[0] + shrinkage),
                10 ** (log10_xvals[1] - shrinkage),
            )
        else:
            xvals_rng = xvals[1] - xvals[0]
            shrinkage = xvals_rng * shrink_factor
            xvals = (xvals[0] + shrinkage, xvals[1] - shrinkage)

    if isinstance(xvals, tuple) and len(xvals) == 2:
        xmin, xmax = xvals
        xscale = ax.get_xscale()

        # Convert datetime objects to numeric values for linspace/geomspace
        x_is_datetime = isinstance(xmin, datetime) or isinstance(xmax, datetime)
        if x_is_datetime:
            if not isinstance(xmin, datetime) or not isinstance(xmax, datetime):
                raise ValueError(
                    f"Cannot mix datetime and numeric values in xvals: {xvals}"
                )
            xmin = plt.matplotlib.dates.date2num(xmin)
            xmax = plt.matplotlib.dates.date2num(xmax)

        if xscale == "log":
            xvals = np.geomspace(xmin, xmax, len(all_lines) + 2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(all_lines) + 2)[1:-1]

        # Convert numeric values back to datetime objects
        if x_is_datetime:
            xvals = plt.matplotlib.dates.num2date(xvals)

        # Build matrix line -> xvalue
        ok_matrix = np.zeros((len(all_lines), len(all_lines)), dtype=bool)

        for i, line in enumerate(all_lines):
            xdata, _ = normalize_xydata(line)
            minx, maxx = np.nanmin(xdata), np.nanmax(xdata)
            for j, xv in enumerate(xvals):  # type: ignore
                xv = line.convert_xunits(xv)
                ok_matrix[i, j] = minx < xv < maxx

        # If some xvals do not fall in their corresponding line,
        # find a better matching using maximum bipartite matching.
        if not np.all(np.diag(ok_matrix)):
            order = maximum_bipartite_matching(ok_matrix)

            # The maximum match may miss a few points, let's add them back
            order[order < 0] = np.setdiff1d(np.arange(len(order)), order[order >= 0])

            # Now reorder the xvalues
            old_xvals = xvals.copy()  # type: ignore
            xvals[order] = old_xvals  # type: ignore
    else:
        xvals = list(always_iterable(xvals))  # force the creation of a copy
        if len(xvals) == 1:
            xvals = [xvals[0]] * len(all_lines)

    lab_lines, labels = [], []
    # Take only the lines which have labels other than the default ones
    for i, (line, xv) in enumerate(zip(all_lines, xvals)):  # type: ignore
        label = all_labels[all_lines.index(line)]
        lab_lines.append(line)
        labels.append(label)

        # Move xlabel if it is outside valid range
        xdata, _ = normalize_xydata(line)
        xmin, xmax = np.nanmin(xdata), np.nanmax(xdata)
        xv = line.convert_xunits(xv)

        if not (xmin <= xv <= xmax):
            warnings.warn(
                (
                    f"The value at position {i} in `xvals` is outside the range of its "
                    f"associated line ({xmin=}, {xmax=}, xval={xv}). "
                    "Clipping it into the allowed range."
                ),
                UserWarning,
                stacklevel=1,
            )
            new_xv = xmin + (xmax - xmin) * 0.9
            xvals[i] = new_xv  # type: ignore

    # Convert float values back to datetime in case of datetime axis
    mpl_version = tuple(int(_) for _ in plt.matplotlib.__version__.split("."))
    if mpl_version < (3, 10, 0):
        converter = ax.xaxis.converter
    else:
        converter = ax.xaxis.get_converter()
    time_classes = (_SwitchableDateConverter, DateConverter, ConciseDateConverter)
    if xvals is None:
        raise ValueError(
            "xvals must be a tuple of two floats or a list of floats."
            f"Got {xvals} instead."
        )
    elif isinstance(converter, time_classes):
        xvals_dates = []
        for x in xvals:
            if isinstance(x, datetime):
                x_datetime = x
            elif isinstance(x, np.datetime64):
                x_datetime = x.astype(datetime)
            else:
                x_datetime = num2date(x)
            xvals_dates.append(x_datetime.replace(tzinfo=ax.xaxis.get_units()))
        xvals = xvals_dates

    txts = []
    try:
        if isinstance(xoffsets, timedelta):
            xoffsets = [xoffsets] * len(all_lines)  # type: ignore
        else:
            xoffsets = [float(xoffsets)] * len(all_lines)  # type: ignore
    except TypeError:
        pass
    try:
        yoffsets = [float(yoffsets)] * len(all_lines)  # type: ignore
    except TypeError:
        pass
    for line, x, xoffset, yoffset, label in zip(
        lab_lines,
        xvals,  # type: ignore
        xoffsets,  # type: ignore
        yoffsets,  # type: ignore
        labels,
    ):
        txts.append(
            labelLine(
                line,
                x,
                label=label,
                align=align,
                drop_label=drop_label,
                xoffset=xoffset,
                yoffset=yoffset,
                outline_color=outline_color,
                outline_width=outline_width,
                rotation=rotation,
                **kwargs,
            )
        )

    return txts
