import warnings
from collections.abc import Iterable
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.dates import DateConverter, num2date
from more_itertools import always_iterable

from .line_label import CurvedLineLabel, LineLabel
from .utils import ensure_float, maximum_bipartite_matching


# Label line with line2D label data
def labelLine(
    line: plt.Line2D,
    x: float,
    curved_text: bool = False,
    label: Optional[str] = None,
    align: bool = True,
    drop_label: bool = False,
    yoffset: float = 0,
    yoffset_logspace: bool = False,
    outline_color: Union[Literal["auto"], None, "str"] = "auto",
    outline_width: float = 8,
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
    curved_text : bool, optional
         If True, the label will be curved to follow the line.
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
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
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """

    label = label or line.get_label()

    try:
        if curved_text:
            txt = CurvedLineLabel(
                line,
                label=label,
                axes=line.axes,
                yoffset=yoffset,
                yoffset_logspace=yoffset_logspace,
                outline_color=outline_color,
                outline_width=outline_width,
                **kwargs,
            )
        else:
            txt = LineLabel(
                line,
                x,
                label=label,
                align=align,
                yoffset=yoffset,
                yoffset_logspace=yoffset_logspace,
                outline_color=outline_color,
                outline_width=outline_width,
                **kwargs,
            )
    except ValueError as err:
        if "does not have a well defined value" in str(err):
            warnings.warn(
                (
                    "%s could not be annotated due to `nans` values. "
                    "Consider using another location via the `x` argument."
                )
                % line,
                UserWarning,
                stacklevel=1,
            )
            return
        raise err

    if drop_label:
        line.set_label(None)

    return txt


def labelLines(
    lines: Optional[List[plt.Line2D]] = None,
    align: bool = True,
    xvals: Union[None, Tuple[float, float], Iterable[float]] = None,
    curved_text: bool = False,
    drop_label: bool = False,
    shrink_factor: float = 0.05,
    yoffsets: Union[float, Iterable[float]] = 0,
    outline_color: Union[Literal["auto"], None, "str"] = "auto",
    outline_width: float = 5,
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
    curved_text : bool, optional
         If True, the labels will be curved to follow the line.
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    shrink_factor : double, optional
       Relative distance from the edges to place closest labels. Defaults to 0.05.
    yoffsets : number or list, optional.
        Distance relative to the line when positioning the labels. If given a number,
        the same value is used for all lines.
    outline_color : None | "auto" | color
        Colour of the outline. If set to "auto", use the background color.
        If set to None, do not draw an outline.
    outline_width : number
        Width of the outline
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
    if lines:
        ax = lines[0].axes
    else:
        ax = plt.gca()

    handles, all_labels = ax.get_legend_handles_labels()

    all_lines = []
    for h in handles:
        if isinstance(h, ErrorbarContainer):
            line = h.lines[0]
        else:
            line = h

        # If the user provided a list of lines to label, only label those
        if (lines is not None) and (line not in lines):
            continue
        all_lines.append(line)

    # Check that the lines passed to the function have all a label
    if lines is not None:
        for line in lines:
            if line in all_lines:
                continue

            warnings.warn(
                "Tried to label line %s, but could not find a label for it." % line,
                UserWarning,
                stacklevel=1,
            )

    # In case no x location was provided, we need to use some heuristics
    # to generate them.
    if xvals is None:
        xvals = ax.get_xlim()
        xvals_rng = xvals[1] - xvals[0]  # type: ignore
        shrinkage = xvals_rng * shrink_factor
        xvals = (xvals[0] + shrinkage, xvals[1] - shrinkage)  # type: ignore

    if isinstance(xvals, tuple) and len(xvals) == 2:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(all_lines) + 2)[
                1:-1
            ]
        else:
            xvals = np.linspace(xmin, xmax, len(all_lines) + 2)[1:-1]

        # Build matrix line -> xvalue
        ok_matrix = np.zeros((len(all_lines), len(all_lines)), dtype=bool)

        for i, line in enumerate(all_lines):
            xdata = ensure_float(line.get_xdata())
            minx, maxx = min(xdata), max(xdata)
            for j, xv in enumerate(xvals):  # type: ignore
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

    lab_lines, labels = [], []
    # Take only the lines which have labels other than the default ones
    for i, (line, xv) in enumerate(zip(all_lines, xvals)):  # type: ignore
        label = all_labels[all_lines.index(line)]
        lab_lines.append(line)
        labels.append(label)

        # Move xlabel if it is outside valid range
        xdata = ensure_float(line.get_xdata())
        if not (min(xdata) <= xv <= max(xdata)):
            warnings.warn(
                (
                    "The value at position %s in `xvals` is outside the range of its "
                    "associated line (xmin=%s, xmax=%s, xval=%s). Clipping it "
                    "into the allowed range."
                )
                % (i, min(xdata), max(xdata), xv),
                UserWarning,
                stacklevel=1,
            )
            new_xv = min(xdata) + (max(xdata) - min(xdata)) * 0.9
            xvals[i] = new_xv  # type: ignore

    # Convert float values back to datetime in case of datetime axis
    if isinstance(ax.xaxis.converter, DateConverter):
        tz = ax.xaxis.get_units()
        xvals = [num2date(x).replace(tzinfo=tz) for x in xvals]  # type: ignore

    txts = []

    if not isinstance(yoffsets, Iterable):
        yoffsets = [float(yoffsets)] * len(all_lines)

    for line, x, yoffset, label in zip(
        lab_lines,
        xvals,  # type: ignore
        yoffsets,
        labels,
    ):
        txts.append(
            labelLine(
                line,
                x,
                label=label,
                align=align,
                drop_label=drop_label,
                yoffset=yoffset,
                outline_color=outline_color,
                outline_width=outline_width,
                **kwargs,
            )
        )

    return txts
