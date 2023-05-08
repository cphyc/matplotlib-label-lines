import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.dates import DateConverter, num2date
from more_itertools import always_iterable

from .line_label import LineLabel
from .utils import maximum_bipartite_matching, normalize_xydata


# Label line with line2D label data
def labelLine(
    line,
    val,
    axis="x",
    label=None,
    align=True,
    drop_label=False,
    offset=0,
    offset_logspace=False,
    outline_color="auto",
    outline_width=8,
    **kwargs,
):
    """
    Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    val : number
       The location in data unit of the label
    axis : "x" | "y"
        Reference axis for `val`.
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    offset : double, optional
        Space to add to label's y position
    offset_logspace : bool, optional
        If True, then offset will be added to the label's y position in
        log10 space
    outline_color : None | "auto" | color
        Colour of the outline. If set to "auto", use the background color.
        If set to None, do not draw an outline.
    outline_width : number
        Width of the outline
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """

    try:
        txt = LineLabel(
            line,
            val,
            axis,
            label=label,
            align=align,
            offset=offset,
            offset_logspace=offset_logspace,
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
    lines=None,
    align=True,
    vals=None,
    axis=None,
    drop_label=False,
    shrink_factor=0.05,
    offsets=0,
    outline_color="auto",
    outline_width=5,
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
    vals : (first, last) or array of float, optional
       The location of the labels. If a tuple, the labels will be
       evenly spaced between first and last (in the axis units).
    axis : None | "x" | "y", optional
        Reference axis for the `vals`.
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent
       calls to e.g. legend do not use it anymore.
    shrink_factor : double, optional
       Relative distance from the edges to place closest labels. Defaults to 0.05.
    offsets : number or list, optional.
        Distance relative to the line when positioning the labels. If given a number,
        the same value is used for all lines. It refers to the *other* axis 
        (i.e. to y if axis=="x")
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

    if axis == "y":
        yaxis = True
    else:
        axis = "x"
        yaxis = False

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
                "Tried to label line %s, but could not find a label for it." % line,
                UserWarning,
                stacklevel=1,
            )

    # In case no x location was provided, we need to use some heuristics
    # to generate them.
    if vals is None:
        if yaxis:
            vals = ax.get_ylim()
        else:
            vals = ax.get_xlim()
        vals_rng = vals[1] - vals[0]
        shrinkage = vals_rng * shrink_factor
        vals = (vals[0] + shrinkage, vals[1] - shrinkage)

    if isinstance(vals, tuple) and len(vals) == 2:
        vmin, vmax = vals
        xscale = ax.get_xscale()
        if xscale == "log":
            vals = np.logspace(np.log10(vmin), np.log10(vmax), len(all_lines) + 2)[1:-1]
        else:
            vals = np.linspace(vmin, vmax, len(all_lines) + 2)[1:-1]

        # Build matrix line -> value
        ok_matrix = np.zeros((len(all_lines), len(all_lines)), dtype=bool)

        for i, line in enumerate(all_lines):
            if yaxis:
                _, data = normalize_xydata(line)
            else:
                data, _ = normalize_xydata(line)
            minv, maxv = min(data), max(data)
            for j, val in enumerate(vals):
                ok_matrix[i, j] = minv < val < maxv

        # If some vals do not fall in their corresponding line,
        # find a better matching using maximum bipartite matching.
        if not np.all(np.diag(ok_matrix)):
            order = maximum_bipartite_matching(ok_matrix)

            # The maximum match may miss a few points, let's add them back
            order[order < 0] = np.setdiff1d(np.arange(len(order)), order[order >= 0])

            # Now reorder the xvalues
            old_xvals = vals.copy()
            vals[order] = old_xvals
    else:
        vals = list(always_iterable(vals))  # force the creation of a copy

    lab_lines, labels = [], []
    # Take only the lines which have labels other than the default ones
    for i, (line, val) in enumerate(zip(all_lines, vals)):
        label = all_labels[all_lines.index(line)]
        lab_lines.append(line)
        labels.append(label)

        # Move xlabel/ylabel if it is outside valid range
        if yaxis:
            _, data = normalize_xydata(line)
        else:
            data, _ = normalize_xydata(line)
        if not (min(data) <= val <= max(data)):
            warnings.warn(
                (
                    "The value at position {} in `vals` is outside the range of its "
                    "associated line (vmin={}, vmax={}, xval={}). Clipping it "
                    "into the allowed range."
                ).format(i, min(data), max(data), val),
                UserWarning,
                stacklevel=1,
            )
            new_val = min(data) + (max(data) - min(data)) * 0.9
            vals[i] = new_val

    # Convert float values back to datetime in case of datetime axis
    if isinstance(ax.xaxis.converter, DateConverter):
        vals = [num2date(x).replace(tzinfo=ax.xaxis.get_units()) for x in vals]

    txts = []
    try:
        offsets = [float(offsets)] * len(all_lines)
    except TypeError:
        pass
    for line, val, offset, label in zip(lab_lines, vals, offsets, labels):
        txts.append(
            labelLine(
                line,
                val,
                axis,
                label=label,
                align=align,
                drop_label=drop_label,
                offset=offset,
                outline_color=outline_color,
                outline_width=outline_width,
                **kwargs,
            )
        )

    return txts
