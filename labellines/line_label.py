from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import numpy as np
from datetime import timedelta
from matplotlib.text import Text

from .utils import normalize_xydata

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Literal, Optional, Union

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    Position = Union[float, datetime, np.datetime64]
    ColorLike = Any  # mpl has no type annotations so this is just a crutch
    AutoLiteral = Literal["auto"]


class LineLabel(Text):
    """This artist adds a label onto a preexisting Line2D object"""

    _line: Line2D
    """Annotated line"""

    _target_x: Position
    """Requested x position of the label, as supplied by the user"""

    _ax: Axes
    """Axes containing the line"""

    _auto_align: bool
    """Align text with the line (True) or parallel to x axis (False)"""

    _xoffset: float
    """An additional x offset for the label"""

    _xoffset_logspace: bool
    """Sets whether to treat _xoffset exponentially"""

    _yoffset: float
    """An additional y offset for the label"""

    _yoffset_logspace: bool
    """Sets whether to treat _yoffset exponentially"""

    _label_pos: np.ndarray
    """Position of the label, computed from _target_x and line data"""

    _anchor_a: np.ndarray
    """Anchor A for rotation calculation, point of _line neighbouring this label"""

    _anchor_b: np.ndarray
    """Anchor B for rotation calculation, point of _line neighbouring this label"""

    def __init__(
        self,
        line: Line2D,
        x: Position,
        label: Optional[str] = None,
        align: Optional[bool] = None,
        xoffset: float = 0,
        xoffset_logspace: bool = False,
        yoffset: float = 0,
        yoffset_logspace: bool = False,
        outline_color: Optional[Union[AutoLiteral, ColorLike]] = "auto",
        outline_width: float = 8,
        rotation: Optional[float] = None,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        line : Line2D
            Line to be decorated.
        x : Position
            Horizontal target position for the label (in data units).
        label : str, optional
            Override for line label, by default None.
        align : bool, optional
            If true, the label is parallel to the line, otherwise horizontal,
            by default True.
        xoffset : float, optional
            An additional x offset for the line label, by default 0.
        xoffset_logspace : bool, optional
            If true xoffset is applied exponentially to appear linear on a log-axis,
            by default False.
        yoffset : float, optional
            An additional y offset for the line label, by default 0.
        yoffset_logspace : bool, optional
            If true yoffset is applied exponentially to appear linear on a log-axis,
            by default False.
        outline_color : None | "auto" | Colorlike
            Colour of the outline. If set to "auto", use the background color.
            If set to None, do not draw an outline, by default "auto".
        outline_width : float
            Width of the outline, by default 8.
        rotation: float, optional
            If set and align = False, controls the angle of the label.
        """
        # When rotation is set, align has to be false or None
        if rotation is not None and align:
            raise ValueError(
                f"When rotation is set, align needs to be false or none was {align=}."
            )
        elif rotation is None:
            align = True if (align or align is None) else False
            rotation = 0
        elif rotation is None and not align:
            align = False
            rotation = 0
        elif not align or align is None:
            align = False
            rotation = rotation

        self._line = line
        self._target_x = x
        self._ax = line.axes
        self._auto_align = align
        self._xoffset = xoffset
        self._xoffset_logspace = xoffset_logspace
        self._yoffset = yoffset
        self._yoffset_logspace = yoffset_logspace
        label = label or line.get_label()

        # Populate self._pos, self._anchor_a, self._anchor_b
        self._update_anchors()
        self._rotation = rotation

        # Set a bunch of default arguments
        kwargs.setdefault("color", self._line.get_color())
        kwargs.setdefault("clip_on", True)
        kwargs.setdefault("zorder", 2.5)
        if "ha" not in kwargs:
            kwargs.setdefault("horizontalalignment", "center")
        if "va" not in kwargs:
            kwargs.setdefault("verticalalignment", "center")

        # Initialize Text Artist
        super().__init__(
            *self._label_pos,
            label,
            rotation=self._rotation,
            rotation_mode="anchor",
            **kwargs,
        )

        # Apply outline effect
        if outline_color is not None:
            if outline_color == "auto":
                outline_color = line.axes.get_facecolor()

            self.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=outline_width, foreground=outline_color
                    ),
                    patheffects.Normal(),
                ]
            )

        # activate clipping if needed and place on axes
        if kwargs["clip_on"]:
            self.set_clip_path(self._ax.patch)
        self._ax._add_text(self)

    def _update_anchors(self):
        """
        This helper method computes the position of the textbox and determines
        the anchor points needed to adjust the rotation
        """
        # Use the mpl-internal float representation (deals with datetime etc)
        x = self._line.convert_xunits(self._target_x)
        xdata, ydata = normalize_xydata(self._line)

        # Convert timedelta to float if needed
        if isinstance(self._xoffset, timedelta):
            xoffset = mdates.date2num(self._xoffset + self._target_x) - x
        else:
            xoffset = self._xoffset

        # Handle nan values
        mask = np.isfinite(ydata)
        if mask.sum() == 0:
            raise ValueError(f"The line {self._line} only contains nan!")
        xdata = xdata[mask]
        ydata = ydata[mask]

        # If the valid data is a single point, then just use that point
        if len(xdata) == 1:
            xa, xb = xdata[0], xdata[0]
            ya, yb = ydata[0], ydata[0]
        else:
            # Find the first line segment surrounding x
            for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
                if min(xa, xb) <= x <= max(xa, xb):
                    ya, yb = ydata[i], ydata[i + 1]
                    break
            else:
                raise ValueError("x label location is outside data range!")

        # Interpolate y position of label, (interp needs sorted data)
        if xa != xb:
            dx = np.array((xa, xb))
            dy = np.array((ya, yb))
            srt = np.argsort(dx)
            y = np.interp(x, dx[srt], dy[srt])
        else:  # Vertical case
            y = (ya + yb) / 2

        # Apply x offset
        if self._xoffset_logspace:
            x *= 10**xoffset
        else:
            x += xoffset

        # Apply y offset
        if self._yoffset_logspace:
            y *= 10**self._yoffset
        else:
            y += self._yoffset

        if not np.isfinite(y):
            raise ValueError(
                f"{self._line} does not have a well defined value"
                f" at x = {self._target_x}. Consider a different position."
            )

        self._label_pos = np.array((x, y))
        self._anchor_a = np.array((xa, ya))
        self._anchor_b = np.array((xb, yb))

    def __auto_align(self, value=None):
        # Helper function  to help  resize the  alignment of
        # the label if the window is resized
        # Providing the _rotation property
        # enables automatic adjustment of the rotation angle
        # Adapted from https://stackoverflow.com/a/53111799
        if self._auto_align:
            # Transform to screen coordinated to make sure the angle is always
            # correct regardless of axis scaling etc.
            xa, ya = self._ax.transData.transform(self._anchor_a)
            xb, yb = self._ax.transData.transform(self._anchor_b)
            angle = np.rad2deg(np.arctan2(yb - ya, xb - xa))

            # Correct the angle to make sure text is always upright-ish
            value = (angle + 90) % 180 - 90
        if isinstance(value, (float, int)):
            self.__rotation = value
        return self.__rotation

    @property
    def _rotation(self):
        return self.__auto_align()

    @_rotation.setter
    def _rotation(self, rotation):
        self.__rotation = self.__auto_align(rotation)
