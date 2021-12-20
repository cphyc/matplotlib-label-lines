from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.text import Text

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Union

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    Position = Union[float, datetime, np.datetime64]


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
        label: str | None = None,
        align: bool = True,
        yoffset: float = 0,
        yoffset_logspace: bool = False,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        line : Line2D
            Line to be decorated.
        x : Position
            Horizontal target position for the label (in data units).
        label : Optional[str], optional
            Override for line label, by default None.
        align : bool, optional
            If true, the label is parallel to the line, otherwise horizontal,
            by default True.
        yoffset : float, optional
            An additional y offset for the line label, by default 0.
        yoffset_logspace : bool, optional
            If true yoffset is applied exponentially to appear linear on a log-axis,
            by default False.
        """
        self._line = line
        self._target_x = x
        self._ax = line.axes
        self._auto_align = align
        self._yoffset = yoffset
        self._yoffset_logspace = yoffset_logspace
        label = label or line.get_label()

        # Populate self._pos, self._anchor_a, self._anchor_b
        self._update_anchors()

        # Set a bunch of default arguments
        kwargs.setdefault("color", self._line.get_color())
        kwargs.setdefault("clip_on", True)
        kwargs.setdefault("zorder", 2.5)
        if "ha" not in kwargs:
            kwargs.setdefault("horizontalalignment", "center")
        if "va" not in kwargs:
            kwargs.setdefault("verticalalignment", "center")

        # Initialize Text Artist, activate clipping if needed and place on axes
        super().__init__(
            *self._label_pos,
            label,
            rotation=self._rotation,
            rotation_mode="anchor",
            **kwargs,
        )
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
        xdata = self._line.get_xdata(orig=False)
        ydata = self._line.get_ydata(orig=False)

        mask = np.isfinite(ydata)
        if mask.sum() == 0:
            raise ValueError(f"The line {self._line} only contains nan!")

        # Find the first line segment surrounding x
        for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= x <= max(xa, xb):
                ya, yb = ydata[i], ydata[i + 1]
                break
        else:
            raise ValueError("x label location is outside data range!")

        # Interpolate y position of label, (interp needs sorted data)
        dx = np.array((xa, xb))
        dy = np.array((ya, yb))
        srt = np.argsort(dx)
        y = np.interp(x, dx[srt], dy[srt])

        # Apply y offset
        if self._yoffset_logspace:
            y *= 10 ** self._yoffset
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

    def _get_rotation(self):
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
            return (angle + 90) % 180 - 90
        return 0

    def _set_rotation(self, rotation):
        pass

    _rotation = property(_get_rotation, _set_rotation)
