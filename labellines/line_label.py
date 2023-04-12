from __future__ import annotations

import re
from itertools import repeat
from typing import TYPE_CHECKING

import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.text import Text

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Literal, Optional, Union

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    Position = Union[float, datetime, np.datetime64]
    ColorLike = Any  # mpl has no type annotations so this is just a crutch
    AutoLiteral = Literal["auto"]

# This matches a dollar sign that is not preceded by a backslash
VALID_MATH_RE = re.compile(r"(?<!\\)\$")


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
        *,
        label: str,
        align: bool = True,
        yoffset: float = 0,
        yoffset_logspace: bool = False,
        outline_color: Optional[Union[AutoLiteral, ColorLike]] = "auto",
        outline_width: float = 8,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        line : Line2D
            Line to be decorated.
        x : Position
            Horizontal target position for the label (in data units).
        label : str
            Line label
        align : bool, optional
            If true, the label is parallel to the line, otherwise horizontal,
            by default True.
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
        """
        self._line = line
        self._target_x = x
        self._ax = line.axes
        self._auto_align = align
        self._yoffset = yoffset
        self._yoffset_logspace = yoffset_logspace

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
                    patheffects.withStroke(
                        linewidth=outline_width, foreground=outline_color
                    ),
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
        if xa != xb:
            dx = np.array((xa, xb))
            dy = np.array((ya, yb))
            srt = np.argsort(dx)
            y = np.interp(x, dx[srt], dy[srt])
        else:  # Vertical case
            y = (ya + yb) / 2

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

    @property
    def _rotation(self):
        return self._get_rotation()

    @_rotation.setter
    def _rotation(self, rotation):
        self._set_rotation(rotation)


class CurvedLineLabel(Text):
    """
    A text object that follows an arbitrary curve.
    """

    _ax: Axes
    """Axes containing the line"""

    _line: Line2D
    """Line to be decorated"""

    _yoffset: float
    """An additional y offset for the label"""

    _yoffset_logspace: bool
    """Sets whether to treat _yoffset exponentially"""

    _x_data: np.ndarray
    """x coordinates of the curve"""

    _y_data: np.ndarray
    """y coordinates of the curve"""

    _zorder: float
    """zorder of the line"""

    _tokens: list[str]
    """List of tokens in the label"""

    __Characters: list[Text]
    """List of characters making up the label"""

    __Outlines: list[Text]
    """List of outlines"""

    def __init__(
        self,
        line: Line2D,
        *,
        label: str,
        axes: Axes,
        yoffset: float = 0,
        yoffset_logspace: bool = False,
        outline_color: Optional[Union[AutoLiteral, ColorLike]] = "auto",
        outline_width: float = 8,
        **kwargs,
    ):
        self.__Characters = []
        self.__Outlines = []

        ##saving the curve:
        self._line = line
        self._x_data = line.get_xdata()
        self._zorder = self.get_zorder()
        self._label = label
        self._ax = axes
        self._yoffset = yoffset
        self._yoffset_logspace = yoffset_logspace
        self._outline_color = outline_color
        self._outline_width = outline_width

        self._tokens = self.tokenize_string(label)

        if yoffset_logspace:
            y = np.asarray(line.get_ydata()) * 10**yoffset
        else:
            y = np.asarray(line.get_ydata()) + yoffset
        self._y_data = y
        self._kwargs = kwargs

        super().__init__(self._x_data[0], self._y_data[0], " ", **kwargs)
        axes.add_artist(self)

        self._tokens = self.tokenize_string(label)

        self.update_texts()

    def update_texts(self):
        for c in self._tokens:
            if c == " ":
                ##make this an invisible 'a':
                t = Text(0, 0, "a")
                t.set_alpha(0.0)
            else:
                t = Text(0, 0, c, **self._kwargs)

            # resetting unnecessary arguments
            t.set_ha("center")
            t.set_rotation(0)
            t.set_zorder(self._zorder + 1)

            self.__Characters.append((c, t))
            self._ax.add_artist(t)

        if self._outline_color is None:
            return

        # Second pass: add the outlines
        if self._outline_color == "auto":
            outline_color = self.get_color()
        else:
            outline_color = self._outline_color

        for c in self._tokens:
            if c == " ":
                ##make this an invisible 'a':
                t = Text(0, 0, "a")
                t.set_alpha(0.0)
            else:
                t = Text(0, 0, c, **self._kwargs)

            t.set_path_effects(
                [
                    patheffects.Stroke(
                        linewidth=self._outline_width, foreground=outline_color
                    ),
                ]
            )

            # resetting unnecessary arguments
            t.set_ha("center")
            t.set_rotation(0)
            t.set_zorder(self._zorder - 10)

            self.__Outlines.append((c, t))
            self._ax.add_artist(t)

    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super().set_zorder(zorder)
        self._zorder = self.get_zorder()
        for _c, t in self.__Characters:
            t.set_zorder(self._zorder + 1)

        for _c, t in self.__Outlines:
            t.set_zorder(self._zorder - 10)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self, renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        # preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        ## Axis size on figure
        figW, figH = self._ax.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self._ax.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w) / (figH * h)) * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

        # points of the curve in figure coordinates:
        x_fig, y_fig = (
            np.array(xy)
            for xy in zip(
                *self._ax.transData.transform(
                    [(i, j) for i, j in zip(self._x_data, self._y_data)]
                )
            )
        )
        # point distances in figure coordinates
        x_fig_dist = x_fig[1:] - x_fig[:-1]
        y_fig_dist = y_fig[1:] - y_fig[:-1]
        r_fig_dist = np.sqrt(x_fig_dist**2 + y_fig_dist**2)

        # arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist), 0, 0)

        # angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]), (x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        characters = self.__Characters
        outlines = self.__Outlines
        if not outlines:
            outlines = repeat((None, None))

        rel_pos = 10
        for (c, t_char), (_c, t_outline) in zip(characters, outlines):
            both_t = [t_char]
            if t_outline is not None:
                both_t.append(t_outline)
            # finding the width of c:
            t_char.set_rotation(0)
            t_char.set_va("center")
            bbox1 = t_char.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            # ignore all letters that don't fit:
            if rel_pos + w / 2 > l_fig[-1]:
                for t in both_t:
                    t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != " ":
                for t in both_t:
                    t.set_alpha(1.0)

            # finding the two data points between which the horizontal
            # center point of the character will be situated
            # left and right indices:
            il = np.where(rel_pos + w / 2 >= l_fig)[0][-1]
            ir = np.where(rel_pos + w / 2 <= l_fig)[0][0]

            # if we exactly hit a data point:
            if ir == il:
                ir += 1

            # how much of the letter width was needed to find il:
            used = l_fig[il] - rel_pos
            rel_pos = l_fig[il]

            # relative distance between il and ir where the center
            # of the character will be
            fraction = (w / 2 - used) / r_fig_dist[il]

            ## setting the character position in data coordinates:
            ## interpolate between the two points:
            x = self._x_data[il] + fraction * (self._x_data[ir] - self._x_data[il])
            y = self._y_data[il] + fraction * (self._y_data[ir] - self._y_data[il])

            # getting the offset when setting correct vertical alignment
            # in data coordinates
            for t in both_t:
                t.set_va(self.get_va())
            bbox2 = t_char.get_window_extent(renderer=renderer)

            bbox1d = self._ax.transData.inverted().transform(bbox1)
            bbox2d = self._ax.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0] - bbox1d[0])

            # the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array(
                [
                    [np.math.cos(rad), np.math.sin(rad) * aspect],
                    [-np.math.sin(rad) / aspect, np.math.cos(rad)],
                ]
            )

            ## computing the offset vector of the rotated character
            drp = np.dot(dr, rot_mat)

            # setting final position and rotation:
            for t in both_t:
                t.set_position(np.array([x, y]) + drp)
                t.set_rotation(degs[il])

                t.set_va("center")
                t.set_ha("center")

            # updating rel_pos to right edge of character
            rel_pos += w - used

    @staticmethod
    def tokenize_string(text: str) -> list[str]:
        # Make sure the string has only valid math (i.e. there is an even number of `$`)
        valid_math = len(re.findall(VALID_MATH_RE, text)) % 2 == 0

        if not valid_math:
            return list(text)

        math_mode = False
        tokens = []
        prev_c = None
        current_token: str = ""
        for _i, c in enumerate(text):
            if c == "$" and prev_c != "\\":
                if math_mode:
                    tokens.append("$" + current_token + "$")
                    math_mode = False
                else:
                    math_mode = True
                    current_token = ""
            elif math_mode:
                current_token += c
            else:
                tokens.append(c)
            prev_c = c
        return tokens
