from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.dates import UTC, DateFormatter, DayLocator
from matplotlib.testing import setup

from .core import labelLine, labelLines


@pytest.fixture()
def setupMpl():
    setup()
    plt.clf()


@pytest.mark.mpl_image_compare
def test_linspace(setupMpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_ylogspace(setupMpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label=r"$f(x)=\exp(%s x)$" % k)

    plt.yscale("log")
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_xlogspace(setupMpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(10**x, k * x, label=r"$f(x)=%s x$" % k)

    plt.xscale("log")
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_xylogspace(setupMpl):
    x = np.geomspace(1e-1, 1e1)
    K = np.arange(-5, 5, 2)

    for k in K:
        plt.plot(x, x**k, label=rf"$f(x)=x^{{{k}}}$")

    plt.xscale("log")
    plt.yscale("log")
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_align(setupMpl):
    x = np.linspace(0, 2 * np.pi)
    y = np.sin(x)

    lines = plt.plot(x, y, label=r"$\sin(x)$")

    labelLines(lines, align=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_rotation_correction(setupMpl):
    # Fix axes limits and plot line
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lines = plt.plot((0, 1), (0, 2), label="rescaled")

    # Now label the line and THEN rescale the axes, to force label rotation
    labelLine(lines[0], 0.5)
    ax.set_ylim(0, 2)

    return fig


@pytest.mark.mpl_image_compare
def test_vertical(setupMpl):
    x = 0.5

    line = plt.axvline(x, label="axvline")

    labelLine(line, x)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_labels_range(setupMpl):
    x = np.linspace(0, 1)

    plt.plot(x, np.sin(x), label=r"$\sin x$")
    plt.plot(x, np.cos(x), label=r"$\cos x$")

    labelLines(plt.gca().get_lines(), xvals=(0, 0.5))
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_naive(setupMpl):
    dates = [datetime(2018, 11, 1), datetime(2018, 11, 2), datetime(2018, 11, 3)]

    plt.plot(dates, [0, 5, 3], label="apples")
    plt.plot(dates, [3, 6, 2], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_advanced(setupMpl):
    dates = [
        datetime(2018, 11, 1, tzinfo=UTC),
        datetime(2018, 11, 2, tzinfo=UTC),
        datetime(2018, 11, 5, tzinfo=UTC),
        datetime(2018, 11, 3, tzinfo=UTC),
    ]

    plt.plot(dates, [0, 5, 3, 0], label="apples")
    plt.plot(dates, [3, 6, 2, 1], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_polar(setupMpl):
    t = np.linspace(0, 2 * np.pi, num=128)
    plt.plot(np.cos(t), np.sin(t), label="$1/1$")
    plt.plot(np.cos(t), np.sin(2 * t), label="$1/2$")
    plt.plot(np.cos(3 * t), np.sin(t), label="$3/1$")
    ax = plt.gca()

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_non_uniform_and_negative_spacing(setupMpl):
    x = [1, -2, -3, 2, -4, -3]
    plt.plot(x, [1, 2, 3, 4, 2, 1], ".-", label="apples")
    plt.plot(x, [6, 5, 4, 2, 5, 5], "o-", label="banana")
    ax = plt.gca()

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_errorbar(setupMpl):
    x = np.linspace(0, 1, 20)

    y = x**0.5
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$\sqrt{x}\pm x$")[0]

    y = x**3
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$x^3\pm x$")[0]

    labelLines()
    return plt.gcf()


def test_nan_warning():
    x = np.array([0, 1, 2, 3])
    y = np.array([np.nan, np.nan, 0, 1])

    line = plt.plot(x, y, label="test")[0]

    warn_msg = (
        ".* could not be annotated due to `nans` values. "
        "Consider using another location via the `x` argument."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        labelLine(line, 0.5)

    labelLine(line, 2.5)


def test_nan_failure():
    x = np.array([0, 1])
    y = np.array([np.nan, np.nan])

    line = plt.plot(x, y, label="test")[0]
    with pytest.raises(Exception):
        labelLine(line, 0.5)


@pytest.mark.mpl_image_compare
def test_label_range(setupMpl):
    x = np.linspace(0, 1)
    line = plt.plot(x, x**2, label="lorem ipsum")[0]

    # This should fail
    with pytest.raises(Exception):
        labelLine(line, -1)
    with pytest.raises(Exception):
        labelLine(line, 2)

    # This should work
    labelLine(line, 0.5)

    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_negative_spacing(setupMpl):
    x = np.linspace(1, -1)
    y = x**2

    line = plt.plot(x, y)[0]

    # Should not throw an error
    labelLine(line, 0.2, label="Test")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_label_datetime_plot(setupMpl):
    plt.clf()
    # data from the chinook database of iTunes music sales
    x = np.array(
        [
            "2009-01-31T00:00:00.000000000",
            "2009-02-28T00:00:00.000000000",
            "2009-03-31T00:00:00.000000000",
            "2009-04-30T00:00:00.000000000",
            "2009-06-30T00:00:00.000000000",
            "2009-09-30T00:00:00.000000000",
            "2009-10-31T00:00:00.000000000",
            "2009-11-30T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )
    y = np.array([13.86, 14.85, 28.71, 42.57, 61.38, 76.23, 77.22, 81.18])

    line = plt.plot_date(x, y, "-")[0]
    plt.xticks(rotation=45)

    # should not throw an error
    xlabel = datetime(2009, 3, 15)
    labelLine(line, xlabel, "USA")
    plt.tight_layout()
    return plt.gcf()


def test_yoffset(setupMpl):
    x = np.linspace(0, 1)

    for yoffset in ([-0.5, 0.5], 1, 1.2):  # try lists  # try int  # try float
        plt.clf()
        ax = plt.gca()
        ax.plot(x, np.sin(x) * 10, label=r"$\sin x$")
        ax.plot(x, np.cos(x) * 10, label=r"$\cos x$")
        lines = ax.get_lines()
        labelLines(
            lines, xvals=(0.2, 0.7), align=False, yoffsets=yoffset, bbox={"alpha": 0}
        )


@pytest.mark.mpl_image_compare
def test_outline(setupMpl):
    x = np.linspace(-2, 2)

    plt.ylim(-1, 5)
    plt.xlim(-2, 2)

    for dy, xlabel, w in zip(
        np.linspace(-1, 1, 5),
        np.linspace(-1.5, 1.5, 5),
        np.linspace(0, 16, 5),
    ):
        y = x**2 + dy
        (line,) = plt.plot(x, y, label=f"width={w}")
        labelLine(line, xlabel, outline_width=w, outline_color="gray")

    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_auto_layout(setupMpl):
    X = [[1, 2], [0, 1]]
    Y = [[0, 1], [0, 1]]

    lines = []
    for i, (x, y) in enumerate(zip(X, Y)):
        lines.extend(plt.plot(x, y, label=f"i={i}"))

    labelLines(lines)
    return plt.gcf()


def test_warning_out_of_range():
    X = [0, 1]
    Y = [0, 1]

    lines = plt.plot(X, Y, label="test")
    with pytest.warns(
        UserWarning,
        match=(
            "The value at position 0 in `xvals` is outside the range of its "
            "associated line"
        ),
    ):
        labelLines(lines, xvals=[-1])

    with pytest.warns(
        UserWarning,
        match=(
            "The value at position 0 in `xvals` is outside the range of its "
            "associated line"
        ),
    ):
        labelLines(lines, xvals=[2])


@pytest.mark.mpl_image_compare
def test_errorbar_with_list(setupMpl):
    np.random.seed(1234)
    fig, ax = plt.subplots()
    samples = ["a", "b"]
    pos = [-1, 1]

    x = list(np.arange(-4, 4.1, 0.1))
    ys = [list(np.random.rand(len(x))), list(np.random.rand(len(x)))]

    lines = []
    for sample, y in zip(samples, ys):
        lines.append(ax.errorbar(x, y, yerr=0.1, label=sample)[0])

    labelLines(lines, align=False, xvals=pos)
    return fig


@pytest.fixture
def create_plot():
    fig, ax = plt.subplots()
    X = [0, 1]
    Y = [0, 1]

    lines = (
        *ax.plot(X, Y, label="label1"),
        *ax.plot(X, Y),  # no label
        *ax.plot(X, Y, label="label2"),
    )
    return fig, ax, lines


def test_warning_line_labeling(create_plot):
    _fig, _ax, lines = create_plot

    warn_msg = "Tried to label line .*, but could not find a label for it."
    with pytest.warns(UserWarning, match=warn_msg):
        txts = labelLines(lines)
    # Make sure only two lines have been labeled
    assert len(txts) == 2

    with pytest.warns(UserWarning, match=warn_msg):
        txts = labelLines(lines[1:])
    # Make sure only one line has been labeled
    assert len(txts) == 1


def test_no_warning_line_labeling(create_plot):
    _fig, _ax, lines = create_plot

    txts = labelLines(lines[0:1])
    assert len(txts) == 1

    txts = labelLines()
    assert len(txts) == 2


def test_labeling_by_axis(create_plot):
    txts = labelLines()
    assert len(txts) == 2
