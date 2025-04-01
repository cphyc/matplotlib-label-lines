from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.dates import UTC, DateFormatter, DayLocator
from matplotlib.testing import setup

from .core import labelLine, labelLines


@pytest.fixture()
def setup_mpl():
    setup()
    plt.clf()


@pytest.mark.mpl_image_compare
def test_linspace(setup_mpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_ylogspace(setup_mpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label=rf"$f(x)=\exp({k} x)$")

    plt.yscale("log")
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_xlogspace(setup_mpl):
    x = np.linspace(0, 10)
    K = [1, 2, 4]

    for k in K:
        plt.plot(10**x, k * x, label=rf"$f(x)={k} x$")

    plt.xscale("log")
    # NOTE: depending on roundoff, the upper limit may be
    # 1e11 or 1e10. See PR #155.
    plt.xlim(1e0, 1e11)
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_xylogspace(setup_mpl):
    x = np.geomspace(0.1, 1e1)
    K = np.arange(-5, 5, 2)

    for k in K:
        plt.plot(x, np.power(x, k), label=rf"$f(x)=x^{{{k}}}$")

    plt.xscale("log")
    plt.yscale("log")

    # We fix the xvals to prevent out-of-range
    labelLines(
        plt.gca().get_lines(),
        xvals=(0.1, 1e1),
        zorder=2.5,
    )
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.ylim(1e-6, 1e6)
    # We need to fix the xlims to prevent funky xlims
    # see PR #115
    plt.xlim(0.1 / 1.2, 1e1 * 1.2)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_align(setup_mpl):
    x = np.linspace(0, 2 * np.pi)
    y = np.sin(x)

    lines = plt.plot(x, y, label=r"$\sin(x)$")

    labelLines(lines, align=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_rotation_correction(setup_mpl):
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
def test_vertical(setup_mpl):
    x = 0.5

    line = plt.axvline(x, label="axvline")

    labelLine(line, x)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_labels_range(setup_mpl):
    x = np.linspace(0, 1)

    plt.plot(x, np.sin(x), label=r"$\sin x$")
    plt.plot(x, np.cos(x), label=r"$\cos x$")

    labelLines(plt.gca().get_lines(), xvals=(0, 0.5))
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_naive(setup_mpl):
    dates = [datetime(2018, 11, 1), datetime(2018, 11, 2), datetime(2018, 11, 3)]

    plt.plot(dates, [0, 5, 3], label="apples")
    plt.plot(dates, [3, 6, 2], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_advanced(setup_mpl):
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

    labelLines(ax.get_lines(), xvals=(dates[0], dates[-1]))
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_timedelta_xoffset(setup_mpl):
    dates = [datetime(2018, 11, 1), datetime(2018, 11, 2), datetime(2018, 11, 3)]
    dt = timedelta(hours=12)

    plt.plot(dates, [0, 1, 2], label="apples")
    plt.plot(dates, [3, 4, 5], label="banana")
    ax = plt.gca()

    labelLines(ax.get_lines(), xoffsets=dt)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_polar(setup_mpl):
    t = np.linspace(0, 2 * np.pi, num=128)
    plt.plot(np.cos(t), np.sin(t), label="$1/1$")
    plt.plot(np.cos(t), np.sin(2 * t), label="$1/2$")
    plt.plot(np.cos(3 * t), np.sin(t), label="$3/1$")
    ax = plt.gca()

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_non_uniform_and_negative_spacing(setup_mpl):
    x = [1, -2, -3, 2, -4, -3]
    plt.plot(x, [1, 2, 3, 4, 2, 1], ".-", label="apples")
    plt.plot(x, [6, 5, 4, 2, 5, 5], "o-", label="banana")
    ax = plt.gca()

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_errorbar(setup_mpl):
    x = np.linspace(0, 1, 20)

    y = x**0.5
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$\sqrt{x}\pm x$")[0]

    y = x**3
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$x^3\pm x$")[0]

    labelLines()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_rotation(setup_mpl):
    x = np.linspace(0, 2 * np.pi)
    y = np.sin(x)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    axes = axes.flatten()

    axes[0].set_title("Rotated by 45")
    (line,) = axes[0].plot(x, y, label=r"$\sin(x)$")
    labelLine(line, np.pi, rotation=45)

    axes[1].set_title("Rotated by 45\naligned explicitly off")
    (line,) = axes[1].plot(x, y, label=r"$\sin(x)$")
    labelLine(line, np.pi, rotation=45, align=False)

    axes[2].set_title("Aligned explicitly on")
    (line,) = axes[2].plot(x, y, label=r"$\sin(x)$")
    labelLine(line, np.pi, align=True)

    axes[3].set_title("Default")
    (line,) = axes[3].plot(x, y, label=r"$\sin(x)$")
    labelLine(line, np.pi)
    return fig


def test_nan_failure():
    x = np.array([0, 1])
    y = np.array([np.nan, np.nan])

    line = plt.plot(x, y, label="test")[0]
    with pytest.raises(ValueError):
        labelLine(line, 0.5)


@pytest.mark.mpl_image_compare
def test_nan_gaps(setup_mpl):
    x = np.linspace(0, 10, 100)
    x[10:30] = np.nan

    for i in range(10):
        y = np.sin(x + i)
        plt.plot(x, y, label=f"y={i}")

    labelLines(plt.gca().get_lines())

    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_label_range(setup_mpl):
    x = np.linspace(0, 1)
    line = plt.plot(x, x**2, label="lorem ipsum")[0]

    # This should fail
    with pytest.raises(ValueError):
        labelLine(line, -1)
    with pytest.raises(ValueError):
        labelLine(line, 2)

    # This should work
    labelLine(line, 0.5)

    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_negative_spacing(setup_mpl):
    x = np.linspace(1, -1)
    y = x**2

    line = plt.plot(x, y)[0]

    # Should not throw an error
    labelLine(line, 0.2, label="Test")
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_label_datetime_plot(setup_mpl):
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

    line = plt.plot(x, y, "-")[0]
    plt.xticks(rotation=45)

    # should not throw an error
    xlabel = datetime(2009, 3, 15)
    labelLine(line, xlabel, "USA")
    plt.tight_layout()
    return plt.gcf()


def test_xyoffset(setup_mpl):
    x = np.linspace(0, 1)

    for offset in ([-0.5, 0.5], 1, 1.2):  # try lists  # try int  # try float
        plt.clf()
        ax = plt.gca()
        ax.plot(x, np.sin(x) * 10, label=r"$\sin x$")
        ax.plot(x, np.cos(x) * 10, label=r"$\cos x$")
        lines = ax.get_lines()
        labelLines(
            lines,
            xvals=(0.2, 0.7),
            xoffsets=offset,
            yoffsets=offset,
            align=False,
            bbox={"alpha": 0},
        )


@pytest.mark.mpl_image_compare
def test_outline(setup_mpl):
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
def test_auto_layout(setup_mpl):
    X = [[1, 2], [0, 1]]
    Y = [[0, 1], [0, 1]]

    lines = []
    for i, (x, y) in enumerate(zip(X, Y)):
        lines.extend(plt.plot(x, y, label=f"i={i}"))

    labelLines(lines)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_single_point_line(setup_mpl):
    plt.plot(1, 1, label="x")
    labelLines(plt.gca().get_lines())
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
def test_errorbar_with_list(setup_mpl):
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


@pytest.mark.mpl_image_compare
def test_labeling_axhline(setup_mpl):
    fig, ax = plt.subplots()
    ax.plot([10, 12, 13], [1, 2, 3], label="plot")
    ax.axhline(y=2, label="axhline")
    labelLines()
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


def test_label_line_input():
    fig, ax = plt.subplots()
    x = [0, 1]
    y = x
    ax.plot(x, y, label="test")
    msg = "When rotation is set, align needs to be false or none was align=True"
    # test non-allowed combinations of inputs: It is not
    # allowed to set a rotation while aligning it with the
    # lines
    with pytest.raises(ValueError, match=msg):
        labelLines(ax.get_lines(), align=True, rotation=0)

    # test default settings
    for label in labelLines(ax.get_lines()):
        assert label._rotation != 0

    # test setting an angle
    for label in labelLines(ax.get_lines(), align=False, rotation=45):
        assert label._rotation == 45

    # test setting an angle, not explicitly setting align
    for label in labelLines(ax.get_lines(), rotation=45):
        assert label._rotation == 45
