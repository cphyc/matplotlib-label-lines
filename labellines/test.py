from labellines import labelLines, labelLine
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter, UTC
from numpy.testing import assert_raises
import warnings

import pytest

@pytest.mark.mpl_image_compare
def test_linspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=r'$f(x)=\sin(%s x)$' % k)

    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_ylogspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label=r'$f(x)=\exp(%s x)$' % k)

    plt.yscale('log')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_xlogspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(10**x, k*x, label=r'$f(x)=%s x$' % k)

    plt.xscale('log')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_xylogspace():
    plt.clf()
    x = np.geomspace(1e-1, 1e1)
    K = np.arange(-5, 5, 2)

    for k in K:
        plt.plot(x, x**k, label=r'$f(x)=x^{%s}$' % k)

    plt.xscale('log')
    plt.yscale('log')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_align():
    x = np.linspace(0, 2*np.pi)
    y = np.sin(x)

    lines = plt.plot(x, y, label=r'$\sin(x)$')

    labelLines(lines, align=False)
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_labels_range():
    x = np.linspace(0, 1)

    plt.plot(x, np.sin(x), label=r'$\sin x$')
    plt.plot(x, np.cos(x), label=r'$\cos x$')

    labelLines(plt.gca().get_lines(), xvals=(0, .5))
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_naive():
    dates = [datetime(2018, 11, 1), datetime(2018, 11, 2), datetime(2018, 11, 3)]

    plt.plot(dates, [0, 5, 3], label='apples')
    plt.plot(dates, [3, 6, 2], label='banana')
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dateaxis_advanced():
    dates = [datetime(2018, 11, 1, tzinfo=UTC), datetime(2018, 11, 2, tzinfo=UTC),
             datetime(2018, 11, 5, tzinfo=UTC), datetime(2018, 11, 3, tzinfo=UTC)]

    plt.plot(dates, [0, 5, 3, 0], label='apples')
    plt.plot(dates, [3, 6, 2, 1], label='banana')
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_polar():
    t = np.linspace(0, 2 * np.pi, num=128)
    plt.plot(np.cos(t), np.sin(t), label='$1/1$')
    plt.plot(np.cos(t), np.sin(2 * t), label='$1/2$')
    plt.plot(np.cos(3 * t), np.sin(t), label='$3/1$')
    ax = plt.gca()

    labelLines(ax.get_lines())
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_non_uniform_and_negative_spacing():
    x = [1, -2, -3, 2, -4, -3]
    plt.plot(x, [1, 2, 3, 4, 2, 1], '.-', label='apples')
    plt.plot(x, [6, 5, 4, 2, 5, 5], 'o-', label='banana')
    ax = plt.gca()

    labelLines(ax.get_lines())
    return plt.gcf()

@pytest.mark.mpl_image_compare
def test_errorbar():
    x = np.linspace(0, 1, 20)

    y = x**0.5
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r'$\sqrt{x}\pm x$')

    y = x**3
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r'$x^3\pm x$')

    ax = plt.gca()
    labelLines(ax.get_lines())
    return plt.gcf()


def test_nan_warning():
    x = np.array([0, 1, 2, 3])
    y = np.array([np.nan, np.nan, 0, 1])

    line = plt.plot(x, y, label='test')[0]

    with warnings.catch_warnings(record=True) as w:
        labelLine(line, 0.5)
        assert issubclass(w[-1].category, UserWarning)
        assert "could not be annotated" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        labelLine(line, 2.5)
        assert len(w) == 0

def test_nan_failure():
    x = np.array([0, 1])
    y = np.array([np.nan, np.nan])

    line = plt.plot(x, y, label='test')[0]
    with assert_raises(Exception):
        labelLine(line, 0.5)

def test_label_range():
    x = np.linspace(0, 1)
    line = plt.plot(x, x**2)[0]

    # This should fail
    with assert_raises(Exception):
        labelLine(line, -1)
    with assert_raises(Exception):
        labelLine(line, 2)

    # This should work
    labelLine(line, 0.5)

def test_negative_spacing():
    plt.clf()
    x = np.linspace(1, -1)
    y = x**2

    line = plt.plot(x, y)[0]

    # Should not throw an error
    labelLine(line, 0.2, label='Test')
