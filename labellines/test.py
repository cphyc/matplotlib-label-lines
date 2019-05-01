from labellines import labelLines, labelLine
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_raises

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
