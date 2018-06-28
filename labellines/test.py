from labellines import labelLines
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison


@image_comparison(baseline_images=['labels_linear'],
                  extensions=['png'])
def test_linspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label='$f(x)=sin(%s x)$' % k)

    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')

@image_comparison(baseline_images=['labels_ylog'],
                  extensions=['png'])
def test_ylogspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label='$f(x)=exp(%s x)$' % k)

    plt.yscale('log')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')

@image_comparison(baseline_images=['labels_xlog'],
                  extensions=['png'])
def test_xlogspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(10**x, k*x, label='$f(x)=%s x$' % k)

    plt.xscale('log')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')


@image_comparison(baseline_images=['labels_xylog'],
                  extensions=['png'])
def test_xylogspace():
    plt.clf()
    x = np.geomspace(1e-1, 1e1)
    K = np.arange(-5, 5, 2)

    for k in K:
        plt.plot(x, x**k, label='$f(x)=x^{%s}$' % k)

    plt.xscale('log')
    plt.yscale('log')
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
