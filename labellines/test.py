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
    plt.savefig('linear.pdf')


@image_comparison(baseline_images=['labels_log'],
                  extensions=['png'])
def test_logspace():
    plt.clf()
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label='$f(x)=exp(%s x)$' % k)

    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.yscale('log')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')

    plt.savefig('log.pdf')
