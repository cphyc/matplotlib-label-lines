# matplotlib-label-lines
[![Build status](https://github.com/cphyc/matplotlib-label-lines/actions/workflows/pytest.yml/badge.svg)](https://github.com/cphyc/matplotlib-label-lines/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cphyc/matplotlib-label-lines/branch/master/graph/badge.svg)](https://codecov.io/gh/cphyc/matplotlib-label-lines)

Easily label line(s) using matplotlib.

The code is heavily based on http://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib (original code from NauticalMile).

## Install

Just do:
```bash
pip install matplotlib-label-lines
```
You can try it online on binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cphyc/matplotlib-label-lines/master), get some inspiration from [the example](https://github.com/cphyc/matplotlib-label-lines/blob/master/example/matplotlib_label_lines.ipynb) or from the following script:
```python
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, loglaplace

from labellines import labelLine, labelLines

X = np.linspace(0, 1, 500)
A = [1, 2, 5, 10, 20]
funcs = [np.arctan, np.sin, loglaplace(4).pdf, chi2(5).pdf]

plt.subplot(321)
for a in A:
    plt.plot(X, np.arctan(a * X), label=str(a))

labelLines(plt.gca().get_lines(), zorder=2.5)

plt.subplot(322)
for a in A:
    plt.plot(X, np.sin(a * X), label=str(a))

labelLines(plt.gca().get_lines(), align=False, fontsize=14)

plt.subplot(323)
for a in A:
    plt.plot(X, loglaplace(4).pdf(a * X), label=str(a))

xvals = [0.8, 0.55, 0.22, 0.104, 0.045]
labelLines(plt.gca().get_lines(), align=False, xvals=xvals, color="k")

plt.subplot(324)
for a in A:
    plt.plot(X, chi2(5).pdf(a * X), label=str(a))

lines = plt.gca().get_lines()
l1 = lines[-1]
labelLine(
    l1,
    0.6,
    label=r"$Re=${}".format(l1.get_label()),
    ha="left",
    va="bottom",
    align=False,
    backgroundcolor='none'
)
labelLines(lines[:-1], yoffsets=0.01, align=False, backgroundcolor='none')

# labelLines also supports log-scaled x-axes
plt.subplot(325)
for a in A:
    plt.semilogx(X, np.arctan(5 * a * X), label=str(a))

labelLines(plt.gca().get_lines(), zorder=2.5)

plt.subplot(326)
for a in A:
    plt.semilogx(X, chi2(5).pdf(a * X), label=str(a))

labelLines(plt.gca().get_lines(), xvals=(0.1, 1), zorder=2.5)

plt.show()

```
![Example](https://raw.githubusercontent.com/cphyc/matplotlib-label-lines/master/example.png)
