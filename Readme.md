# matplotlib-label-lines
Label line using matplotlib.

From http://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib (original code from NauticalMile).

## Install

Just do:
```bash
pip install matplotlib-label-lines
```
and then:
```python
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import loglaplace,chi2

from labellines import labelLine, labelLines

X = np.linspace(0,1,500)
A = [1,2,5,10,20]
funcs = [np.arctan,np.sin,loglaplace(4).pdf,chi2(5).pdf]

plt.subplot(221)
for a in A:
    plt.plot(X,np.arctan(a*X),label=str(a))

labelLines(plt.gca().get_lines(),zorder=2.5)

plt.subplot(222)
for a in A:
    plt.plot(X,np.sin(a*X),label=str(a))

labelLines(plt.gca().get_lines(),align=False,fontsize=14)

plt.subplot(223)
for a in A:
    plt.plot(X,loglaplace(4).pdf(a*X),label=str(a))

xvals = [0.8,0.55,0.22,0.104,0.045]
labelLines(plt.gca().get_lines(),align=False,xvals=xvals,color='k')

plt.subplot(224)
for a in A:
    plt.plot(X,chi2(5).pdf(a*X),label=str(a))

lines = plt.gca().get_lines()
l1=lines[-1]
labelLine(l1,0.6,label=r'$Re=${}'.format(l1.get_label()),ha='left',va='bottom',align = False)
labelLines(lines[:-1],align=False)

plt.show()
```
