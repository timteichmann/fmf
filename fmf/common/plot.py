# FreeMolecularFlow - common - plot
# (c) 2024 Tim Teichmann

import numpy as np
import matplotlib.pyplot as plt

def scalar_field(xc, yc, field, contours=None, label=None):
    nx = len(xc)
    ny = len(yc)
    dx = (xc[-1] - xc[0])/nx
    dy = (yc[-1] - yc[0])/ny
    x = np.linspace(xc[0] - 0.5*dx, xc[-1] + 0.5*dx, nx + 1)
    y = np.linspace(yc[0] - 0.5*dy, yc[-1] + 0.5*dy, ny + 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ax.pcolormesh(x, y, field, cmap="bwr")
    plt.axis("scaled")
    cb = fig.colorbar(c, ax=ax)
    if not label is None:
        cb.set_label(label)
    if not contours is None:
        xc, yc = np.meshgrid(xc, yc)
        xc = xc.flatten()
        yc = yc.flatten()
        ax.tricontour(xc, yc, field.flatten(), levels=contours, colors="black", linewidths=0.5)
        
    plt.show()

def xy(x, ys, labels=None, xlabel=None, ylabel=None):
    x = np.array(x)
    ys = np.array(ys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if ys.ndim == 2:
        for i in range(len(ys)):
            li = None
            if not labels is None:
                li = labels[i]
            ax.plot(x, ys[i], label=li)
    else:
        ax.plot(x, ys, label=labels)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if labels:
        ax.legend()

    plt.show()
