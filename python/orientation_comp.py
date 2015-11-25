import numpy as np
import sys
import matplotlib.pyplot as plt

def calc(th, x0=4000.0, y0=500.0):
    th = np.deg2rad(th + 90)
    r0 = np.sqrt(x0**2 + y0**2)
    th0 = np.arctan(y0 / x0)
    y1 = r0 * np.cos(th - th0)
    x1 = r0 * np.sin(th - th0)
    return x1,y1

for h in np.linspace(-500.0, 500.0):
    x = []
    y = []
    for i in np.linspace(-3.5, 3.5):
        dx, dy = calc(i, y0=h)
        x.append(dx)
        y.append(dy)
    plt.plot(x,y, c='black')
plt.show()
