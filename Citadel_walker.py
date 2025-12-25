import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

def create_street_contour():
    """to create an irregular closed shape representing the citadel shape"""
    t = list(np.arange(0, 361,10))
    x = []
    for i in t:
        x.append((33 * np.cos(i*(math.pi/180))))

    y = []
    for i in t:
        y.append((33 * np.sin(i*(math.pi/180)))+random.randint(-10,10))

    return x,y

x,y=create_street_contour()
plt.plot(x,y)
plt.show()

