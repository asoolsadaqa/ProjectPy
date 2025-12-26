import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import turtle

L = 200
r = L/2
# def create_street_contour():
#     """to create an irregular closed shape representing the citadel shape"""
#     t = list(np.arange(0, 361,10))
#     x = []
#     for i in t:
#         x.append((33 * np.cos(i*(math.pi/180))))
#
#     y = []
#     for i in t:
#         y.append((33 * np.sin(i*(math.pi/180)))+random.randint(-10,10))
#
#     return x,y
#
# x,y=create_street_contour()
# plt.plot(x,y)
# plt.show()
#

def create_street_contour():
    """to create an irregular closed shape representing the citadel shape"""
    x_axis = np.arange(-r,r,10)
    y_axis = []
    for x in x_axis:
        y_axis.append(math.sqrt(pow(r, 2) - pow(x, 2))+random.randint(-10,10))

    for x in x_axis:
        y_axis.append(-math.sqrt(pow(r, 2) - pow(x, 2)) + random.randint(-10, 10))

    y_axis[0] = 0
    y_axis[len(x_axis)-1] = 0
    y_axis[len(x_axis)] = 0
    y_axis[-1] = 0

    return x_axis,y_axis

x,y=create_street_contour()
plt.plot(x,y[0:len(x)],x,y[len(x):])


def d_right(X):
    return (L - X) % L
def d_left(X):
    return X % L
def run_simulation(Force_direction = None):
    sigmoid = 0.25
    delta_x = 1

    X = L / 2

    prob_right = 1 / (1 + math.exp(-sigmoid * (d_right - d_left)))

    positions = []
    probabilities = []
    if Force_direction == 'right':
        X = X - 10 * delta_x
    elif Force_direction == 'left':
        X = X + 10 * delta_x

    while X > 0.0001 or X < -0.0001:
        d_right = (L - X) % L
        d_left = X % L
        prob_right = 1 / (1 + math.exp(-sigmoid * (d_right - d_left)))
        if prob_right < 0.5:
            X = X - delta_x
        elif prob_right > 0.5:
            X = X + delta_x
        positions.append(X)
        probabilities.append(prob_right)
    print(X)
    print(positions)
    print(probabilities)

run_simulation('right')