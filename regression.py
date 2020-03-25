import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


x = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
y = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# remove outlier
del x[5]
del y[5]
x = np.asarray(x)
y = np.asarray(y)


def least_square_method():
    global x, y
    # estimate a slope a and y-intercept b
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    a = np.sum((x - x_bar)*(y - y_bar)) / np.sum((x - x_bar)**2)
    b = y_bar - a*x_bar
    print('slope: {}, y-intercept: {}'.format(a, b))

    # true regression line
    line_x = np.arange(np.min(x), np.max(x), 0.01)
    line_y = a*line_x + b
    plt.plot(line_x, line_y, 'r-')

    # plot a origin data
    plt.plot(x, y, 'bo')
    plt.xlabel('Population Growth Rate (%)')
    plt.ylabel('Elderly Population Rate (%)')
    plt.show()


def tf_regression():
    global x, y


