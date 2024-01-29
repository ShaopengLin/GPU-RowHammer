import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from mplcursors import cursor
import math

if __name__ == '__main__':
    # Open the file
    with open('stride_latencies.txt', 'r') as f:
        lines = f.readlines()

    # Parse the lines
    x_values = []
    y_values = []
    with open('numbers.txt', 'w') as f:
        for line in lines:
            x, y = map(float, line.split())

            # if y > 1050:
            #     x_values.append(x / 128.0)
            #     y_values.append(y)
            #     print(x/128.0,y)

            x_values.append(math.log2(x / 128.0))
            y_values.append(y)

            # interval = 128
            # grace = 1
            # offset = 0
            # if ((x / 128) - offset) % interval < grace or ((x / 128) - offset) % interval > (interval - grace):
            #     x_values.append(x / 128.0)
            #     y_values.append(y)

    # Plot the values
    fig, ax = plt.subplots()
    scatter = ax.plot(x_values, y_values)

    # Enable the hover functionality
    crs = cursor(scatter, hover=False)

    plt.ylim([800,950])
    plt.xlabel('Strides')
    plt.ylabel('Cycles')
    plt.title('DRAM latencies with different strides')
    plt.show()
    # Parse the lines
    # x_values = []
    # y_values = []
    # for line in lines:
    #     x, y = map(float, line.split())
    #     x_values.append(x)
    #     y_values.append(y)
    #
    # # Create a function that interpolates the data points
    # spl = make_interp_spline(x_values, y_values)
    #
    # # Create an array of x values representing the "smooth" line
    # x_smooth = np.linspace(min(x_values), max(x_values), 500000)
    #
    # # Generate the y values for the smooth line
    # y_smooth = spl(x_smooth)
    #
    # # Plot the original data points
    # # plt.plot(x_values, y_values, 'bo', label='Original points')
    #
    # # Plot the smooth line
    # plt.plot(x_smooth, y_smooth, 'r-', label='Smooth line')
    #
    # plt.xlabel('X values')
    # plt.ylabel('Y values')
    # plt.title('Smooth Line Graph')
    # plt.legend()
    # plt.show()
