import numpy as np 
from matplotlib import pyplot as plt

def plotter(horizontal_variable, vertical_variable, horizontal_label, vertical_label, function_label, plot_size=16, figure_size=(8,6)):
    '''
    Something to make it easier to plot
    Flesh out later
    '''

    plt.rc("font", size = plot_size)
    plt.figure(figsize=figure_size)
    plt.plot(horizontal_variable, vertical_variable,  '-', function_label)
    plt.xlabel(horizontal_label)
    plt.ylabel(vertical_label);
    plt.legend()
    plt.show()