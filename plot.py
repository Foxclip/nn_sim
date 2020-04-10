import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # noqa
import simulation
import numpy as np


def plot_loss():
    loss_list = [sim.loss for sim in simulation.simulations]
    ln, = plt.plot(loss_list)
    max_x = np.max(ln.get_data()[0])
    max_y = np.max(ln.get_data()[1])
    axes = plt.gca()
    axes.set_xlim([0, max_x + 1])
    axes.set_ylim([0, max_y])
    plt.show()


def surface_plot(x, y, z, xlabel="", ylabel=""):
    x, y = np.meshgrid(x, y)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot_surface(x, y, z, cmap='plasma')
    plt.show()


def loss_surface_plot(x, y, xlabel, ylabel):
    x = np.array(x[:])
    y = np.array(y[:])
    balance_arr = np.array([sim.loss for sim in simulation.simulations])
    z = balance_arr.reshape(len(y), len(x))
    surface_plot(x, y, z, xlabel=xlabel, ylabel=ylabel)
