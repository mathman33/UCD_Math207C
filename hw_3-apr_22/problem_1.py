from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
from timeit import default_timer
from mpl_toolkits.mplot3d import Axes3D
import gc as garbage
import os
import json
import itertools
import argparse
import shlex
from time import sleep
from subprocess import Popen, PIPE
from datetime import datetime


def get_actual(epsilon):
    disc = sqrt(1 - 4*epsilon)
    r_1 = (-1 + disc)/(2*epsilon)
    r_2 = (-1 - disc)/(2*epsilon)
    A = 1/(epsilon*(r_1 - r_2))
    def f(t):
        return A*(np.exp(r_1*t) - np.exp(r_2*t))
    return f

def get_composite(epsilon):
    def f(t):
        order_1 = np.exp(-t) - (1+t)*np.exp(-t/epsilon)
        order_epsilon = (2 - t)*np.exp(-t) - np.exp(-t/epsilon)
        return order_1 + epsilon*order_epsilon
    def g(t):
        return np.exp(-t) - (1+t)*np.exp(-t/epsilon)
    return f, g

def main():
    epsilons = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    after_epsilons = epsilons[1::]
    y = np.linspace(10**(-4), 10**(-2), 1000)
    z = np.linspace(10**(-2), 1, 1000)
    x = np.concatenate((y, z), axis=0)
    data = {}
    for epsilon in epsilons:
        actual = get_actual(epsilon)
        [composite, first_term] = get_composite(epsilon)
        y_actual = actual(x)
        y_composite = composite(x)
        y_first_term = first_term(x)
        error = abs(y_actual - y_composite)
        first_term_error = abs(y_actual - y_first_term)
        data[str(epsilon)] = [y_actual, y_composite, error, y_first_term, first_term_error]

    plt.figure()
    plt.semilogx(x, data[str(epsilons[0])][0], "-", color="k", linewidth=2, label="actual solution")
    plt.semilogx(x, data[str(epsilons[0])][1], "--", color="0.75", linewidth=2, label=r"$O(\varepsilon)$ composite expansion")
    plt.semilogx(x, data[str(epsilons[0])][3], "--", color="0.5", linewidth=2, label=r"$(1) composite expansion")
    for epsilon in epsilons:
        plt.semilogx(x, data[str(epsilon)][0], "-", color="k", linewidth=2)
        plt.semilogx(x, data[str(epsilon)][1], "--", color="0.75", linewidth=2)
        plt.semilogx(x, data[str(epsilon)][3], "--", color="0.5", linewidth=2)
        plt.ylim([0, 1.075])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$u(x)$", rotation=0)
    plt.text(1.5*10**(-4), 1.015, r"$\varepsilon=10^{-5}$")
    plt.text(1.5*10**(-4), 0.74, r"$\varepsilon=10^{-4}$")
    plt.text(1.3*10**(-3), 0.7, r"$\varepsilon=10^{-3}$")
    plt.text(10**(-2), 0.6, r"$\varepsilon=10^{-2}$")
    plt.text(0.8*10**(-1), 0.625, r"$\varepsilon=10^{-1}$", rotation=60)
    plt.legend(loc=4, fontsize=10)
    plt.savefig("semilog.png", filetype="png", dpi=300)
    plt.close()
    garbage.collect()

    plt.figure()
    plt.plot(x, data[str(epsilons[0])][0], "-", color="k", linewidth=2, label="actual solution")
    plt.plot(x, data[str(epsilons[0])][1], "--", color="0.75", linewidth=2, label=r"$O(\varepsilon)$ composite expansion")
    plt.plot(x, data[str(epsilons[0])][3], "--", color="0.5", linewidth=2, label=r"$(1) composite expansion")
    for epsilon in epsilons:
        plt.plot(x, data[str(epsilon)][0], "-", color="k", linewidth=2)
        plt.plot(x, data[str(epsilon)][1], "--", color="0.75", linewidth=2)
        plt.plot(x, data[str(epsilon)][3], "--", color="0.5", linewidth=2)
        plt.ylim([0, 1.1])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$u(x)$", rotation=0)
    plt.text(1.5*10**(-4), 1.015, r"$\varepsilon=10^{-5}$")
    plt.text(2.5*10**(-2), 0.98, r"$\varepsilon=10^{-4}$")
    plt.text(0.05, 0.88, r"$\varepsilon=10^{-3}$", rotation=-30)
    plt.text(0.14, 0.9, r"$\varepsilon=10^{-2}$", rotation=-30)
    plt.text(0.1, 0.68, r"$\varepsilon=10^{-1}$", rotation=40)
    plt.legend(loc=0, fontsize=10)
    plt.savefig("standard.png", filetype="png", dpi=300)
    plt.close()
    garbage.collect()

    plt.figure()
    for epsilon in epsilons:
        plt.loglog(x, data[str(epsilon)][2], color="k")
        plt.ylim([0, 1.1])
        plt.xlabel(r"$x$")
        plt.ylabel("absolute error")
    plt.title(r"Absolute Error of $O(\varepsilon)$ Composite Expansion")
    plt.text(10**(-1), 8*10**(-10), r"$\varepsilon=10^{-5}$")
    plt.text(4*10**(-2), 8*10**(-8), r"$\varepsilon=10^{-4}$")
    plt.text(7*10**(-3), 6.5*10**(-6), r"$\varepsilon=10^{-3}$")
    plt.text(2*10**(-3), 10**(-2), r"$\varepsilon=10^{-2}$")
    plt.text(4*10**(-4), 1.1*10**(-1), r"$\varepsilon=10^{-1}$")
    plt.savefig("errors_epsilon.png", filetype="png", dpi=300)
    plt.close()
    garbage.collect()

    plt.figure()
    for epsilon in epsilons:
        plt.loglog(x, data[str(epsilon)][4], color="k")
        plt.ylim([0, 1.1])
        plt.xlabel(r"$x$")
        plt.ylabel("absolute error")
    plt.title(r"Absolute Error of $O(1)$ Composite Expansion")
    plt.text(10**(-1), 2*10**(-5), r"$\varepsilon=10^{-5}$")
    plt.text(10**(-1), 2*10**(-4), r"$\varepsilon=10^{-4}$")
    plt.text(10**(-1), 2*10**(-3), r"$\varepsilon=10^{-3}$")
    plt.text(10**(-1), 2*10**(-2), r"$\varepsilon=10^{-2}$")
    plt.text(10**(-1), 2*10**(-1), r"$\varepsilon=10^{-1}$")
    plt.savefig("errors_1.png", filetype="png", dpi=300)
    plt.close()
    garbage.collect()


if __name__ == "__main__":
    main()
