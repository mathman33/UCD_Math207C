from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from timeit import default_timer
from mpl_toolkits.mplot3d import Axes3D
import gc as garbage
import os
import json
import argparse
import shlex
from time import sleep
from subprocess import Popen, PIPE
from datetime import datetime

initial_conditions = [1, 0]
final_time = 100
steps = 1000

t = np.linspace(0, final_time, steps)

epsilon = 0.01

def f(a, phi):
    f_1 = epsilon*((1/2)*math.cos(2*phi) - 1)
    f_2 = epsilon*(-(1/2)*math.sin(2*phi))
    return [f_1, f_2]

soln = odeint(f, initial_conditions, t)

u = []
for i in xrange(0, steps):
    u.append(soln[i,0]*math.cos(t[i] + soln[i,1]))

plt.figure()
plt.plot(t, u)
plt.show()
plt.close()
garbage.collect()

def F(a,phi):
    pass

