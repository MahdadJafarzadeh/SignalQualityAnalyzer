# -*- coding: utf-8 -*-
"""
SigQual
CopyRight: Mahdad Jafarzadeh - 2020

Simulation of synchronizing two sinusoidal waves.

"""

# import libs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define sampling freq and time
fs = 250
t1 = np.arange(0, 5, 1/fs)

# Generate sig 1 
f1 = 1
x1 = 1e-6 * np.sin(2*3.14*f1*t1)

# Generate sig 2
f2 = 1
x2 = 1e-6*np.cos(2*3.14*f2*t1)

# plot original sigs
plt.plot(t1,x1, color = 'blue', label = 'sin')
plt.plot(t1,x2, color = 'red', label = 'cos')

# correlation
corr = signal.correlate(x1, x2)
       
# find lag (either positive or negative)
min_max_corr = np.abs([np.min(corr), np.max(corr)])

# Is the lag neg or pos?
corr_sign = np.max(min_max_corr)

# Shift sig 1 forward if the corr < 0
if corr_sign == min_max_corr[0]: # if negative corr
    lag  = np.argmin(corr) - len(x1) + 1
    plt.plot(t1+lag/fs , x1 , color = 'black')
    
# Shift sig 1 backward if the corr > 0
else:                            # if positive corr
    lag  = np.argmax(corr) - len(x1) + 1
    plt.plot(t1-lag/fs , x1 , color = 'black')

    
# Plot correlation
plt.figure()
plt.plot(np.arange(len(corr)), corr)
plt.title('corr')    