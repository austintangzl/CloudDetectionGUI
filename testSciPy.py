import numpy as np
import scipy.signal as sproc
import matplotlib.pyplot as plt
from mag2db import mag2db

# b = np.array([1, -0.5])
# a = 1
# n = 1024
# w, h = sproc.freqz(b, a=a, worN=n)
#
# # Convert to dB
# dB = mag2db(abs(h))
#
# # Plot the graph
# plt.plot(w/np.pi, dB)
# plt.show()

numtaps = 15
bands = np.array([0, .2, .3, .5])
desired = np.array([1, 0])
h = sproc.remez(numtaps, bands, desired)

plt.plot(h)
plt.show()
