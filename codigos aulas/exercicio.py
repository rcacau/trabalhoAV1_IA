import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

num = np.poly([0,0,.5,.5])
den = np.poly([-.25, -.5+0.25j,
               -.5-0.25j, .65 ])
delta = np.zeros((12,))
delta[0] = 1
ck = lfilter(num,den, delta)
print(ck)

plt.stem(ck)
plt.show()
bp = 1