import numpy as np
import matplotlib.pyplot as plt


N = 1000
p  = 8000



X = np.random.uniform(-4,90, (N,p))

X = np.hstack((
    np.ones((N,1)), X
))
XTX = X.T@X
print(XTX.shape)
y = np.random.uniform(0,1, (N,1))
beta = np.random.normal(0,6, (p+1,1))

print(y.T@X@beta)
print((X@beta).T@y)
print(beta.T@X.T@y)

input()
bp = 1