import numpy as np
import matplotlib.pyplot as plt



data = np.loadtxt("EMG.csv", delimiter=' ')
classes = np.unique(data[:,-1])
classes = [2,4,5]
C = len(classes)

Y = np.empty((0, C))
X = np.empty((0, 2))
for i,classe in enumerate(classes):
    X_classe = data[data[:,-1]==classe,:-1]
    X = np.vstack((
        X, X_classe
    ))

    y = -np.ones((1, C))
    y[0,i] = 1
    Y = np.vstack((
        Y, np.tile(y, (X_classe.shape[0],1))
    ))


    plt.scatter(X_classe[:,0],X_classe[:,1], edgecolors='k')

X = np.hstack((
    np.ones((X.shape[0],1)), X
))


W = np.linalg.pinv(X)@Y

x1 = np.linspace(-300,5000,1000)
for i in range(C):
    x2 = -W[1,i]/W[2,i]*x1 -W[0,i]/W[2,i]
    plt.plot(x1,x2,c ='k')

X1,X2 = np.meshgrid(x1,x1)
X_plot = np.concatenate((
    np.ones((X1.shape[0],X1.shape[1],1)),
    X1[:,:,None],
    X2[:,:,None]
),axis=2)
Y_pred = X_plot @ W
x_novo = np.array([[1, 1070, 1897]])
J = np.argmax(Y_pred,axis=2)
plt.contourf(X1,X2,J, alpha=.4)
y = x_novo @ W
plt.scatter(1070,1897, marker='x',s=90)
plt.xlim(-200,4095)
plt.ylim(-200,4095)


plt.show()
    

bp = 1