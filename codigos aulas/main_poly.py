import numpy as np
import matplotlib.pyplot as plt
from scirillo_kitlearn import PolynomialRegression


data = np.loadtxt("Solubilidade.csv", delimiter=',')

X = data[:,:-1]
Y = data[:,-1:]
rodadas = 1000
N, p = X.shape
for i in range(rodadas):
    idx = np.random.permutation(N)
    Xr = np.copy(X)[idx, :]
    Yr = np.copy(Y)[idx, :]

    X_treino = Xr[:int(.8*N),:]
    Y_treino = Yr[:int(.8*N),:]


    pr = PolynomialRegression(X_treino, Y_treino, 13)

    pr.fit()
    X_teste = Xr[int(.8*N):,:]
    Y_teste = Yr[int(.8*N):,:]
    Y_pred = pr.predict(X_teste)


    bp = 1
    if i == 0:
        fig = plt.figure(1)
        ax = fig.add_subplot(1,2,1,projection='3d')

        ax.scatter(X_treino[:,0],
                   X_treino[:,1],
                   Y_treino[:,0], c='r',edgecolor='k')

        x1 = np.linspace(0,30,300)
        x2 = np.linspace(0,700,300)
        X1,X2 = np.meshgrid(x1,x2)
        X_plot = np.concatenate((
            X1[:,:,None],
            X2[:,:,None],
        ),axis = 2)
        Y_pred = pr.predict(X_plot)
        ax.plot_surface(X1,X2, Y_pred[:,:,0])
        ax = fig.add_subplot(1,2,2,projection='3d')

        ax.set_zlim(-8,8)
        ax.scatter(X_teste[:,0],
                   X_teste[:,1],
                   Y_teste[:,0], c='g',edgecolor='k')
        ax.plot_surface(X1,X2, Y_pred[:,:,0])

        
        plt.show()
    bp = 1