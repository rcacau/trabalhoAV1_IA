import numpy as np
import matplotlib.pyplot as plt
from scirillo_kitlearn import PolynomialRegression

data = np.loadtxt("aerogerador.dat", delimiter='\t')


X = data[:,0].reshape(data.shape[0],1)
Y = data[:,-1:]

#Normalização (Padronização - z-Score)
media_x = np.mean(X)
desv_pad = np.std(X)
# X = (X-media_x)/desv_pad
# Y = (Y-np.mean(Y))/np.std(Y)

lr = PolynomialRegression(X,Y, 4 )
lr.fit()

x_novo = np.linspace(-2,20)
Y_pred = lr.predict(x_novo.reshape(len(x_novo),1))
plt.figure(0)
plt.plot(x_novo, Y_pred, c='yellow')
plt.scatter(X[:], Y[:], c='r',edgecolors='k')
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência Gerada")
plt.ylim(-1, 600)
plt.xlim(-1, 15)
plt.grid()

plt.figure(2)
Y_pred = lr.predict(X)
E = Y - Y_pred
plt.hist(E,bins=30,color='teal',edgecolor='k')

# #PURAMENTE PARA FINS DIDÁTICOS
# b0 = np.linspace(-100,100,300)
# b1 = np.linspace(-100,100,300)
# B0, B1 = np.meshgrid(b0,b1)

# Y_pred =B0[:,:,None] + B1[:,:,None]*X[:,0]
# E = Y[:,0] - Y_pred
# J = np.sum(E**2, axis=2)
# print(J.shape)
# fig = plt.figure(3)
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(B0,B1,J, cmap='jet',alpha=.5)
# ax.scatter(lr.beta[0], lr.beta[1], 0, s=120)

plt.show()
bp=1