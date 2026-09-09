import numpy as np
import matplotlib.pyplot as plt

def superficie(X,Y):
    return np.exp(-(X**2 + Y**2)) + np.exp(-((X-1.7)**2 + (Y-1.7)**2))*2
def pdf_gaussiana(x, mu = 0, sigma_2 = 1):
    return 1/(np.sqrt(2*sigma_2))*np.exp(-((x - mu)**2/(2*sigma_2)))

def seno(x):
    return np.sin(x * 2*np.pi)
def cosseno(x):
    return np.cos(x * 2*np.pi)

x = np.linspace(-5,5,1000)
y = seno(x)
plt.figure(1, facecolor='k')
plt.plot(x,y,'--r', label = 'seno')
y = cosseno(x)
plt.plot(x,y, label = 'cosseno',lw = 4)
plt.legend(loc='upper right')
plt.gca().set_facecolor('k')
plt.grid()
plt.title("Funções seno e cosseno")
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")



fig = plt.figure(2)
x = np.linspace(-4,4,13)
ax = fig.add_subplot(2,2,1)
plt.plot(x, pdf_gaussiana(x))
ax = fig.add_subplot(2,2,2)
ax.stem(x , pdf_gaussiana(x))
ax = fig.add_subplot(2,2,3)
ax.scatter(x, pdf_gaussiana(x), c='r', marker=r'$\frac{\pi}{\Sigma^2}$',s = 500)
ax = fig.add_subplot(2,2,4)
normal1 = np.random.normal(2,5,(1000))
normal2 = np.random.uniform(2,5,(1000))
ax.boxplot([normal1, normal2], positions=[1,2])



fig = plt.figure(3)
ax = fig.add_subplot(2,2,1)
normal2 = np.random.normal(0, 4, (1000)).astype(int)
ax.hist(normal2, color= 'teal', edgecolor= 'k', bins = 20)
print(normal2)

ax = fig.add_subplot(2,2,2)
contadores = [np.sum(i == normal2) for i in np.unique(normal2)]
ax.pie(contadores, labels=np.unique(normal2))

ax = fig.add_subplot(2,1,2)
ax.violinplot(normal2, vert= False)



fig = plt.figure(4)
ax = fig.add_subplot(projection='3d')
x = np.linspace(-4,4, 1000)
ax.scatter(seno(x)*np.exp(-.1*x),cosseno(x)*np.exp(-.4*x),x, marker="x")



fig = plt.figure(4)
ax = fig.add_subplot(projection='3d')
X,Y = np.meshgrid(x,x)

Z = superficie(X,Y)
ax.plot_surface(X,Y,Z, cmap='turbo', rstride=50, cstride=50)


plt.figure(5)
dados = np.loadtxt("spiral_d.csv",delimiter=',')
plt.scatter(dados[:,0], dados[:,1])
plt.figure(6)
plt.scatter(dados[dados[:,-1]==1,0], dados[dados[:,-1]==1,1],c='r',edgecolors='purple')
plt.scatter(dados[dados[:,-1]==-1,0], dados[dados[:,-1]==-1,1],c='b',edgecolors='k')

plt.show()





bp = 1