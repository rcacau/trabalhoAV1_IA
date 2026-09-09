import numpy as np

x = np.array([
    [1,2,3,4,5],
    [2,2,3,4,5],
    [3,2,3,4,5],
    [4,2,3,4,5],
])


y = np.linspace(1, 30,5)
y = np.arange(2, 10, 0.01)

uns = np.ones((3,2))
zeros = np.zeros((10,2)).astype(int)
I = np.eye(6)*.5

R = np.random.randint(4, 10, (10,2))
Rf = np.random.uniform(4, 10, (10,2))

X = np.hstack((R,Rf))



print(np.max(X, axis = 1))
print(np.min(X, axis = 1))
print(np.mean(X, axis = 1))
print(np.sum(X, axis = 1))
print(np.var(X, axis = 1))
print(np.std(X, axis = 1))
print(np.argmax(X, axis = 1))


x = np.random.randint(0,4,(10,3))
y = np.random.randint(0,4,(10,5))
print(x)
print(y)
# print(x * y) #Produto de hadamard (element-wise product)

#      A       B
#   [n x m][m x p]  
print(x.T @ y)


A = np.random.randint(0,10,(10,4))
print(A)
print(A[:,:-1])


dados = np.loadtxt("spiral_d.csv",delimiter=',')
rotulos = np.unique(dados[:,-1])
print(rotulos)



X1 = dados[dados[:,-1] == 1, :-1]
X2 = dados[dados[:,-1] == -1, :-1]


bp=1