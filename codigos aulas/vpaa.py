import numpy as np
import matplotlib.pyplot as plt


XTX = np.array([
    [1,2,13],
    [2,4,22],
    [4,8,14],
])
#Regularização (Ridge Regression -> Regressão pela Cumeeira) Tikhonov
lbd = 0
I = np.eye(3)
print(np.linalg.inv(XTX + I*lbd))


















#Validação: Random subsampling validation 

Rodadas = 100000

#k-fold cross validation
for i in range(Rodadas):
    #Embaralhar os N dados (características e rotulos corretos)
    #Particionamento (treino e teste) (90/10, 80/20, 70/30)
    #Estima parâmetros (treino)
    #Predição (teste)
    #Acumular a métrica desempenho (SSR, acurácia)


    ...
#Extrair estatísticas das métricas calculadas (média, dp, maior, menor, mediana, moda)

bp = 1