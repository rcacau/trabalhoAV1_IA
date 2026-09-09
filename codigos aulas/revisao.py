from biblioteca import *

nbc = NaiveBayesClassifier()
def nome_funcao(a:int,b:float,c:int, *args, **kwargs) -> str:
    print(a)
    print(b)
    print(c)
    print(args)
    print(kwargs)
   

x = 0.1
y = 0.2

if x + y == 0.3:
    print("é igual a 0,3")
else:
    print("não é")

i = 0
while i < 10:
    print(i)
    if i == 3:
        break
    i+=1
else:
    print("cheguei")

# lista = [1,2,3,4,5,'sdajkjsd','ß']
# for i,elemento in enumerate(lista):
#     print(i,elemento)

for i in range(3,50,2):
    print(i)




nome_funcao(*[1,2,[1,2,3,4]],38,90,'a',True, nome = 'paulo', idade = 'siebenunddrießig')