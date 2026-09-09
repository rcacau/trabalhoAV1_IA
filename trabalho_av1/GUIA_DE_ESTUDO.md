# Guia de estudo: modelos de regressao e classificacao por MQO

Este documento explica o trabalho, as escolhas feitas na implementacao, os
resultados obtidos e os conceitos que podem aparecer na apresentacao ou prova.
Ele deve ser estudado junto com os scripts numerados: cada script corresponde a
um item do enunciado.

## 1. Visao geral

O trabalho resolve dois problemas supervisionados:

1. Regressao: estimar uma quantidade continua, o PIB da China, a partir do ano.
2. Classificacao: estimar uma categoria, a expressao facial, a partir de dois
   sinais de eletromiografia.

Aprendizado supervisionado significa que cada observacao de treinamento possui
entradas e uma resposta conhecida. O modelo ajusta seus parametros para reduzir
o erro entre a resposta conhecida e sua previsao.

## 2. Relacao com os codigos da aula

A implementacao preserva as ideias centrais dos exemplos fornecidos:

- `scirillo_kitlearn.py`: classes de regressao, matriz de intercepto, potencias e
  estimacao de `beta` por algebra linear.
- `main.py`: construcao e visualizacao de uma regressao polinomial.
- `main_poly.py`: embaralhamento, divisao 80/20, treinamento e teste.
- `main_classificacao.py`: matriz de alvos com `-1/+1`, estimacao da matriz `W` e
  decisao multiclasse com `argmax`.
- `vpaa.py`: regularizacao de Tikhonov e roteiro de validacao aleatoria.

As adaptacoes necessarias ficaram em `comum.py`. Elas evitam repetir o mesmo
algoritmo em todos os itens, mas cada questao continua em um arquivo independente.

Nenhum modelo pronto de aprendizado de maquina foi usado. NumPy realiza apenas
as operacoes com vetores e matrizes; Matplotlib gera os graficos.

## 3. Notacao e dimensoes

As letras usadas no trabalho possuem significados especificos:

- `N`: quantidade de amostras.
- `p`: quantidade de caracteristicas de entrada.
- `C`: quantidade de classes.
- `X`: matriz de entradas, com dimensao `N x p`.
- `y`: resposta da regressao, com dimensao `N x 1`.
- `Y`: alvos da classificacao, com dimensao `N x C`.
- `beta`: parametros de um modelo com uma saida.
- `W`: parametros de um modelo com varias saidas. No codigo, `beta` pode ser uma
  matriz e cumprir o papel de `W`.

Cada linha de `X` representa uma observacao. Cada coluna representa uma
caracteristica.

## 4. Padronizacao por z-score

Antes de formar as potencias, cada caracteristica e padronizada:

```text
z = (x - media_do_treino) / desvio_padrao_do_treino
```

Depois dessa transformacao, a caracteristica fica aproximadamente com media zero
e desvio-padrao um. Isso e importante porque:

- o ano esta perto de 2000, enquanto suas potencias ficam enormes;
- os sensores variam aproximadamente entre 0 e 4095;
- a regularizacao depende da escala dos parametros;
- matrizes com valores muito diferentes podem ficar numericamente instaveis.

A media e o desvio sao calculados somente no conjunto de treino. O teste e
transformado com os valores aprendidos no treino. Calcular a media usando tambem
o teste seria vazamento de dados, pois o treinamento estaria recebendo informacao
do conjunto usado para avaliacao.

## 5. Matriz de projeto e intercepto

No modelo linear com uma entrada, a matriz de projeto e:

```text
Phi = [1, x]
```

A primeira coluna, composta por uns, cria o intercepto. A previsao e:

```text
y_estimado = beta_0 + beta_1 * x
```

No modelo polinomial de ordem `q`:

```text
Phi = [1, X, X^2, ..., X^q]
```

Para duas caracteristicas, a expansao segue o codigo da aula e nao cria termos
cruzados. Para `q=2`, por exemplo:

```text
Phi = [1, x1, x2, x1^2, x2^2]
```

Um modelo polinomial continua sendo linear nos parametros `beta`, embora seja
nao linear em relacao as entradas.

## 6. MQO tradicional

MQO significa Minimos Quadrados Ordinarios. O objetivo e minimizar a soma dos
quadrados dos residuos:

```text
J(beta) = ||Y - Phi * beta||^2
```

Residuo e a diferenca entre o valor real e o valor previsto. A solucao pelas
equacoes normais e:

```text
beta = pseudoinversa(Phi.T * Phi) * Phi.T * Y
```

A pseudoinversa foi mantida porque tambem aparece nos exemplos da aula. Ao
contrario da inversa comum, ela permite obter uma solucao mesmo quando a matriz e
singular ou quase singular.

## 7. MQO regularizado ou Tikhonov

A regularizacao adiciona uma penalidade ao tamanho dos parametros:

```text
J(beta) = ||Y - Phi * beta||^2 + lambda * ||beta||^2
```

Sua solucao e:

```text
beta = pseudoinversa(Phi.T * Phi + lambda * I) * Phi.T * Y
```

Interpretacao de `lambda`:

- `lambda = 0`: o resultado coincide com o MQO tradicional.
- `lambda` pequeno: penalidade fraca.
- `lambda` grande: coeficientes menores, modelo menos sensivel, mas com risco de
  subajuste.

O codigo regulariza todos os coeficientes, incluindo o intercepto, seguindo a
matriz identidade completa mostrada em `vpaa.py`.

## 8. Metricas

### 8.1 MSE

O Erro Quadratico Medio e:

```text
MSE = media((y - y_estimado)^2)
```

Quanto menor, melhor. Como o PIB esta medido em dolares, o MSE fica em dolares ao
quadrado e apresenta numeros muito grandes. Isso nao significa erro no calculo.

### 8.2 Coeficiente R2

```text
R2 = 1 - soma_dos_residuos_quadrados / soma_total_dos_quadrados
```

- `R2 = 1`: previsao perfeita.
- `R2 = 0`: equivale a sempre prever a media do teste.
- `R2 < 0`: pior do que usar a media como previsao.

Um `R2` negativo e permitido. Ele nao deve ser artificialmente limitado a zero.

### 8.3 Acuracia

```text
acuracia = quantidade_de_acertos / quantidade_de_amostras
```

Uma acuracia de `0.9928` corresponde a `99.28%` de previsoes corretas.

## 9. Validacao Random Subsampling ou Monte Carlo

Em cada uma das 500 rodadas:

1. Os indices das amostras sao embaralhados.
2. Os primeiros 80% formam o treino.
3. Os 20% restantes formam o teste.
4. Um novo modelo e ajustado usando somente o treino.
5. A metrica e calculada somente no teste.
6. A metrica e armazenada em uma lista.

Ao final sao calculados media, desvio-padrao, maior e menor valor.

A media estima o desempenho esperado. O desvio-padrao mede a variacao causada por
diferentes amostras de treino e teste. Um bom modelo deve combinar media boa e
variacao pequena.

A semente `42` torna a sequencia de particoes reproduzivel. Isso nao elimina a
aleatoriedade do metodo; apenas permite repetir exatamente o experimento.

# Parte I - Regressao

## 10. Dados da regressao

O arquivo `china_gdp.csv` possui 55 observacoes:

- `X`, dimensao `55 x 1`: ano de 1960 a 2014.
- `y`, dimensao `55 x 1`: PIB em dolares americanos.

O grafico da questao 1 mostra crescimento lento no inicio e muito acelerado nos
anos finais. Portanto, a relacao nao parece linear. Essa observacao justifica
testar uma curva polinomial.

## 11. Questoes da regressao

### Questao 1 - visualizacao

`questao_01_visualizacao.py` carrega os dados e cria o grafico de dispersao. O PIB
e dividido por `10^12` apenas no eixo do grafico, para ser mostrado em trilhoes de
dolares. O modelo continua trabalhando com os valores originais.

### Questao 2 - organizacao

`questao_02_organizacao.py` separa a primeira coluna em `X` e a segunda em `y`,
preservando matrizes bidimensionais.

### Questao 3 - modelos

`questao_03_modelos.py` demonstra os tres tipos pedidos. A reta tradicional e a
regularizada quase se sobrepoem. O polinomio acompanha melhor a curvatura, embora
o exemplo inicial use apenas `q=3`.

### Questao 4 - selecao da ordem

`questao_04_selecao_polinomial.py` testa ordens de 1 a 10. Cada ordem e avaliada
em 50 particoes preliminares e comparada pelo `R2` medio de teste. Isso corresponde
a poda: os modelos aninhados sao comparados para decidir quantos termos manter.

Resultados principais:

| q | R2 medio | Desvio-padrao |
|---:|---:|---:|
| 3 | 0.8105 | 0.3030 |
| 4 | 0.9601 | 0.0670 |
| 5 | 0.9850 | 0.0155 |
| 8 | 0.9881 | 0.0206 |
| 9 | **0.9965** | **0.0044** |
| 10 | 0.9953 | 0.0063 |

Foi selecionado `q=9`. A ordem 10 nao trouxe melhoria e, portanto, o termo extra
foi podado.

### Questao 5 - regularizacao

`questao_05_regularizacao.py` calcula o MQO tradicional e os cinco valores de
`lambda` pedidos. A execucao confirma que MQO tradicional e regularizado com
`lambda=0` geram o mesmo vetor:

```text
beta = [1.43704182e+12, 1.79317964e+12]
```

Conforme `lambda` cresce, a norma dos coeficientes diminui. Esse e exatamente o
efeito esperado da penalidade de Tikhonov.

### Questao 6 - validacao

`questao_06_validacao.py` compara seis modelos unicos em 500 rodadas:

- polinomial de ordem 9;
- MQO tradicional;
- regularizado com `lambda` 0.25, 0.5, 0.75 e 1.

O regularizado com `lambda=0` nao e repetido porque ja e o MQO tradicional.

### Questao 7 - resultados

Resumo das 500 rodadas:

| Modelo | MSE medio | R2 medio | Desvio do R2 |
|---|---:|---:|---:|
| Polinomial q=9 | **1.4544e22** | **0.9949** | **0.0145** |
| MQO tradicional | 3.3781e24 | -1.3214 | 9.5537 |
| Regularizado 0.25 | 3.3763e24 | -1.2942 | 9.4380 |
| Regularizado 0.50 | 3.3750e24 | -1.2676 | 9.3244 |
| Regularizado 0.75 | 3.3739e24 | -1.2415 | 9.2128 |
| Regularizado 1.00 | 3.3732e24 | -1.2160 | 9.1031 |

Interpretacao:

- O polinomio reduz o MSE em mais de duas ordens de grandeza.
- O `R2` medio do polinomio esta muito proximo de 1 e varia pouco.
- Os modelos lineares apresentam `R2` medio negativo porque uma reta nao descreve
  o crescimento acelerado do PIB.
- Alguns conjuntos de teste possuem pequena variacao do PIB. Neles, o denominador
  do `R2` fica pequeno e um erro linear gera valores negativos extremos, explicando
  o desvio-padrao alto.
- Aumentar `lambda` melhora levemente o resultado linear, mas nao resolve o erro
  de forma do modelo. Regularizacao controla coeficientes; ela nao transforma uma
  reta em curva.

# Parte II - Classificacao

## 12. Dados da classificacao

O arquivo `EMG1.csv` possui:

- `N = 50000` amostras;
- `p = 2` sensores;
- `C = 5` classes;
- 10000 amostras de cada classe.

As duas primeiras colunas formam `X`. A terceira contem os rotulos:

| Rotulo | Expressao |
|---:|---|
| 1 | Neutro |
| 2 | Sorriso |
| 3 | Sobrancelhas levantadas |
| 4 | Surpreso |
| 5 | Rabugento |

## 13. Codificacao um-contra-todos

O exemplo da aula usa uma coluna de saida para cada classe. A classe correta
recebe `+1` e as outras recebem `-1`. Para uma amostra da classe 3:

```text
Y = [-1, -1, +1, -1, -1]
```

O modelo retorna cinco pontuacoes. A classe prevista e a coluna com maior valor:

```text
indice = argmax(pontuacoes)
classe = classes[indice]
```

Nao se trata de probabilidade. As saidas sao escores usados para comparacao.

## 14. Questoes da classificacao

### Questao 1 - organizacao

`questao_01_organizacao.py` produz:

- `X` com dimensao `50000 x 2`;
- `Y` com dimensao `50000 x 5`;
- um vetor de rotulos usado para calcular a acuracia.

### Questao 2 - visualizacao

`questao_02_visualizacao.py` separa as cinco cores. Algumas classes possuem formas
curvas ou alongadas e ocupam regioes que uma unica fronteira reta nao consegue
isolar adequadamente. Logo, o problema nao e completamente linearmente separavel.

### Questao 3 - modelos

`questao_03_modelos.py` estima uma matriz de parametros para cada classificador.
O MQO linear cria fronteiras retas. O polinomial cria fronteiras curvas porque os
escores incluem potencias dos sensores.

Foi adotado `lambda=0.25` no classificador regularizado. O PDF nao define esse
valor para a classificacao; a escolha usa o primeiro valor regularizado pedido na
etapa de regressao e esta centralizada na constante `LAMBDA_CLASSIFICACAO`.

### Questao 4 - selecao da ordem

As ordens de 1 a 6 foram comparadas:

| q | Acuracia inicial | Tempo de ajuste aproximado |
|---:|---:|---:|
| 1 | 72.53% | ~0.004 s |
| 2 | 94.49% | ~0.005 s |
| 3 | 97.64% | ~0.008 s |
| 4 | **99.24%** | ~0.011 s |
| 5 | 99.26% | ~0.014 s |
| 6 | 99.30% | ~0.021 s |

Foi escolhido `q=4`. O criterio aceita uma diferenca maxima de 0.5 ponto
percentual para a melhor acuracia e, dentro dessa faixa, seleciona a menor ordem.
A pequena melhoria das ordens 5 e 6 nao compensa o aumento de complexidade.

Tempos dependem do computador e nao devem ser apresentados como constantes
universais. O que importa e a tendencia de aumento com `q`.

### Questao 5 - validacao

`questao_05_validacao.py` repete 500 vezes a divisao 80/20. Em cada rodada sao
treinados do zero:

- MQO tradicional;
- MQO regularizado com `lambda=0.25`;
- MQO polinomial com `q=4`.

### Questao 6 - resultados

| Modelo | Acuracia media | Desvio-padrao | Maior | Menor |
|---|---:|---:|---:|---:|
| MQO tradicional | 72.3888% | 0.6430% | 74.18% | 70.13% |
| MQO regularizado 0.25 | 72.3890% | 0.6428% | 74.18% | 70.13% |
| MQO polinomial q=4 | **99.2799%** | **0.0829%** | **99.48%** | **99.06%** |

Interpretacao:

- O polinomial possui media muito maior e variacao muito menor.
- O pior resultado polinomial ainda supera 99%, mostrando estabilidade.
- Tradicional e regularizado sao praticamente identicos. Com 40000 amostras de
  treino e `lambda=0.25`, a penalidade e pequena diante da matriz `Phi.T * Phi`.
- O problema principal dos modelos lineares e a forma das fronteiras, nao apenas
  coeficientes grandes. Por isso a regularizacao quase nao altera a acuracia.
- As potencias ate ordem 4 fornecem flexibilidade suficiente para acompanhar a
  geometria das classes.

## 15. Inconsistencias e decisoes sobre o PDF

O enunciado possui alguns trechos contraditorios:

1. Na regressao, o texto menciona cinco modelos, mas a tabela final tem seis.
2. A lista de lambdas inclui zero, embora regularizado com zero seja igual ao MQO.
3. Na classificacao, o texto menciona cinco modelos, mas lista e tabela mostram tres.
4. O item de organizacao menciona modelos gaussianos bayesianos, que nao aparecem
   em nenhuma lista de implementacoes ou na tabela final.
5. Nao e indicado um valor de `lambda` para a classificacao.

Decisoes adotadas:

- seguir as linhas das tabelas finais;
- nao implementar Bayes gaussiano, pois ele nao e solicitado na lista de modelos;
- mostrar `lambda=0` na questao de parametros, mas nao duplica-lo na validacao;
- usar `lambda=0.25` na classificacao e documentar a escolha.

Esses pontos devem ser confirmados com o professor antes da entrega definitiva,
principalmente o `lambda` da classificacao.

## 16. Roteiro curto para apresentacao

Uma apresentacao objetiva pode seguir esta ordem:

1. Problemas e dados: PIB continuo versus cinco categorias de EMG.
2. Metodo: matriz de projeto, intercepto e solucao por MQO.
3. Regularizacao: papel de `lambda`.
4. Polinomio: papel de `q` e fronteiras curvas.
5. Validacao: 500 rodadas, 80/20, metricas armazenadas.
6. Regressao: mostrar que `q=9` vence porque o PIB cresce de forma nao linear.
7. Classificacao: mostrar que `q=4` vence porque as classes exigem fronteiras curvas.
8. Conclusao: complexidade deve ser suficiente para o padrao, mas sem termos
   desnecessarios.

## 17. Perguntas que podem aparecer

### Por que usar pseudoinversa?

Porque a inversa comum pode nao existir. A pseudoinversa fornece a solucao de
minimos quadrados mesmo para matrizes singulares.

### Por que elevar ao quadrado o erro?

Para impedir que erros positivos e negativos se cancelem e para penalizar erros
grandes com mais intensidade.

### Regularizacao sempre melhora o resultado?

Nao. Ela ajuda quando ha coeficientes instaveis ou sobreajuste. Ela nao corrige um
modelo cuja forma e inadequada, como uma reta tentando representar uma curva.

### Por que o polinomio ainda e resolvido por MQO?

Porque as entradas sao transformadas, mas os parametros continuam aparecendo
linearmente na equacao.

### Qual a diferenca entre parametro e hiperparametro?

`beta` e aprendido diretamente dos dados e e um parametro. `q` e `lambda` sao
escolhidos antes do ajuste final e sao hiperparametros.

### Por que separar treino e teste?

Treino mede a capacidade de ajustar dados conhecidos. Teste mede generalizacao
para amostras que nao participaram do ajuste.

### O que significa um desvio-padrao pequeno?

Significa que o desempenho muda pouco quando a divisao dos dados e alterada. E um
indicio de estabilidade.

### Por que usar `argmax`?

Cada coluna do classificador produz um escore para uma classe. `argmax` encontra
a classe com o maior escore.

## 18. Arquivos de resultados

As tabelas CSV contem todos os numeros usados neste guia. Os arquivos de 500
rodadas permitem conferir cada repeticao individual. Os PNG podem ser usados no
relatorio e na apresentacao.

Os resultados sao reproduzidos executando:

```powershell
trabalho_av1\.venv\Scripts\python.exe trabalho_av1\executar_tudo.py
```
