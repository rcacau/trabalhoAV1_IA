# Trabalho AV1 - regressao e classificacao

Implementacao baseada nos codigos desenvolvidos em sala. Os modelos foram feitos
com NumPy, sem `scikit-learn` ou bibliotecas que fornecam os algoritmos prontos.

## Organizacao

```text
trabalho_av1/
|-- comum.py                       # MQO, regularizacao, metricas e dados
|-- executar_tudo.py               # executor opcional
|-- GUIA_DE_ESTUDO.md              # teoria, resultados e roteiro de apresentacao
|-- regressao/
|   |-- questao_01_visualizacao.py
|   |-- questao_02_organizacao.py
|   |-- questao_03_modelos.py
|   |-- questao_04_selecao_polinomial.py
|   |-- questao_05_regularizacao.py
|   |-- questao_06_validacao.py
|   `-- questao_07_resultados.py
|-- classificacao/
|   |-- questao_01_organizacao.py
|   |-- questao_02_visualizacao.py
|   |-- questao_03_modelos.py
|   |-- questao_04_selecao_polinomial.py
|   |-- questao_05_validacao.py
|   `-- questao_06_resultados.py
`-- resultados/                    # graficos e tabelas gerados
```

## Como executar

A partir da pasta que contem `trabalho_av1`:

```powershell
python -m venv trabalho_av1\.venv
trabalho_av1\.venv\Scripts\python.exe -m pip install -r trabalho_av1\requirements.txt
trabalho_av1\.venv\Scripts\python.exe trabalho_av1\executar_tudo.py
```

Cada questao tambem pode ser executada separadamente. Exemplo:

```powershell
trabalho_av1\.venv\Scripts\python.exe trabalho_av1\regressao\questao_01_visualizacao.py
```

O executor completo leva aproximadamente um minuto neste computador. Os arquivos
gerados sao colocados em `trabalho_av1/resultados`.

## Observacao sobre o enunciado

O PDF possui inconsistencias na quantidade de modelos. A implementacao segue as
tabelas fornecidas: seis modelos unicos na regressao e tres na classificacao. O
MQO regularizado com `lambda=0` foi demonstrado na questao 5, mas nao foi repetido
na tabela final porque e igual ao MQO tradicional.

