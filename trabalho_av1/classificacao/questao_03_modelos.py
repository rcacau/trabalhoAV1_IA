"""Questao 3: classificadores MQO tradicional, regularizado e polinomial."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trabalho_av1.classificacao.questao_01_organizacao import organizar_dados
from trabalho_av1.comum import ModeloMQO, PASTA_RESULTADOS, prever_classes


LAMBDA_CLASSIFICACAO = 0.25


def ajustar_modelos(ordem_inicial: int = 2):
    x, y, rotulos, classes = organizar_dados()
    modelos = {
        "MQO tradicional": ModeloMQO(ordem=1),
        f"MQO regularizado (lambda={LAMBDA_CLASSIFICACAO})": ModeloMQO(
            ordem=1, lbd=LAMBDA_CLASSIFICACAO
        ),
        f"MQO polinomial (q={ordem_inicial})": ModeloMQO(ordem=ordem_inicial),
    }
    for modelo in modelos.values():
        modelo.fit(x, y)
    return x, rotulos, classes, modelos


def gerar_regioes_decisao() -> Path:
    x, rotulos, classes, modelos = ajustar_modelos()
    margem = 100
    eixo_1 = np.linspace(x[:, 0].min() - margem, x[:, 0].max() + margem, 220)
    eixo_2 = np.linspace(x[:, 1].min() - margem, x[:, 1].max() + margem, 220)
    grade_1, grade_2 = np.meshgrid(eixo_1, eixo_2)
    grade = np.column_stack((grade_1.ravel(), grade_2.ravel()))

    figura, eixos = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for eixo, (nome, modelo) in zip(eixos, modelos.items()):
        previstos = prever_classes(modelo.predict(grade), classes).reshape(grade_1.shape)
        eixo.contourf(grade_1, grade_2, previstos, levels=np.arange(0.5, 6), alpha=0.25)
        for classe in classes:
            pontos = x[rotulos == classe]
            eixo.scatter(pontos[::20, 0], pontos[::20, 1], s=5, alpha=0.5)
        eixo.set_title(nome)
        eixo.set_xlabel("Sensor 1")
        eixo.grid(alpha=0.2)
    eixos[0].set_ylabel("Sensor 2")
    figura.suptitle("Regioes de decisao dos classificadores")
    figura.tight_layout()

    destino = PASTA_RESULTADOS / "classificacao" / "03_regioes_decisao.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=160)
    plt.close(figura)
    return destino


if __name__ == "__main__":
    print(f"Grafico salvo em: {gerar_regioes_decisao()}")
