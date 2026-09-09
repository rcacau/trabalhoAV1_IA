"""Questao 5: estimativas para os valores de lambda pedidos."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from trabalho_av1.comum import ModeloMQO, PASTA_RESULTADOS, carregar_china, salvar_csv


LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def estimar_betas() -> list[tuple[str, float, np.ndarray]]:
    x, y = carregar_china()
    estimativas = []

    tradicional = ModeloMQO(ordem=1).fit(x, y)
    estimativas.append(("MQO tradicional", 0.0, tradicional.beta.copy()))

    for lbd in LAMBDAS:
        modelo = ModeloMQO(ordem=1, lbd=lbd).fit(x, y)
        estimativas.append(("MQO regularizado", lbd, modelo.beta.copy()))
    return estimativas


def salvar_estimativas(estimativas: list[tuple[str, float, np.ndarray]]) -> Path:
    destino = PASTA_RESULTADOS / "regressao" / "05_estimativas_beta.csv"
    linhas = []
    for nome, lbd, beta in estimativas:
        linhas.append([nome, lbd, *beta[:, 0].tolist(), float(np.linalg.norm(beta))])
    salvar_csv(destino, ["modelo", "lambda", "beta_0", "beta_1", "norma_beta"], linhas)
    return destino


if __name__ == "__main__":
    betas = estimar_betas()
    for nome, lbd, beta in betas:
        print(f"{nome:18s} lambda={lbd:>4}: beta={beta[:, 0]}")
    print("\nCom lambda=0, o regularizado coincide com o MQO tradicional.")
    print(f"Tabela salva em: {salvar_estimativas(betas)}")

