"""Questao 3: MQO tradicional, regularizado e polinomial."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trabalho_av1.comum import ModeloMQO, PASTA_RESULTADOS, carregar_china


def ajustar_modelos(ordem_inicial: int = 3, lbd_inicial: float = 0.25):
    x, y = carregar_china()
    modelos = {
        "MQO tradicional": ModeloMQO(ordem=1),
        f"MQO regularizado (lambda={lbd_inicial})": ModeloMQO(
            ordem=1, lbd=lbd_inicial
        ),
        f"MQO polinomial (q={ordem_inicial})": ModeloMQO(ordem=ordem_inicial),
    }
    for modelo in modelos.values():
        modelo.fit(x, y)
    return x, y, modelos


def gerar_comparacao() -> Path:
    x, y, modelos = ajustar_modelos()
    anos = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    destino = PASTA_RESULTADOS / "regressao" / "03_modelos_iniciais.png"
    destino.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 0], y[:, 0] / 1e12, color="black", s=24, label="Dados")
    cores = ["royalblue", "darkorange", "seagreen"]
    for (nome, modelo), cor in zip(modelos.items(), cores):
        previsao = modelo.predict(anos)
        plt.plot(anos[:, 0], previsao[:, 0] / 1e12, label=nome, color=cor)
    plt.xlabel("Ano")
    plt.ylabel("PIB (trilhoes de US$)")
    plt.title("Comparacao inicial dos modelos de regressao")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destino, dpi=160)
    plt.close()
    return destino


if __name__ == "__main__":
    print(f"Grafico salvo em: {gerar_comparacao()}")
