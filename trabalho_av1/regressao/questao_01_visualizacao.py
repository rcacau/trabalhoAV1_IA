"""Questao 1: visualizacao inicial do PIB da China."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trabalho_av1.comum import PASTA_RESULTADOS, carregar_china


def gerar_grafico() -> Path:
    x, y = carregar_china()
    destino = PASTA_RESULTADOS / "regressao" / "01_dispersao_china.png"
    destino.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.scatter(x[:, 0], y[:, 0] / 1e12, color="teal", edgecolors="black")
    plt.title("Evolucao do PIB da China")
    plt.xlabel("Ano")
    plt.ylabel("PIB (trilhoes de US$)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(destino, dpi=160)
    plt.close()
    return destino


if __name__ == "__main__":
    caminho = gerar_grafico()
    print(f"Grafico salvo em: {caminho}")
    print("O crescimento possui curvatura; uma reta tende a subestimar esse padrao.")
