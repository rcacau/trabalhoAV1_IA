"""Questao 2: dispersao dos sensores, destacando as cinco classes."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trabalho_av1.classificacao.questao_01_organizacao import (
    NOMES_CLASSES,
    organizar_dados,
)
from trabalho_av1.comum import PASTA_RESULTADOS


def gerar_grafico() -> Path:
    x, _, rotulos, classes = organizar_dados()
    destino = PASTA_RESULTADOS / "classificacao" / "02_dispersao_emg.png"
    destino.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    for classe in classes:
        pontos = x[rotulos == classe]
        plt.scatter(
            pontos[:, 0], pontos[:, 1], s=8, alpha=0.35, label=NOMES_CLASSES[classe]
        )
    plt.xlabel("Sensor 1 - Corrugador do supercilio")
    plt.ylabel("Sensor 2 - Zigomatico maior")
    plt.title("Sinais de eletromiografia por expressao facial")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(destino, dpi=160)
    plt.close()
    return destino


if __name__ == "__main__":
    print(f"Grafico salvo em: {gerar_grafico()}")
    print("Sobreposicoes entre cores indicam classes que nao sao linearmente separaveis.")
