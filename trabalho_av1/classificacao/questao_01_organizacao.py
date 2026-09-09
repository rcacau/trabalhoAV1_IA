"""Questao 1: organizacao de X, rotulos e matriz-alvo Y."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from trabalho_av1.comum import carregar_emg, codificar_um_contra_todos


NOMES_CLASSES = {
    1: "Neutro",
    2: "Sorriso",
    3: "Sobrancelhas levantadas",
    4: "Surpreso",
    5: "Rabugento",
}


def organizar_dados():
    x, rotulos = carregar_emg()
    y, classes = codificar_um_contra_todos(rotulos)
    return x, y, rotulos, classes


if __name__ == "__main__":
    X, Y, rotulos, classes = organizar_dados()
    print(f"X: {X.shape} = (N, p)")
    print(f"Y: {Y.shape} = (N, C)")
    print(f"Classes: {classes.tolist()}")
    print("Distribuicao:")
    for classe in classes:
        print(f"  {classe} - {NOMES_CLASSES[classe]}: {np.sum(rotulos == classe)} amostras")

