"""Questao 2: organizacao das variaveis regressoras X e da saida y."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trabalho_av1.comum import carregar_china


def organizar_dados():
    return carregar_china()


if __name__ == "__main__":
    X, y = organizar_dados()
    print(f"Dimensao de X: {X.shape} = (N, p)")
    print(f"Dimensao de y: {y.shape} = (N, 1)")
    print("\nCinco primeiras observacoes:")
    for ano, pib in zip(X[:5, 0], y[:5, 0]):
        print(f"ano={ano:.0f}, PIB=US$ {pib:,.2f}")

