"""Questao 4: poda da ordem q por desempenho de validacao em R2."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trabalho_av1.comum import (
    ModeloMQO,
    PASTA_RESULTADOS,
    SEMENTE,
    carregar_china,
    coeficiente_determinacao,
    dividir_indices,
    salvar_csv,
)


ORDENS_TESTADAS = range(1, 11)
RODADAS_SELECAO = 50


def avaliar_ordens() -> list[dict[str, float]]:
    x, y = carregar_china()
    resultados = []
    for ordem in ORDENS_TESTADAS:
        valores_r2 = []
        gerador = np.random.default_rng(SEMENTE)
        for _ in range(RODADAS_SELECAO):
            treino, teste = dividir_indices(x.shape[0], gerador)
            modelo = ModeloMQO(ordem=ordem).fit(x[treino], y[treino])
            valores_r2.append(coeficiente_determinacao(y[teste], modelo.predict(x[teste])))
        resultados.append(
            {
                "ordem": float(ordem),
                "r2_medio": float(np.mean(valores_r2)),
                "r2_desvio": float(np.std(valores_r2, ddof=1)),
            }
        )
    return resultados


def selecionar_ordem(resultados: list[dict[str, float]] | None = None) -> int:
    resultados = resultados or avaliar_ordens()
    melhor = max(resultados, key=lambda item: item["r2_medio"])
    return int(melhor["ordem"])


def salvar_resultados(resultados: list[dict[str, float]]) -> tuple[Path, Path]:
    pasta = PASTA_RESULTADOS / "regressao"
    csv_path = pasta / "04_selecao_ordem.csv"
    figura_path = pasta / "04_selecao_ordem.png"
    salvar_csv(
        csv_path,
        ["ordem", "r2_medio", "r2_desvio_padrao"],
        [[int(r["ordem"]), r["r2_medio"], r["r2_desvio"]] for r in resultados],
    )

    ordens = [int(r["ordem"]) for r in resultados]
    medias = [r["r2_medio"] for r in resultados]
    desvios = [r["r2_desvio"] for r in resultados]
    plt.figure(figsize=(9, 5))
    plt.errorbar(ordens, medias, yerr=desvios, marker="o", capsize=4)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Ordem polinomial q")
    plt.ylabel("R2 medio no teste")
    plt.title("Poda da ordem do modelo polinomial")
    plt.xticks(ordens)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figura_path, dpi=160)
    plt.close()
    return csv_path, figura_path


if __name__ == "__main__":
    avaliacao = avaliar_ordens()
    arquivos = salvar_resultados(avaliacao)
    print(f"Ordem selecionada: q={selecionar_ordem(avaliacao)}")
    print(f"Resultados salvos em: {arquivos[0]} e {arquivos[1]}")
