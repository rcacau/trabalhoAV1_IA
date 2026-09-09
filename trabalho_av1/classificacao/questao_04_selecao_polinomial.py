"""Questao 4: escolha de q pelo compromisso entre acuracia e complexidade."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trabalho_av1.classificacao.questao_01_organizacao import organizar_dados
from trabalho_av1.comum import (
    ModeloMQO,
    PASTA_RESULTADOS,
    SEMENTE,
    acuracia,
    ajustar_cronometrado,
    dividir_indices,
    prever_classes,
    salvar_csv,
)


ORDENS_TESTADAS = range(1, 7)
TOLERANCIA_ACURACIA = 0.005


def avaliar_ordens() -> list[dict[str, float]]:
    x, y, rotulos, classes = organizar_dados()
    gerador = np.random.default_rng(SEMENTE)
    treino, teste = dividir_indices(x.shape[0], gerador)
    resultados = []

    for ordem in ORDENS_TESTADAS:
        modelo, tempo = ajustar_cronometrado(ModeloMQO(ordem=ordem), x[treino], y[treino])
        previstos = prever_classes(modelo.predict(x[teste]), classes)
        resultados.append(
            {"ordem": float(ordem), "acuracia": acuracia(rotulos[teste], previstos), "tempo": tempo}
        )
    return resultados


def selecionar_ordem(resultados: list[dict[str, float]] | None = None) -> int:
    resultados = resultados or avaliar_ordens()
    melhor_acuracia = max(item["acuracia"] for item in resultados)
    candidatos = [
        item for item in resultados if item["acuracia"] >= melhor_acuracia - TOLERANCIA_ACURACIA
    ]
    # Entre resultados praticamente empatados, fica com o modelo mais simples.
    return int(min(candidatos, key=lambda item: item["ordem"])["ordem"])


def salvar_resultados(resultados: list[dict[str, float]]) -> tuple[Path, Path]:
    pasta = PASTA_RESULTADOS / "classificacao"
    csv_path = pasta / "04_selecao_ordem.csv"
    figura_path = pasta / "04_selecao_ordem.png"
    salvar_csv(
        csv_path,
        ["ordem", "acuracia", "tempo_ajuste_segundos"],
        [[int(r["ordem"]), r["acuracia"], r["tempo"]] for r in resultados],
    )

    figura, eixo_1 = plt.subplots(figsize=(9, 5))
    ordens = [int(r["ordem"]) for r in resultados]
    eixo_1.plot(ordens, [r["acuracia"] for r in resultados], marker="o", color="teal")
    eixo_1.set_xlabel("Ordem polinomial q")
    eixo_1.set_ylabel("Acuracia", color="teal")
    eixo_1.tick_params(axis="y", labelcolor="teal")
    eixo_1.grid(alpha=0.3)
    eixo_2 = eixo_1.twinx()
    eixo_2.plot(ordens, [r["tempo"] for r in resultados], marker="s", color="darkorange")
    eixo_2.set_ylabel("Tempo de ajuste (s)", color="darkorange")
    eixo_2.tick_params(axis="y", labelcolor="darkorange")
    plt.title("Compromisso entre acuracia e tempo")
    figura.tight_layout()
    figura.savefig(figura_path, dpi=160)
    plt.close(figura)
    return csv_path, figura_path


if __name__ == "__main__":
    avaliacao = avaliar_ordens()
    arquivos = salvar_resultados(avaliacao)
    for item in avaliacao:
        print(item)
    print(f"Ordem selecionada: q={selecionar_ordem(avaliacao)}")
    print(f"Resultados salvos em: {arquivos[0]} e {arquivos[1]}")
