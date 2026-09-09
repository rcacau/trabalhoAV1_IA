"""Questao 7: estatisticas, tabela e graficos finais da regressao."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trabalho_av1.comum import PASTA_RESULTADOS, resumir, salvar_csv
from trabalho_av1.regressao.questao_06_validacao import executar_validacao, salvar_rodadas


def gerar_resultados_finais():
    resultados = executar_validacao()
    salvar_rodadas(resultados)
    linhas = []
    for nome, metricas in resultados.items():
        for metrica in ("mse", "r2"):
            resumo = resumir(metricas[metrica])
            linhas.append([nome, metrica.upper(), *resumo.values()])

    pasta = PASTA_RESULTADOS / "regressao"
    tabela = pasta / "07_resumo_regressao.csv"
    salvar_csv(
        tabela,
        ["modelo", "metrica", "media", "desvio_padrao", "maior_valor", "menor_valor"],
        linhas,
    )

    figura = pasta / "07_distribuicao_r2.png"
    plt.figure(figsize=(11, 6))
    plt.boxplot([v["r2"] for v in resultados.values()], tick_labels=list(resultados), showfliers=False)
    plt.ylabel("R2 no conjunto de teste")
    plt.title("Distribuicao do R2 em 500 rodadas")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figura, dpi=160)
    plt.close()
    return tabela, figura, linhas


if __name__ == "__main__":
    tabela, figura, linhas = gerar_resultados_finais()
    for linha in linhas:
        print(linha)
    print(f"\nTabela: {tabela}\nGrafico: {figura}")
