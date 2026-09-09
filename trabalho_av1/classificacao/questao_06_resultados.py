"""Questao 6: resumo estatistico e grafico das acuracias."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trabalho_av1.classificacao.questao_05_validacao import executar_validacao, salvar_rodadas
from trabalho_av1.comum import PASTA_RESULTADOS, resumir, salvar_csv


def gerar_resultados_finais():
    resultados = executar_validacao()
    salvar_rodadas(resultados)
    linhas = []
    for nome, valores in resultados.items():
        resumo = resumir(valores)
        linhas.append([nome, *resumo.values()])

    pasta = PASTA_RESULTADOS / "classificacao"
    tabela = pasta / "06_resumo_classificacao.csv"
    salvar_csv(
        tabela,
        ["modelo", "media", "desvio_padrao", "maior_valor", "menor_valor"],
        linhas,
    )

    figura = pasta / "06_distribuicao_acuracia.png"
    plt.figure(figsize=(10, 6))
    plt.boxplot(list(resultados.values()), tick_labels=list(resultados), showfliers=False)
    plt.ylabel("Acuracia no conjunto de teste")
    plt.title("Distribuicao das acuracias em 500 rodadas")
    plt.xticks(rotation=15, ha="right")
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
