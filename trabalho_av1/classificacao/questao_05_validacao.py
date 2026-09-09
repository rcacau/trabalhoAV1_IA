"""Questao 5: validacao Monte Carlo com 500 particoes 80/20."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from trabalho_av1.classificacao.questao_01_organizacao import organizar_dados
from trabalho_av1.classificacao.questao_03_modelos import LAMBDA_CLASSIFICACAO
from trabalho_av1.classificacao.questao_04_selecao_polinomial import selecionar_ordem
from trabalho_av1.comum import (
    ModeloMQO,
    PASTA_RESULTADOS,
    RODADAS_VALIDACAO,
    SEMENTE,
    acuracia,
    dividir_indices,
    prever_classes,
    salvar_csv,
)


def definir_modelos(ordem_polinomial: int) -> dict[str, tuple[int, float]]:
    return {
        "MQO tradicional": (1, 0.0),
        f"MQO regularizado lambda={LAMBDA_CLASSIFICACAO:g}": (1, LAMBDA_CLASSIFICACAO),
        f"MQO polinomial q={ordem_polinomial}": (ordem_polinomial, 0.0),
    }


def executar_validacao(
    rodadas: int = RODADAS_VALIDACAO, ordem_polinomial: int | None = None
) -> dict[str, np.ndarray]:
    x, y, rotulos, classes = organizar_dados()
    ordem_polinomial = ordem_polinomial or selecionar_ordem()
    configuracoes = definir_modelos(ordem_polinomial)
    resultados = {nome: np.empty(rodadas) for nome in configuracoes}
    gerador = np.random.default_rng(SEMENTE)

    for rodada in range(rodadas):
        treino, teste = dividir_indices(x.shape[0], gerador)
        for nome, (ordem, lbd) in configuracoes.items():
            modelo = ModeloMQO(ordem=ordem, lbd=lbd).fit(x[treino], y[treino])
            previstos = prever_classes(modelo.predict(x[teste]), classes)
            resultados[nome][rodada] = acuracia(rotulos[teste], previstos)
    return resultados


def salvar_rodadas(resultados: dict[str, np.ndarray]) -> Path:
    nomes = list(resultados)
    quantidade = len(resultados[nomes[0]])
    linhas = [[rodada + 1, *[resultados[nome][rodada] for nome in nomes]] for rodada in range(quantidade)]
    destino = PASTA_RESULTADOS / "classificacao" / "05_validacao_500_rodadas.csv"
    salvar_csv(destino, ["rodada", *nomes], linhas)
    return destino


if __name__ == "__main__":
    metricas = executar_validacao()
    print(f"Foram avaliados {len(metricas)} modelos em {RODADAS_VALIDACAO} rodadas.")
    print(f"Resultados brutos salvos em: {salvar_rodadas(metricas)}")

