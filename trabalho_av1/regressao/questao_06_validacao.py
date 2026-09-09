"""Questao 6: Random Subsampling Validation com 500 rodadas 80/20."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from trabalho_av1.comum import (
    ModeloMQO,
    PASTA_RESULTADOS,
    RODADAS_VALIDACAO,
    SEMENTE,
    carregar_china,
    coeficiente_determinacao,
    dividir_indices,
    erro_quadratico_medio,
    salvar_csv,
)
from trabalho_av1.regressao.questao_04_selecao_polinomial import selecionar_ordem


LAMBDAS_AVALIADOS = [0.25, 0.5, 0.75, 1.0]


def definir_modelos(ordem_polinomial: int) -> dict[str, tuple[int, float]]:
    modelos = {
        f"Polinomial q={ordem_polinomial}": (ordem_polinomial, 0.0),
        "MQO tradicional": (1, 0.0),
    }
    modelos.update({f"MQO regularizado lambda={lbd:g}": (1, lbd) for lbd in LAMBDAS_AVALIADOS})
    return modelos


def executar_validacao(
    rodadas: int = RODADAS_VALIDACAO, ordem_polinomial: int | None = None
) -> dict[str, dict[str, np.ndarray]]:
    x, y = carregar_china()
    ordem_polinomial = ordem_polinomial or selecionar_ordem()
    configuracoes = definir_modelos(ordem_polinomial)
    resultados = {
        nome: {"mse": np.empty(rodadas), "r2": np.empty(rodadas)}
        for nome in configuracoes
    }
    gerador = np.random.default_rng(SEMENTE)

    for rodada in range(rodadas):
        treino, teste = dividir_indices(x.shape[0], gerador)
        for nome, (ordem, lbd) in configuracoes.items():
            modelo = ModeloMQO(ordem=ordem, lbd=lbd).fit(x[treino], y[treino])
            previsao = modelo.predict(x[teste])
            resultados[nome]["mse"][rodada] = erro_quadratico_medio(y[teste], previsao)
            resultados[nome]["r2"][rodada] = coeficiente_determinacao(y[teste], previsao)
    return resultados


def salvar_rodadas(resultados: dict[str, dict[str, np.ndarray]]) -> Path:
    nomes = list(resultados)
    cabecalho = ["rodada"]
    for nome in nomes:
        cabecalho.extend([f"{nome} - MSE", f"{nome} - R2"])
    linhas = []
    quantidade = len(resultados[nomes[0]]["mse"])
    for rodada in range(quantidade):
        linha = [rodada + 1]
        for nome in nomes:
            linha.extend([resultados[nome]["mse"][rodada], resultados[nome]["r2"][rodada]])
        linhas.append(linha)
    destino = PASTA_RESULTADOS / "regressao" / "06_validacao_500_rodadas.csv"
    salvar_csv(destino, cabecalho, linhas)
    return destino


if __name__ == "__main__":
    metricas = executar_validacao()
    print(f"Foram avaliados {len(metricas)} modelos em {RODADAS_VALIDACAO} rodadas.")
    print(f"Resultados brutos salvos em: {salvar_rodadas(metricas)}")

