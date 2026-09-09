"""Funcoes comuns, adaptadas dos exemplos desenvolvidos em sala."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np


PASTA_TRABALHO = Path(__file__).resolve().parent
PASTA_RAIZ = PASTA_TRABALHO.parent
PASTA_DADOS = PASTA_RAIZ / "data set"
PASTA_RESULTADOS = PASTA_TRABALHO / "resultados"

SEMENTE = 42
RODADAS_VALIDACAO = 500
PROPORCAO_TREINO = 0.8


def carregar_china() -> tuple[np.ndarray, np.ndarray]:
    dados = np.loadtxt(PASTA_DADOS / "china_gdp.csv", delimiter=",", skiprows=1)
    x = dados[:, 0].reshape(-1, 1)
    y = dados[:, 1].reshape(-1, 1)
    return x, y


def carregar_emg() -> tuple[np.ndarray, np.ndarray]:
    dados = np.loadtxt(PASTA_DADOS / "EMG1.csv")
    x = dados[:, :2]
    rotulos = dados[:, 2].astype(int)
    return x, rotulos


def codificar_um_contra_todos(
    rotulos: np.ndarray, classes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Repete a codificacao -1/+1 usada em main_classificacao.py."""
    if classes is None:
        classes = np.unique(rotulos)
    y = -np.ones((rotulos.size, classes.size))
    for indice, classe in enumerate(classes):
        y[rotulos == classe, indice] = 1
    return y, classes


def dividir_indices(
    quantidade: int,
    gerador: np.random.Generator,
    proporcao_treino: float = PROPORCAO_TREINO,
) -> tuple[np.ndarray, np.ndarray]:
    indices = gerador.permutation(quantidade)
    limite = int(proporcao_treino * quantidade)
    return indices[:limite], indices[limite:]


def expansao_polinomial(x: np.ndarray, ordem: int) -> np.ndarray:
    """Gera [1, X, X**2, ..., X**q], sem termos cruzados, como na aula."""
    if ordem < 1:
        raise ValueError("A ordem polinomial deve ser pelo menos 1.")
    blocos = [np.ones((x.shape[0], 1))]
    blocos.extend(x**potencia for potencia in range(1, ordem + 1))
    return np.hstack(blocos)


class ModeloMQO:
    """MQO tradicional, regularizado ou polinomial para uma ou varias saidas."""

    def __init__(self, ordem: int = 1, lbd: float = 0.0, padronizar: bool = True):
        self.ordem = ordem
        self.lbd = lbd
        self.padronizar = padronizar
        self.beta: np.ndarray | None = None
        self.media: np.ndarray | None = None
        self.desvio: np.ndarray | None = None

    def _ajustar_escala(self, x: np.ndarray) -> np.ndarray:
        self.media = np.mean(x, axis=0, keepdims=True)
        self.desvio = np.std(x, axis=0, keepdims=True)
        self.desvio[self.desvio == 0] = 1.0
        return (x - self.media) / self.desvio

    def _aplicar_escala(self, x: np.ndarray) -> np.ndarray:
        if not self.padronizar:
            return x
        if self.media is None or self.desvio is None:
            raise RuntimeError("O modelo precisa ser ajustado antes da predicao.")
        return (x - self.media) / self.desvio

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ModeloMQO":
        x_escala = self._ajustar_escala(x) if self.padronizar else x.copy()
        matriz_projeto = expansao_polinomial(x_escala, self.ordem)

        # Equacoes normais apresentadas em aula. A pseudoinversa deixa a
        # solucao definida mesmo se a matriz nao possuir inversa comum.
        gram = matriz_projeto.T @ matriz_projeto
        identidade = np.eye(matriz_projeto.shape[1])
        sistema = gram + self.lbd * identidade
        self.beta = np.linalg.pinv(sistema) @ matriz_projeto.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("O modelo precisa ser ajustado antes da predicao.")
        x_escala = self._aplicar_escala(x)
        return expansao_polinomial(x_escala, self.ordem) @ self.beta


def erro_quadratico_medio(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y - y_pred) ** 2))


def coeficiente_determinacao(y: np.ndarray, y_pred: np.ndarray) -> float:
    soma_residuos = np.sum((y - y_pred) ** 2)
    soma_total = np.sum((y - np.mean(y)) ** 2)
    return float(1 - soma_residuos / soma_total) if soma_total != 0 else 0.0


def acuracia(rotulos: np.ndarray, previstos: np.ndarray) -> float:
    return float(np.mean(rotulos == previstos))


def prever_classes(pontuacoes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(pontuacoes, axis=1)]


def ajustar_cronometrado(
    modelo: ModeloMQO, x: np.ndarray, y: np.ndarray
) -> tuple[ModeloMQO, float]:
    inicio = perf_counter()
    modelo.fit(x, y)
    return modelo, perf_counter() - inicio


def resumir(valores: np.ndarray) -> dict[str, float]:
    return {
        "media": float(np.mean(valores)),
        "desvio_padrao": float(np.std(valores, ddof=1)),
        "maior_valor": float(np.max(valores)),
        "menor_valor": float(np.min(valores)),
    }


def salvar_csv(caminho: Path, cabecalho: list[str], linhas: list[list[object]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)
