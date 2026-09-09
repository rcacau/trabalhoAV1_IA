"""Executa todas as questoes e gera os arquivos da pasta resultados."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trabalho_av1.classificacao.questao_02_visualizacao import gerar_grafico as grafico_emg
from trabalho_av1.classificacao.questao_03_modelos import gerar_regioes_decisao
from trabalho_av1.classificacao.questao_04_selecao_polinomial import (
    avaliar_ordens as avaliar_ordens_classificacao,
    salvar_resultados as salvar_selecao_classificacao,
    selecionar_ordem as selecionar_ordem_classificacao,
)
from trabalho_av1.classificacao.questao_06_resultados import (
    gerar_resultados_finais as resultados_classificacao,
)
from trabalho_av1.regressao.questao_01_visualizacao import gerar_grafico as grafico_china
from trabalho_av1.regressao.questao_03_modelos import gerar_comparacao
from trabalho_av1.regressao.questao_04_selecao_polinomial import (
    avaliar_ordens as avaliar_ordens_regressao,
    salvar_resultados as salvar_selecao_regressao,
    selecionar_ordem as selecionar_ordem_regressao,
)
from trabalho_av1.regressao.questao_05_regularizacao import estimar_betas, salvar_estimativas
from trabalho_av1.regressao.questao_07_resultados import (
    gerar_resultados_finais as resultados_regressao,
)


def main() -> None:
    print("Gerando resultados da regressao...")
    grafico_china()
    gerar_comparacao()
    avaliacao_regressao = avaliar_ordens_regressao()
    salvar_selecao_regressao(avaliacao_regressao)
    print(f"Ordem escolhida na regressao: q={selecionar_ordem_regressao(avaliacao_regressao)}")
    salvar_estimativas(estimar_betas())
    resultados_regressao()

    print("Gerando resultados da classificacao...")
    grafico_emg()
    gerar_regioes_decisao()
    avaliacao_classificacao = avaliar_ordens_classificacao()
    salvar_selecao_classificacao(avaliacao_classificacao)
    print(
        "Ordem escolhida na classificacao: "
        f"q={selecionar_ordem_classificacao(avaliacao_classificacao)}"
    )
    resultados_classificacao()
    print("Concluido. Consulte trabalho_av1/resultados.")


if __name__ == "__main__":
    main()
