import argparse
import sys
from hashlib import md5
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


#  CONSTANTES CONFIGURÁVEIS 
PERGUNTA_INSTITUICAO_FEDERAL_PADRAO = "Informe a sua Instituição Federal de Ensino Superior"
PERGUNTA_UNIDADE_REPRESENTADA_PADRAO = "Informe unidade que você está representando"

MAPA_CONVERSAO_RESPOSTA_PARA_FATOR = {
    "4": 1.00,
    "3": 0.75,
    "2": 0.50,
    "1": 0.25,
    "0": 0.00,
}

FAIXAS_DE_CLASSIFICACAO_DO_INDICE = [
    (0.00, 0.19, "Incipiente"),
    (0.20, 0.39, "Em planejamento"),
    (0.40, 0.59, "Em implantação"),
    (0.60, 0.79, "Implementado, porém otimizável"),
    (0.80, 1.00, "Excelência"),
]

LISTA_DELIMITADORES_CANDIDATOS = [",", ";", "\t", "|"]


#  FUNÇÕES AUXILIARES 
def classificar_nivel_de_maturidade(pontuacao_normalizada: float) -> str:
    """Retorna a classificação textual do nível de maturidade a partir da pontuação [0,1]."""
    if pd.isna(pontuacao_normalizada):
        return np.nan
    valor_float = float(pontuacao_normalizada)
    if valor_float < 0.0 or valor_float > 1.0:
        raise ValueError(
            f"Pontuação fora do intervalo [0,1]: {valor_float}. Verifique os dados de entrada."
        )
    for indice_faixa, (limite_inferior, limite_superior, rotulo_textual) in enumerate(FAIXAS_DE_CLASSIFICACAO_DO_INDICE):
        if indice_faixa < len(FAIXAS_DE_CLASSIFICACAO_DO_INDICE) - 1:
            if limite_inferior <= valor_float < limite_superior:
                return rotulo_textual
        else:
            if limite_inferior <= valor_float <= limite_superior:
                return rotulo_textual
    return np.nan


def interpretar_resposta_para_fator_e_validade(valor_bruto):
    """Converte a resposta bruta em (fator [0..1], valida[bool]) considerando NA/N/A/Não se aplica."""
    if pd.isna(valor_bruto):
        return np.nan, False
    texto_normalizado = str(valor_bruto).strip().upper()
    if texto_normalizado in ("NA", "N/A", "NÃO SE APLICA", "NAO SE APLICA"):
        return np.nan, False
    primeiro_digito_encontrado = None
    for caractere in texto_normalizado:
        if caractere.isdigit():
            primeiro_digito_encontrado = caractere
            break
    if primeiro_digito_encontrado is None:
        return np.nan, False
    fator_convertido = MAPA_CONVERSAO_RESPOSTA_PARA_FATOR.get(primeiro_digito_encontrado, np.nan)
    return fator_convertido, (not np.isnan(fator_convertido))


def normalizar_coluna_area_responsavel(tabela_perguntas: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o nome da coluna 'Área responsável' e cria a coluna caso não exista."""
    mapeamento_colunas_limpo = {nome: nome.strip() for nome in tabela_perguntas.columns}
    tabela_perguntas = tabela_perguntas.rename(columns=mapeamento_colunas_limpo)
    coluna_encontrada = False
    for nome_coluna in list(tabela_perguntas.columns):
        if nome_coluna.strip().lower() in ("área responsável", "area responsável", "área responsavel", "area responsavel"):
            if nome_coluna != "Área responsável":
                tabela_perguntas = tabela_perguntas.rename(columns={nome_coluna: "Área responsável"})
            coluna_encontrada = True
    if not coluna_encontrada and "Área responsável" not in tabela_perguntas.columns:
        tabela_perguntas["Área responsável"] = np.nan
    return tabela_perguntas


def gerar_identificador_estavel_da_resposta(
    linha_da_tabela: pd.Series,
    coluna_instituicao: str,
    coluna_unidade: str
) -> str:
    """Cria um identificador estável para cada resposta (hash dos principais metadados)."""
    lista_chaves_componentes = []
    for nome_campo in ["Carimbo de data/hora", "E-mail de contato", coluna_instituicao, coluna_unidade]:
        if nome_campo in linha_da_tabela and pd.notna(linha_da_tabela[nome_campo]):
            lista_chaves_componentes.append(str(linha_da_tabela[nome_campo]))
    base_para_hash = "|".join(lista_chaves_componentes) if lista_chaves_componentes else str(linha_da_tabela.get("RespostaIndex", ""))
    hash_gerado = md5(base_para_hash.encode("utf-8")).hexdigest()[:12]
    return f"RSP-{hash_gerado}"


def detectar_delimitador_arquivo_texto(caminho_arquivo: Path, quantidade_linhas_amostra: int = 5) -> str:
    """Detecta o delimitador mais frequente entre vírgula, ponto e vírgula, tab e pipe."""
    try:
        contagem_por_delimitador = {delim: 0 for delim in LISTA_DELIMITADORES_CANDIDATOS}
        with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as manipulador:
            for indice_linha, linha_lida in enumerate(manipulador):
                if indice_linha >= quantidade_linhas_amostra:
                    break
                for delimitador in contagem_por_delimitador:
                    contagem_por_delimitador[delimitador] += linha_lida.count(delimitador)
        delimitador_mais_frequente = max(contagem_por_delimitador, key=contagem_por_delimitador.get)
        return delimitador_mais_frequente if contagem_por_delimitador[delimitador_mais_frequente] > 0 else ","
    except Exception:
        return ","


def carregar_tabela_de_arquivo(
    caminho_arquivo: Path,
    *,
    codificacao_arquivo: str = "utf-8",
    delimitador_arquivo: str | None = None,
    nome_ou_indice_planilha: str | int | None = None,
    rotulo_legivel: str = ""
) -> pd.DataFrame:
    """Carrega CSV (.csv/.txt) ou Excel (.xlsx/.xls). Se CSV, detecta delimitador quando não informado."""
    sufixo = caminho_arquivo.suffix.lower()
    if sufixo in {".csv", ".txt"}:
        separador = delimitador_arquivo if delimitador_arquivo else detectar_delimitador_arquivo_texto(caminho_arquivo)
        try:
            return pd.read_csv(caminho_arquivo, encoding=codificacao_arquivo, sep=separador, engine="python")
        except Exception as excecao:
            raise ValueError(f"Falha ao ler {rotulo_legivel or caminho_arquivo.name} como CSV (sep='{separador}'): {excecao}")
    elif sufixo in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(caminho_arquivo, sheet_name=nome_ou_indice_planilha if nome_ou_indice_planilha is not None else 0)
        except Exception as excecao:
            raise ValueError(f"Falha ao ler {rotulo_legivel or caminho_arquivo.name} como Excel: {excecao}")
    else:
        raise ValueError(
            f"Formato não suportado para {rotulo_legivel or caminho_arquivo.name}: '{sufixo}'. "
            f"Use .xlsx/.xls ou .csv/.txt"
        )


#  TXT / FORMATAÇÃO 
def gerar_identificador_slug(texto_livre: str) -> str:
    texto_convertido = "" if texto_livre is None else str(texto_livre)
    texto_convertido = unicodedata.normalize("NFKD", texto_convertido)
    texto_convertido = texto_convertido.encode("ascii", "ignore").decode("ascii")
    texto_convertido = re.sub(r"[^A-Za-z0-9]+", "_", texto_convertido).strip("_")
    return texto_convertido[:120] or "NA"


def calcular_quantil_da_serie(serie_pandas: pd.Series, probabilidade: float) -> float:
    try:
        return float(serie_pandas.quantile(probabilidade))
    except Exception:
        return np.nan


def formatar_numero(valor, casas_decimais: int = 2):
    if pd.isna(valor):
        return "—"
    try:
        return f"{float(valor):.{casas_decimais}f}"
    except Exception:
        return str(valor)


def escrever_arquivo_texto(caminho_arquivo: Path, lista_de_linhas: list[str]):
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_arquivo, "w", encoding="utf-8") as manipulador:
        manipulador.write("\n".join(lista_de_linhas))


def calcular_larguras_de_colunas(nomes_colunas: list[str], linhas_tabela: list[list[str]]) -> list[int]:
    larguras_calculadas = [len(str(nome)) for nome in nomes_colunas]
    for linha in linhas_tabela:
        for indice, celula in enumerate(linha):
            if indice < len(larguras_calculadas):
                larguras_calculadas[indice] = max(larguras_calculadas[indice], len(str(celula)))
            else:
                larguras_calculadas.append(len(str(celula)))
    return larguras_calculadas


def montar_tabela_ascii(
    cabecalhos_colunas: list[str],
    linhas_da_tabela: list[list[str]],
    alinhamento_por_coluna: list[str] | None = None
) -> list[str]:
    """
    Gera uma tabela ASCII alinhada  
    """
    if alinhamento_por_coluna is None:
        alinhamento_por_coluna = ["left"] + ["right"] * (len(cabecalhos_colunas) - 1)

    larguras = calcular_larguras_de_colunas(cabecalhos_colunas, linhas_da_tabela)

    def formatar_celula(texto, largura, alinhamento):
        texto_str = str(texto)
        if alinhamento == "right":
            return texto_str.rjust(largura)
        if alinhamento == "center":
            return texto_str.center(largura)
        return texto_str.ljust(largura)

    borda_superior = "+" + "+".join("-" * (largura + 2) for largura in larguras) + "+"
    separador_cabecalho = "+" + "+".join("=" * (largura + 2) for largura in larguras) + "+"
    borda_inferior = "+" + "+".join("-" * (largura + 2) for largura in larguras) + "+"

    linhas_formatadas = [borda_superior]
    linhas_formatadas.append("| " + " | ".join(formatar_celula(h, w, "center") for h, w in zip(cabecalhos_colunas, larguras)) + " |")
    linhas_formatadas.append(separador_cabecalho)
    for linha in linhas_da_tabela:
        if len(linha) < len(larguras):
            linha = list(linha) + [""] * (len(larguras) - len(linha))
        linhas_formatadas.append(
            "| " + " | ".join(formatar_celula(c, w, a) for c, w, a in zip(linha, larguras, alinhamento_por_coluna)) + " |"
        )
    linhas_formatadas.append(borda_inferior)
    return linhas_formatadas


#  RELATÓRIOS TXT 
def gerar_relatorios_em_texto(
    caminho_arquivo_saida_excel: Path,
    tabela_dados_em_formato_longo: pd.DataFrame,
    tabela_consolidada_dimensao_por_unidade: pd.DataFrame,
    tabela_consolidada_dimensao_por_instituicao: pd.DataFrame,
    tabela_consolidada_indice_por_instituicao: pd.DataFrame,
):
    """
    Gera relatórios TXT  
    - Desvio-padrão amostral (ddof=1) 
    """

    # Pastas de saída
    pasta_base_relatorios = Path(caminho_arquivo_saida_excel).parent / (Path(caminho_arquivo_saida_excel).stem + "_txt")
    pasta_relatorios_por_instituicao = pasta_base_relatorios / "IFES"
    pasta_relatorios_por_unidade = pasta_base_relatorios / "unidades"

    # Apenas respostas válidas para cálculos de fatores de pergunta
    tabela_somente_respostas_validas = tabela_dados_em_formato_longo.loc[
        tabela_dados_em_formato_longo["Valida"] == True
    ].copy()

    # IM por unidade (média das dimensões da unidade)
    tabela_indice_por_unidade = (
        tabela_consolidada_dimensao_por_unidade
        .groupby(["IFES", "Unidade", "ifes_unidade"], dropna=False)["Pontuação IM da Dimensão na ifes_unidade"]
        .mean()
        .reset_index(name="IM da ifes_unidade")
    )

    # --- Relatório IM – IFES (um arquivo por IFES) --------------------
    for nome_instituicao, tabela_dimensao_unidade_da_instituicao in tabela_consolidada_dimensao_por_unidade.groupby("IFES", dropna=False):
        rotulo_instituicao = str(nome_instituicao) if pd.notna(nome_instituicao) else "(Sem IFES)"
        slug_instituicao = gerar_identificador_slug(rotulo_instituicao)

        # 1) IM da IFES (média das dimensões)
        linha_instituicao = tabela_consolidada_indice_por_instituicao.loc[
            tabela_consolidada_indice_por_instituicao["IFES"] == nome_instituicao
        ]
        if len(linha_instituicao):
            indice_medio_da_instituicao = float(linha_instituicao["Pontuação IM na IFES"].iloc[0])
        else:
            indice_medio_da_instituicao = float(
                tabela_consolidada_dimensao_por_instituicao.loc[
                    tabela_consolidada_dimensao_por_instituicao["IFES"] == nome_instituicao,
                    "Pontuação IM da Dimensão na IFES"
                ].mean()
            )

        # 2) Tabela de análise por IFES (por unidade)
        serie_indice_unidades = tabela_indice_por_unidade.loc[
            tabela_indice_por_unidade["IFES"] == nome_instituicao, "IM da ifes_unidade"
        ].dropna()

        if len(serie_indice_unidades):
            valor_minimo = serie_indice_unidades.min()
            quartil_um = calcular_quantil_da_serie(serie_indice_unidades, 0.25)
            mediana_valores = serie_indice_unidades.median()
            quartil_tres = calcular_quantil_da_serie(serie_indice_unidades, 0.75)
            valor_maximo = serie_indice_unidades.max()
            desvio_padrao_amostral = serie_indice_unidades.std(ddof=1)
            media_valores = serie_indice_unidades.mean()
            amplitude_interquartil = quartil_tres - quartil_um if pd.notna(quartil_tres) and pd.notna(quartil_um) else np.nan
        else:
            valor_minimo = quartil_um = mediana_valores = quartil_tres = valor_maximo = desvio_padrao_amostral = media_valores = amplitude_interquartil = np.nan

        # 3) Tabela de análise de dimensão por IFES
        tabela_dimensao_indice_da_instituicao = (
            tabela_consolidada_dimensao_por_instituicao.loc[
                tabela_consolidada_dimensao_por_instituicao["IFES"] == nome_instituicao,
                ["Dimensão", "Pontuação IM da Dimensão na IFES"]
            ]
            .copy()
        )
        tabela_dimensao_indice_da_instituicao["Classificação"] = tabela_dimensao_indice_da_instituicao[
            "Pontuação IM da Dimensão na IFES"
        ].apply(classificar_nivel_de_maturidade)

        linhas_relatorio = []
        linhas_relatorio.append(f"Relatório IM – IFES: {rotulo_instituicao}")
        linhas_relatorio.append(f"1) IM da IFES (média das dimensões): {formatar_numero(indice_medio_da_instituicao)}")
        linhas_relatorio.append("Obs.: Desvio-padrão amostral (ddof=1).")
        linhas_relatorio.append("")

        # 2) tabela por unidade (estatística)
        linhas_relatorio.append("2) Tabela de análise por IFES (por unidade)")
        cabecalho_tabela_2 = [
            "(IM por unidade)",
            "Valor mínimo de IM",
            "Q1",
            "Mediana",
            "Q3",
            "Valor Máximo IM",
            "Desvio-padrão",
            "Média",
            "IQR",
        ]
        linhas_da_tabela_2 = [[
            "IM (por unidade)",
            formatar_numero(valor_minimo),
            formatar_numero(quartil_um),
            formatar_numero(mediana_valores),
            formatar_numero(quartil_tres),
            formatar_numero(valor_maximo),
            formatar_numero(desvio_padrao_amostral),
            formatar_numero(media_valores),
            formatar_numero(amplitude_interquartil),
        ]]
        for linha_formatada in montar_tabela_ascii(cabecalho_tabela_2, linhas_da_tabela_2, alinhamento_por_coluna=["left"] + ["right"] * (len(cabecalho_tabela_2) - 1)):
            linhas_relatorio.append(linha_formatada)
        linhas_relatorio.append("")

        # 3) tabela por dimensão
                # 3) Tabela de análise de dimensão por IFES
        linhas_relatorio.append("3) Tabela de análise de dimensão por IFES")
        cabecalho_tabela_3 = [
            "Dimensão",
            "Valor mínimo de IM",
            "Valor Máximo IM",
            "Média (IM)",
            "Diferença do IM em relação ao IM Institucional",
            "Classificação",
        ]
        linhas_da_tabela_3 = []
        for nome_dimensao, agrupamento_unidades in tabela_dimensao_unidade_da_instituicao.groupby("Dimensão", dropna=False):
            rotulo_dimensao = str(nome_dimensao) if pd.notna(nome_dimensao) else "(Sem dimensão)"
            serie_pontos_dimensao = agrupamento_unidades["Pontuação IM da Dimensão na ifes_unidade"].dropna()
            if len(serie_pontos_dimensao):
                dim_min = serie_pontos_dimensao.min()
                dim_max = serie_pontos_dimensao.max()
                dim_media = serie_pontos_dimensao.mean()
            else:
                dim_min = dim_max = dim_media = np.nan

            # Recuperar IM da dimensão na IFES e sua classificação para calcular a diferença
            linha_dim_ifes = tabela_dimensao_indice_da_instituicao.loc[
                tabela_dimensao_indice_da_instituicao["Dimensão"] == nome_dimensao,
                ["Pontuação IM da Dimensão na IFES", "Classificação"]
            ]
            if len(linha_dim_ifes):
                im_dim_ifes_valor = float(linha_dim_ifes["Pontuação IM da Dimensão na IFES"].iloc[0])
                classificacao_texto = linha_dim_ifes["Classificação"].iloc[0]
            else:
                im_dim_ifes_valor = np.nan
                classificacao_texto = "—"

            diferenca_em_relacao_ao_im_institucional = (
                im_dim_ifes_valor - indice_medio_da_instituicao
                if pd.notna(im_dim_ifes_valor) and pd.notna(indice_medio_da_instituicao) else np.nan
            )

            linhas_da_tabela_3.append([
                rotulo_dimensao,
                formatar_numero(dim_min),
                formatar_numero(dim_max),
                formatar_numero(dim_media),
                formatar_numero(diferenca_em_relacao_ao_im_institucional),
                classificacao_texto,
            ])

        for linha_formatada in montar_tabela_ascii(
            cabecalho_tabela_3,
            linhas_da_tabela_3,
            alinhamento_por_coluna=["left", "right", "right", "right", "right", "left"]
        ):
            linhas_relatorio.append(linha_formatada)
        linhas_relatorio.append("")


        # 4) Dimensões com IM menor que o IM da IFES
        linhas_relatorio.append("4) Dimensões com IM menor que o IM da IFES")
        subconjunto_dimensoes_abaixo = tabela_dimensao_indice_da_instituicao.loc[
            tabela_dimensao_indice_da_instituicao["Pontuação IM da Dimensão na IFES"] < indice_medio_da_instituicao
        ]
        if subconjunto_dimensoes_abaixo.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_dimensoes_abaixo.sort_values("Pontuação IM da Dimensão na IFES").iterrows():
                linhas_relatorio.append(f" - {linha['Dimensão']}: {formatar_numero(linha['Pontuação IM da Dimensão na IFES'])}")
        linhas_relatorio.append("")

        # 5) Dimensões com IM maior que o IM da IFES
        linhas_relatorio.append("5) Dimensões com IM maior que o IM da IFES")
        subconjunto_dimensoes_acima = tabela_dimensao_indice_da_instituicao.loc[
            tabela_dimensao_indice_da_instituicao["Pontuação IM da Dimensão na IFES"] > indice_medio_da_instituicao
        ]
        if subconjunto_dimensoes_acima.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_dimensoes_acima.sort_values("Pontuação IM da Dimensão na IFES", ascending=False).iterrows():
                linhas_relatorio.append(f" - {linha['Dimensão']}: {formatar_numero(linha['Pontuação IM da Dimensão na IFES'])}")
        linhas_relatorio.append("")

        # 6) Tabela — IM por unidade
        linhas_relatorio.append("6) Tabela — IM por unidade")
        cabecalho_tabela_6 = ["Unidade", "IM Unidade"]
        tabela_unidades_da_instituicao = (
            tabela_indice_por_unidade.loc[tabela_indice_por_unidade["IFES"] == nome_instituicao, ["Unidade", "IM da ifes_unidade"]]
            .sort_values("Unidade", na_position="last")
        )
        linhas_da_tabela_6 = [
            [str(nome_unidade) if pd.notna(nome_unidade) else "(Sem unidade)", formatar_numero(valor_indice)]
            for nome_unidade, valor_indice in tabela_unidades_da_instituicao.to_numpy()
        ]
        for linha_formatada in montar_tabela_ascii(cabecalho_tabela_6, linhas_da_tabela_6, alinhamento_por_coluna=["left", "right"]):
            linhas_relatorio.append(linha_formatada)
        linhas_relatorio.append("")

        # 7) Unidades com IM abaixo do IM da IFES
        linhas_relatorio.append("7) Unidades com IM abaixo do IM da IFES")
        subconjunto_unidades_abaixo = tabela_unidades_da_instituicao.loc[
            tabela_unidades_da_instituicao["IM da ifes_unidade"] < indice_medio_da_instituicao
        ]
        if subconjunto_unidades_abaixo.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_unidades_abaixo.sort_values("IM da ifes_unidade").iterrows():
                linhas_relatorio.append(f" - {linha['Unidade']}: {formatar_numero(linha['IM da ifes_unidade'])}")
        linhas_relatorio.append("")

        # 8 e 9) Perguntas vs IM da respectiva dimensão na IFES
        base_dimensao_im_instituicao = tabela_consolidada_dimensao_por_instituicao.loc[
            tabela_consolidada_dimensao_por_instituicao["IFES"] == nome_instituicao,
            ["Dimensão", "Pontuação IM da Dimensão na IFES"]
        ].rename(columns={"Pontuação IM da Dimensão na IFES": "IM_dimensao_IFES"})

        tabela_media_pergunta_na_instituicao = (
            tabela_somente_respostas_validas.loc[tabela_somente_respostas_validas["IFES"] == nome_instituicao]
            .groupby(["Dimensão", "Pergunta"], dropna=False)["Fator"]
            .mean()
            .reset_index(name="media_por_pergunta")
            .merge(base_dimensao_im_instituicao, on="Dimensão", how="left")
        )

        # 8) abaixo
        linhas_relatorio.append("8) Perguntas cuja média do fator ficou abaixo do IM da respectiva dimensão na IFES")
        subconjunto_perguntas_abaixo = tabela_media_pergunta_na_instituicao.loc[
            tabela_media_pergunta_na_instituicao["media_por_pergunta"] < tabela_media_pergunta_na_instituicao["IM_dimensao_IFES"]
        ]
        if subconjunto_perguntas_abaixo.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for nome_dim, grupo in subconjunto_perguntas_abaixo.groupby("Dimensão", dropna=False):
                linhas_relatorio.append(f" - {nome_dim}:")
                for _, linha in grupo.sort_values("media_por_pergunta").iterrows():
                    linhas_relatorio.append(
                        f"    • {linha['Pergunta']} = {formatar_numero(linha['media_por_pergunta'])} "
                        f"(IM dim IFES {formatar_numero(linha['IM_dimensao_IFES'])})"
                    )
        linhas_relatorio.append("")

        # 9) acima
        linhas_relatorio.append("9) Perguntas cuja média do fator ficou acima do IM da respectiva dimensão na IFES")
        subconjunto_perguntas_acima = tabela_media_pergunta_na_instituicao.loc[
            tabela_media_pergunta_na_instituicao["media_por_pergunta"] > tabela_media_pergunta_na_instituicao["IM_dimensao_IFES"]
        ]
        if subconjunto_perguntas_acima.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for nome_dim, grupo in subconjunto_perguntas_acima.groupby("Dimensão", dropna=False):
                linhas_relatorio.append(f" - {nome_dim}:")
                for _, linha in grupo.sort_values("media_por_pergunta", ascending=False).iterrows():
                    linhas_relatorio.append(
                        f"    • {linha['Pergunta']} = {formatar_numero(linha['media_por_pergunta'])} "
                        f"(IM dim IFES {formatar_numero(linha['IM_dimensao_IFES'])})"
                    )
        linhas_relatorio.append("")

        escrever_arquivo_texto(pasta_relatorios_por_instituicao / f"IFES_{slug_instituicao}.txt", linhas_relatorio)

    # Relatório por IFES_UNIDADE (um arquivo por unidade)
    for _, linha_unidade in tabela_indice_por_unidade.iterrows():
        nome_instituicao = linha_unidade["IFES"]
        nome_unidade = linha_unidade["Unidade"]
        rotulo_instituicao_unidade = linha_unidade["ifes_unidade"]

        rotulo_instituicao = str(nome_instituicao) if pd.notna(nome_instituicao) else "(Sem IFES)"
        rotulo_unidade = str(nome_unidade) if pd.notna(nome_unidade) else "(Sem Unidade)"
        slug_instituicao = gerar_identificador_slug(rotulo_instituicao)
        slug_unidade = gerar_identificador_slug(rotulo_unidade)

        indice_medio_da_unidade = float(linha_unidade["IM da ifes_unidade"]) if pd.notna(linha_unidade["IM da ifes_unidade"]) else np.nan

        linha_instituicao = tabela_consolidada_indice_por_instituicao.loc[
            tabela_consolidada_indice_por_instituicao["IFES"] == nome_instituicao
        ]
        indice_medio_da_instituicao = float(linha_instituicao["Pontuação IM na IFES"].iloc[0]) if len(linha_instituicao) else np.nan

        linhas_relatorio = []
        linhas_relatorio.append(f"Relatório IM – Unidade: {rotulo_instituicao_unidade}")
        linhas_relatorio.append(f"1) IM da unidade (média das dimensões): {formatar_numero(indice_medio_da_unidade)}")
        linhas_relatorio.append("Obs.: Desvio-padrão amostral (ddof=1).")
        linhas_relatorio.append("")

        # 2) Tabela de análise de dimensão por IFES_UNIDADE
        tabela_dimensao_na_unidade = (
            tabela_consolidada_dimensao_por_unidade
            .loc[
                (tabela_consolidada_dimensao_por_unidade["IFES"] == nome_instituicao) &
                (tabela_consolidada_dimensao_por_unidade["Unidade"] == nome_unidade),
                ["Dimensão", "Pontuação IM da Dimensão na ifes_unidade", "Classificação IM da dimensão na ifes_unidade"]
            ]
            .rename(columns={
                "Pontuação IM da Dimensão na ifes_unidade": "IM_dimensao_unidade",
                "Classificação IM da dimensão na ifes_unidade": "Classificação"
            })
            .copy()
        )
        tabela_dimensao_na_instituicao = (
            tabela_consolidada_dimensao_por_instituicao.loc[
                tabela_consolidada_dimensao_por_instituicao["IFES"] == nome_instituicao,
                ["Dimensão", "Pontuação IM da Dimensão na IFES"]
            ]
            .rename(columns={"Pontuação IM da Dimensão na IFES": "IM_dimensao_IFES"})
        )
        tabela_dimensao_comparativa = tabela_dimensao_na_unidade.merge(
            tabela_dimensao_na_instituicao, on="Dimensão", how="left"
        )
        tabela_dimensao_comparativa["Delta_para_IM_da_IFES"] = (
            tabela_dimensao_comparativa["IM_dimensao_unidade"] - tabela_dimensao_comparativa["IM_dimensao_IFES"]
        )

        linhas_relatorio.append("2) Tabela de análise de dimensão por IFES_UNIDADE")
        cabecalho_tabela_unidade = ["Dimensão", "IM da unidade", "IM IFES", "Diferença do IM em relação ao IM Institucional", "Classificação"]
        linhas_da_tabela_unidade = [
            [
                str(linha["Dimensão"]) if pd.notna(linha["Dimensão"]) else "(Sem dimensão)",
                formatar_numero(linha["IM_dimensao_unidade"]),
                formatar_numero(linha["IM_dimensao_IFES"]),
                formatar_numero(linha["Delta_para_IM_da_IFES"]),
                linha.get("Classificação", "—"),
            ]
            for _, linha in tabela_dimensao_comparativa.sort_values("Dimensão", na_position="last").iterrows()
        ]
        for linha_formatada in montar_tabela_ascii(
            cabecalho_tabela_unidade, linhas_da_tabela_unidade, alinhamento_por_coluna=["left", "right", "right", "right", "left"]
        ):
            linhas_relatorio.append(linha_formatada)
        linhas_relatorio.append("")

        # 3) Dimensões com IM menor que o IM da IFES_UNIDADE
        linhas_relatorio.append("3) Dimensões com IM menor que o IM da IFES_UNIDADE")
        subconjunto_dimensoes_menor_que_unidade = tabela_dimensao_comparativa.loc[
            tabela_dimensao_comparativa["IM_dimensao_unidade"] < indice_medio_da_unidade
        ]
        if subconjunto_dimensoes_menor_que_unidade.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_dimensoes_menor_que_unidade.sort_values("IM_dimensao_unidade").iterrows():
                linhas_relatorio.append(
                    f" - {linha['Dimensão']}: {formatar_numero(linha['IM_dimensao_unidade'])} "
                    f"(IM unidade {formatar_numero(indice_medio_da_unidade)})"
                )
        linhas_relatorio.append("")

        # 4) Dimensões com IM menor que o IM da IFES
        linhas_relatorio.append("4) Dimensões com IM menor que o IM da IFES")
        subconjunto_dimensoes_menor_que_instituicao = tabela_dimensao_comparativa.loc[
            tabela_dimensao_comparativa["IM_dimensao_unidade"] < indice_medio_da_instituicao
        ]
        if subconjunto_dimensoes_menor_que_instituicao.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_dimensoes_menor_que_instituicao.sort_values("IM_dimensao_unidade").iterrows():
                linhas_relatorio.append(
                    f" - {linha['Dimensão']}: {formatar_numero(linha['IM_dimensao_unidade'])} "
                    f"(IM IFES {formatar_numero(indice_medio_da_instituicao)})"
                )
        linhas_relatorio.append("")

        # 5) Dimensões com IM maior que o IM da IFES_UNIDADE
        linhas_relatorio.append("5) Dimensões com IM maior que o IM da IFES_UNIDADE")
        subconjunto_dimensoes_maior_que_unidade = tabela_dimensao_comparativa.loc[
            tabela_dimensao_comparativa["IM_dimensao_unidade"] > indice_medio_da_unidade
        ]
        if subconjunto_dimensoes_maior_que_unidade.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_dimensoes_maior_que_unidade.sort_values("IM_dimensao_unidade", ascending=False).iterrows():
                linhas_relatorio.append(
                    f" - {linha['Dimensão']}: {formatar_numero(linha['IM_dimensao_unidade'])} "
                    f"(IM unidade {formatar_numero(indice_medio_da_unidade)})"
                )
        linhas_relatorio.append("")

        # 6) Dimensões com IM maior que o IM da IFES
        linhas_relatorio.append("6) Dimensões com IM maior que o IM da IFES")
        subconjunto_dimensoes_maior_que_instituicao = tabela_dimensao_comparativa.loc[
            tabela_dimensao_comparativa["IM_dimensao_unidade"] > indice_medio_da_instituicao
        ]
        if subconjunto_dimensoes_maior_que_instituicao.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for _, linha in subconjunto_dimensoes_maior_que_instituicao.sort_values("IM_dimensao_unidade", ascending=False).iterrows():
                linhas_relatorio.append(
                    f" - {linha['Dimensão']}: {formatar_numero(linha['IM_dimensao_unidade'])} "
                    f"(IM IFES {formatar_numero(indice_medio_da_instituicao)})"
                )
        linhas_relatorio.append("")

        # 7 e 8) Perguntas na unidade vs IM da respectiva dimensão na UNIDADE
        base_dimensao_indice_unidade = tabela_dimensao_comparativa[["Dimensão", "IM_dimensao_unidade"]].rename(
            columns={"IM_dimensao_unidade": "IM_dimensao_unidade_valor"}
        )
        tabela_media_pergunta_na_unidade = (
            tabela_somente_respostas_validas.loc[
                (tabela_somente_respostas_validas["IFES"] == nome_instituicao) &
                (tabela_somente_respostas_validas["Unidade"] == nome_unidade)
            ]
            .groupby(["Dimensão", "Pergunta"], dropna=False)["Fator"]
            .mean()
            .reset_index(name="media_da_pergunta_na_unidade")
            .merge(base_dimensao_indice_unidade, on="Dimensão", how="left")
        )

        # 7) abaixo
        linhas_relatorio.append("7) Perguntas cuja média do fator ficou abaixo do IM da respectiva dimensão na IFES_UNIDADE")
        subconjunto_perguntas_abaixo_unidade = tabela_media_pergunta_na_unidade.loc[
            tabela_media_pergunta_na_unidade["media_da_pergunta_na_unidade"] < tabela_media_pergunta_na_unidade["IM_dimensao_unidade_valor"]
        ]
        if subconjunto_perguntas_abaixo_unidade.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for nome_dim, grupo in subconjunto_perguntas_abaixo_unidade.groupby("Dimensão", dropna=False):
                linhas_relatorio.append(f" - {nome_dim}:")
                for _, linha in grupo.sort_values("media_da_pergunta_na_unidade").iterrows():
                    linhas_relatorio.append(
                        f"    • {linha['Pergunta']} = {formatar_numero(linha['media_da_pergunta_na_unidade'])} "
                        f"(IM dim unid {formatar_numero(linha['IM_dimensao_unidade_valor'])})"
                    )
        linhas_relatorio.append("")

        # 8) acima
        linhas_relatorio.append("8) Perguntas cuja média do fator ficou acima do IM da respectiva dimensão na IFES_UNIDADE")
        subconjunto_perguntas_acima_unidade = tabela_media_pergunta_na_unidade.loc[
            tabela_media_pergunta_na_unidade["media_da_pergunta_na_unidade"] > tabela_media_pergunta_na_unidade["IM_dimensao_unidade_valor"]
        ]
        if subconjunto_perguntas_acima_unidade.empty:
            linhas_relatorio.append(" - (nenhuma)")
        else:
            for nome_dim, grupo in subconjunto_perguntas_acima_unidade.groupby("Dimensão", dropna=False):
                linhas_relatorio.append(f" - {nome_dim}:")
                for _, linha in grupo.sort_values("media_da_pergunta_na_unidade", ascending=False).iterrows():
                    linhas_relatorio.append(
                        f"    • {linha['Pergunta']} = {formatar_numero(linha['media_da_pergunta_na_unidade'])} "
                        f"(IM dim unid {formatar_numero(linha['IM_dimensao_unidade_valor'])})"
                    )
        linhas_relatorio.append("")

        escrever_arquivo_texto(
            pasta_relatorios_por_unidade / f"IFES_{slug_instituicao}__UNIDADE_{slug_unidade}.txt",
            linhas_relatorio
        )

    print(f"[OK] Relatórios TXT gerados em: {pasta_base_relatorios}")


#  PIPELINE PRINCIPAL 
def executar_pipeline_principal(
    caminho_arquivo_questionario: Path,
    caminho_arquivo_respostas: Path,
    caminho_arquivo_saida_excel: Path,
    texto_pergunta_instituicao: str = PERGUNTA_INSTITUICAO_FEDERAL_PADRAO,
    texto_pergunta_unidade: str = PERGUNTA_UNIDADE_REPRESENTADA_PADRAO,
    codificacao_questionario: str = "utf-8",
    codificacao_respostas: str = "utf-8",
    delimitador_questionario: str | None = None,
    delimitador_respostas: str | None = None,
    planilha_questionario: str | int | None = None,
    planilha_respostas: str | int | None = None
) -> None:

    # Leitura
    tabela_perguntas = carregar_tabela_de_arquivo(
        caminho_arquivo_questionario,
        codificacao_arquivo=codificacao_questionario,
        delimitador_arquivo=delimitador_questionario,
        nome_ou_indice_planilha=planilha_questionario,
        rotulo_legivel="questionário",
    )
    tabela_perguntas = normalizar_coluna_area_responsavel(tabela_perguntas)

    tabela_respostas = carregar_tabela_de_arquivo(
        caminho_arquivo_respostas,
        codificacao_arquivo=codificacao_respostas,
        delimitador_arquivo=delimitador_respostas,
        nome_ou_indice_planilha=planilha_respostas,
        rotulo_legivel="respostas",
    )

    # Perguntas do questionário e metadados
    tabela_perguntas["Pergunta"] = tabela_perguntas["Pergunta"].astype(str).str.strip()
    conjunto_perguntas = set(tabela_perguntas["Pergunta"].dropna().tolist())

    lista_colunas_respostas = list(tabela_respostas.columns)
    lista_colunas_metadados = [col for col in lista_colunas_respostas if col not in conjunto_perguntas]

    # Formato longo de respostas
    tabela_respostas = tabela_respostas.reset_index().rename(columns={"index": "RespostaIndex"})
    lista_colunas_com_respostas = [col for col in lista_colunas_respostas if col in conjunto_perguntas]

    tabela_dados_em_formato_longo = tabela_respostas.melt(
        id_vars=["RespostaIndex"] + lista_colunas_metadados,
        value_vars=lista_colunas_com_respostas,
        var_name="Pergunta",
        value_name="Resposta",
    )

    # IFES/Unidade
    tabela_dados_em_formato_longo["IFES"] = (
        tabela_dados_em_formato_longo.get(texto_pergunta_instituicao, np.nan).astype(str).str.strip()
    )
    tabela_dados_em_formato_longo["Unidade"] = (
        tabela_dados_em_formato_longo.get(texto_pergunta_unidade, np.nan).astype(str).str.strip()
    )

    # ifes_unidade
    tabela_dados_em_formato_longo["ifes_unidade"] = (
        tabela_dados_em_formato_longo["IFES"].fillna("").astype(str).str.strip()
        + " - " +
        tabela_dados_em_formato_longo["Unidade"].fillna("").astype(str).str.strip()
    )

    # RespostaID estável
    tabela_mapa_identificadores = tabela_respostas[["RespostaIndex"] + lista_colunas_metadados].drop_duplicates().copy()
    tabela_mapa_identificadores["RespostaID"] = tabela_mapa_identificadores.apply(
        lambda linha: gerar_identificador_estavel_da_resposta(
            linha, texto_pergunta_instituicao, texto_pergunta_unidade
        ),
        axis=1,
    )
    tabela_dados_em_formato_longo = tabela_dados_em_formato_longo.merge(
        tabela_mapa_identificadores[["RespostaIndex", "RespostaID"]],
        on="RespostaIndex",
        how="left",
    )

    # Metadados do questionário
    tabela_metadados_perguntas = tabela_perguntas[["Pergunta", "Área responsável", "Dimensão"]].copy()
    tabela_metadados_perguntas["Pergunta"] = tabela_metadados_perguntas["Pergunta"].astype(str).str.strip()
    tabela_dados_em_formato_longo["Pergunta"] = tabela_dados_em_formato_longo["Pergunta"].astype(str).str.strip()
    tabela_dados_em_formato_longo = tabela_dados_em_formato_longo.merge(
        tabela_metadados_perguntas, on="Pergunta", how="left"
    )

    # Fator e validade
    serie_interpretada = tabela_dados_em_formato_longo["Resposta"].apply(interpretar_resposta_para_fator_e_validade)
    tabela_dados_em_formato_longo["Fator"] = serie_interpretada.apply(lambda tupla: tupla[0])
    tabela_dados_em_formato_longo["Valida"] = serie_interpretada.apply(lambda tupla: tupla[1])

    # Peso por dimensão (1/n_validas) por resposta/unidade
    chaves_para_agrupamento = ["IFES", "Unidade", "ifes_unidade", "Dimensão", "RespostaID"]
    tabela_quantidade_validas = (
        tabela_dados_em_formato_longo.assign(validas=tabela_dados_em_formato_longo["Valida"].astype(int))
        .groupby(chaves_para_agrupamento)["validas"].sum().rename("n_validas").reset_index()
    )
    tabela_dados_em_formato_longo = tabela_dados_em_formato_longo.merge(
        tabela_quantidade_validas, on=chaves_para_agrupamento, how="left"
    )

    tabela_dados_em_formato_longo["Peso"] = np.where(
        (tabela_dados_em_formato_longo["Valida"]) & (tabela_dados_em_formato_longo["n_validas"] > 0),
        1.0 / tabela_dados_em_formato_longo["n_validas"],
        np.nan,
    )
    tabela_dados_em_formato_longo["Pontuação da pergunta"] = (
        tabela_dados_em_formato_longo["Fator"] * tabela_dados_em_formato_longo["Peso"]
    )

    # Consolidações 
    # Dimensão na unidade (média ponderada por RespostaID)
    tabela_dimensao_por_resposta = (
        tabela_dados_em_formato_longo
        .groupby(["IFES", "Unidade", "ifes_unidade", "Dimensão", "RespostaID"], dropna=False)["Pontuação da pergunta"]
        .sum()
        .reset_index(name="Pontuação_dimensao_por_resposta")
    )

    tabela_consolidada_dimensao_por_unidade = (
        tabela_dimensao_por_resposta
        .groupby(["IFES", "Unidade", "ifes_unidade", "Dimensão"], dropna=False)["Pontuação_dimensao_por_resposta"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação_dimensao_por_resposta": "Pontuação IM da Dimensão na ifes_unidade"})
    )
    tabela_consolidada_dimensao_por_unidade["Classificação IM da dimensão na ifes_unidade"] = \
        tabela_consolidada_dimensao_por_unidade["Pontuação IM da Dimensão na ifes_unidade"].apply(
            classificar_nivel_de_maturidade
        )

    # Dimensão na IFES (média das unidades)
    tabela_consolidada_dimensao_por_instituicao = (
        tabela_consolidada_dimensao_por_unidade
        .groupby(["IFES", "Dimensão"], dropna=False)["Pontuação IM da Dimensão na ifes_unidade"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação IM da Dimensão na ifes_unidade": "Pontuação IM da Dimensão na IFES"})
    )
    tabela_consolidada_dimensao_por_instituicao["Classificação IM da dimensão na IFES"] = \
        tabela_consolidada_dimensao_por_instituicao["Pontuação IM da Dimensão na IFES"].apply(
            classificar_nivel_de_maturidade
        )

    # IM por IFES (média do IM das dimensões)
    tabela_consolidada_indice_por_instituicao = (
        tabela_consolidada_dimensao_por_instituicao
        .groupby(["IFES"], dropna=False)["Pontuação IM da Dimensão na IFES"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação IM da Dimensão na IFES": "Pontuação IM na IFES"})
    )
    tabela_consolidada_indice_por_instituicao["Classificação IM na IFES"] = \
        tabela_consolidada_indice_por_instituicao["Pontuação IM na IFES"].apply(
            classificar_nivel_de_maturidade
        )

    # Excel: Dados agregados 
    tabela_dados_finais = tabela_dados_em_formato_longo[[
        "IFES", "Unidade", "ifes_unidade", "RespostaID",
        "Pergunta", "Resposta", "Área responsável", "Dimensão",
        "Peso", "Fator", "Pontuação da pergunta"
    ] + lista_colunas_metadados].copy()

    tabela_dados_agregados = (
        tabela_dados_finais
        .merge(tabela_consolidada_dimensao_por_unidade[[
            "IFES","Unidade","ifes_unidade","Dimensão",
            "Pontuação IM da Dimensão na ifes_unidade",
            "Classificação IM da dimensão na ifes_unidade"
        ]], how="left", on=["IFES","Unidade","ifes_unidade","Dimensão"])
        .merge(tabela_consolidada_dimensao_por_instituicao[[
            "IFES","Dimensão",
            "Pontuação IM da Dimensão na IFES",
            "Classificação IM da dimensão na IFES"
        ]], how="left", on=["IFES","Dimensão"])
        .merge(tabela_consolidada_indice_por_instituicao[[
            "IFES",
            "Pontuação IM na IFES",
            "Classificação IM na IFES"
        ]], how="left", on=["IFES"])
    )

    caminho_arquivo_saida_excel = Path(caminho_arquivo_saida_excel)
    with pd.ExcelWriter(caminho_arquivo_saida_excel, engine="xlsxwriter") as escritor_excel:
        tabela_dados_agregados.to_excel(escritor_excel, index=False, sheet_name="dados agregados")
    print(f"[OK] Arquivo gerado: {caminho_arquivo_saida_excel}")

    # TXT
    gerar_relatorios_em_texto(
        caminho_arquivo_saida_excel,
        tabela_dados_em_formato_longo,
        tabela_consolidada_dimensao_por_unidade,
        tabela_consolidada_dimensao_por_instituicao,
        tabela_consolidada_indice_por_instituicao
    )

    # Informações gerais no stdout
    print(f"[INFO] COLUNA_IFES = {texto_pergunta_instituicao}")
    print(f"[INFO] COLUNA_UNIDADE = {texto_pergunta_unidade}")
    try:
        colunas_metadados_detectadas = [c for c in tabela_respostas.columns if c not in tabela_perguntas["Pergunta"].tolist()]
        print("[META] Campos de metadados detectados (não entram no cálculo):")
        if colunas_metadados_detectadas:
            for nome_coluna in colunas_metadados_detectadas:
                print(f" - {nome_coluna}")
        else:
            print(" (nenhum metadado detectado)")
    except Exception as excecao:
        print("[META][AVISO] Não foi possível imprimir metadados:", excecao)


#  CLI - Interface de Linha de Comando 
if __name__ == "__main__":
    analisador_argumentos = argparse.ArgumentParser(
        description="Gera Excel com IM e relatórios TXT por IFES e Unidade (aceita .xlsx/.xls e .csv/.txt)."
    )
    analisador_argumentos.add_argument("--questionario", required=True, type=Path, help="Caminho para questionário (.xlsx/.xls/.csv)")
    analisador_argumentos.add_argument("--respostas", required=True, type=Path, help="Caminho para respostas (.xlsx/.xls/.csv)")
    analisador_argumentos.add_argument("--saida", required=True, type=Path, help="Caminho para o Excel de saída (ex.: im_resultado.xlsx)")

    # Mantidos por compatibilidade, porém nomes internos em português
    analisador_argumentos.add_argument("--ifes-col", dest="coluna_ifes", default=PERGUNTA_INSTITUICAO_FEDERAL_PADRAO,
                                       help="Texto da pergunta/coluna que identifica a IFES")
    analisador_argumentos.add_argument("--unidade-col", dest="coluna_unidade", default=PERGUNTA_UNIDADE_REPRESENTADA_PADRAO,
                                       help="Texto da pergunta/coluna que identifica a Unidade da IFES")

    # Opções de leitura (CSV/Excel)
    analisador_argumentos.add_argument("--questionario-encoding", dest="codificacao_q", default="utf-8", help="Encoding do questionário (CSV)")
    analisador_argumentos.add_argument("--respostas-encoding", dest="codificacao_r", default="utf-8", help="Encoding das respostas (CSV)")
    analisador_argumentos.add_argument("--questionario-delim", dest="delimitador_q", default=None, help="Delimitador do questionário (CSV). Se omitido, detecta automaticamente (, ; \\t |)")
    analisador_argumentos.add_argument("--respostas-delim", dest="delimitador_r", default=None, help="Delimitador das respostas (CSV). Se omitido, detecta automaticamente (, ; \\t |)")
    analisador_argumentos.add_argument("--questionario-sheet", dest="planilha_q", default=None, help="Nome/índice da planilha do questionário (Excel). Padrão: 0")
    analisador_argumentos.add_argument("--respostas-sheet", dest="planilha_r", default=None, help="Nome/índice da planilha das respostas (Excel). Padrão: 0")

    argumentos = analisador_argumentos.parse_args()

    try:
        executar_pipeline_principal(
            caminho_arquivo_questionario=argumentos.questionario,
            caminho_arquivo_respostas=argumentos.respostas,
            caminho_arquivo_saida_excel=argumentos.saida,
            texto_pergunta_instituicao=argumentos.coluna_ifes,
            texto_pergunta_unidade=argumentos.coluna_unidade,
            codificacao_questionario=argumentos.codificacao_q,
            codificacao_respostas=argumentos.codificacao_r,
            delimitador_questionario=argumentos.delimitador_q,
            delimitador_respostas=argumentos.delimitador_r,
            planilha_questionario=argumentos.planilha_q,
            planilha_respostas=argumentos.planilha_r,
        )
    except Exception as excecao_execucao:
        print("[ERRO]", excecao_execucao, file=sys.stderr)
        sys.exit(1)
