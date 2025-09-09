import argparse 
import sys
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd


# CONSTANTES CONFIGURÁVEIS (podem ser sobrescritas por --ifes-col e --unidade-col)
PERGUNTA_IFES_PADRAO = "Informe a sua Instituição Federal de Ensino Superior"
PERGUNTA_UNIDADE_PADRAO = "Informe unidade que você está representando"

MAPA_FATOR = {
    "4": 1.00,
    "3": 0.75,
    "2": 0.50,
    "1": 0.25,
    "0": 0.00,
}

FAIXAS_CLASSIFICACAO = [
    (0.00, 0.19, "Incipiente"),
    (0.20, 0.39, "Em planejamento"),
    (0.40, 0.59, "Em implantação"),
    (0.60, 0.79, "Implementado, porém otimizável"),
    (0.80, 1.00, "Excelência"),
]


def classificar(pontuacao: float) -> str:
    if pd.isna(pontuacao):
        return np.nan
    pontuacao = float(pontuacao)
    if pontuacao < 0.0 or pontuacao > 1.0:
        raise ValueError(f"Pontuação fora do intervalo [0,1]: {pontuacao}. Verifique os dados de entrada.")
    # Arredondar para 2 casas ANTES de classificar
    pontuacao = round(pontuacao, 2)
    for i, (a, b, label) in enumerate(FAIXAS_CLASSIFICACAO):
        if i < len(FAIXAS_CLASSIFICACAO) - 1:
            if a <= pontuacao < b:
                return label
        else:
            if a <= pontuacao <= b:
                return label
    return np.nan

def parse_fator(val):
    """Converte a resposta em "fator" e "valida"
    - 'NA' e variantes -> (nan, False)
    - '4 - ...' etc -> pega o primeiro dígito [0-4] e mapeia via MAPA_FATOR
    - vazio/NaN -> (nan, False)
    """
    if pd.isna(val):
        return np.nan, False
    s = str(val).strip().upper()
    if s in ("NA", "N/A", "NÃO SE APLICA", "NAO SE APLICA"):
        return np.nan, False
    d = None
    for ch in s:
        if ch.isdigit():
            d = ch
            break
    if d is None:
        return np.nan, False
    fator = MAPA_FATOR.get(d, np.nan)
    return fator, (not np.isnan(fator))


def normalizar_coluna_area_responsavel(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nome da coluna 'Área responsável' (remove espaços, acentos já vêm prontos)"""
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Unificar se vier com espaços estranhos ou variações
    for c in list(df.columns):
        if c.strip().lower() in ("área responsável", "area responsável", "área responsavel", "area responsavel"):
            if c != "Área responsável":
                df = df.rename(columns={c: "Área responsável"})
    return df


def gerar_uid(row: pd.Series, ifes_col: str, unidade_col: str) -> str:
    """Cria um Identificador estável por resposta (hash dos principais metadados)."""
    keys = []
    for k in ["Carimbo de data/hora", "E-mail de contato", ifes_col, unidade_col]:
        if k in row and pd.notna(row[k]):
            keys.append(str(row[k]))
    base = "|".join(keys) if keys else str(row.get("RespostaIndex", ""))
    h = md5(base.encode("utf-8")).hexdigest()[:12]
    return f"RSP-{h}"


def main(questionario_path: Path,
         respostas_path: Path,
         saida_path: Path,
         ifes_question: str = PERGUNTA_IFES_PADRAO,
         unidade_question: str = PERGUNTA_UNIDADE_PADRAO) -> None:
    # Leitura
    q_df = pd.read_excel(questionario_path, sheet_name=0)
    q_df = normalizar_coluna_area_responsavel(q_df)
    r_df = pd.read_excel(respostas_path, sheet_name=0)

    # Metadados: colunas das respostas que NÃO são perguntas do questionário
    q_df["Pergunta"] = q_df["Pergunta"].astype(str).str.strip()
    perguntas_questionario = set(q_df["Pergunta"].dropna().tolist())
    respostas_cols = list(r_df.columns)
    meta_cols = [c for c in respostas_cols if c not in perguntas_questionario]

    # Transformar respostas para formato longo
    r_df = r_df.reset_index().rename(columns={"index": "RespostaIndex"})
    value_vars = [c for c in respostas_cols if c in perguntas_questionario]

    dados_long = r_df.melt(
        id_vars=["RespostaIndex"] + meta_cols,
        value_vars=value_vars,
        var_name="Pergunta",
        value_name="Resposta"
    )

    # Extrair IFES/Unidade
    dados_long["IFES"] = dados_long.get(ifes_question, np.nan).astype(str).str.strip()
    dados_long["Unidade"] = dados_long.get(unidade_question, np.nan).astype(str).str.strip()

    # RespostaUID
    uid_map = r_df[["RespostaIndex"] + meta_cols].drop_duplicates().copy()
    uid_map["RespostaUID"] = uid_map.apply(lambda row: gerar_uid(row, ifes_question, unidade_question), axis=1)
    dados_long = dados_long.merge(uid_map[["RespostaIndex", "RespostaUID"]], on="RespostaIndex", how="left")

    # Juntar metadados do questionário
    q_meta = q_df[["Pergunta", "Área responsável", "Dimensão"]].copy()
    q_meta["Pergunta"] = q_meta["Pergunta"].astype(str).str.strip()
    dados_long["Pergunta"] = dados_long["Pergunta"].astype(str).str.strip()
    dados_long = dados_long.merge(q_meta, on="Pergunta", how="left")

    # Calcular Fator e Valida
    parsed = dados_long["Resposta"].apply(parse_fator)
    dados_long["Fator"] = parsed.apply(lambda x: x[0])
    dados_long["Valida"] = parsed.apply(lambda x: x[1])

    # Peso por dimensão (1/n_válidas) por resposta/unidade
    grp_keys = ["IFES", "Unidade", "Dimensão", "RespostaUID"]
    n_validas = (dados_long.assign(validas=dados_long["Valida"].astype(int))
                 .groupby(grp_keys)["validas"].sum().rename("n_validas").reset_index())
    dados_long = dados_long.merge(n_validas, on=grp_keys, how="left")

    dados_long["Peso"] = np.where(
        (dados_long["Valida"]) & (dados_long["n_validas"] > 0),
        1.0 / dados_long["n_validas"],
        np.nan
    )
    dados_long["Pontuação da pergunta"] = dados_long["Fator"] * dados_long["Peso"]

    # ifes_unidade
    dados_long["ifes_unidade"] = dados_long["IFES"].fillna("").astype(str).str.strip() + " - " + \
                                 dados_long["Unidade"].fillna("").astype(str).str.strip()

    # Consolidações
    # Dimensão na unidade
    dim_por_resposta = (dados_long
        .groupby(["IFES", "Unidade", "ifes_unidade", "Dimensão", "RespostaUID"], dropna=False)["Pontuação da pergunta"]
        .sum()
        .reset_index(name="Pontuação_dim_por_resposta")
    )
    consolidacao_dim_unidade = (dim_por_resposta
        .groupby(["IFES", "Unidade", "ifes_unidade", "Dimensão"], dropna=False)["Pontuação_dim_por_resposta"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação_dim_por_resposta": "Pontuação IM da Dimensão na ifes_unidade"})
    )
    consolidacao_dim_unidade["Pontuação IM da Dimensão na ifes_unidade"] = consolidacao_dim_unidade["Pontuação IM da Dimensão na ifes_unidade"].round(2)
    consolidacao_dim_unidade["Classificação IM da dimensão na ifes_unidade"] = \
        consolidacao_dim_unidade["Pontuação IM da Dimensão na ifes_unidade"].apply(classificar)

    # Dimensão na IFES
    consolidacao_dim_ifes = (consolidacao_dim_unidade
        .groupby(["IFES", "Dimensão"], dropna=False)["Pontuação IM da Dimensão na ifes_unidade"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação IM da Dimensão na ifes_unidade": "Pontuação IM da Dimensão na IFES"})
    )
    consolidacao_dim_ifes["Pontuação IM da Dimensão na IFES"] = consolidacao_dim_ifes["Pontuação IM da Dimensão na IFES"].round(2)
    consolidacao_dim_ifes["Classificação IM da dimensão na IFES"] = \
        consolidacao_dim_ifes["Pontuação IM da Dimensão na IFES"].apply(classificar)

    # IM por IFES
    consolidacao_ifes = (consolidacao_dim_ifes
        .groupby(["IFES"], dropna=False)["Pontuação IM da Dimensão na IFES"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação IM da Dimensão na IFES": "Pontuação IM na IFES"})
    )
    consolidacao_ifes["Pontuação IM na IFES"] = consolidacao_ifes["Pontuação IM na IFES"].round(2)
    consolidacao_ifes["Classificação IM na IFES"] = \
        consolidacao_ifes["Pontuação IM na IFES"].apply(classificar)

    # Aba 'dados' com metadados + colunas questionário
    dados_cols = meta_cols + [
        "IFES", "Unidade", "ifes_unidade",
        "RespostaUID", "Pergunta", "Resposta",
        "Área responsável", "Dimensão",
        "Peso", "Fator", "Pontuação da pergunta"
    ]
    for c in dados_cols:
        if c not in dados_long.columns:
            dados_long[c] = np.nan
    dados_final = dados_long[dados_cols].copy() 

    
    # ABA 'dados agregados': left join:
    # Base: dados_final
    # Join 1: consolidacao_dimensao_unidade (chaves: IFES, Unidade, ifes_unidade, Dimensão)
    # Join 2: consolidacao_dimensao_ifes (chaves: IFES, Dimensão)
    # Join 3: consolidacao_ifes (chaves: IFES)
    colunas_unidade = [
        "IFES","Unidade","ifes_unidade","Dimensão",
        "Pontuação IM da Dimensão na ifes_unidade",
        "Classificação IM da dimensão na ifes_unidade"
    ]
    colunas_ifes_dim = [
        "IFES","Dimensão",
        "Pontuação IM da Dimensão na IFES",
        "Classificação IM da dimensão na IFES"
    ]
    colunas_ifes = [
        "IFES",
        "Pontuação IM na IFES",
        "Classificação IM na IFES"
    ]
    dados_agregados = (
        dados_final
            .merge(consolidacao_dim_unidade[colunas_unidade], how="left",
                   on=["IFES","Unidade","ifes_unidade","Dimensão"])
            .merge(consolidacao_dim_ifes[colunas_ifes_dim], how="left",
                   on=["IFES","Dimensão"])
            .merge(consolidacao_ifes[colunas_ifes], how="left",
                   on=["IFES"])
    )
    

    # Gravar Excel
    saida_path = Path(saida_path)
    with pd.ExcelWriter(saida_path, engine="xlsxwriter") as writer:
        dados_final.to_excel(writer, index=False, sheet_name="dados")

        consolidacao_dim_unidade[[
            "IFES", "Unidade", "ifes_unidade", "Dimensão",
            "Pontuação IM da Dimensão na ifes_unidade",
            "Classificação IM da dimensão na ifes_unidade"
        ]].to_excel(writer, index=False, sheet_name="consolidacao_dimensao_unidade")

        consolidacao_dim_ifes[[
            "IFES", "Dimensão",
            "Pontuação IM da Dimensão na IFES",
            "Classificação IM da dimensão na IFES"
        ]].to_excel(writer, index=False, sheet_name="consolidacao_dimensao_ifes")

        consolidacao_ifes[[
            "IFES", "Pontuação IM na IFES", "Classificação IM na IFES"
        ]].to_excel(writer, index=False, sheet_name="consolidacao_ifes")

    
        dados_agregados.to_excel(writer, index=False, sheet_name="dados agregados")
    print(f"[OK] Arquivo gerado: {saida_path}")
    print(f"[INFO] IFES_QUESTION = {ifes_question}")
    print(f"[INFO] UNIDADE_QUESTION = {unidade_question}")
    # Imprimir em tela as perguntas consideradas metadados
    try:
        metas_list = list(meta_cols)
        print("[META] Campos de metadados detectados (não entram no cálculo):")
        if metas_list:
            for c in metas_list:
                print(f" - {c}")
        else:
            print(" (nenhum metadado detectado)")
    except Exception as _e:
        print("[META][AVISO] Não foi possível imprimir metadados:", _e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera Excel com IM por Unidade e IFES.")
    parser.add_argument("--questionario", required=True, type=Path, help="Caminho para questionario.xlsx")
    parser.add_argument("--respostas", required=True, type=Path, help="Caminho para respostas.xlsx")
    parser.add_argument("--saida", required=True, type=Path, help="Caminho para o Excel de saída (ex.: im_resultado.xlsx)")
    parser.add_argument("--ifes-col", dest="ifes_col", default=PERGUNTA_IFES_PADRAO, help="Texto da pergunta que identifica a IFES")
    parser.add_argument("--unidade-col", dest="unidade_col", default=PERGUNTA_UNIDADE_PADRAO, help="Texto da pergunta que identifica a Unidade da IFES")

    args = parser.parse_args()

    try:
        main(
            questionario_path=args.questionario,
            respostas_path=args.respostas,
            saida_path=args.saida,
            ifes_question=args.ifes_col,
            unidade_question=args.unidade_col,
        )
    except Exception as e:
        print("[ERRO]", e, file=sys.stderr)
        sys.exit(1)
