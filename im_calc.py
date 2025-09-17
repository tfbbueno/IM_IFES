
import argparse 
import sys
from hashlib import md5
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ======================= CONSTANTES CONFIGURAVEIS =======================
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

DELIMITADORES_CANDIDATOS = [",", ";", "\t", "|"]


# ======================= FUNCOES AUXILIARES =======================
def classificar_nivel(pontuacao: float) -> str:
    """Retorna a faixa de classificacao textual a partir da pontuacao [0,1]."""
    if pd.isna(pontuacao):
        return np.nan
    pontuacao = float(pontuacao)
    if pontuacao < 0.0 or pontuacao > 1.0:
        raise ValueError(f"Pontuacao fora do intervalo [0,1]: {pontuacao}. Verifique os dados de entrada.")
    #pontuacao = round(pontuacao, 2)
    for i, (a, b, rotulo) in enumerate(FAIXAS_CLASSIFICACAO):
        if i < len(FAIXAS_CLASSIFICACAO) - 1:
            if a <= pontuacao < b:
                return rotulo
        else:
            if a <= pontuacao <= b:
                return rotulo
    return np.nan


def interpretar_fator(valor):
    """Converte a resposta bruta em (fator [0..1], valida[bool]) considerando NA/N/A/NAO SE APLICA."""
    if pd.isna(valor):
        return np.nan, False
    s = str(valor).strip().upper()
    if s in ("NA", "N/A", "NÃO SE APLICA", "NAO SE APLICA"):
        return np.nan, False
    dig = None
    for ch in s:
        if ch.isdigit():
            dig = ch
            break
    if dig is None:
        return np.nan, False
    fator = MAPA_FATOR.get(dig, np.nan)
    return fator, (not np.isnan(fator))


def normalizar_nome_coluna_area_responsavel(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o nome da coluna 'Área responsável' para exatamente esse texto e cria caso não exista."""
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    achou = False
    for c in list(df.columns):
        if c.strip().lower() in ("área responsável", "area responsável", "área responsavel", "area responsavel"):
            if c != "Área responsável":
                df = df.rename(columns={c: "Área responsável"})
            achou = True
    if not achou and "Área responsável" not in df.columns:
        df["Área responsável"] = np.nan
    return df


def gerar_id_resposta(linha: pd.Series, col_ifes: str, col_unidade: str) -> str:
    """Cria um identificador estavel para cada resposta (hash dos principais metadados)."""
    chaves = []
    for k in ["Carimbo de data/hora", "E-mail de contato", col_ifes, col_unidade]:
        if k in linha and pd.notna(linha[k]):
            chaves.append(str(linha[k]))
    base = "|".join(chaves) if chaves else str(linha.get("RespostaIndex", ""))
    h = md5(base.encode("utf-8")).hexdigest()[:12]
    return f"RSP-{h}"


def detectar_delimitador(caminho: Path, amostra_linhas: int = 5) -> str:
    """Detecta o delimitador mais frequente entre virgula, ponto-e-virgula, tab e pipe."""
    try:
        contagens = {d: 0 for d in [",", ";", "\t", "|"]}
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            for i, linha in enumerate(f):
                if i >= amostra_linhas:
                    break
                for d in contagens:
                    contagens[d] += linha.count(d)
        melhor = max(contagens, key=contagens.get)
        return melhor if contagens[melhor] > 0 else ","
    except Exception:
        return ","


def carregar_tabela_arquivo(caminho: Path, *, codificacao: str = "utf-8", delimitador: str | None = None,
                            planilha: str | int | None = None, rotulo: str = "") -> pd.DataFrame:
    """Carrega CSV (.csv/.txt) ou Excel (.xlsx/.xls). Se CSV, detecta delimitador quando nao informado."""
    sufixo = caminho.suffix.lower()
    if sufixo in {".csv", ".txt"}:
        sep = delimitador if delimitador else detectar_delimitador(caminho)
        try:
            return pd.read_csv(caminho, encoding=codificacao, sep=sep, engine="python")
        except Exception as e:
            raise ValueError(f"Falha ao ler {rotulo or caminho.name} como CSV (sep='{sep}'): {e}")
    elif sufixo in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(caminho, sheet_name=planilha if planilha is not None else 0)
        except Exception as e:
            raise ValueError(f"Falha ao ler {rotulo or caminho.name} como Excel: {e}")
    else:
        raise ValueError(f"Formato nao suportado para {rotulo or caminho.name}: '{sufixo}'. Use .xlsx/.xls ou .csv/.txt")


# ======================= TXT / FORMATACAO =======================
def gerar_slug(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:120] or "NA"


def quantil(serie: pd.Series, p: float) -> float:
    try:
        return float(serie.quantile(p))
    except Exception:
        return np.nan


def fmt_num(x, nd=2):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def escrever_txt(caminho: Path, linhas: list[str]):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

def _calc_larguras(colunas: list[str], linhas: list[list[str]]) -> list[int]:
    larguras = [len(str(c)) for c in colunas]
    for linha in linhas:
        for i, cel in enumerate(linha):
            if i < len(larguras):
                larguras[i] = max(larguras[i], len(str(cel)))
            else:
                larguras.append(len(str(cel)))
    return larguras

def montar_tabela_txt(cabecalhos: list[str], linhas: list[list[str]], alinhamento: list[str] | None = None) -> list[str]:
    """
    Gera uma tabela ASCII alinhada (sem dependências externas).
    alinhamento: lista com 'left'/'right'/'center' por coluna (default: left p/ 1a, right p/ demais).
    """
    if alinhamento is None:
        alinhamento = ["left"] + ["right"] * (len(cabecalhos) - 1)
    larg = _calc_larguras(cabecalhos, linhas)

    def fmt_cell(txt, w, align):
        s = str(txt)
        if align == "right":
            return s.rjust(w)
        if align == "center":
            return s.center(w)
        return s.ljust(w)

    top = "+" + "+".join("-" * (w + 2) for w in larg) + "+"
    sep = "+" + "+".join("=" * (w + 2) for w in larg) + "+"
    mid = "+" + "+".join("-" * (w + 2) for w in larg) + "+"

    out = [top]
    out.append("| " + " | ".join(fmt_cell(h, w, "center") for h, w in zip(cabecalhos, larg)) + " |")
    out.append(sep)
    for lin in linhas:
        if len(lin) < len(larg):
            lin = list(lin) + [""] * (len(larg) - len(lin))
        out.append("| " + " | ".join(fmt_cell(c, w, a) for c, w, a in zip(lin, larg, alinhamento)) + " |")
    out.append(mid)
    return out

# ======================= RELATORIOS TXT =======================

def gerar_relatorios_txt(
    caminho_saida: Path,
    dados_longos: pd.DataFrame,
    consolidacao_dimensao_unidade: pd.DataFrame,
    consolidacao_dimensao_ifes: pd.DataFrame,
    consolidacao_ifes: pd.DataFrame,
):
    """
    Gera relatórios TXT por IFES e por ifes_unidade.
    Regras aplicadas (APENAS MÉDIAS):
      - Comparações por IFES e Dimensão usam **média** como limiar (ex.: "< média" e ">= média").
      - Desvio-padrão amostral (ddof=1).
      - Tabelas mostram apenas: Mínimo, Máximo, Desvio-padrão e Média.
    """
    dir_base = Path(caminho_saida).parent / (Path(caminho_saida).stem + "_txt")
    dir_ifes = dir_base / "IFES"
    dir_unidades = dir_base / "unidades"

    dados_validos = dados_longos.loc[dados_longos["Valida"] == True].copy()

    # IM por unidade (média das dimensões da unidade)
    im_por_unidade = (
        consolidacao_dimensao_unidade
        .groupby(["IFES", "Unidade", "ifes_unidade"], dropna=False)["Pontuação IM da Dimensão na ifes_unidade"]
        .mean()
        .reset_index(name="IM da ifes_unidade")
    )

    # Limiar: média do IM das unidades por IFES
    limiares_im_ifes = (
        im_por_unidade.groupby("IFES", dropna=False)["IM da ifes_unidade"]
        .agg(media="mean")
        .reset_index()
    )

    # (Entre unidades) média por dimensão na IFES — para relatórios IFES/unidade
    media_dimensao_por_ifes = (
        consolidacao_dimensao_unidade
        .groupby(["IFES","Dimensão"], dropna=False)["Pontuação IM da Dimensão na ifes_unidade"]
        .mean()
        .reset_index(name="Media_dim_IFES")
    )

    # Baseline de respostas por dimensão (na IFES): apenas média
    base_dimensao_respostas = (
        dados_validos
        .groupby(["IFES","Dimensão"], dropna=False)["Fator"]
        .agg(dim_media="mean")
        .reset_index()
    )

    # Média das respostas por PERGUNTA dentro da IFES
    media_pergunta_por_ifes = (
        dados_validos
        .groupby(["IFES","Dimensão","Pergunta"], dropna=False)["Fator"]
        .mean()
        .reset_index(name="media_pergunta")
    )

    # ------------------- Relatórios por IFES -------------------
    for ifes, df_ifes in consolidacao_dimensao_unidade.groupby("IFES", dropna=False):
        ifes_rotulo = str(ifes) if pd.notna(ifes) else "(Sem IFES)"
        ifes_slug = gerar_slug(ifes_rotulo)

        # IM da IFES: média do IM das dimensões
        linha_ifes = consolidacao_ifes.loc[consolidacao_ifes["IFES"] == ifes]
        if len(linha_ifes):
            im_da_ifes = linha_ifes["Pontuação IM na IFES"].iloc[0]
        else:
            im_da_ifes = consolidacao_dimensao_ifes.loc[
                consolidacao_dimensao_ifes["IFES"] == ifes, "Pontuação IM da Dimensão na IFES"
            ].mean()

        linhas = []
        linhas.append(f"Relatório IM – IFES: {ifes_rotulo}")
        linhas.append(f"IM da IFES (média das dimensões): {fmt_num(im_da_ifes)}")
        linhas.append("Obs.: Desvio-padrão amostral (ddof=1). Regras: '< média' e '>= média'.")
        linhas.append("")

        # A.1) Tabela de análise por IFES (IM por unidade) — estatísticas simples com média
        linhas.append("A.1) Tabela de análise por IFES (IM por unidade)")
        cab = ["Métrica","Mínimo","Máximo","Desvio-padrão","Média","Q1 (25%)","Mediana (50%)","Q3 (75%)","IQR (Q3-Q1)"]

        s2 = im_por_unidade.loc[im_por_unidade["IFES"] == ifes, "IM da ifes_unidade"].dropna()
        if len(s2) == 0:
            stats = ["—"] * 8
        else:
            minimo  = s2.min()
            maximo  = s2.max()
            desvio  = s2.std(ddof=1)
            media   = s2.mean()
            q1      = s2.quantile(0.25)
            med     = s2.quantile(0.50)
            q3      = s2.quantile(0.75)
            iqr     = q3 - q1
            stats   = [fmt_num(minimo), fmt_num(maximo), fmt_num(desvio), fmt_num(media), fmt_num(q1), fmt_num(med), fmt_num(q3), fmt_num(iqr)]
        linhas_tabela = [["IM (por unidade)"] + stats]
        for l in montar_tabela_txt(cab, linhas_tabela, alinhamento=["left"] + ["right"] * (len(cab)-1)):
            linhas.append(l)
        linhas.append("")

        # A.2) Tabela de análise de dimensão por IFES — estatísticas com média
        linhas.append("A.2) Tabela de análise de dimensão por IFES")
        cab = ["Dimensão","Mínimo","Máximo","Desvio-padrão","Média","Δ p/ IM da IFES","Q1 (25%)","Mediana (50%)","Q3 (75%)","IQR (Q3-Q1)"]
        linhas_tabela = []
        for dimensao, df_dim in df_ifes.groupby("Dimensão", dropna=False):
            rotulo_dim = str(dimensao) if pd.notna(dimensao) else "(Sem dimensão)"
            s = df_dim["Pontuação IM da Dimensão na ifes_unidade"].dropna()
            if len(s) == 0:
                stats = ["—"]*9
            else:
                minimo = s.min()
                maximo = s.max()
                desvio = s.std(ddof=1)
                media  = s.mean()
                delta  = media - im_da_ifes
                q1     = s.quantile(0.25)
                med    = s.quantile(0.50)
                q3     = s.quantile(0.75)
                iqr    = q3 - q1
                stats  = [fmt_num(minimo), fmt_num(maximo), fmt_num(desvio), fmt_num(media), fmt_num(delta), fmt_num(q1), fmt_num(med), fmt_num(q3), fmt_num(iqr)]
            linhas_tabela.append([rotulo_dim] + stats)
        for l in montar_tabela_txt(cab, linhas_tabela, alinhamento=["left"] + ["right"]*(len(cab)-1)):
            linhas.append(l)
        linhas.append("")

        # Limiar por IFES (média do IM das unidades)
        im_da_ifes_unidades = im_por_unidade.loc[im_por_unidade["IFES"] == ifes].copy()
        limiares = limiares_im_ifes.loc[limiares_im_ifes["IFES"] == ifes]
        mean_im = limiares["media"].iloc[0] if len(limiares) else np.nan

        # Média por dimensão (entre unidades) nesta IFES
        dim_mean_tbl = media_dimensao_por_ifes.loc[media_dimensao_por_ifes["IFES"] == ifes][["Dimensão","Media_dim_IFES"]]

        # A.3A
        linhas.append("A.3A Dimensões com média do IM menor que a média do IM da IFES")
        if pd.isna(mean_im) or dim_mean_tbl.empty:
            linhas.append(" - (sem média calculada)")
        else:
            sub = dim_mean_tbl.loc[dim_mean_tbl["Media_dim_IFES"] < mean_im].sort_values("Media_dim_IFES")
            if sub.empty:
                linhas.append(" - (nenhuma)")
            else:
                for _, rr in sub.iterrows():
                    linhas.append(f" - {rr['Dimensão']}: {fmt_num(rr['Media_dim_IFES'])} (limiar {fmt_num(mean_im)})")
        linhas.append("")

        # A.3B
        linhas.append('A.3B "Dimensões referência": Dimensões com média do IM >= média do IM da IFES')
        if pd.isna(mean_im) or dim_mean_tbl.empty:
            linhas.append(" - (sem média calculada)")
        else:
            sub = dim_mean_tbl.loc[dim_mean_tbl["Media_dim_IFES"] >= mean_im].sort_values("Media_dim_IFES", ascending=False)
            if sub.empty:
                linhas.append(" - (nenhuma)")
            else:
                for _, rr in sub.iterrows():
                    linhas.append(f" - {rr['Dimensão']}: {fmt_num(rr['Media_dim_IFES'])} (limiar {fmt_num(mean_im)})")
        linhas.append("")

        # A.3 Tabela — IM por unidade (média das dimensões por unidade)
        linhas.append("A.3 Tabela — IM por unidade")
        cab = ["Unidade","IM da unidade"]
        try:
            tab = (
                im_da_ifes_unidades[["Unidade","IM da ifes_unidade"]]
                .sort_values("Unidade", na_position="last")
            )
            linhas_tabela = [
                [str(u) if pd.notna(u) else "(Sem unidade)", fmt_num(v)]
                for u, v in tab.to_numpy()
            ]
        except Exception:
            linhas_tabela = []
        for l in montar_tabela_txt(cab, linhas_tabela, alinhamento=["left","right"]):
            linhas.append(l)
        linhas.append("")

        # B) Unidades com IM < média da IFES
        linhas.append("B) Unidades com IM abaixo da média da IFES")
        if pd.isna(mean_im):
            linhas.append(" - (sem média calculada)")
        else:
            sub = im_da_ifes_unidades.loc[im_da_ifes_unidades["IM da ifes_unidade"] < mean_im].sort_values("IM da ifes_unidade")
            if sub.empty:
                linhas.append(" - (nenhuma)")
            else:
                for _, r in sub.iterrows():
                    linhas.append(f" - {r['ifes_unidade']} = {fmt_num(r['IM da ifes_unidade'])}")
        linhas.append("")

        # C) Unidades Referência: IM >= média
        linhas.append('C) "Unidades Referência": IM >= média da IFES')
        if pd.isna(mean_im):
            linhas.append(" - (sem média calculada)")
        else:
            sub = im_da_ifes_unidades.loc[im_da_ifes_unidades["IM da ifes_unidade"] >= mean_im].sort_values("IM da ifes_unidade", ascending=False)
            if sub.empty:
                linhas.append(" - (nenhuma)")
            else:
                for _, r in sub.iterrows():
                    linhas.append(f" - {r['ifes_unidade']} = {fmt_num(r['IM da ifes_unidade'])}")
        linhas.append("")

        # D/E) Perguntas vs baseline da dimensão (na IFES) — baseline = MÉDIA DA DIMENSÃO DA IFES (IM da Dimensão)
        base_dim_ifes_media = (
            consolidacao_dimensao_ifes.loc[consolidacao_dimensao_ifes["IFES"] == ifes, ["IFES","Dimensão","Pontuação IM da Dimensão na IFES"]]
            .rename(columns={"Pontuação IM da Dimensão na IFES": "Media_dim_IFES"})
        )
        perg_mean = media_pergunta_por_ifes.loc[media_pergunta_por_ifes["IFES"] == ifes].copy()
        perg_mean = perg_mean.merge(base_dim_ifes_media, on=["IFES","Dimensão"], how="left")

        # D) media_pergunta < média da Dimensão da IFES
        linhas.append("D) Perguntas por dimensão com média < média da Dimensão da IFES")
        if perg_mean.empty:
            linhas.append(" - (sem dados)")
        else:
            for dimensao, df_dim in perg_mean.groupby("Dimensão", dropna=False):
                rotulo_dim = str(dimensao) if pd.notna(dimensao) else "(Sem dimensão)"
                base_md = df_dim["Media_dim_IFES"].iloc[0]
                subset = df_dim.loc[df_dim["media_pergunta"] < base_md].sort_values("media_pergunta")
                if subset.empty:
                    continue
                linhas.append(f" - {rotulo_dim}:")
                for _, rrr in subset.iterrows():
                    linhas.append(f"    • {rrr['Pergunta']} = {fmt_num(rrr['media_pergunta'])} (limiar {fmt_num(base_md)})")
        linhas.append("")

        # E) media_pergunta >= média da Dimensão da IFES
        linhas.append('E) "Perguntas Referência": perguntas com média >= média da Dimensão da IFES')
        if perg_mean.empty:
            linhas.append(" - (sem dados)")
        else:
            for dimensao, df_dim in perg_mean.groupby("Dimensão", dropna=False):
                rotulo_dim = str(dimensao) if pd.notna(dimensao) else "(Sem dimensão)"
                base_md = df_dim["Media_dim_IFES"].iloc[0]
                subset = df_dim.loc[df_dim["media_pergunta"] >= base_md].sort_values("media_pergunta", ascending=False)
                if subset.empty:
                    continue
                linhas.append(f" - {rotulo_dim}:")
                for _, rrr in subset.iterrows():
                    linhas.append(f"    • {rrr['Pergunta']} = {fmt_num(rrr['media_pergunta'])} (limiar {fmt_num(base_md)})")
        linhas.append("")

        escrever_txt(dir_ifes / f"IFES_{ifes_slug}.txt", linhas)

    # ------------------- Relatórios por ifes_unidade -------------------
    for _, r in im_por_unidade.iterrows():
        ifes = r["IFES"]
        unidade = r["Unidade"]
        ifes_unid = r["ifes_unidade"]
        ifes_rotulo = str(ifes) if pd.notna(ifes) else "(Sem IFES)"
        unid_rotulo = str(unidade) if pd.notna(unidade) else "(Sem Unidade)"
        ifes_slug = gerar_slug(ifes_rotulo)
        unid_slug = gerar_slug(unid_rotulo)

        linhas = []
        linhas.append(f"Relatório IM – Unidade: {ifes_unid}")
        linhas.append(f"IM da unidade (média das dimensões): {fmt_num(r['IM da ifes_unidade'])}")
        linhas.append("Obs.: Desvio-padrão amostral (ddof=1). Regras: '< média' e '>= média'.")
        linhas.append("")

        # A.2) Tabela de análise de dimensão por IFES_UNIDADE
        linhas.append("A.2) Tabela de análise de dimensão por IFES_UNIDADE")

        # Base da unidade: IM por dimensão + classificação
        dim_unid_tab = (
            consolidacao_dimensao_unidade
            .loc[
                (consolidacao_dimensao_unidade["IFES"] == ifes) &
                (consolidacao_dimensao_unidade["Unidade"] == unidade)
            ][["Dimensão",
               "Pontuação IM da Dimensão na ifes_unidade",
               "Classificação IM da dimensão na ifes_unidade"]]
            .rename(columns={
                "Pontuação IM da Dimensão na ifes_unidade": "IM_dim_unidade",
                "Classificação IM da dimensão na ifes_unidade": "Classif"
            })
        )

        # Baselines na IFES: **média** por dimensão
        base_mean_dim2 = media_dimensao_por_ifes.loc[
            media_dimensao_por_ifes["IFES"] == ifes, ["Dimensão", "Media_dim_IFES"]
        ]

        # Junta e calcula delta p/ média
        dim_unid_tab = (
            dim_unid_tab
            .merge(base_mean_dim2, on="Dimensão", how="left")
        )
        dim_unid_tab["delta_mean"] = dim_unid_tab["IM_dim_unidade"] - dim_unid_tab["Media_dim_IFES"]

        # Monta tabela ASCII alinhada
        cab = ["Dimensão", "IM da unidade", "Média (IFES)",
               "Δ p/ média", "Classificação"]
        linhas_tabela = [
            [
                str(row["Dimensão"]) if pd.notna(row["Dimensão"]) else "(Sem dimensão)",
                fmt_num(row["IM_dim_unidade"]),
                fmt_num(row["Media_dim_IFES"]),
                fmt_num(row["delta_mean"]),
                row.get("Classif", "—"),
            ]
            for _, row in dim_unid_tab.sort_values("Dimensão", na_position="last").iterrows()
        ]
        for l in montar_tabela_txt(
            cab,
            linhas_tabela,
            alinhamento=["left"] + ["right"] * 3 + ["left"],
        ):
            linhas.append(l)
        linhas.append("")

        # A/B usam IM por dimensão da unidade
        dim_unid = (
            consolidacao_dimensao_unidade
            .loc[(consolidacao_dimensao_unidade["IFES"] == ifes) & (consolidacao_dimensao_unidade["Unidade"] == unidade)]
            [["Dimensão","Pontuação IM da Dimensão na ifes_unidade"]]
            .rename(columns={"Pontuação IM da Dimensão na ifes_unidade":"IM_dim_unidade"})
        )

        # A) IM_dim_unidade < média(IFES, dimensão)
        base_mean_dim = media_dimensao_por_ifes.loc[media_dimensao_por_ifes["IFES"] == ifes][["Dimensão","Media_dim_IFES"]]
        a_tbl = dim_unid.merge(base_mean_dim, on="Dimensão", how="left")
        a_bad = a_tbl.loc[a_tbl["IM_dim_unidade"] < a_tbl["Media_dim_IFES"]].sort_values("IM_dim_unidade")
        linhas.append("A) Dimensões com IM da unidade < média das unidades da IFES")
        if a_bad.empty:
            linhas.append(" - (nenhuma)")
        else:
            for _, rr in a_bad.iterrows():
                linhas.append(f" - {rr['Dimensão']}: {fmt_num(rr['IM_dim_unidade'])} (limiar {fmt_num(rr['Media_dim_IFES'])})")
        linhas.append("")

        # B) IM_dim_unidade < IM da dimensão na IFES (média) — já equivalente à A), mantido por clareza
        base_mean_dim_ifes = (
            consolidacao_dimensao_ifes.loc[consolidacao_dimensao_ifes["IFES"] == ifes][["Dimensão","Pontuação IM da Dimensão na IFES"]]
            .rename(columns={"Pontuação IM da Dimensão na IFES":"Media_dim_IFES"})
        )
        b_tbl = dim_unid.merge(base_mean_dim_ifes, on="Dimensão", how="left")
        b_bad = b_tbl.loc[b_tbl["IM_dim_unidade"] < b_tbl["Media_dim_IFES"]].sort_values("IM_dim_unidade")
        linhas.append("B) Dimensões com IM da unidade < IM da dimensão na IFES (média)")
        if b_bad.empty:
            linhas.append(" - (nenhuma)")
        else:
            for _, rr in b_bad.iterrows():
                linhas.append(f" - {rr['Dimensão']}: {fmt_num(rr['IM_dim_unidade'])} (limiar {fmt_num(rr['Media_dim_IFES'])})")
        linhas.append("")

        # C/D) Perguntas da unidade vs baseline da dimensão na IFES (respostas) — média
        base_dim_ifes = base_dimensao_respostas.loc[base_dimensao_respostas["IFES"] == ifes][["Dimensão","dim_media"]]
        perg_mean_unid = (
            dados_validos
            .loc[(dados_validos["IFES"] == ifes) & (dados_validos["Unidade"] == unidade)]
            .groupby(["Dimensão","Pergunta"], dropna=False)["Fator"]
            .mean()
            .reset_index(name="media_pergunta_unidade")
        )

        perg_mean_unid = perg_mean_unid.merge(base_dim_ifes, on="Dimensão", how="left")

        # C) média_unidade < média(IFES, dimensão)
        linhas.append("C) Perguntas por dimensão com média da unidade < média das respostas da Dimensão na IFES")
        c_bad = perg_mean_unid.loc[perg_mean_unid["media_pergunta_unidade"] < perg_mean_unid["dim_media"]]
        if c_bad.empty:
            linhas.append(" - (nenhuma)")
        else:
            for dimensao, df_dim in c_bad.groupby("Dimensão", dropna=False):
                linhas.append(f" - {dimensao}:")
                for _, rr in df_dim.sort_values("media_pergunta_unidade").iterrows():
                    linhas.append(f"    • {rr['Pergunta']} = {fmt_num(rr['media_pergunta_unidade'])} (limiar {fmt_num(rr['dim_media'])})")
        linhas.append("")

        # D) média_unidade >= média(IFES, dimensão)
        linhas.append('D) "Perguntas Referência": perguntas com média da unidade >= média das respostas da Dimensão na IFES')
        d_good = perg_mean_unid.loc[perg_mean_unid["media_pergunta_unidade"] >= perg_mean_unid["dim_media"]]
        if d_good.empty:
            linhas.append(" - (nenhuma)")
        else:
            for dimensao, df_dim in d_good.groupby("Dimensão", dropna=False):
                linhas.append(f" - {dimensao}:")
                for _, rr in df_dim.sort_values("media_pergunta_unidade", ascending=False).iterrows():
                    linhas.append(f"    • {rr['Pergunta']} = {fmt_num(rr['media_pergunta_unidade'])} (limiar {fmt_num(rr['dim_media'])})")
        linhas.append("")

        ifes_unidades_dir = dir_unidades / f"IFES_{ifes_slug}__UNIDADE_{unid_slug}.txt"
        escrever_txt(ifes_unidades_dir, linhas)

    print(f"[OK] Relatórios TXT gerados em: {dir_base}")



# ======================= PIPELINE PRINCIPAL =======================
def executar(caminho_questionario: Path,
             caminho_respostas: Path,
             caminho_saida: Path,
             pergunta_ifes: str = PERGUNTA_IFES_PADRAO,
             pergunta_unidade: str = PERGUNTA_UNIDADE_PADRAO,
             codificacao_questionario: str = "utf-8",
             codificacao_respostas: str = "utf-8",
             delimitador_questionario: str | None = None,
             delimitador_respostas: str | None = None,
             planilha_questionario: str | int | None = None,
             planilha_respostas: str | int | None = None) -> None:
    # Leitura
    q_df = carregar_tabela_arquivo(
        caminho_questionario,
        codificacao=codificacao_questionario,
        delimitador=delimitador_questionario,
        planilha=planilha_questionario,
        rotulo="questionário"
    )
    q_df = normalizar_nome_coluna_area_responsavel(q_df)
    r_df = carregar_tabela_arquivo(
        caminho_respostas,
        codificacao=codificacao_respostas,
        delimitador=delimitador_respostas,
        planilha=planilha_respostas,
        rotulo="respostas"
    )

    # Perguntas do questionario e metadados
    q_df["Pergunta"] = q_df["Pergunta"].astype(str).str.strip()
    perguntas_questionario = set(q_df["Pergunta"].dropna().tolist())
    colunas_respostas = list(r_df.columns)
    colunas_meta = [c for c in colunas_respostas if c not in perguntas_questionario]

    # Formato longo de respostas
    r_df = r_df.reset_index().rename(columns={"index": "RespostaIndex"})
    colunas_valores = [c for c in colunas_respostas if c in perguntas_questionario]

    dados_longos = r_df.melt(
        id_vars=["RespostaIndex"] + colunas_meta,
        value_vars=colunas_valores,
        var_name="Pergunta",
        value_name="Resposta"
    )

    # IFES/Unidade
    dados_longos["IFES"] = dados_longos.get(pergunta_ifes, np.nan).astype(str).str.strip()
    dados_longos["Unidade"] = dados_longos.get(pergunta_unidade, np.nan).astype(str).str.strip()

    # ifes_unidade (precisa existir antes dos agrupamentos)
    dados_longos["ifes_unidade"] = (
        dados_longos["IFES"].fillna("").astype(str).str.strip()
        + " - " +
        dados_longos["Unidade"].fillna("").astype(str).str.strip()
    )

    # RespostaID estavel
    mapa_ids = r_df[["RespostaIndex"] + colunas_meta].drop_duplicates().copy()
    mapa_ids["RespostaID"] = mapa_ids.apply(lambda linha: gerar_id_resposta(linha, pergunta_ifes, pergunta_unidade), axis=1)
    dados_longos = dados_longos.merge(mapa_ids[["RespostaIndex", "RespostaID"]], on="RespostaIndex", how="left")

    # Metadados do questionario
    q_meta = q_df[["Pergunta", "Área responsável", "Dimensão"]].copy()
    q_meta["Pergunta"] = q_meta["Pergunta"].astype(str).str.strip()
    dados_longos["Pergunta"] = dados_longos["Pergunta"].astype(str).str.strip()
    dados_longos = dados_longos.merge(q_meta, on="Pergunta", how="left")

    # Fator e validade
    interpretado = dados_longos["Resposta"].apply(interpretar_fator)
    dados_longos["Fator"] = interpretado.apply(lambda x: x[0])
    dados_longos["Valida"] = interpretado.apply(lambda x: x[1])

    # Peso por dimensao (1/n_validas) por resposta/unidade
    chaves_grp = ["IFES", "Unidade", "ifes_unidade", "Dimensão", "RespostaID"]
    n_validas = (
        dados_longos.assign(validas=dados_longos["Valida"].astype(int))
        .groupby(chaves_grp)["validas"].sum().rename("n_validas").reset_index()
    )
    dados_longos = dados_longos.merge(n_validas, on=chaves_grp, how="left")

    dados_longos["Peso"] = np.where(
        (dados_longos["Valida"]) & (dados_longos["n_validas"] > 0),
        1.0 / dados_longos["n_validas"],
        np.nan
    )
    dados_longos["Pontuação da pergunta"] = dados_longos["Fator"] * dados_longos["Peso"]


    # ------------------- Consolidações -------------------
    # Dimensao na unidade (media das respostas ponderadas por RespostaID)
    dim_por_resposta = (
        dados_longos
        .groupby(["IFES", "Unidade", "ifes_unidade", "Dimensão", "RespostaID"], dropna=False)["Pontuação da pergunta"]
        .sum()
        .reset_index(name="Pontuação_dim_por_resposta")
    )

    consolidacao_dimensao_unidade = (
        dim_por_resposta
        .groupby(["IFES", "Unidade", "ifes_unidade", "Dimensão"], dropna=False)["Pontuação_dim_por_resposta"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação_dim_por_resposta": "Pontuação IM da Dimensão na ifes_unidade"})
    )
    #consolidacao_dimensao_unidade["Pontuação IM da Dimensão na ifes_unidade"] = \
        #consolidacao_dimensao_unidade["Pontuação IM da Dimensão na ifes_unidade"].round(2)
    consolidacao_dimensao_unidade["Classificação IM da dimensão na ifes_unidade"] = \
        consolidacao_dimensao_unidade["Pontuação IM da Dimensão na ifes_unidade"].apply(classificar_nivel)

    # Dimensao na IFES (media das unidades)
    consolidacao_dimensao_ifes = (
        consolidacao_dimensao_unidade
        .groupby(["IFES", "Dimensão"], dropna=False)["Pontuação IM da Dimensão na ifes_unidade"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação IM da Dimensão na ifes_unidade": "Pontuação IM da Dimensão na IFES"})
    )
    #consolidacao_dimensao_ifes["Pontuação IM da Dimensão na IFES"] = consolidacao_dimensao_ifes["Pontuação IM da Dimensão na IFES"].round(2)
    consolidacao_dimensao_ifes["Classificação IM da dimensão na IFES"] = \
        consolidacao_dimensao_ifes["Pontuação IM da Dimensão na IFES"].apply(classificar_nivel)

    # IM por IFES (media do IM das dimensoes)
    consolidacao_ifes = (
        consolidacao_dimensao_ifes
        .groupby(["IFES"], dropna=False)["Pontuação IM da Dimensão na IFES"]
        .mean()
        .reset_index()
        .rename(columns={"Pontuação IM da Dimensão na IFES": "Pontuação IM na IFES"})
    )
    #consolidacao_ifes["Pontuação IM na IFES"] = consolidacao_ifes["Pontuação IM na IFES"].round(2)
    consolidacao_ifes["Classificação IM na IFES"] = consolidacao_ifes["Pontuação IM na IFES"].apply(classificar_nivel)

    # ------------------- Excel: Dados agregados -------------------
    dados_finais = dados_longos[[
        "IFES", "Unidade", "ifes_unidade", "RespostaID",
        "Pergunta", "Resposta", "Área responsável", "Dimensão",
        "Peso", "Fator", "Pontuação da pergunta"
    ] + colunas_meta].copy()

    dados_agregados = (
        dados_finais
        .merge(consolidacao_dimensao_unidade[[
            "IFES","Unidade","ifes_unidade","Dimensão",
            "Pontuação IM da Dimensão na ifes_unidade",
            "Classificação IM da dimensão na ifes_unidade"
        ]], how="left", on=["IFES","Unidade","ifes_unidade","Dimensão"])
        .merge(consolidacao_dimensao_ifes[[
            "IFES","Dimensão",
            "Pontuação IM da Dimensão na IFES",
            "Classificação IM da dimensão na IFES"
        ]], how="left", on=["IFES","Dimensão"])
        .merge(consolidacao_ifes[[
            "IFES",
            "Pontuação IM na IFES",
            "Classificação IM na IFES"
        ]], how="left", on=["IFES"])
    )

    caminho_saida = Path(caminho_saida)
    with pd.ExcelWriter(caminho_saida, engine="xlsxwriter") as escritor:
        dados_agregados.to_excel(escritor, index=False, sheet_name="dados agregados")
    print(f"[OK] Arquivo gerado: {caminho_saida}")

    # ------------------- TXT -------------------
    gerar_relatorios_txt(caminho_saida, dados_longos, consolidacao_dimensao_unidade, consolidacao_dimensao_ifes, consolidacao_ifes)

    # INFO gerais no stdout
    print(f"[INFO] COL_IFES = {pergunta_ifes}")
    print(f"[INFO] COL_UNIDADE = {pergunta_unidade}")
    try:
        metas_cols = [c for c in r_df.columns if c not in q_df["Pergunta"].tolist()]
        print("[META] Campos de metadados detectados (não entram no cálculo):")
        if metas_cols:
            for c in metas_cols:
                print(f" - {c}")
        else:
            print(" (nenhum metadado detectado)")
    except Exception as _e:
        print("[META][AVISO] Não foi possível imprimir metadados:", _e)


# ======================= CLI - Interface de Linha de Comando =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera Excel com IM e relatórios TXT por IFES e Unidade (aceita .xlsx/.xls e .csv/.txt).")
    parser.add_argument("--questionario", required=True, type=Path, help="Caminho para questionario (.xlsx/.xls/.csv)")
    parser.add_argument("--respostas", required=True, type=Path, help="Caminho para respostas (.xlsx/.xls/.csv)")
    parser.add_argument("--saida", required=True, type=Path, help="Caminho para o Excel de saída (ex.: im_resultado.xlsx)")
    parser.add_argument("--ifes-col", dest="col_ifes", default=PERGUNTA_IFES_PADRAO, help="Texto da pergunta/coluna que identifica a IFES")
    parser.add_argument("--unidade-col", dest="col_unidade", default=PERGUNTA_UNIDADE_PADRAO, help="Texto da pergunta/coluna que identifica a Unidade da IFES")

    # Opcoes de leitura (CSV/Excel)
    parser.add_argument("--questionario-encoding", dest="enc_q", default="utf-8", help="Encoding do questionário (CSV)")
    parser.add_argument("--respostas-encoding", dest="enc_r", default="utf-8", help="Encoding das respostas (CSV)")
    parser.add_argument("--questionario-delim", dest="delim_q", default=None, help="Delimitador do questionário (CSV). Se omitido, detecta automaticamente (, ; \\t |)")
    parser.add_argument("--respostas-delim", dest="delim_r", default=None, help="Delimitador das respostas (CSV). Se omitido, detecta automaticamente (, ; \\t |)")
    parser.add_argument("--questionario-sheet", dest="sheet_q", default=None, help="Nome/índice da planilha do questionário (Excel). Padrão: 0")
    parser.add_argument("--respostas-sheet", dest="sheet_r", default=None, help="Nome/índice da planilha das respostas (Excel). Padrão: 0")

    args = parser.parse_args()

    try:
        executar(
            caminho_questionario=args.questionario,
            caminho_respostas=args.respostas,
            caminho_saida=args.saida,
            pergunta_ifes=args.col_ifes,
            pergunta_unidade=args.col_unidade,
            codificacao_questionario=args.enc_q,
            codificacao_respostas=args.enc_r,
            delimitador_questionario=args.delim_q,
            delimitador_respostas=args.delim_r,
            planilha_questionario=args.sheet_q,
            planilha_respostas=args.sheet_r,
        )
    except Exception as e:
        print("[ERRO]", e, file=sys.stderr)
        sys.exit(1)
