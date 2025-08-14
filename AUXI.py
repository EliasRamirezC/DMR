# =============================================================================
# Dato Mata Relato – Dashboard (HOME + REPORTE)
# =============================================================================

# --- Core UI / datos / gráficos ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Visualización extra (nubes) ---
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Descargas/HTTP (si descargas de Drive o haces requests) ---
import gdown
from requests.exceptions import ChunkedEncodingError  # si lo usas en try/except

# --- Utilidades de la stdlib (puedes quitar las que no uses) ---
import os
import re
import ast
import hashlib
import time
import shutil
import unicodedata
import math
from collections import Counter, defaultdict

import os, shutil, time
import gdown

# -----------------------------
# CONFIGURACIÓN BÁSICA
# -----------------------------

st.set_page_config(page_title="Dato Mata Relato", layout="wide")

# -----------------------------
# CARGA DIRECTA DESDE DRIVE (IDs fijos)
# -----------------------------
FILE_ID_COV_ENT            = "1CN2IeRS6R23e9tLpPi5q5iZXAVKNw0CJ"
FILE_ID_COV_PROG           = "1uzGDFC3Qqd9PbfUJ1VVhPkCE5iq2ODgT"
FILE_ID_SUBTEMAS           = "19E98gR9Sggn-jia2ibXXae8EOGeF_lWZ"
FILE_ID_DF_CHUNKS          = "13pFYFGyFif4jUxgx_9Fn-3Elkdl1Kw4t"
FILE_ID_DF_CHUNKS_PROGRAMA = "1IHgcC7l4om8idmJD-ua-OFMYCYHRUSxs"
FILE_ID_SENT_ENT           = "1b7piWVqSs_tOfSYKDT7iNsBhDXmSwWUx"
FILE_ID_SENT_PROG          = "10bTAED0o6Zak6sW3SXUdILQIHokpOQdJ"
FILE_ID_SEEDS              = "1iEjSOzHITSEBTigXI6RoE5j8yyu4IxCG"

CACHE_DIR = ".cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# Mapeo de IDs → nombre local (para mantener consistente en disco)
DRIVE_FILES = {
    FILE_ID_COV_ENT:            "coverage_entrevistas_topm.parquet",
    FILE_ID_COV_PROG:           "coverage_programa_topm.parquet",
    FILE_ID_SUBTEMAS:           "df_subtemas.parquet",
    FILE_ID_DF_CHUNKS:          "df_chunks.parquet",
    FILE_ID_DF_CHUNKS_PROGRAMA: "df_chunks_programas.parquet",
    FILE_ID_SENT_ENT:           "sent_ent.parquet",
    FILE_ID_SENT_PROG:          "sent_prog.parquet",
    FILE_ID_SEEDS:              "seeds_semanticos.parquet",
}

# ---------- Utilidades ----------
def _local_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def _assert_min_size(path: str, min_bytes: int = 1024) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"Archivo inválido o demasiado pequeño: {path}")

@st.cache_data(show_spinner=False, ttl=3600)
def _load_parquet_cached(file_id: str, local_name: str) -> pd.DataFrame:
    """
    Función pura (sin st.*). Lee del disco si existe; si no, descarga y lee.
    Cachea el DataFrame para evitar recomputar en reruns.
    """
    dest = _local_path(local_name)

    # 1) Si ya existe en disco y abre, úsalo
    if os.path.exists(dest):
        return pd.read_parquet(dest)

    # 2) Descargar a .part y mover (atómico)
    tmp = dest + ".part"
    if os.path.exists(tmp):
        try: os.remove(tmp)
        except Exception: pass

    # gdown soporta pasar el id directo
    # (evitamos params extra para máxima compatibilidad)
    gdown.download(id=file_id, output=tmp, quiet=True)

    _assert_min_size(tmp)
    shutil.move(tmp, dest)

    # 3) Leer parquet
    return pd.read_parquet(dest)

def _safe_load(file_id: str, human_name: str) -> pd.DataFrame:
    """Wrapper con UI: spinner, fallback claro y stop si no se puede cargar."""
    local_name = DRIVE_FILES[file_id]
    try:
        with st.spinner(f"Cargando {human_name}…"):
            return _load_parquet_cached(file_id, local_name)
    except (ChunkedEncodingError, Exception) as e:
        # Mensaje de ayuda y posibilidad de carga manual
        url = f"https://drive.google.com/uc?id={file_id}"
        st.error(
            f"No pude cargar **{human_name}**.\n\n"
            f"1) Intenta recargar la página.\n"
            f"2) Si persiste, descarga manualmente desde:\n{url}\n"
            f"   y coloca el archivo en `{os.path.abspath(CACHE_DIR)}` "
            f"con el nombre **{local_name}**.\n\n"
            f"Detalle: {type(e).__name__}: {e}"
        )
        st.stop()

def _validate_sent_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    req = {"chunk_id", "stars", "star_score", "conf"}
    miss = req - set(df.columns)
    if miss:
        raise KeyError(f"[{name}] faltan columnas requeridas: {miss}")
    out = df.copy()
    out["chunk_id"] = out["chunk_id"].astype(str)
    for c in ("stars", "star_score", "conf"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop_duplicates(subset=["chunk_id"])

# ---------- Carga de todas las tablas ----------
coverage_entrevistas = _safe_load(FILE_ID_COV_ENT,   "Coverage ENT (top-m)")
coverage_programa    = _safe_load(FILE_ID_COV_PROG,  "Coverage PROG (top-m)")
df_subtemas          = _safe_load(FILE_ID_SUBTEMAS,  "Catálogo de subtemas")
df_chunks            = _safe_load(FILE_ID_DF_CHUNKS, "Chunks ENT (con embeddings)")
df_chunks_programas  = _safe_load(FILE_ID_DF_CHUNKS_PROGRAMA, "Chunks PROG (con embeddings)")
sent_ent             = _safe_load(FILE_ID_SENT_ENT,  "Sentimiento ENT (precomputado)")
sent_prog            = _safe_load(FILE_ID_SENT_PROG, "Sentimiento PROG (precomputado)")
seeds_df             = _safe_load(FILE_ID_SEEDS,     "Semillas semánticas")

# ---------- Normalizaciones mínimas ----------
# chunk_id como string en bases clave para joins/merges
for _df in (coverage_entrevistas, coverage_programa, df_chunks, df_chunks_programas):
    if "chunk_id" in _df.columns:
        _df["chunk_id"] = _df["chunk_id"].astype(str)

# Validar sentimiento precomputado (y tipar)
sent_ent  = _validate_sent_df(sent_ent,  "sent_ent")
sent_prog = _validate_sent_df(sent_prog, "sent_prog")

# -----------------------------
# FUNCIONES DE PROCESO Y GRAFICADO
# -----------------------------
def _check_cols(df, cols, name="df"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en {name}: {missing}")

def dedupe_chunk_lvl3(coverage_df: pd.DataFrame) -> pd.DataFrame:
    _check_cols(coverage_df, ["chunk_id","codigo_lvl3","sim_subtema"], "coverage_df")
    return (coverage_df.sort_values(['chunk_id','sim_subtema'], ascending=[True, False])
            .drop_duplicates(subset=['chunk_id','codigo_lvl3'], keep='first')
            .reset_index(drop=True))

def compute_w_tokens(coverage: pd.DataFrame, mode: str = "soft") -> pd.DataFrame:
    _check_cols(coverage, ["chunk_id","n_tokens_chunk","sim_subtema","codigo_lvl3"], "coverage")
    cov = coverage.copy()
    if mode == "soft":
        denom = cov.groupby("chunk_id")["sim_subtema"].transform(lambda x: x.clip(lower=1e-9).sum())
        w_local = cov["sim_subtema"].clip(lower=1e-9) / denom
    else:
        k = cov.groupby("chunk_id")["codigo_lvl3"].transform("count").clip(lower=1)
        w_local = 1.0 / k
    cov["w_tokens"] = cov["n_tokens_chunk"] * w_local
    return cov

def build_resumen_lvl3(coverage_df: pd.DataFrame,
                       df_subtemas: pd.DataFrame,
                       weight_col: str = "w_tokens") -> pd.DataFrame:
    _check_cols(coverage_df, ["candidato","codigo_lvl3","chunk_id","sim_subtema",weight_col], "coverage_df")
    _check_cols(df_subtemas, ["codigo_lvl3","tema_nivel3","codigo_lvl2","tema_nivel2","pilar"], "df_subtemas")
    g = (coverage_df.groupby(['candidato','codigo_lvl3'], as_index=False)
         .agg(peso=(weight_col, 'sum'),
              n_chunks=('chunk_id','nunique'),
              sim_mediana=('sim_subtema','median')))
    g['share'] = g['peso'] / g.groupby('candidato')['peso'].transform('sum').replace({0: np.nan})
    g['share'] = g['share'].fillna(0.0)
    temas = (df_subtemas[['codigo_lvl3','tema_nivel3','codigo_lvl2','tema_nivel2','pilar','horizonte_simple']]
             .drop_duplicates('codigo_lvl3'))
    g = g.merge(temas, on='codigo_lvl3', how='left')
    g['pilar'] = g['pilar'].fillna('Sin pilar')
    return g

def build_share_por_pilar_df(res_df: pd.DataFrame) -> pd.DataFrame:
    _check_cols(res_df, ["candidato","pilar","share"], "res_df")
    tmp = res_df.groupby(['candidato','pilar'], as_index=False)['share'].sum()
    tmp['share_norm'] = tmp['share'] / tmp.groupby('candidato')['share'].transform('sum').replace({0: np.nan})
    tmp['share_norm'] = tmp['share_norm'].fillna(0.0)
    return tmp

def _make_color_map(df_subtemas):
    _pilares = (df_subtemas['pilar'].dropna().unique().tolist()
                if 'pilar' in df_subtemas.columns else [])
    _palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    cmap = {p: _palette[i % len(_palette)] for i, p in enumerate(_pilares)}
    if 'Sin pilar' not in cmap:
        cmap['Sin pilar'] = _palette[max(0, len(_pilares) % len(_palette))]
    return cmap, _pilares

def plotly_share_por_pilar(res_df: pd.DataFrame, titulo: str,
                           df_subtemas: pd.DataFrame,
                           min_label: float = 0.03, height: int = 520,
                           candidate_order: list | None = None,
                           pillar_order: list | None = None):
    COLOR_MAP_PILAR, _pilares = _make_color_map(df_subtemas)
    df_plot = build_share_por_pilar_df(res_df).copy()
    if candidate_order is None:
        candidate_order = sorted(df_plot['candidato'].unique().tolist())
    if pillar_order is None:
        pillar_order = _pilares if _pilares else sorted(df_plot['pilar'].unique().tolist())
    df_plot['label'] = np.where(df_plot['share_norm'] >= min_label,
                                (df_plot['share_norm']*100).round(1).astype(str) + '%', '')
    fig = px.bar(
        df_plot, x='candidato', y='share_norm', color='pilar',
        category_orders={'candidato': candidate_order, 'pilar': pillar_order},
        color_discrete_map=COLOR_MAP_PILAR, barmode='stack',
        title=titulo, text='label'
    )
    fig.update_layout(
        height=height, template="plotly_white", bargap=0.25,
        yaxis_title="Share (%)", xaxis_title="Candidato",
        yaxis_tickformat=".0%", legend_title="Pilar", margin=dict(t=70, l=40, r=20, b=40)
    )
    fig.update_traces(
        textposition='inside', textfont_size=12, insidetextanchor='middle',
        hovertemplate="<b>%{x}</b><br>Pilar: %{legendgroup}<br>Share: %{y:.1%}<extra></extra>"
    )
    return fig

def build_share_por_horizonte_df(res_df: pd.DataFrame) -> pd.DataFrame:
    _check_cols(res_df, ["candidato","horizonte_simple","share"], "res_df")
    tmp = res_df.groupby(['candidato','horizonte_simple'], as_index=False)['share'].sum()
    denom = tmp.groupby('candidato')['share'].transform('sum').replace({0: np.nan})
    tmp['share_norm'] = (tmp['share'] / denom).fillna(0.0)
    return tmp

H_ORDER = ["CP", "MP", "LP", "Sin horizonte"]
H_COLORS = {
    "CP": "#4C78A8",
    "MP": "#72B7B2",
    "LP": "#F58518",
    "Sin horizonte": "#B0B0B0"
}

def plotly_share_por_horizonte(res_df: pd.DataFrame, titulo: str,
                               min_label: float = 0.03,
                               height: int = 520,
                               candidate_order: list | None = None,
                               horizon_order: list | None = None):
    df_plot = build_share_por_horizonte_df(res_df).copy()
    if candidate_order is None:
        candidate_order = sorted(df_plot['candidato'].unique().tolist())
    if horizon_order is None:
        seen = df_plot['horizonte_simple'].unique().tolist()
        horizon_order = [h for h in H_ORDER if h in seen] + [h for h in sorted(seen) if h not in H_ORDER]

    df_plot['label'] = np.where(
        df_plot['share_norm'] >= min_label,
        (df_plot['share_norm']*100).round(1).astype(str) + '%',
        ''
    )
    color_map = {h: H_COLORS.get(h, "#999999") for h in horizon_order}

    fig = px.bar(
        df_plot,
        x='candidato', y='share_norm', color='horizonte_simple',
        category_orders={'candidato': candidate_order, 'horizonte_simple': horizon_order},
        color_discrete_map=color_map, barmode='stack',
        title=titulo, text='label'
    )
    fig.update_layout(
        height=height, template="plotly_white", bargap=0.25,
        yaxis_title="Share (%)", xaxis_title="Candidato",
        yaxis_tickformat=".0%", legend_title="Horizonte",
        margin=dict(t=70, l=40, r=20, b=40)
    )
    fig.update_traces(
        textposition='inside', textfont_size=12, insidetextanchor='middle',
        hovertemplate="<b>%{x}</b><br>Horizonte: %{legendgroup}<br>Share: %{y:.1%}<extra></extra>"
    )
    return fig

def _map_h(h):
    if pd.isna(h): 
        return np.nan
    h = str(h).strip().upper()
    return {"CP-MP": "CP", "MP-LP": "MP"}.get(h, h)

def _unpad_blocks(code: str) -> str:
    if pd.isna(code):
        return code
    s = str(code).strip()
    parts = re.split(r'(\d+)', s)
    parts = [str(int(p)) if p.isdigit() else p for p in parts]
    return ''.join(parts)

def make_SUB_pivot(df_subtemas: pd.DataFrame) -> pd.DataFrame:
    S = df_subtemas.drop_duplicates("codigo_lvl3").copy()
    if "codigo_lvl3" in S.columns:
        S["codigo_lvl3"] = S["codigo_lvl3"].astype(str).str.strip()
    if "horizonte_simple" not in S.columns and "horizonte" in S.columns:
        S["horizonte_simple"] = S["horizonte"]
    if "horizonte_simple" not in S.columns:
        S["horizonte_simple"] = np.nan
    S["horizonte_simple"] = S["horizonte_simple"].map(_map_h)
    S.loc[~S["horizonte_simple"].isin({"CP","MP","LP"}), "horizonte_simple"] = np.nan
    S["key_unpadded"] = S["codigo_lvl3"].map(_unpad_blocks)
    return S[["codigo_lvl3","horizonte_simple","key_unpadded"]]

def ensure_horizonte_in_cov(cov: pd.DataFrame, SUB: pd.DataFrame) -> pd.DataFrame:
    cov = cov.copy()
    if "codigo_lvl3" in cov.columns:
        cov["codigo_lvl3"] = cov["codigo_lvl3"].astype(str).str.strip()

    if "horizonte_simple" in cov.columns:
        cov["horizonte_simple"] = cov["horizonte_simple"].map(_map_h)
        cov.loc[~cov["horizonte_simple"].isin({"CP","MP","LP"}), "horizonte_simple"] = np.nan
        return cov

    out = cov.merge(SUB[["codigo_lvl3","horizonte_simple"]], on="codigo_lvl3", how="left")
    if out["horizonte_simple"].notna().sum() == 0:
        cov["key_unpadded"] = cov["codigo_lvl3"].map(_unpad_blocks)
        SUB_key = SUB.drop_duplicates("key_unpadded")[["key_unpadded","horizonte_simple"]]
        out = cov.merge(SUB_key, on="key_unpadded", how="left").drop(columns=["key_unpadded"])

    if "horizonte_simple" not in out.columns:
        out["horizonte_simple"] = np.nan
    return out

def universe_totals(SUB: pd.DataFrame, H_ORDER=("CP","MP","LP")) -> pd.Series:
    tot = (SUB.dropna(subset=["horizonte_simple"])
              .groupby("horizonte_simple")["codigo_lvl3"].nunique())
    return tot.reindex(H_ORDER, fill_value=0)

def cobertura_pct_por_h(cov_h: pd.DataFrame, tot_h: pd.Series, H_ORDER=("CP","MP","LP")) -> pd.DataFrame:
    cand_list = sorted(cov_h["candidato"].dropna().astype(str).unique())
    if "horizonte_simple" not in cov_h.columns:
        cov_h = cov_h.copy(); cov_h["horizonte_simple"] = np.nan

    counts = (cov_h.dropna(subset=["horizonte_simple"])
                    .groupby(["candidato","horizonte_simple"])["codigo_lvl3"]
                    .nunique()
                    .unstack("horizonte_simple"))
    counts = (counts
              .reindex(index=cand_list, fill_value=0)
              .reindex(columns=H_ORDER, fill_value=0))

    denom = tot_h.reindex(H_ORDER).replace({0: np.nan})
    pct = 100 * counts.divide(denom, axis=1)
    pct = pct.fillna(0.0)
    return (pct.reset_index()
               .melt(id_vars="candidato", var_name="horizonte_simple", value_name="pct"))

def plot_cobertura_pct_simple(cob: pd.DataFrame, title: str, H_ORDER=("CP","MP","LP")):
    H_COLORS = {"CP":"#4C78A8","MP":"#72B7B2","LP":"#F58518"}
    cand_order = sorted(cob["candidato"].astype(str).unique())
    h_order    = [h for h in H_ORDER if h in cob["horizonte_simple"].unique()]
    cob = cob.copy()
    cob["pct_lbl"] = cob["pct"].map(lambda v: f"{v:.1f}%")

    fig = px.bar(
        cob, x="candidato", y="pct", color="horizonte_simple", text="pct_lbl",
        category_orders={"candidato": cand_order, "horizonte_simple": h_order},
        color_discrete_map=H_COLORS,
        barmode="group",
        title=title,
        labels={"pct":"% del universo cubierto","horizonte_simple":"Horizonte"}
    )
    fig.update_layout(template="plotly_white", yaxis_ticksuffix="%", bargap=0.25,
                      margin=dict(t=70, l=40, r=20, b=40))
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

def _truncate(s, maxlen=60):
    s = str(s) if s is not None else ""
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

def _format_val(v, usar):
    if usar == "share":
        try:
            return f"{100*float(v):.1f}%"
        except Exception:
            return ""
    else:
        try:
            return f"{float(v):,.0f}"
        except Exception:
            return ""

def _ensure_color_map_pilar(res_df, df_subtemas=None):
    if df_subtemas is not None and "pilar" in df_subtemas.columns:
        pilares = df_subtemas["pilar"].dropna().unique().tolist()
    else:
        pilares = res_df["pilar"].dropna().unique().tolist()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    cmap = {p: palette[i % len(palette)] for i, p in enumerate(sorted(pilares))}
    cmap.setdefault("Sin pilar", "#A0A0A0")
    return cmap

def plotly_topN_por_candidato_lvl3(
    res_df: pd.DataFrame,
    candidato: str,
    N: int = 8,
    usar: str = "share",
    max_label_len: int = 60,
    largest_at_top: bool = True,
    pilar_filter: list | None = None,
    horizonte_filter: list | None = None,
    df_subtemas: pd.DataFrame | None = None,
    show_legend: bool = True,
):
    _check_cols(res_df, ["candidato","tema_nivel3","pilar","share","peso","sim_mediana"], "res_df")
    dfc = res_df[res_df["candidato"].astype(str).str.strip() == str(candidato).strip()].copy()
    if pilar_filter is not None:
        dfc = dfc[dfc["pilar"].isin(pilar_filter)]
    if horizonte_filter is not None and "horizonte_simple" in dfc.columns:
        dfc = dfc[dfc["horizonte_simple"].isin(horizonte_filter)]
    if dfc.empty:
        raise ValueError(f"No hay datos para el candidato '{candidato}' con los filtros aplicados.")

    usar = usar if usar in {"share","peso"} else "share"
    dfc = dfc.sort_values(usar, ascending=False).head(N).copy()
    dfc["tema_nivel3_lbl"] = dfc["tema_nivel3"].map(lambda x: _truncate(x, max_label_len))
    color_map = _ensure_color_map_pilar(res_df, df_subtemas=df_subtemas)

    fig = px.bar(
        dfc, x=usar, y="tema_nivel3_lbl", color="pilar",
        color_discrete_map=color_map, orientation="h",
        title=f"Top {len(dfc)} temas (Nivel 3) de {candidato} por {usar}",
        text=dfc[usar].map(lambda v: _format_val(v, usar))
    )
    if usar == "share":
        fig.update_xaxes(tickformat=".0%"); hover_val = "%{x:.2%}"; x_title = "Share"
    else:
        hover_val = "%{x:,.0f}"; x_title = "Peso (tokens)"

    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title="Tema (Nivel 3)",
        bargap=0.15,
        legend=dict(orientation="h", y=-0.25, yanchor="bottom", x=0.5, xanchor="center",
                    font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        showlegend=show_legend,
        margin=dict(t=60, l=80, r=50, b=50)
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Pilar: %{marker.color}<br>Valor: " + hover_val + "<extra></extra>"
    )
    if largest_at_top:
        fig.update_yaxes(autorange="reversed")
    return fig

def _weighted_median(values, weights):
    v = np.asarray(values, dtype="float32"); w = np.asarray(weights, dtype="float32")
    if v.size == 0 or w.sum() == 0: return np.nan
    idx = np.argsort(v); v = v[idx]; w = w[idx]
    cw = np.cumsum(w)/w.sum()
    k = np.searchsorted(cw, 0.5); k = np.clip(k, 0, len(v)-1)
    return float(v[k])

def resumir_a_lvl2(res_lvl3: pd.DataFrame) -> pd.DataFrame:
    _check_cols(res_lvl3, ["candidato","codigo_lvl2","tema_nivel2","peso","n_chunks","sim_mediana"], "res_lvl3")
    g = (res_lvl3.groupby(["candidato","codigo_lvl2","tema_nivel2"], as_index=False)
         .agg(peso=("peso","sum"), n_chunks=("n_chunks","sum")))
    sm = (res_lvl3.groupby(["candidato","codigo_lvl2","tema_nivel2"])[["sim_mediana","peso"]]
          .apply(lambda df: _weighted_median(df["sim_mediana"], df["peso"]))
          .rename("sim_mediana").reset_index())
    out = g.merge(sm, on=["candidato","codigo_lvl2","tema_nivel2"], how="left")
    out["share"] = out["peso"] / out.groupby("candidato")["peso"].transform("sum").replace({0: np.nan})
    out["share"] = out["share"].fillna(0.0)
    return out

def plotly_topN_por_candidato_lvl2(
    res_lvl2: pd.DataFrame,
    candidato: str,
    N: int = 8,
    usar: str = "share",
    max_label_len: int = 60,
    largest_at_top: bool = True
):
    _check_cols(res_lvl2, ["candidato","tema_nivel2","share","peso","sim_mediana"], "res_lvl2")
    dfc = res_lvl2[res_lvl2["candidato"].astype(str).str.strip() == str(candidato).strip()].copy()
    if dfc.empty:
        raise ValueError(f"No hay datos para el candidato '{candidato}'.")
    usar = usar if usar in {"share","peso"} else "share"
    dfc = dfc.sort_values(usar, ascending=False).head(N).copy()
    dfc["tema_nivel2_lbl"] = dfc["tema_nivel2"].map(lambda x: _truncate(x, max_label_len))

    fig = px.bar(
        dfc, x=usar, y="tema_nivel2_lbl", orientation="h",
        title=f"Top {len(dfc)} temas (Nivel 2) de {candidato} por {usar}",
        text=dfc[usar].map(lambda v: _format_val(v, usar))
    )
    if usar == "share":
        fig.update_xaxes(tickformat=".0%"); hover_val = "%{x:.2%}"; x_title = "Share"
    else:
        hover_val = "%{x:,.0f}"; x_title = "Peso (tokens)"
    fig.update_layout(template="plotly_white", xaxis_title=x_title, yaxis_title="Tema (Nivel 2)", bargap=0.15, margin=dict(t=70, l=40, r=20, b=40))
    fig.update_traces(textposition="outside", hovertemplate="<b>%{y}</b><br>Valor: " + hover_val + "<extra></extra>")
    if largest_at_top:
        fig.update_yaxes(autorange="reversed")
    return fig

def top_low_share(res_nivel: pd.DataFrame, candidato: str, N: int = 8) -> pd.DataFrame:
    _check_cols(res_nivel, ["candidato","share"], "res_nivel")
    df_cand = res_nivel[res_nivel["candidato"].astype(str) == str(candidato)].copy()
    df_cand = df_cand.sort_values("share", ascending=True).head(N)
    return df_cand

def plotly_top_low_share(df_low: pd.DataFrame, nivel: str, titulo: str):
    df_low = df_low.copy()
    tema_col = "tema_nivel3" if nivel=="lvl3" else "tema_nivel2"
    df_low["tema_lbl"] = df_low[tema_col].astype(str).str.slice(0, 60)

    fig = px.bar(
        df_low,
        x="share", y="tema_lbl",
        orientation="h",
        title=titulo,
        text=df_low["share"].map(lambda v: f"{100*float(v):.1f}%")
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Share (%)",
        yaxis_title=f"Tema ({nivel.upper()})",
        bargap=0.15,
        margin=dict(t=60, b=90, l=60, r=60)
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Share: %{x:.2%}<extra></extra>"
    )
    fig.update_yaxes(autorange="reversed")
    return fig

# ==== Cobertura vs Diversidad: helpers ====
def _entropy_normalized(shares: np.ndarray) -> float:
    p = np.asarray(shares, dtype=float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    ent = -(p * np.log(p)).sum()
    return float(ent / np.log(len(p)))

def _hhi(shares: np.ndarray) -> float:
    p = np.asarray(shares, dtype=float)
    return float((p ** 2).sum())

def resumen_cobertura_diversidad(
    coverage_df: pd.DataFrame,
    df_subtemas: pd.DataFrame | None = None,
    origen_label: str = "ENT",
    weight_mode: str = "soft",
    use_pct_universe: bool = False
) -> pd.DataFrame:
    req = ["candidato","chunk_id","codigo_lvl3","n_tokens_chunk","sim_subtema"]
    _check_cols(coverage_df, req, "coverage_df")
    cov = coverage_df.copy()
    cov["candidato"] = cov["candidato"].astype(str).str.strip()
    cov["codigo_lvl3"] = cov["codigo_lvl3"].astype(str).str.strip()

    cov_nodup = dedupe_chunk_lvl3(cov)
    n_l3 = (cov_nodup.groupby("candidato")["codigo_lvl3"]
                   .nunique()
                   .rename("n_l3"))

    cov_w = compute_w_tokens(cov, mode=weight_mode)
    g = (cov_w.groupby(["candidato","codigo_lvl3"])["w_tokens"]
             .sum()
             .rename("peso")
             .reset_index())
    tot = g.groupby("candidato")["peso"].transform("sum").replace({0: np.nan})
    g["share"] = (g["peso"] / tot).fillna(0.0)

    ent = g.groupby("candidato")["share"].apply(lambda s: _entropy_normalized(s.values)).rename("entropia_norm")
    hhi = g.groupby("candidato")["share"].apply(lambda s: _hhi(s.values)).rename("HHI")
    vol = g.groupby("candidato")["peso"].sum().rename("vol_tokens")

    out = pd.concat([n_l3, ent, hhi, vol], axis=1).reset_index()
    out["origen"] = origen_label

    if use_pct_universe and df_subtemas is not None and "codigo_lvl3" in df_subtemas.columns:
        n_uni = df_subtemas["codigo_lvl3"].astype(str).str.strip().nunique()
        out["n_l3_universo"] = n_uni
        out["n_l3_pct"] = np.where(n_uni > 0, 100 * out["n_l3"] / n_uni, 0.0)

    return out

def plot_cobertura_vs_diversidad(df: pd.DataFrame,
                                 usar_pct_universo: bool = False,
                                 title: str = "Cobertura vs Diversidad (resumen ejecutivo)",
                                 size_max: int = 42):
    _check_cols(df, ["candidato","origen","n_l3","entropia_norm","vol_tokens"], "df")
    x_col = "n_l3_pct" if (usar_pct_universo and "n_l3_pct" in df.columns) else "n_l3"
    x_title = "% de L3 cubiertos" if x_col == "n_l3_pct" else "# de L3 cubiertos"

    fig = px.scatter(
        df, x=x_col, y="entropia_norm",
        size="vol_tokens", size_max=size_max,
        color="origen", symbol="origen", symbol_sequence=["circle","diamond"],
        text="candidato", hover_name="candidato",
        hover_data={"vol_tokens":":,.0f","HHI":":.3f","n_l3":True,"entropia_norm":":.3f"},
        template="plotly_white", title=title
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Diversidad (entropía normalizada 0–1)",
        legend_title="Origen",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_traces(textposition="top center")

    x_med = df[x_col].median()
    y_med = df["entropia_norm"].median()
    fig.add_hline(y=y_med, line_width=1, line_dash="dot", opacity=0.4)
    fig.add_vline(x=x_med, line_width=1, line_dash="dot", opacity=0.4)
    return fig

def _maybe_parse_vec(x, dtype="float32"):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None
    try:
        a = np.asarray(x, dtype=dtype)
        return a if (a.ndim==1 and a.size>0 and np.isfinite(a).all()) else None
    except Exception:
        return None

def _unit(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v/(n if n>0 else 1.0)

def _ensure_chunk_id(df, origen):
    out = df.copy()
    out["origen"] = origen
    if "chunk_id" not in out.columns:
        def _mkid(r):
            s = f"{r.get('candidato','')}|{origen}|{r.get('texto_chunk','')}"
            return hashlib.sha1(str(s).encode("utf-8")).hexdigest()
        out["chunk_id"] = out.apply(_mkid, axis=1)
    return out

def attach_pilar_siempre(cov: pd.DataFrame, df_subtemas: pd.DataFrame) -> pd.DataFrame:
    cat = df_subtemas[["codigo_lvl3","pilar"]].drop_duplicates("codigo_lvl3").copy()
    cat["codigo_lvl3"] = cat["codigo_lvl3"].astype(str).str.strip()
    cat["pilar"] = cat["pilar"].astype(str).str.strip().fillna("Sin pilar")
    out = cov.copy()
    out["codigo_lvl3"] = out["codigo_lvl3"].astype(str).str.strip()
    out = out.drop(columns=[c for c in out.columns if c.lower()=="pilar"], errors="ignore")
    out = out.merge(cat, on="codigo_lvl3", how="left")
    out["pilar"] = out["pilar"].fillna("Sin pilar")
    return out

def _attach_embeddings(coverage_df, base_df, emb_col):
    base = base_df[["chunk_id", emb_col]].copy()
    base[emb_col] = base[emb_col].apply(_maybe_parse_vec)
    base = base[base[emb_col].notna()].copy()
    cov  = coverage_df.merge(base, on="chunk_id", how="left")
    cov  = cov[cov[emb_col].notna()].copy()
    cov["emb_unit"] = cov[emb_col].apply(_unit)
    return cov

def _weighted_centroid(group, vec_col="emb_unit", w_col="w_tokens"):
    V = np.stack(group[vec_col].to_list(), axis=0)
    w = group[w_col].to_numpy().astype("float32")
    w = w/(w.sum() if w.sum()>0 else 1.0)
    c = (V * w[:,None]).sum(axis=0)
    return _unit(c)

def coherencia_ent_prog_por_pilar_topm(
    coverage_ent_topm: pd.DataFrame,
    coverage_prog_topm: pd.DataFrame,
    df_chunks: pd.DataFrame,
    df_prog: pd.DataFrame,
    df_subtemas: pd.DataFrame,
    emb_col_ent="embedding",
    emb_col_prog="embedding_programa",
    weight_mode="soft",
    min_chunks_l3=1,
    min_w_tokens_l3=0.0
):
    ENT = compute_w_tokens(attach_pilar_siempre(coverage_ent_topm,  df_subtemas), mode=weight_mode)
    PRO = compute_w_tokens(attach_pilar_siempre(coverage_prog_topm, df_subtemas), mode=weight_mode)

    base_ent  = _ensure_chunk_id(df_chunks, "ENT")
    base_prog = _ensure_chunk_id(df_prog,   "PROG")

    ENT = _attach_embeddings(ENT, base_ent,  emb_col_ent)
    PRO = _attach_embeddings(PRO, base_prog, emb_col_prog)

    C_ent = []
    for (cand, l3), g in ENT.groupby(["candidato","codigo_lvl3"]):
        c = _weighted_centroid(g); wsum = float(g["w_tokens"].sum()); n = int(g["chunk_id"].nunique())
        C_ent.append(dict(candidato=cand, codigo_lvl3=l3, centroid_ent=c, w_tokens_sum_ent=wsum, n_chunks_ent=n))
    C_ent = pd.DataFrame(C_ent)

    C_prog = []
    for (cand, l3), g in PRO.groupby(["candidato","codigo_lvl3"]):
        c = _weighted_centroid(g); wsum = float(g["w_tokens"].sum()); n = int(g["chunk_id"].nunique())
        C_prog.append(dict(candidato=cand, codigo_lvl3=l3, centroid_prog=c, w_tokens_sum_prog=wsum, n_chunks_prog=n))
    C_prog = pd.DataFrame(C_prog)

    M = C_ent.merge(C_prog, on=["candidato","codigo_lvl3"], how="inner")
    if min_chunks_l3 > 1 or min_w_tokens_l3 > 0:
        M = M[(M["n_chunks_ent"]>=min_chunks_l3) & (M["n_chunks_prog"]>=min_chunks_l3) &
              (M["w_tokens_sum_ent"]>=min_w_tokens_l3) & (M["w_tokens_sum_prog"]>=min_w_tokens_l3)]
    if M.empty:
        return pd.DataFrame(), pd.DataFrame()

    M["sim_ENT_PROG_L3"] = M.apply(lambda r: float(np.dot(_unit(r["centroid_ent"]), _unit(r["centroid_prog"]))), axis=1)

    cat = df_subtemas[["codigo_lvl3","pilar"]].drop_duplicates("codigo_lvl3")
    M = M.merge(cat, on="codigo_lvl3", how="left").fillna({"pilar":"Sin pilar"})
    M["w_pair_min"]  = np.minimum(M["w_tokens_sum_ent"], M["w_tokens_sum_prog"])
    M["w_pair_geom"] = np.sqrt(M["w_tokens_sum_ent"] * M["w_tokens_sum_prog"])

    rows = []
    for (cand, pil), g in M.groupby(["candidato","pilar"]):
        med   = float(np.median(g["sim_ENT_PROG_L3"]))
        wmed1 = _weighted_median(g["sim_ENT_PROG_L3"], g["w_pair_min"])
        wmed2 = _weighted_median(g["sim_ENT_PROG_L3"], g["w_pair_geom"])
        # 👉 añade soporte total por origen para ponderación global posterior
        rows.append(dict(
            candidato=cand, pilar=pil,
            sim_mediana=med,
            sim_mediana_w_min=wmed1,
            sim_mediana_w_geom=wmed2,
            w_tokens_ent=float(g["w_tokens_sum_ent"].sum()),
            w_tokens_prog=float(g["w_tokens_sum_prog"].sum()),
        ))
    P = pd.DataFrame(rows)
    return M, P

def plot_heatmap_coherencia(P: pd.DataFrame,
                            value_col="sim_mediana_w_min",
                            title="Coherencia ENT↔PROG por pilar",
                            annotate=True):
    if P.empty:
        return None
    piv = P.pivot(index="candidato", columns="pilar", values=value_col)
    row_order = piv.mean(axis=1).sort_values(ascending=False).index
    col_order = piv.mean(axis=0).sort_values(ascending=False).index
    piv = piv.reindex(index=row_order, columns=col_order)

    fig = px.imshow(
        piv,
        x=piv.columns, y=piv.index,
        color_continuous_scale=[(0.0, "#fde725"), (0.5, "#21918c"), (1.0, "#440154")],
        origin="upper", zmin=0, zmax=1,
        labels=dict(color="Similitud (cos)"),
        title=title
    )
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title="Similitud"),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(title="Pilar")
    fig.update_yaxes(title="Candidato")

    if annotate:
        for yi, yv in enumerate(piv.index):
            for xi, xv in enumerate(piv.columns):
                val = piv.iloc[yi, xi]
                if pd.notna(val):
                    txt_color = "white" if val >= 0.55 else "black"
                    fig.add_annotation(
                        x=xv, y=yv, text=f"{val:.2f}",
                        showarrow=False, font=dict(size=11, color=txt_color),
                        xanchor="center", yanchor="middle"
                    )
    return fig

def coherencia_global_bar(
    P: pd.DataFrame,
    value_col: str = "sim_mediana_w_geom",
    weighting: str = "common",          # 'common' | 'equal' | 'ent' | 'prog'
    ent_col: str = "w_tokens_ent",
    prog_col: str = "w_tokens_prog",
    show_plot: bool = False
):
    if "candidato" not in P.columns or value_col not in P.columns:
        raise KeyError(f"P debe tener columnas 'candidato' y '{value_col}'.")

    P2 = P.copy()
    have_ent  = ent_col  in P2.columns
    have_prog = prog_col in P2.columns
    wants_weight = weighting != "equal"

    if wants_weight and not (have_ent and have_prog):
        weighting = "equal"
        try:
            st.warning("No se encontraron columnas de soporte (w_tokens_ent / w_tokens_prog). "
                       "Se usará promedio simple por pilar.", icon="⚠️")
        except Exception:
            pass

    if weighting == "common" and have_ent and have_prog:
        P2["w"] = np.minimum(P2[ent_col].fillna(0.0), P2[prog_col].fillna(0.0))
    elif weighting == "ent" and have_ent:
        P2["w"] = P2[ent_col].fillna(0.0)
    elif weighting == "prog" and have_prog:
        P2["w"] = P2[prog_col].fillna(0.0)
    else:
        P2["w"] = 1.0

    P2 = P2.dropna(subset=[value_col]).copy()
    if weighting != "equal":
        P2 = P2[P2["w"] > 0]

    if P2.empty:
        raise ValueError("No hay filas válidas para calcular el promedio global.")

    def _avg(g):
        vals = g[value_col].to_numpy()
        if weighting == "equal":
            return float(np.mean(vals))
        w = g["w"].to_numpy()
        return float(np.average(vals, weights=w))

    agg = (P2.groupby("candidato", as_index=False)
             .apply(lambda g: pd.Series({"coherencia_global": _avg(g)}))
             .sort_values("coherencia_global", ascending=False)
             .reset_index(drop=True))

    titulo = ("Coherencia ENT↔PROG promedio por candidato " +
              ("(ponderada por soporte común)" if weighting != "equal" else "(promedio simple)"))
    fig = px.bar(
        agg, x="coherencia_global", y="candidato", orientation="h",
        text=agg["coherencia_global"].map(lambda v: f"{v:.2f}"),
        labels={"coherencia_global":"Coherencia promedio (0–1)", "candidato":"Candidato"},
        title=titulo, template="plotly_white"
    )
    fig.update_xaxes(range=[0,1])
    fig.update_traces(textposition="outside", cliponaxis=False)

    if show_plot:
        fig.show()

    return agg, fig

def similitud_a_centro_programa_por_mes_centrado(
    coverage_ent: pd.DataFrame,
    df_chunks: pd.DataFrame,
    df_prog: pd.DataFrame,
    emb_col_ent: str = 'embedding',
    emb_col_prog: str = 'embedding_programa',
    fecha_col: str = 'Fecha publicación'
):
    base_ent  = _ensure_chunk_id(df_chunks, origen="ENT")
    base_prog = _ensure_chunk_id(df_prog,   origen="PROG")

    P = base_prog[['candidato','chunk_id', emb_col_prog]].copy()
    P[emb_col_prog] = P[emb_col_prog].apply(_maybe_parse_vec)
    P = P[P[emb_col_prog].notna()].copy()
    if P.empty:
        raise ValueError("No hay embeddings válidos en df_prog para calcular centroides.")
    P['emb_unit'] = P[emb_col_prog].apply(_unit)
    Cprog = (P.groupby('candidato')['emb_unit']
               .apply(lambda s: _unit(np.stack(s.to_list(), axis=0).mean(axis=0)))
               .to_dict())

    if fecha_col not in base_ent.columns:
        raise KeyError(f"La columna de fecha '{fecha_col}' no existe en df_chunks.")
    ENT = coverage_ent[['chunk_id','candidato']].merge(
        base_ent[['chunk_id', fecha_col, emb_col_ent]],
        on='chunk_id', how='left'
    )
    ENT[emb_col_ent] = ENT[emb_col_ent].apply(_maybe_parse_vec)
    ENT = ENT[ENT[emb_col_ent].notna()].copy()
    if ENT.empty:
        raise ValueError("No hay embeddings válidos en ENT tras el merge.")
    ENT['emb_unit'] = ENT[emb_col_ent].apply(_unit)
    ENT['fecha'] = pd.to_datetime(ENT[fecha_col], errors='coerce')
    ENT = ENT.dropna(subset=['fecha'])

    ENT['sim_to_prog_center'] = ENT.apply(
        lambda r: float((r['emb_unit'] * Cprog.get(r['candidato'], r['emb_unit']*0)).sum()),
        axis=1
    )

    ENT['mes_period'] = ENT['fecha'].dt.to_period('M')
    ENT['mes_inicio'] = ENT['mes_period'].dt.to_timestamp(how='start')
    ENT['mes_fin']    = ENT['mes_period'].dt.to_timestamp(how='end')
    ENT['mes_centro'] = ENT['mes_inicio'] + (ENT['mes_fin'] - ENT['mes_inicio'])/2

    ts = (ENT.groupby(['candidato','mes_period'], as_index=False)
              .agg(sim_to_prog_center=('sim_to_prog_center','mean'),
                   mes_inicio=('mes_inicio','first'),
                   mes_fin=('mes_fin','first'),
                   mes_centro=('mes_centro','first')))

    if ts.empty:
        raise ValueError("No hay datos válidos tras agrupar por mes.")

    ts = ts.sort_values(['mes_period','candidato']).reset_index(drop=True)
    ts['mes_label'] = ts['mes_period'].dt.strftime('%m %y')
    xmin = ts['mes_inicio'].min()
    xmax = ts['mes_fin'].max()

    fig = px.line(
        ts, x='mes_centro', y='sim_to_prog_center', color='candidato', markers=True,
        title='Deriva semántica: similitud de ENT al centro del PROG (promedio mensual, punto centrado)',
        labels={'sim_to_prog_center':'Similitud coseno (↑ = más cerca del programa)',
                'mes_centro':'Mes'}
    )
    fig.update_layout(template='plotly_white')
    fig.update_yaxes(range=[0.4, 0.7])

    tickvals = (ts[['mes_period','mes_centro']]
                .drop_duplicates('mes_period')
                .sort_values('mes_period')['mes_centro'].tolist())
    ticktext = (ts[['mes_period','mes_label']]
                .drop_duplicates('mes_period')
                .sort_values('mes_period')['mes_label'].tolist())
    fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext, range=[xmin, xmax])
    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color="gray")

    return ts, fig

def vecindades_por_pilar(
    coverage_topm: pd.DataFrame,
    base_df: pd.DataFrame,
    df_subtemas: pd.DataFrame,
    emb_col: str,
    base_origen: str = "ENT",
    weight_mode: str = "soft",
    alpha: float = 0.8,
    beta: float  = 0.2,
    temperature: float = 0.25,
    min_chunks: int = 3
):
    if emb_col not in base_df.columns:
        raise KeyError(f"'{emb_col}' no existe en base_df.")

    base_df = _ensure_chunk_id(base_df, origen=base_origen).copy()

    cov = compute_w_tokens(coverage_topm.copy(), mode=weight_mode)
    cov = attach_pilar_siempre(cov, df_subtemas)
    cov["candidato"] = cov["candidato"].astype(str)

    w_chunk_pil = (cov.groupby(["chunk_id","candidato","pilar"], as_index=False)["w_tokens"].sum()
                     .rename(columns={"w_tokens":"w_chunk"}))

    base = base_df[["chunk_id", emb_col]].copy()
    base[emb_col] = base[emb_col].apply(_maybe_parse_vec)
    base = base[base[emb_col].notna()].copy()
    if base.empty:
        raise ValueError("base_df no tiene embeddings válidos tras parseo.")

    B = w_chunk_pil.merge(base, on="chunk_id", how="inner").dropna(subset=[emb_col])
    if B.empty:
        raise ValueError("No hay cruce entre coverage_topm y base_df por 'chunk_id' con embeddings.")

    B["emb_unit"] = B[emb_col].apply(_unit)

    groups, centroids = {}, {}
    for (cand, pil), g in B.groupby(["candidato","pilar"]):
        if g["chunk_id"].nunique() < min_chunks:
            continue
        w = g["w_chunk"].to_numpy("float32"); w = w/(w.sum() if w.sum()>0 else 1.0)
        E = np.vstack(g["emb_unit"].values)
        groups[(cand,pil)]    = (E, w)
        centroids[(cand,pil)] = _unit((E * w[:,None]).sum(axis=0))

    if not groups:
        raise ValueError("No hay grupos suficientes por pilar (sube datos o baja min_chunks).")

    res_tables, heatmaps = {}, {}
    pilares = sorted({pil for (_, pil) in groups.keys()})

    for pil in pilares:
        keys = sorted([k for k in groups.keys() if k[1]==pil])
        if len(keys) < 2:
            continue

        E_all, w_all = [], []
        for k in keys:
            E, w = groups[k]
            E_all.append(E); w_all.append(w)
        E_all = np.vstack(E_all); w_all = np.concatenate(w_all)
        w_all = w_all/(w_all.sum() if w_all.sum()>0 else 1.0)
        center_pil = _unit((E_all * w_all[:,None]).sum(axis=0))

        intens = {}
        cands = [k[0] for k in keys]
        for (cand, _p) in keys:
            c = centroids[(cand, pil)]
            intens[cand] = 1.0 - float(np.dot(c, center_pil))

        i_vals  = np.array(list(intens.values()), dtype="float32")
        i_range = float(i_vals.max() - i_vals.min()) if i_vals.size>0 else 1.0
        if i_range <= 1e-9: i_range = 1.0

        n = len(cands)
        S_cmp = np.zeros((n,n), dtype="float32")
        for i, ci in enumerate(cands):
            vi = centroids[(ci, pil)]
            for j, cj in enumerate(cands):
                vj = centroids[(cj, pil)]
                cos = float(np.dot(vi, vj))
                sim_cos = (cos + 1.0)/2.0
                sim_int = 1.0 - abs(intens[ci] - intens[cj]) / i_range
                S_cmp[i, j] = alpha*sim_cos + beta*sim_int

        Pm = np.zeros_like(S_cmp)
        tau = max(temperature, 1e-6)
        for i in range(n):
            logits = S_cmp[i].copy()
            logits[i] = -np.inf
            m = np.nanmax(logits)
            ex = np.exp((logits - m)/tau)
            ex[np.isinf(ex)] = 0.0
            s = ex.sum()
            Pm[i] = ex / (s if s>0 else 1.0)

        rows = []
        for i, ci in enumerate(cands):
            for j, cj in enumerate(cands):
                if i==j: continue
                rows.append(dict(
                    pilar=pil, source=ci, target=cj,
                    score=float(S_cmp[i,j]), prob=float(Pm[i,j])
                ))
        res_tables[pil] = (pd.DataFrame(rows)
                           .sort_values(["pilar","source","prob"], ascending=[True, True, False])
                           .reset_index(drop=True))
        heatmaps[pil]   = pd.DataFrame(S_cmp, index=cands, columns=cands)

    return res_tables, heatmaps

def plot_heatmap_vecindades(heatmap_df: pd.DataFrame, pilar: str):
    if heatmap_df.empty:
        raise ValueError("Heatmap vacío.")
    fig = px.imshow(
        heatmap_df, x=heatmap_df.columns, y=heatmap_df.index,
        color_continuous_scale="YlGn", origin="upper", zmin=0, zmax=1,
        labels=dict(color="Score (0–1)"),
        title=f"Vecindades por pilar — {pilar} (score compuesto)"
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, l=60, r=20, b=40))
    for yi, yv in enumerate(heatmap_df.index):
        for xi, xv in enumerate(heatmap_df.columns):
            val = heatmap_df.iloc[yi, xi]
            if np.isfinite(val):
                fig.add_annotation(x=xv, y=yv, text=f"{val:.2f}",
                                   showarrow=False, font=dict(size=11, color="black"),
                                   xanchor="center", yanchor="middle")
    fig.update_xaxes(title="Se parece a →")
    fig.update_yaxes(title="Candidato origen")
    return fig

# ================================================
# HELPERS — Sentimiento (parquet precomputado)
# ================================================
def _validate_sent_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    req = {"chunk_id","stars","star_score","conf"}
    miss = req - set(df.columns)
    if miss:
        raise KeyError(f"[{name}] faltan columnas requeridas: {miss}")
    out = df.copy()
    out["chunk_id"] = out["chunk_id"].astype(str)
    for c in ("stars","star_score","conf"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(subset=["chunk_id"])
    return out

def _merge_precomputed_sent(cov_base: pd.DataFrame, sent_df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "chunk_id" not in cov_base.columns:
        raise KeyError("cov_base no tiene 'chunk_id'.")
    cov = cov_base.copy()
    cov["chunk_id"] = cov["chunk_id"].astype(str)

    before = cov["chunk_id"].nunique()
    cov = cov.merge(sent_df[["chunk_id","stars","star_score","conf"]], on="chunk_id", how="left")

    matched_chunks = cov.dropna(subset=["stars"])["chunk_id"].nunique()
    coverage_pct = 100.0 * matched_chunks / max(1, before)
    if coverage_pct < 60:
        st.warning(
            f"Solo {coverage_pct:.1f}% de los chunks tienen sentimiento adjunto para {name}. "
            "Revisa que el parquet precomputado corresponda al mismo coverage.",
            icon="⚠️"
        )
    return cov

def _agg_sent_por_pilar(df_cov_sent: pd.DataFrame,
                        weight_by_tokens: bool = False,
                        conf_min: float = 0.0) -> pd.DataFrame:
    need = {"chunk_id","candidato","pilar","n_tokens_chunk","stars","star_score","conf"}
    miss = need - set(df_cov_sent.columns)
    if miss:
        raise KeyError(f"Faltan columnas para agregación: {miss}")

    df = df_cov_sent[df_cov_sent["conf"] >= float(conf_min)].copy()

    if weight_by_tokens:
        w = df["n_tokens_chunk"].clip(lower=1)
        def wavg(x, ww):
            x = np.asarray(x, float); ww = np.asarray(ww, float)
            m = np.isfinite(x) & np.isfinite(ww)
            return float(np.average(x[m], weights=ww[m])) if m.any() else np.nan
        agg = (df.groupby(["candidato","pilar"], as_index=False)
                 .agg(
                     n_chunks=("chunk_id","nunique"),
                     tokens=("n_tokens_chunk","sum"),
                     stars_mean=("stars",       lambda x: wavg(x, w.loc[x.index])),
                     score_mean=("star_score",  lambda x: wavg(x, w.loc[x.index])),
                     conf_mean=("conf",         lambda x: wavg(x, w.loc[x.index])),
                     pct_1_2=("stars",          lambda x: float(wavg((x<=2).astype(float), w.loc[x.index]))),
                     pct_3  =("stars",          lambda x: float(wavg((x==3).astype(float), w.loc[x.index]))),
                     pct_4_5=("stars",          lambda x: float(wavg((x>=4).astype(float), w.loc[x.index]))),
                 ))
    else:
        agg = (df.groupby(["candidato","pilar"], as_index=False)
                 .agg(
                     n_chunks=("chunk_id","nunique"),
                     tokens=("n_tokens_chunk","sum"),
                     stars_mean=("stars","mean"),
                     score_mean=("star_score","mean"),
                     conf_mean=("conf","mean"),
                     pct_1_2=("stars", lambda x: float((x<=2).mean())),
                     pct_3  =("stars", lambda x: float((x==3).mean())),
                     pct_4_5=("stars", lambda x: float((x>=4).mean())),
                 ))
    agg["SBI"] = agg["pct_4_5"] - agg["pct_1_2"]
    return agg

def _heatmap_sent(df: pd.DataFrame, value_col: str, title: str):
    piv = df.pivot(index="candidato", columns="pilar", values=value_col)
    row_order = piv.mean(axis=1, skipna=True).sort_values(ascending=False).index
    col_order = piv.mean(axis=0, skipna=True).sort_values(ascending=False).index
    piv = piv.reindex(index=row_order, columns=col_order)

    fig = px.imshow(
        piv, text_auto=".2f", color_continuous_scale="RdYlGn",
        origin="upper", labels=dict(color=value_col), title=title
    )
    if value_col == "SBI":
        fig.update_traces(zmin=-1, zmax=1, zauto=False)
    fig.update_layout(template="plotly_white", margin=dict(t=60,l=60,r=20,b=40))
    return fig

def _agg_sent_by_levels(df_cov_sent: pd.DataFrame, levels: list) -> pd.DataFrame:
    need = {"chunk_id","candidato","n_tokens_chunk","stars","star_score","conf"}
    miss = need - set(df_cov_sent.columns)
    if miss:
        raise KeyError(f"Faltan columnas para agregación por nivel: {miss}")

    df = df_cov_sent.copy()
    agg = (df.groupby(levels, as_index=False)
             .agg(
                 n_chunks=("chunk_id","nunique"),
                 tokens=("n_tokens_chunk","sum"),
                 stars_mean=("stars","mean"),
                 score_mean=("star_score","mean"),
                 pct_1_2=("stars", lambda x: float((x<=2).mean())),
                 pct_3  =("stars", lambda x: float((x==3).mean())),
                 pct_4_5=("stars", lambda x: float((x>=4).mean())),
             ))
    agg["SBI"] = agg["pct_4_5"] - agg["pct_1_2"]
    return agg

def _truncate_label(s: str, maxlen: int = 60) -> str:
    s = str(s) if s is not None else ""
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

def _xrange_for_metric(metric: str):
    if metric in ("SBI","score_mean"):
        return [-1, 1]
    if metric == "stars_mean":
        return [1, 5]
    return None

def _make_top_bar_smart(df, metric, positive=True, title="", x_range=None):
    d = df.sort_values(metric).copy()
    y_vals = d["tema_lbl"].tolist()
    x_vals = d[metric].astype(float).tolist()
    texts  = [f"{v:.2f}" for v in x_vals]

    fig = go.Figure()
    if positive:
        fig.add_bar(
            y=y_vals, x=x_vals, orientation="h",
            text=texts, textposition="outside",
            marker=dict(color="#2CA02C", line=dict(width=0.5, color="rgba(0,0,0,.25)")),
            hovertemplate="<b>%{y}</b><br>"+metric+": %{x:.2f}<extra></extra>",
            cliponaxis=False
        )
    else:
        fig.add_bar(
            y=y_vals, x=x_vals, orientation="h",
            text=texts, textposition="inside", insidetextanchor="start",
            marker=dict(color="#D62728", line=dict(width=0.5, color="rgba(0,0,0,.25)")),
            hovertemplate="<b>%{y}</b><br>"+metric+": %{x:.2f}<extra></extra>",
            cliponaxis=False
        )

    if x_range:
        fig.update_xaxes(range=x_range)
    if metric in ("SBI","score_mean"):
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")

    if metric in ("SBI","score_mean"):
        fig.update_xaxes(dtick=0.5)
    elif metric == "stars_mean":
        fig.update_xaxes(dtick=1)

    fig.update_layout(
        title=title, title_x=0.02,
        template="plotly_white",
        height=420,
        margin=dict(t=58, l=110, r=20, b=40),
        xaxis_title=metric, yaxis_title=None,
        xaxis=dict(gridcolor="rgba(0,0,0,.08)"),
        font=dict(size=12),
        showlegend=False
    )
    return fig

def _seed_centroids(seeds_df: pd.DataFrame) -> dict:
    need = {"axis","embedding_seed"}
    if not need.issubset(seeds_df.columns):
        raise KeyError(f"seeds_df requiere columnas {need}")
    S = seeds_df.copy()
    S["embedding_seed"] = S["embedding_seed"].apply(_maybe_parse_vec)
    S = S[S["embedding_seed"].notna()]
    if S.empty: raise ValueError("Semillas sin embeddings válidos.")
    cents = {}
    for ax, g in S.groupby("axis"):
        V = np.stack([_unit(v) for v in g["embedding_seed"].to_list()], axis=0)
        cents[ax] = _unit(V.mean(axis=0))
    for a in ["pop","inst","tec","vago"]:
        if a not in cents: raise ValueError(f"Falta axis '{a}' en seeds_df.")
    return cents

def _pair_softmax(a, b, tau=10.0):
    exps = np.exp(tau*np.array([a, b], dtype="float64"))
    Z = exps.sum()
    return float(exps[0]/Z) if Z>0 else 0.5

def _ensure_w_tokens(cov: pd.DataFrame) -> pd.DataFrame:
    cov = cov.copy()
    if "w_tokens" in cov.columns and np.isfinite(cov["w_tokens"]).any(): 
        return cov
    if "sim_subtema" in cov.columns:
        denom = cov.groupby("chunk_id")["sim_subtema"].transform(lambda s: s.clip(lower=1e-9).sum())
        wloc  = cov["sim_subtema"].clip(lower=1e-9) / denom
    else:
        k = cov.groupby("chunk_id")["codigo_lvl3"].transform("count").clip(lower=1)
        wloc = 1.0 / k
    cov["w_tokens"] = cov["n_tokens_chunk"] * wloc
    return cov

def align_embeddings_with_coverage(coverage: pd.DataFrame, df_chunks: pd.DataFrame, emb_col="embedding"):
    if emb_col not in df_chunks.columns:
        raise KeyError(f"'{emb_col}' no está en df_chunks.")
    base = df_chunks.copy()
    base[emb_col] = base[emb_col].apply(_maybe_parse_vec)
    base = base[base[emb_col].notna()].copy()

    # “spine” desde coverage
    map_ids = coverage[["chunk_id","candidato","texto_chunk"]].drop_duplicates()
    if "chunk_id" not in base.columns or base["chunk_id"].isna().any():
        if {"candidato","texto_chunk"}.issubset(base.columns):
            base = base.merge(map_ids, on=["candidato","texto_chunk"], how="left", suffixes=("","_cov"))
            if "chunk_id_cov" in base.columns:
                base["chunk_id"] = base["chunk_id"].fillna(base["chunk_id_cov"])
                base.drop(columns=["chunk_id_cov"], inplace=True)

    out = coverage.merge(base[["chunk_id", emb_col]], on="chunk_id", how="left")
    if out[emb_col].notna().sum() == 0:
        raise RuntimeError("No se alinearon embeddings: revisa que 'texto_chunk' sea idéntico en coverage y df_chunks.")
    return out.dropna(subset=[emb_col]).copy()

def _aggregate_weighted(df: pd.DataFrame, keys, cols=("POP_INDEX","TEC_INDEX","INST_INDEX","VAGO_INDEX")) -> pd.DataFrame:
    X = df.copy()
    for c in cols:
        X[f"{c}_w"] = X[c] * X["w_tokens"]
    agg = X.groupby(keys, as_index=False).agg(
        w_total=("w_tokens","sum"),
        n_chunks=("chunk_id","nunique"),
        **{f"sum_{c}": (f"{c}_w","sum") for c in cols}
    )
    for c in cols:
        agg[c] = agg[f"sum_{c}"] / agg["w_total"].where(agg["w_total"]>0, np.nan)
        agg.drop(columns=[f"sum_{c}"], inplace=True)
    return agg

# ===== Helpers para nubes de palabras (ES) =====
def _norm_basic(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

@st.cache_resource
def build_stop_es(extra_muletillas=None, drop_domain_words=False, extra_stop=None):
    stops = set()
    try:
        from nltk.corpus import stopwords
        try:
            _ = stopwords.words("spanish")
        except LookupError:
            import nltk; nltk.download("stopwords", quiet=True)
        stops |= set(stopwords.words("spanish"))
    except Exception:
        pass
    try:
        from stop_words import get_stop_words
        stops |= set(get_stop_words("spanish"))
    except Exception:
        pass

    muletillas = {
        "eh","ehh","mmm","este","esto","osea","o sea","oseaque","osea que",
        "bueno","ok","vale","digamos","tipo","nada","cosas","tema","temas",
        "porfa","por favor","ya","a ver","en fin","pues","ehmm","aja","ajá",
        "entonces","ademas","además","igual","cosa","cosita","cositas"
    }
    if extra_muletillas: muletillas |= set(extra_muletillas)

    domain_words = set()
    if drop_domain_words:
        domain_words |= {"gobierno","estado","pais","país","bolivia","boliviano","boliviana","bolivianos","bolivianas"}
    if extra_stop: domain_words |= set(extra_stop)

    STOP = {_norm_basic(w) for w in (stops | muletillas | domain_words)}
    STOP = {w for w in STOP if len(w) >= 2}
    return STOP

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
MENT_RE  = re.compile(r"[@#]\w+")
NUM_RE   = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)

def tokenize_es(text: str, STOP: set, min_len=3):
    s = (text or "")
    s = URL_RE.sub(" ", s)
    s = MENT_RE.sub(" ", s)
    s = NUM_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _norm_basic(s)

    toks = TOKEN_RE.findall(s)
    toks = [t for t in toks if len(t) >= min_len and t not in STOP]
    toks = [toks[i] for i in range(len(toks)) if i == 0 or toks[i] != toks[i-1]]
    return toks

def _build_chunk_pilar_weights(cov: pd.DataFrame) -> pd.DataFrame:
    total = cov.groupby("chunk_id")["w_tokens"].transform("sum").clip(lower=1e-9)
    cov = cov.assign(w_frac = cov["w_tokens"] / total)
    w = (cov.groupby(["chunk_id","candidato","pilar"], as_index=False)
            .agg(w_frac=("w_frac","sum"),
                 texto_chunk=("texto_chunk","first"),
                 n_tokens_chunk=("n_tokens_chunk","first")))
    return w

def build_group_counters(cov: pd.DataFrame,
                         min_group_support_tokens: float = 30.0,
                         min_len_token: int = 3) -> tuple[dict, dict]:
    STOP = build_stop_es(extra_muletillas={"digo","esteee"}, drop_domain_words=True)
    chunk_tokens = {}
    for _, row in cov[["chunk_id","texto_chunk"]].drop_duplicates("chunk_id").iterrows():
        toks = tokenize_es(row["texto_chunk"], STOP=STOP, min_len=min_len_token)
        chunk_tokens[row["chunk_id"]] = Counter(toks)

    wp = _build_chunk_pilar_weights(cov)
    counters = defaultdict(Counter)
    support  = Counter()

    for _, r in wp.iterrows():
        cid = r["chunk_id"]; cand = r["candidato"]; pil = r["pilar"]; frac = float(r["w_frac"])
        cnt = chunk_tokens.get(cid)
        if not cnt: continue
        for t, c in cnt.items():
            counters[(cand, pil)][t] += c * frac
        support[(cand, pil)] += float(r.get("n_tokens_chunk", 0.0)) * frac

    counters = {k:v for k,v in counters.items() if support[k] >= float(min_group_support_tokens)}
    support  = {k:v for k,v in support.items()  if v >= float(min_group_support_tokens)}
    return counters, support

def counters_to_tfidf(counters: dict) -> dict:
    """TF-IDF suave por grupo para resaltar términos distintivos."""
    groups = list(counters.keys())
    N = len(groups)
    df = Counter()
    for g in groups:
        df.update(counters[g].keys())
    #            ↓↓↓ aquí estaba el error: dfc_t -> df_t
    idf = {t: math.log((1.0 + N) / (1.0 + df_t)) + 1.0 for t, df_t in df.items()}
    tfidf = {g: {t: c * idf[t] for t, c in counters[g].items()} for g in groups}
    return tfidf

def _wc_from_freq(freqs: dict, width=900, height=600, max_words=120):
    wc = WordCloud(width=width, height=height, background_color="white",
                   max_words=max_words, prefer_horizontal=0.9, collocations=False)
    return wc.generate_from_frequencies(freqs)

# -----------------------------
# PREPARACIÓN DE DATOS (sin parámetros)
# -----------------------------
for _cov in (coverage_entrevistas, coverage_programa):
    if 'candidato' in _cov.columns:
        _cov['candidato'] = _cov['candidato'].astype(str).str.strip()

if 'horizonte_simple' not in df_subtemas.columns and 'horizonte' in df_subtemas.columns:
    df_subtemas = df_subtemas.copy()
    df_subtemas['horizonte_simple'] = df_subtemas['horizonte']

coverage_ent_nodup  = dedupe_chunk_lvl3(coverage_entrevistas)
coverage_prog_nodup = dedupe_chunk_lvl3(coverage_programa)

SUB_pivot = make_SUB_pivot(df_subtemas)  # pivote con horizonte_simple limpio
H_ORDER = ("CP","MP","LP")
TOT_H   = universe_totals(SUB_pivot, H_ORDER)

ENT_h = ensure_horizonte_in_cov(coverage_ent_nodup, SUB_pivot)
PRO_h = ensure_horizonte_in_cov(coverage_prog_nodup, SUB_pivot)

ent_pct_univ  = cobertura_pct_por_h(ENT_h, TOT_H, H_ORDER)
prog_pct_univ = cobertura_pct_por_h(PRO_h, TOT_H, H_ORDER)

coverage_ent_w  = compute_w_tokens(coverage_ent_nodup, mode='soft')
coverage_prog_w = compute_w_tokens(coverage_prog_nodup, mode='soft')

res_ent_lvl3  = build_resumen_lvl3(coverage_ent_w,  df_subtemas, weight_col='w_tokens')
res_prog_lvl3 = build_resumen_lvl3(coverage_prog_w, df_subtemas, weight_col='w_tokens')

res_exec_ent = resumen_cobertura_diversidad(
    coverage_entrevistas, df_subtemas=df_subtemas,
    origen_label="Entrevistas", weight_mode="soft", use_pct_universe=False
)
res_exec_prog = resumen_cobertura_diversidad(
    coverage_programa, df_subtemas=df_subtemas,
    origen_label="Programa", weight_mode="soft", use_pct_universe=False
)
res_exec = pd.concat([res_exec_ent, res_exec_prog], ignore_index=True)

res_ent_lvl2  = resumir_a_lvl2(res_ent_lvl3)
res_prog_lvl2 = resumir_a_lvl2(res_prog_lvl3)

# Ejemplo ENT:
cov_ent = attach_pilar_siempre(coverage_entrevistas, df_subtemas)
cov_ent = _merge_precomputed_sent(cov_ent, sent_ent, name="ENT")

# Ejemplo PROG:
cov_prog = attach_pilar_siempre(coverage_programa, df_subtemas)
cov_prog = _merge_precomputed_sent(cov_prog, sent_prog, name="PROG")

seeds_df["embedding_seed"] = seeds_df["embedding_seed"].apply(_maybe_parse_vec)

# =============================
# CONFIG INICIAL (opcional)
# =============================
st.set_page_config(page_title="Dato Mata Relato", page_icon="📊", layout="wide")

# =============================
# NAV: sidebar
# =============================
page = st.sidebar.radio("Navegación", ["HOME", "REPORTE"], index=0)

# =============================
# HOME (presentación + descripción completa)
# =============================
def render_home():
    # ---------- HERO ----------
    title = "📊 Dato Mata Relato"
    subtitle = "¿Qué dicen, cómo lo dicen y cuán alineados están?"  # cambia si quieres
    st.markdown(f"<h1 style='margin-bottom:0'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:#374151; font-size:1.1rem; margin-top:0.35rem;'>{subtitle}</div>",
        unsafe_allow_html=True
    )

    # En breve (antes TL;DR)
    st.markdown("### En breve")
    st.markdown(
        "✅ **Compara** entrevistas (**discurso**) vs **programas** con una taxonomía MECE  \n"
        "✅ **Mide** enfoque temporal (**CP/MP/LP**), **cobertura** de subtemas, **sentimiento** y **estilo**  \n"
        "✅ **Evalúa** la **coherencia** ENT↔PROG y cómo **cambia en el tiempo**"
    )

    # Fila de 3 bullets con iconos
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🎙️ **ASR + diarización**  \n*Solo el candidato*")
    with c2:
        st.markdown("🧠 **Embeddings + MECE**  \n*Alinear* *chunk → subtema*")
    with c3:
        st.markdown("📊 **KPIs**  \nShare CP/MP/LP · Cobertura · Sentimiento · Estilo · Coherencia")

    st.write("---")

    # Chips
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("🗓️ **Periodo**  \nFeb → inicios de Ago")
    with c2:
        st.markdown("🧩 **Unidad**  \n*Chunk* (200–300 tok)")
    with c3:
        st.markdown("⏳ **Horizontes**  \nCP · MP · LP")
    with c4:
        st.markdown("🔁 **Comparación**  \nENT ↔ PROG")
        
    st.divider()

    # Objetivo y Por qué
    st.subheader("Objetivo")
    st.markdown(
        "Encontrar **patrones** en el discurso, **correlacionarlos** con entrevistas y "
        "**compararlos** con los **programas de gobierno**: qué temas se priorizan, con qué "
        "**sentimiento**, qué **estilo** predomina (populismo/tecnicismo/vaguedad/institucionalismo) "
        "y **cuán alineado** está el discurso con el programa en el tiempo."
    )

    st.subheader("¿Por qué este proyecto?")
    st.markdown(
        "- No podemos **escuchar todo**: hay ruido (ataques, anécdotas, eslóganes).\n"
        "- Priorizamos lo **propositivo**: promesas, medidas, metas, plazos, verbos de acción.\n"
        "- La matriz pivote **MECE** permite **comparar manzanas con manzanas** entre candidatos y en el tiempo.\n"
        "- Evitamos clustering 100% automático (miles de microtemas) y usamos una **taxonomía curada**."
    )

    st.divider()

    # Matriz pivote (MECE)
    st.subheader("Matriz pivote (ejes y pilares)")
    st.markdown(
        "**Esfera → Pregunta clave → Pilar (MECE)**\n\n"
        "- **Económica** — ¿Cómo crecer con estabilidad y diversificar la producción?\n"
        "  **Pilares**: 1) Estabilidad macroeconómica; 2) Transformación productiva y digital; "
        "3) Gobernanza de recursos; 4) Sistema financiero e inclusión\n"
        "- **Social** — ¿Cómo mejorar salud, educación, empleo y protección social?\n"
        "  **Pilar**: 5) Capital humano y bienestar social\n"
        "- **Ambiental-climática** — ¿Cómo gestionar recursos, riesgos y transición verde?\n"
        "  *(Eje transversal articulado con Gobernanza de recursos)*\n"
        "- **Institucional** — ¿Cómo garantizar Estado de Derecho, transparencia y seguridad?\n"
        "  **Pilar**: 6) Estado de Derecho y gobernabilidad\n"
        "- **Territorial-global** — ¿Cómo integrar regiones, gestionar fronteras y proyectarse al mundo?\n"
        "  **Pilar**: 7) Cohesión territorial e integración global\n\n"
        "**Principio MECE** — *Mutuamente excluyente* y *colectivamente exhaustiva*."
    )
    with st.expander("Ver pilares (lista rápida)"):
        st.markdown(
            "1) Estabilidad macroeconómica  \n"
            "2) Transformación productiva y digital  \n"
            "3) Gobernanza de recursos  \n"
            "4) Sistema financiero e inclusión  \n"
            "5) Capital humano y bienestar social  \n"
            "6) Estado de Derecho y gobernabilidad  \n"
            "7) Cohesión territorial e integración global"
        )

    st.divider()

    # Inputs y alcance
    st.subheader("Inputs (datos de origen) y alcance")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(
            "- **Entrevistas**: Febrero → primeros de agosto; transcritas con ASR (p.ej., Turboscribe).\n"
            "- **Programas**: textos oficiales del Órgano Electoral (limpios/estructurados)."
        )
    with c6:
        st.markdown(
            "**Incluye:** intervenciones **propositivas** del candidato.  \n"
            "**Excluye:** preguntas del entrevistador, ataques personales, menciones a terceros, chistes."
        )
    st.info("**Unidad de análisis:** *chunk* (≈200–300 tokens) mapeado a **subtema L3** por candidato.")

    st.divider()

    # Flujo de procesamiento
    st.subheader("Flujo de procesamiento (punta a punta)")
    st.markdown(
        "1) **Ingesta & normalización** → 2) **Diarización** → 3) **Filtro propositivo** → "
        "4) **Chunking** → 5) **Embeddings** → 6) **Matriz pivote** → "
        "7) **Matching** (coseno, umbral ~0.50, dedupe, pesos) → 8) **KPIs/Gráficos**"
    )
    st.code(
        "Audio/Programas → Transcripción/Limpieza → Diarización → Filtro propositivo → Chunking "
        "→ Embeddings → Matching con matriz pivote → Dedupe/Pesos → KPIs/Gráficos",
        language="text"
    )

    st.divider()

    # Métricas
    st.subheader("Métricas y tableros")
    st.markdown(
        "- **Share por horizonte (CP/MP/LP)** — intensidad **dentro** del candidato.\n"
        "- **Cobertura del universo** — % de **subtemas distintos** tocados.\n"
        "- **Cobertura por pilar/subtema** — distribución y ranking.\n"
        "- **Top frases** — evidencia textual.\n"
        "- **Evolución temporal** — cambios por entrevista/fecha.\n"
        "- **Coherencia ENT ↔ PROG** — convergencia/divergencia por horizonte/pilar/subtema."
    )

    # Sentimiento
    st.subheader("Análisis de sentimiento")
    st.markdown(
        "- **Léxico** base (negativo/neutral/positivo o 1–5) + opción de **modelo preentrenado (HuggingFace)**.  \n"
        "Resultados por **subtema**, **pilar** y **candidato** (ENT y PROG)."
    )

    # Estilo
    st.subheader("Indicadores de estilo discursivo")
    st.markdown(
        "- **IP** (populismo−institucional) y **IT** (técnico−vago).  \n"
        "- **PTS (−1…1)**:  \\( PTS = IT - |IP| \\) — alto = técnico sin populismo fuerte.  \n"
        "Estandarización robusta por candidato (mediana/IQR)."
    )

    st.divider()

    # Cómo leer
    st.subheader("Cómo leer los gráficos")
    st.markdown(
        "**Share** = intensidad; **Cobertura** = amplitud; **Sentimiento/Estilo** = tono; "
        "**Coherencia** = ENT↔PROG y su cambio temporal."
    )

    st.divider()

    # Calidad y límites
    st.subheader("Calidad y validación")
    st.markdown(
        "- Muestra etiquetada para estimar **precisión/recall** y calibrar umbral.  \n"
        "- **Sanity checks**: outliers de similitud, horizonte faltante, duplicados.  \n"
        "- Revisión de ASR/diarización en fragmentos dudosos."
    )
    st.subheader("Limitaciones")
    st.markdown(
        "- Posibles errores de ASR/diarización.  \n"
        "- Umbral de similitud afecta recall/precision.  \n"
        "- Ironía/ambigüedad política: mitigada con dedupe, pesos y agregación."
    )

    st.divider()

    # Reproducibilidad
    st.subheader("Datos mínimos y reproducibilidad")
    with st.expander("Diccionario mínimo de columnas"):
        st.markdown(
            "**`coverage_entrevistas` y `coverage_programa`**: `candidato`, `chunk_id`, `codigo_lvl3`, `sim_subtema`  \n"
            "*(Para **share** por tokens: agregar `n_tokens_chunk`)*  \n\n"
            "**`df_subtemas`**: `codigo_lvl3`, `tema_nivel3`, `codigo_lvl2`, `tema_nivel2`, `pilar`, `horizonte_simple` (solo {CP,MP,LP})  \n"
            "Normalización: `CP-MP→CP`, `MP-LP→MP`"
        )

    # FAQ
    with st.expander("Preguntas frecuentes (FAQ)"):
        st.markdown(
            "**¿Por qué no clustering puro?** Microtemas poco interpretables; MECE es estable.  \n"
            "**¿Qué es propositivo?** Promesas/medidas/metas/plazos.  \n"
            "**¿Un chunk puede tocar varios subtemas?** Sí; se pondera y se deduplica.  \n"
            "**¿Limitaciones del sentimiento?** Ironía/eslóganes; por eso agregados por tema/pilar."
        )

    st.divider()

    # Ética y Roadmap
    st.subheader("Ética y uso responsable")
    st.warning(
        "Fuentes públicas, resultados agregados. Cuidar sesgos y errores de ASR/diarización. "
        "Las métricas son **indicadores analíticos**, no juicios categóricos."
    )
    st.subheader("Roadmap y versión")
    st.markdown(
        "**Próximo:** co-ocurrencias (red temática), promesas cuantificables, panel de calibración de umbral, tracking por medio/foro.  \n"
        "**Versión actual:** v0.1"
    )

# =============================
# REPORTE (usa tu encabezado exacto)
# =============================
def render_reporte():
    st.markdown("### Distribución temática por pilar (%)")
    st.markdown("**De todo lo que dijo cada candidato, ¿qué % corresponde a cada pilar?**")
    # Aquí insertas tus filtros/gráficos:
    # - selectbox de candidato
    # - gráfico plotly de barras apiladas por pilar
    # - tabla de cobertura/share por pilar
    # Ejemplo placeholder:
    st.info("Pronto: filtros, gráficos de barras/stack y tablas.")

# =============================
# RENDER
# =============================
# HOME
if page == "HOME":
    render_home()

# =============================
# REPORTE (dashboard)
# =============================
elif page == "REPORTE":

    # --------- helpers básicos ---------
    def _lock_uirev(fig, name: str):
        try:
            fig.update_layout(uirevision=name)
        except Exception:
            pass
        return fig

    def _plotly(fig, uirev: str):
        ph = st.empty()
        if fig is None:
            with ph: st.info("Sin datos para graficar.")
            return
        fig = _lock_uirev(fig, uirev)
        with ph:
            # ⚠️ sin key; uirevision mantiene el DOM estable
            st.plotly_chart(fig, use_container_width=True)

    def _df(df: pd.DataFrame | None, caption: str | None = None):
        ph = st.empty()
        with ph:
            if df is None or (hasattr(df, "empty") and df.empty):
                st.info("Sin datos para mostrar.")
                return
            if caption:
                st.caption(caption)
            # ⚠️ sin key para evitar reconciliación agresiva del árbol
            st.dataframe(df, use_container_width=True)

    def _ensure_cols(df: pd.DataFrame, name: str, cols: set[str]) -> bool:
        miss = cols - set(df.columns)
        if miss:
            st.warning(f"‘{name}’ no tiene columnas requeridas: {sorted(miss)}")
            return False
        return True

    # fallback de orden de horizontes si no estaba en el entorno
    try:
        H_ORDER
    except NameError:
        H_ORDER = ["CP", "MP", "LP"]

    # ===========================
    # Distribución por pilar
    # ===========================
    st.markdown("### Distribución temática por pilar (%)")
    st.markdown("**De todo lo que dijo cada candidato, ¿qué % corresponde a cada pilar?**")
    st.info(
        "**¿Qué es un chunk?** Un *chunk* es un fragmento de texto —unidad mínima de análisis— "
        "extraído de una entrevista o de un programa de gobierno.\n\n"
        "**¿Qué es el share?** Proporción del discurso (o programa) que un candidato dedica a un pilar específico.\n\n"
        "**Cómo leer el gráfico:** cada barra es un candidato; el 100% representa su total de discurso (o su programa). "
        "Los colores son pilares y el porcentaje en cada segmento es su *share* dentro del candidato. "
        "Los shares por candidato suman **100%**."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Entrevistas")
        try:
            fig_ent = plotly_share_por_pilar(res_ent_lvl3, "Entrevistas — share por pilar", df_subtemas)
        except Exception as e:
            fig_ent = None; st.warning(f"No fue posible generar el gráfico: {e}")
        _plotly(fig_ent, uirev="pilar_ent")
    with col2:
        st.markdown("#### Programa")
        try:
            fig_prog = plotly_share_por_pilar(res_prog_lvl3, "Programa — share por pilar", df_subtemas)
        except Exception as e:
            fig_prog = None; st.warning(f"No fue posible generar el gráfico: {e}")
        _plotly(fig_prog, uirev="pilar_prog")

    st.divider()

    # ===========================
    # Share por horizonte
    # ===========================
    st.markdown("### ¿En qué horizonte se concentra el discurso? — Share por horizonte (CP/MP/LP)")
    st.markdown("**De todo lo que dijo cada candidato en entrevistas y en programas, "
                "¿qué proporción estuvo enfocada en corto (CP), mediano (MP) o largo plazo (LP)?**")
    st.info("Lectura: si un candidato muestra **60% en CP**, ~60% de sus *tokens ponderados* "
            "mapeados a subtemas estuvieron en **corto plazo**.")

    hcol1, hcol2 = st.columns(2)
    with hcol1:
        st.markdown("#### Entrevistas — share por horizonte")
        try:
            fig_ent_h = plotly_share_por_horizonte(res_ent_lvl3, "Entrevistas — share por horizonte (CP/MP/LP)")
        except Exception as e:
            fig_ent_h = None; st.warning(f"No fue posible generar el gráfico: {e}")
        _plotly(fig_ent_h, uirev="horiz_ent")
    with hcol2:
        st.markdown("#### Programa — share por horizonte")
        try:
            fig_prog_h = plotly_share_por_horizonte(res_prog_lvl3, "Programa — share por horizonte (CP/MP/LP)")
        except Exception as e:
            fig_prog_h = None; st.warning(f"No fue posible generar el gráfico: {e}")
        _plotly(fig_prog_h, uirev="horiz_prog")

    st.divider()

    # ===========================
    # % del universo cubierto
    # ===========================
    st.markdown("### ¿Qué porcentaje del universo de subtemas cubre cada candidato?")
    st.markdown("**Cobertura del universo de subtemas por horizonte (CP/MP/LP).**")
    st.info("Mide **amplitud**: para cada horizonte, % del **universo de subtemas** tocado al menos una vez.")

    ucol1, ucol2 = st.columns(2)
    with ucol1:
        st.markdown("#### Entrevistas — % del universo cubierto")
        try:
            fig_ent_univ = plot_cobertura_pct_simple(
                ent_pct_univ, "Entrevistas — % del universo de subtemas cubierto (CP/MP/LP)", H_ORDER
            )
        except Exception as e:
            fig_ent_univ = None; st.warning(f"No fue posible generar el gráfico: {e}")
        _plotly(fig_ent_univ, uirev="univ_ent")
    with ucol2:
        st.markdown("#### Programa — % del universo cubierto")
        try:
            fig_prog_univ = plot_cobertura_pct_simple(
                prog_pct_univ, "Programa — % del universo de subtemas cubierto (CP/MP/LP)", H_ORDER
            )
        except Exception as e:
            fig_prog_univ = None; st.warning(f"No fue posible generar el gráfico: {e}")
        _plotly(fig_prog_univ, uirev="univ_prog")

    st.divider()

    # ===========================
    # Top 8 L3
    # ===========================
    st.markdown("### Top 8 temas (Nivel 3) por candidato — Share")
    st.markdown("**Selecciona un candidato para ver su Top-8 en entrevistas y en programa.**")
    st.info("El *share* es la proporción del total del candidato (tokens ponderados) que cae en cada tema; colores = pilar.")

    try:
        cand_list_l3 = sorted(set(res_ent_lvl3["candidato"].astype(str)) | set(res_prog_lvl3["candidato"].astype(str)))
    except Exception:
        cand_list_l3 = []
    if cand_list_l3:
        cand_sel_l3 = st.selectbox("Candidato (Nivel 3)", cand_list_l3, index=0, key="cand_l3")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Entrevistas — Top 8 L3")
            try:
                fig_top_ent_l3 = plotly_topN_por_candidato_lvl3(
                    res_ent_lvl3, cand_sel_l3, N=8, usar="share", df_subtemas=df_subtemas, show_legend=False
                )
            except Exception as e:
                fig_top_ent_l3 = None; st.warning(f"No fue posible generar el gráfico: {e}")
            _plotly(fig_top_ent_l3, uirev="top_l3_ent")
        with c2:
            st.markdown("#### Programa — Top 8 L3")
            try:
                fig_top_prog_l3 = plotly_topN_por_candidato_lvl3(
                    res_prog_lvl3, cand_sel_l3, N=8, usar="share", df_subtemas=df_subtemas, show_legend=False
                )
            except Exception as e:
                fig_top_prog_l3 = None; st.warning(f"No fue posible generar el gráfico: {e}")
            _plotly(fig_top_prog_l3, uirev="top_l3_prog")
    else:
        st.info("No hay candidatos para Top-8 L3.")

    st.divider()

    # ===========================
    # Top 8 L2
    # ===========================
    st.markdown("### Top 8 temas (Nivel 2) por candidato — Share")
    st.markdown("**Selecciona un candidato para ver su Top-8 agregado en Nivel 2.**")

    try:
        cand_list_l2 = sorted(set(res_ent_lvl2["candidato"].astype(str)) | set(res_prog_lvl2["candidato"].astype(str)))
    except Exception:
        cand_list_l2 = []
    if cand_list_l2:
        cand_sel_l2 = st.selectbox("Candidato (Nivel 2)", cand_list_l2, index=0, key="cand_l2")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Entrevistas — Top 8 L2")
            try:
                fig_top_ent_l2 = plotly_topN_por_candidato_lvl2(res_ent_lvl2, cand_sel_l2, N=8, usar="share")
            except Exception as e:
                fig_top_ent_l2 = None; st.warning(f"No fue posible generar el gráfico: {e}")
            _plotly(fig_top_ent_l2, uirev="top_l2_ent")
        with c4:
            st.markdown("#### Programa — Top 8 L2")
            try:
                fig_top_prog_l2 = plotly_topN_por_candidato_lvl2(res_prog_lvl2, cand_sel_l2, N=8, usar="share")
            except Exception as e:
                fig_top_prog_l2 = None; st.warning(f"No fue posible generar el gráfico: {e}")
            _plotly(fig_top_prog_l2, uirev="top_l2_prog")
    else:
        st.info("No hay candidatos para Top-8 L2.")

    st.divider()

    # ===========================
    # Menor share (L3 y L2)
    # ===========================
    st.markdown("### Temas con menor share — Top 8 por candidato")
    try:
        cand_all = sorted(set(res_ent_lvl3["candidato"].astype(str)) | set(res_prog_lvl3["candidato"].astype(str)))
    except Exception:
        cand_all = []
    if cand_all:
        cand_sel = st.selectbox("Elige candidato", cand_all, index=0, key="cand_low_share")
        colA, colB = st.columns(2)
        with colA:
            try:
                df_low_ent_l3 = top_low_share(res_ent_lvl3, cand_sel, N=8)
                fig_low_ent_l3 = plotly_top_low_share(df_low_ent_l3, "lvl3",
                                                      f"Top 8 temas L3 con menor share — {cand_sel} (Entrevistas)")
                _plotly(fig_low_ent_l3, uirev="low_l3_ent")
            except Exception as e:
                st.warning(f"No fue posible calcular: {e}", icon="⚠️")
        with colB:
            try:
                df_low_prog_l3 = top_low_share(res_prog_lvl3, cand_sel, N=8)
                fig_low_prog_l3 = plotly_top_low_share(df_low_prog_l3, "lvl3",
                                                       f"Top 8 temas L3 con menor share — {cand_sel} (Programa)")
                _plotly(fig_low_prog_l3, uirev="low_l3_prog")
            except Exception as e:
                st.warning(f"No fue posible calcular: {e}", icon="⚠️")

        colC, colD = st.columns(2)
        with colC:
            try:
                df_low_ent_l2 = top_low_share(res_ent_lvl2, cand_sel, N=8)
                fig_low_ent_l2 = plotly_top_low_share(df_low_ent_l2, "lvl2",
                                                      f"Top 8 temas L2 con menor share — {cand_sel} (Entrevistas)")
                _plotly(fig_low_ent_l2, uirev="low_l2_ent")
            except Exception as e:
                st.warning(f"No fue posible calcular: {e}", icon="⚠️")
        with colD:
            try:
                df_low_prog_l2 = top_low_share(res_prog_lvl2, cand_sel, N=8)
                fig_low_prog_l2 = plotly_top_low_share(df_low_prog_l2, "lvl2",
                                                       f"Top 8 temas L2 con menor share — {cand_sel} (Programa)")
                _plotly(fig_low_prog_l2, uirev="low_l2_prog")
            except Exception as e:
                st.warning(f"No fue posible calcular: {e}", icon="⚠️")
    else:
        st.info("No hay candidatos para esta sección.")

    st.divider()

    # ===========================
    # Cobertura vs Diversidad
    # ===========================
    st.markdown("### Cobertura vs Diversidad (resumen ejecutivo)")
    st.info("- **X (Cobertura):** # de L3 distintos.  - **Y (Diversidad):** entropía 0–1.  - **Tamaño:** w_tokens.  - **Color:** ENT vs PROG.")
    try:
        fig_exec = plot_cobertura_vs_diversidad(
            res_exec, usar_pct_universo=False,
            title="Cobertura (#L3) vs Diversidad (entropía) — ENT vs PROG", size_max=42
        )
    except Exception as e:
        fig_exec = None; st.warning(f"No fue posible generar el gráfico: {e}")
    _plotly(fig_exec, uirev="exec_cov_div")

    st.divider()

    # ===========================
    # Coherencia ENT↔PROG
    # ===========================
    st.markdown("### ¿Qué tan coherente es el discurso vs. el programa?")
    st.markdown("**Coherencia ENT↔PROG por pilar (similitud de centroides de embeddings)**")

    ph_coh_heat = st.empty()
    ph_coh_bar  = st.empty()
    ph_coh_tab  = st.expander("Ver tabla", expanded=False)

    try:
        M_l3, P_pilar = coherencia_ent_prog_por_pilar_topm(
            coverage_entrevistas, coverage_programa,
            df_chunks, df_chunks_programas, df_subtemas,
            emb_col_ent="embedding", emb_col_prog="embedding_programa",
            weight_mode="soft", min_chunks_l3=1, min_w_tokens_l3=0.0
        )
        if P_pilar is None or P_pilar.empty:
            with ph_coh_heat: st.warning("No hay L3 en común ENT–PROG con soporte suficiente.")
        else:
            fig_heat = plot_heatmap_coherencia(
                P_pilar, value_col="sim_mediana_w_min",
                title="Coherencia ENT↔PROG por pilar — similitud (0–1)", annotate=True
            )
            with ph_coh_heat:
                _plotly(fig_heat, uirev="coh_heat")

            agg_global, fig_global = coherencia_global_bar(
                P_pilar, value_col="sim_mediana_w_geom", weighting="common", show_plot=False
            )
            with ph_coh_bar:
                _plotly(fig_global, uirev="coh_bar")
                st.caption("Barra = coherencia promedio ENT↔PROG (0–1) ponderada por soporte común.")
            with ph_coh_tab:
                _df(agg_global, caption=None)
    except ValueError as e:
        with ph_coh_heat: st.warning(f"No fue posible calcular la coherencia: {e}", icon="⚠️")
    except Exception as e:
        with ph_coh_heat: st.error(f"Ocurrió un error inesperado al graficar la coherencia: {e}")

    st.divider()

    # ===========================
    # Deriva semántica ENT→PROG
    # ===========================
    st.markdown("### Deriva semántica ENT→PROG (promedio mensual centrado)")
    st.markdown("**¿Cómo evoluciona en el tiempo la similitud de ENT respecto al centro semántico del PROG?**")

    ph_ts_plot = st.empty()
    exp_ts = st.expander("Ver tabla mensual (promedio por candidato)", expanded=False)
    try:
        ts_ent_prog, fig_ts = similitud_a_centro_programa_por_mes_centrado(
            coverage_entrevistas, df_chunks, df_chunks_programas,
            emb_col_ent='embedding', emb_col_prog='embedding_programa',
            fecha_col='Fecha publicación'
        )
        with ph_ts_plot:
            _plotly(fig_ts, uirev="deriva_ts")
        with exp_ts:
            _df(ts_ent_prog)
    except (KeyError, ValueError) as e:
        with ph_ts_plot: st.warning(f"No fue posible calcular la serie temporal: {e}")
    except Exception as e:
        with ph_ts_plot: st.error(f"Ocurrió un error al generar la deriva semántica: {e}")

    st.divider()

    # ===========================
    # Vecindades por Pilar
    # ===========================
    st.markdown("### Vecindades (¿quién se parece más a quién?) por Pilar")
    st.info("α = peso de similitud semántica; β = similitud de intensidad; temperatura = nitidez del ‘vecino’.")

    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        origen_sel = st.radio("Origen", ["Entrevistas","Programa"], index=0, horizontal=True, key="vec_origen")
    with colB:
        alpha = st.slider("Peso coseno (α)", 0.0, 1.0, 0.8, 0.05, key="vec_alpha")
    with colC:
        beta  = st.slider("Peso intensidad (β)", 0.0, 1.0, 0.2, 0.05, key="vec_beta")
    with colD:
        temp  = st.slider("Temperatura softmax", 0.05, 1.0, 0.25, 0.05, key="vec_temp")
    min_chunks = st.slider("Mín. chunks por (candidato,pilar)", 1, 10, 3, 1, key="vec_min_chunks")

    if origen_sel == "Entrevistas":
        cov_use, base_use, emb_col, origen_id = coverage_entrevistas, df_chunks, "embedding", "ENT"
    else:
        cov_use, base_use, emb_col, origen_id = coverage_programa, df_chunks_programas, "embedding_programa", "PROG"

    ph_vec_top  = st.empty()
    ph_vec_det  = st.empty()
    ph_vec_heat = st.empty()

    try:
        res_tables, heatmaps = vecindades_por_pilar(
            coverage_topm=cov_use, base_df=base_use, df_subtemas=df_subtemas,
            emb_col=emb_col, base_origen=origen_id,
            weight_mode="soft", alpha=alpha, beta=beta, temperature=temp, min_chunks=min_chunks
        )
        if not res_tables:
            with ph_vec_top: st.warning("No hay pilares suficientes para calcular vecindades.")
        else:
            pilares_disp = sorted(res_tables.keys())
            pil_sel = st.selectbox("Pilar", pilares_disp, index=0, key="vec_pilar")

            df_pil = res_tables[pil_sel].copy()
            if df_pil.empty:
                with ph_vec_top: st.warning("No hay pares suficientes en este pilar.")
            else:
                top_por_source = (df_pil.sort_values(["source","prob"], ascending=[True, False])
                                    .groupby("source", as_index=False).head(1))
                top_por_source = top_por_source.sort_values("prob", ascending=False).reset_index(drop=True)
                with ph_vec_top:
                    _df(top_por_source, caption="Top vecino por candidato (probabilidad máxima)")

                c1, c2 = st.columns([1,3])
                with c1:
                    cand_source_opts = sorted(top_por_source["source"].unique()) if not top_por_source.empty else []
                    cand_source = st.selectbox("Candidato origen (fila)", cand_source_opts, index=0, key="vec_cand_source") if cand_source_opts else None
                    top_k = st.slider("Top-K vecinos a mostrar", 1, 10, 5, 1, key="vec_topk")
                with c2:
                    with ph_vec_det:
                        if cand_source is not None:
                            df_det = (df_pil[df_pil["source"]==cand_source]
                                        .sort_values("prob", ascending=False)
                                        .head(top_k)
                                        .reset_index(drop=True))
                            _df(df_det, caption=f"Vecinos de `{cand_source}` en `{pil_sel}`")
                        else:
                            st.info("Selecciona un candidato origen para ver sus vecinos.")

                fig_hm = plot_heatmap_vecindades(heatmaps[pil_sel], pil_sel)
                with ph_vec_heat:
                    _plotly(fig_hm, uirev="vec_heat")
    except KeyError as e:
        with ph_vec_top: st.warning(f"Falta columna requerida: {e}")
    except ValueError as e:
        with ph_vec_top: st.warning(f"No fue posible calcular vecindades: {e}")
    except Exception as e:
        with ph_vec_top: st.error(f"Ocurrió un error en vecindades: {e}")

    st.divider()

    # ===========================
    # Sentimiento candidato × pilar
    # ===========================
    st.markdown("### Sentimiento por candidato × pilar")
    st.info("`SBI` (−1..1), `score_mean` (−1..1), `stars_mean` (1–5). Heatmap o barras agrupadas.")

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        origen_vista = st.radio("Origen", ["Entrevistas","Programa"], index=0, horizontal=True, key="sent_vista_origen")
    with c2:
        metrica_vista = st.selectbox("Métrica", ["SBI","score_mean","stars_mean"], index=0, key="sent_vista_metric")
    with c3:
        tipo_vista = st.selectbox("Vista", ["Heatmap","Barras agrupadas"], index=0, key="sent_vista_tipo")
    with c4:
        ponderar_vista = st.checkbox("Ponderar por tokens", value=False, key="sent_vista_ponderar")
    conf_min_vista = st.slider("Confianza mínima", 0.0, 1.0, 0.0, 0.05, key="sent_vista_conf")

    if origen_vista == "Entrevistas":
        cov_sent = attach_pilar_siempre(coverage_entrevistas.copy(), df_subtemas)
        cov_sent = _merge_precomputed_sent(cov_sent, sent_ent, name="ENT"); label_origen = "ENT"
    else:
        cov_sent = attach_pilar_siempre(coverage_programa.copy(), df_subtemas)
        cov_sent = _merge_precomputed_sent(cov_sent, sent_prog, name="PROG"); label_origen = "PROG"

    ph_sent_plot = st.empty()
    ph_sent_tab  = st.expander("Ver tabla (candidato × pilar)", expanded=False)
    try:
        agg_cp = _agg_sent_por_pilar(cov_sent, weight_by_tokens=ponderar_vista, conf_min=conf_min_vista)
        if agg_cp.empty:
            with ph_sent_plot: st.warning("No hay resultados de sentimiento para los filtros actuales.")
        else:
            if tipo_vista == "Heatmap":
                titulo = f"{label_origen} — {metrica_vista} por candidato × pilar"
                fig_hm = _heatmap_sent(agg_cp, metrica_vista, titulo)
                with ph_sent_plot: _plotly(fig_hm, uirev="sent_heat")
            else:
                order_pilar = (agg_cp.groupby("pilar")[metrica_vista]
                               .mean().sort_values(ascending=False).index.tolist())
                order_cand = sorted(agg_cp["candidato"].astype(str).unique().tolist())
                df_plot = agg_cp.sort_values(["pilar", metrica_vista], ascending=[True, False]).copy()
                df_plot["text_lbl"] = df_plot[metrica_vista].map(lambda v: f"{float(v):.2f}" if pd.notna(v) else "")
                fig_bar = px.bar(
                    df_plot, x="pilar", y=metrica_vista, color="candidato", barmode="group",
                    category_orders={"pilar": order_pilar, "candidato": order_cand},
                    text="text_lbl", title=f"{label_origen} — {metrica_vista} por pilar (barras agrupadas)",
                    labels={metrica_vista: metrica_vista, "pilar":"Pilar"}
                )
                if metrica_vista in ("SBI","score_mean"): fig_bar.update_yaxes(range=[-1,1])
                elif metrica_vista == "stars_mean":       fig_bar.update_yaxes(range=[1,5])
                fig_bar.update_traces(textposition="outside", cliponaxis=False)
                fig_bar.update_layout(template="plotly_white", legend_title="Candidato",
                                      margin=dict(t=70, l=40, r=20, b=50))
                with ph_sent_plot: _plotly(fig_bar, uirev="sent_bar")

            with ph_sent_tab:
                cols_show = ["candidato","pilar","n_chunks","tokens","stars_mean","score_mean","pct_1_2","pct_3","pct_4_5","SBI"]
                _df(agg_cp[cols_show].sort_values(["pilar", metrica_vista], ascending=[True, False]))
    except KeyError as e:
        with ph_sent_plot: st.warning(f"Falta columna requerida para sentimiento: {e}", icon="⚠️")
    except Exception as e:
        with ph_sent_plot: st.error(f"Error al renderizar sentimiento candidato×pilar: {e}")

    st.divider()

    # ===========================
    # Top ± sentimiento
    # ===========================
    st.markdown("### Top 8 temas más positivos vs. más negativos (sentimiento)")

    selA, selB, selC, selD = st.columns([1,1,1,2])
    with selA:
        origen_top = st.radio("Origen", ["Entrevistas","Programa"], index=0, horizontal=True, key="sent_top_origen")
    with selB:
        nivel_top  = st.radio("Nivel", ["L2","L3"], index=1, horizontal=True, key="sent_top_nivel")
    with selC:
        metrica_top = st.selectbox("Métrica", ["SBI","stars_mean","score_mean"], index=0, key="sent_top_metric")

    if origen_top == "Entrevistas":
        cov_b = attach_pilar_siempre(coverage_entrevistas.copy(), df_subtemas); cov_b = _merge_precomputed_sent(cov_b, sent_ent, name="ENT")
    else:
        cov_b = attach_pilar_siempre(coverage_programa.copy(), df_subtemas);    cov_b = _merge_precomputed_sent(cov_b, sent_prog, name="PROG")

    def _agg_sent_by_levels_local(df_cov_sent: pd.DataFrame, levels: list) -> pd.DataFrame:
        need = {"chunk_id","candidato","n_tokens_chunk","stars","star_score","conf"}
        if not _ensure_cols(df_cov_sent, "cov_sent", need): return pd.DataFrame()
        df = df_cov_sent.copy()
        agg = (df.groupby(levels, as_index=False)
                 .agg(n_chunks=("chunk_id","nunique"),
                      tokens=("n_tokens_chunk","sum"),
                      stars_mean=("stars","mean"),
                      score_mean=("star_score","mean"),
                      pct_1_2=("stars", lambda x: float((x<=2).mean())),
                      pct_3=("stars", lambda x: float((x==3).mean())),
                      pct_4_5=("stars", lambda x: float((x>=4).mean()))))
        agg["SBI"] = agg["pct_4_5"] - agg["pct_1_2"]
        return agg

    if nivel_top == "L2":
        levels = ["candidato","codigo_lvl2","tema_nivel2","pilar"]; name_col = "tema_nivel2"
    else:
        levels = ["candidato","codigo_lvl3","tema_nivel3","codigo_lvl2","tema_nivel2","pilar"]; name_col = "tema_nivel3"

    ph_toppos = st.empty(); ph_topneg = st.empty()
    ph_toptab = st.expander("Ver tabla (candidato y nivel seleccionados)", expanded=False)
    try:
        agg_nivel = _agg_sent_by_levels_local(cov_b, levels)
        cands_disp = sorted(agg_nivel["candidato"].dropna().astype(str).unique().tolist()) if not agg_nivel.empty else []
        if not cands_disp:
            with ph_toppos: st.warning("No hay datos de sentimiento para esta vista.")
        else:
            with selD:
                cand_sel = st.selectbox("Candidato", cands_disp, index=0, key="sent_top_cand")
            df_cand = agg_nivel[agg_nivel["candidato"]==cand_sel].copy()
            if df_cand.empty:
                with ph_toppos: st.warning("No hay datos para el candidato seleccionado.")
            else:
                M = metrica_top
                top_pos = df_cand.sort_values(M, ascending=False).head(8).copy()
                top_neg = df_cand.sort_values(M, ascending=True).head(8).copy()
                def _lbl(s): s = str(s) if s is not None else ""; return s if len(s) <= 60 else s[:59]+"…"
                top_pos["tema_lbl"] = top_pos[name_col].map(_lbl); top_neg["tema_lbl"] = top_neg[name_col].map(_lbl)
                x_range = [-1,1] if M in ("SBI","score_mean") else ([1,5] if M=="stars_mean" else None)
                colL, colR = st.columns(2)
                with colL:
                    fig_pos = _make_top_bar_smart(top_pos, metric=M, positive=True, x_range=x_range,
                                                  title=f"{cand_sel} — Top 8 {nivel_top} más **positivos** ({M})")
                    _plotly(fig_pos, uirev="top_pos")
                with colR:
                    fig_neg = _make_top_bar_smart(top_neg, metric=M, positive=False, x_range=x_range,
                                                  title=f"{cand_sel} — Top 8 {nivel_top} más **negativos** ({M})")
                    _plotly(fig_neg, uirev="top_neg")
                with ph_toptab:
                    cols_show = ["candidato", name_col, "pilar", "n_chunks", "tokens", "stars_mean", "score_mean", "SBI"]
                    _df(df_cand[cols_show].sort_values(M, ascending=False))
    except Exception as e:
        with ph_toppos: st.error(f"Error en Top± sentimiento: {e}")

    st.divider()

    # ===========================
    # Indicadores semánticos (pop/inst/tec/vago)
    # ===========================
    st.markdown("### Indicadores semánticos: Populismo / Institucionalidad / Tecnicismo / Vaguedad")
    st.info("Proyección de embeddings a ejes seed. Índices 0..1 con pair-softmax; agregación ponderada por w_tokens.")

    cc1, cc2, cc3 = st.columns([1,1,2])
    with cc1:
        idx_origen = st.radio("Origen", ["Entrevistas","Programa"], index=0, horizontal=True, key="idx_origen_sem")
    with cc2:
        indicador = st.selectbox("Indicador", ["Tecnicismo","Populismo","Institucionalidad","Vaguedad"], index=0, key="idx_indicador")
    with cc3:
        tau_soft = st.slider("Suavizado (τ) del softmax par (↑ = más contraste)", 2.0, 20.0, 10.0, 1.0, key="idx_tau")

    ph_idx_heat = st.empty()
    ph_idx_bar  = st.empty()
    ph_idx_tab  = st.expander("Ver tabla de candidato × indicadores", expanded=False)
    try:
        cents = _seed_centroids(seeds_df)
        if idx_origen == "Entrevistas":
            cov = coverage_entrevistas.copy(); base_df, emb_col, origen_tag = df_chunks, "embedding", "ENT"
        else:
            cov = coverage_programa.copy();   base_df, emb_col, origen_tag = df_chunks_programas, "embedding_programa", "PROG"

        cov = attach_pilar_siempre(cov, df_subtemas)
        cov = compute_w_tokens(cov, mode="soft")
        cov = align_embeddings_with_coverage(cov, base_df, emb_col=emb_col)

        cov["emb_unit"] = cov[emb_col].apply(_unit)
        cov["sim_pop"]  = cov["emb_unit"].apply(lambda v: float(np.dot(v, cents["pop"])))
        cov["sim_inst"] = cov["emb_unit"].apply(lambda v: float(np.dot(v, cents["inst"])))
        cov["sim_tec"]  = cov["emb_unit"].apply(lambda v: float(np.dot(v, cents["tec"])))
        cov["sim_vago"] = cov["emb_unit"].apply(lambda v: float(np.dot(v, cents["vago"])))

        tau = float(tau_soft)
        cov["POP_INDEX"]  = cov.apply(lambda r: _pair_softmax(r["sim_pop"],  r["sim_inst"], tau=tau), axis=1)
        cov["INST_INDEX"] = cov.apply(lambda r: _pair_softmax(r["sim_inst"], r["sim_pop"],  tau=tau), axis=1)
        cov["TEC_INDEX"]  = cov.apply(lambda r: _pair_softmax(r["sim_tec"],  r["sim_vago"], tau=tau), axis=1)
        cov["VAGO_INDEX"] = cov.apply(lambda r: _pair_softmax(r["sim_vago"], r["sim_tec"],  tau=tau), axis=1)

        P_pilar = _aggregate_weighted(cov, ["candidato","pilar"], cols=("POP_INDEX","TEC_INDEX","INST_INDEX","VAGO_INDEX"))
        Cand    = _aggregate_weighted(cov, ["candidato"],       cols=("POP_INDEX","TEC_INDEX","INST_INDEX","VAGO_INDEX"))

        col_map = {
            "Tecnicismo": ("TEC_INDEX", "Blues",  "↑ = más técnico"),
            "Populismo": ("POP_INDEX", "Reds",   "↑ = más populista"),
            "Institucionalidad": ("INST_INDEX", "Greens","↑ = más institucional"),
            "Vaguedad": ("VAGO_INDEX", "Purples","↑ = más vago")
        }
        val_col, palette, legend_note = col_map[indicador]

        piv = P_pilar.pivot(index="candidato", columns="pilar", values=val_col)
        if piv.size == 0:
            with ph_idx_heat: st.warning("No hay datos para el heatmap con los filtros actuales.")
        else:
            row_order = piv.mean(axis=1).sort_values(ascending=False).index
            col_order = piv.mean(axis=0).sort_values(ascending=False).index
            piv = piv.reindex(index=row_order, columns=col_order)
            fig_hm = px.imshow(
                piv, x=piv.columns, y=piv.index, zmin=0, zmax=1,
                color_continuous_scale=palette,
                labels={"color": f"{val_col} (0..1) {legend_note}"},
                title=f"{origen_tag} — {indicador} por pilar (ponderado por w_tokens)"
            )
            fig_hm.update_layout(template="plotly_white", margin=dict(t=70,l=60,r=20,b=40))
            with ph_idx_heat: _plotly(fig_hm, uirev="idx_heat")

        Cand_long = (Cand
            .melt(id_vars=["candidato","w_total","n_chunks"],
                  value_vars=["TEC_INDEX","POP_INDEX","INST_INDEX","VAGO_INDEX"],
                  var_name="indicador", value_name="valor"))
        Cand_long["indicador"] = Cand_long["indicador"].map({
            "TEC_INDEX":"Tecnicismo", "POP_INDEX":"Populismo",
            "INST_INDEX":"Institucionalidad", "VAGO_INDEX":"Vaguedad"
        })
        order_cand = (Cand_long[Cand_long["indicador"] == {"Tecnicismo":"Tecnicismo","Populismo":"Populismo",
                        "Institucionalidad":"Institucionalidad","Vaguedad":"Vaguedad"}[indicador]]
                      .sort_values("valor", ascending=False)["candidato"].tolist())

        fig_bars = px.bar(
            Cand_long, x="candidato", y="valor", color="indicador", barmode="group",
            category_orders={"candidato": order_cand, "indicador": ["Tecnicismo","Populismo","Institucionalidad","Vaguedad"]},
            labels={"valor":"Índice (0..1)"}, title=f"{origen_tag} — Perfil global por candidato (4 indicadores)"
        )
        fig_bars.update_layout(template="plotly_white", yaxis=dict(range=[0,1]), legend_title="Indicador",
                               margin=dict(t=70,l=40,r=20,b=50))
        fig_bars.update_traces(hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:.2f}<extra></extra>")
        with ph_idx_bar: _plotly(fig_bars, uirev="idx_bar")

        with ph_idx_tab:
            show_cols = ["candidato","POP_INDEX","INST_INDEX","TEC_INDEX","VAGO_INDEX","n_chunks","w_total"]
            _df(Cand[show_cols].sort_values(val_col, ascending=False))
    except NameError as e:
        with ph_idx_heat: st.warning(f"Falta un objeto necesario (`seeds_df`, bases, etc.). Detalle: {e}", icon="⚠️")
    except KeyError as e:
        with ph_idx_heat: st.warning(f"Falta columna requerida: {e}", icon="⚠️")
    except Exception as e:
        with ph_idx_heat: st.error(f"Error en indicadores semánticos: {e}")

    st.divider()

    # ===========================
    # Nubes de palabras
    # ===========================
    st.markdown("### Nubes de palabras por pilar/candidato")
    st.info("Palabras más frecuentes o distintivas (TF-IDF) con asignación soft por pilar (w_tokens).")

    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except Exception as e:
        st.error(f"Dependencias para nubes no disponibles: {e}")

    colA, colB, colC, colD, colE = st.columns([1,1,1,1,1])
    with colA:
        origen_wc = st.radio("Origen", ["Entrevistas","Programa"], index=0, horizontal=True, key="wc_origen")
    with colB:
        vista_wc = st.radio("Vista", ["Por pilar → candidatos","Por candidato → pilares"], index=0, key="wc_vista")
    with colC:
        usar_tfidf = st.radio("Ponderación", ["TF-IDF","Frecuencia"], index=0, key="wc_mode") == "TF-IDF"
    with colD:
        min_support = st.slider("Soporte mínimo (tokens)", 10, 200, 40, 10, key="wc_support")
    with colE:
        max_paneles = st.slider("Máx. paneles", 2, 12, 6, 1, key="wc_maxpan")

    cov_wc = coverage_entrevistas.copy() if origen_wc == "Entrevistas" else coverage_programa.copy()
    ph_wc = st.empty()
    try:
        cov_wc = attach_pilar_siempre(cov_wc, df_subtemas)
        cov_wc = compute_w_tokens(cov_wc, mode="soft")
        need_cols = {"chunk_id","candidato","pilar","texto_chunk","n_tokens_chunk","w_tokens"}
        if not _ensure_cols(cov_wc, "coverage (wc)", need_cols):
            with ph_wc: st.warning("No hay columnas suficientes para nubes.")
        else:
            counters, support = build_group_counters(
                cov_wc[list(need_cols)].copy(),
                min_group_support_tokens=float(min_support),
                min_len_token=3
            )
            if not counters:
                with ph_wc: st.warning("No hay grupos con soporte suficiente para generar nubes.")
            else:
                # función local para renderizar mosaico de nubes
                def _render_wcs(keys, label_getter):
                    data = counters if not usar_tfidf else counters_to_tfidf({k: counters[k] for k in keys})
                    n = len(keys); cols = 2; rows = (n + cols - 1) // cols
                    fig = plt.figure(figsize=(14, 6*rows))
                    for i, k in enumerate(keys, 1):
                        freqs = data.get(k, {})
                        ax = fig.add_subplot(rows, cols, i)
                        ax.axis("off")
                        if not freqs:
                            ax.set_title(f"{label_getter(k)} — sin términos", fontsize=11)
                            continue
                        wc = _wc_from_freq(freqs, max_words=120)
                        ax.imshow(wc, interpolation="bilinear")
                        sup = int(support.get(k, 0.0))
                        ax.set_title(f"{label_getter(k)}  • soporte≈{sup}", fontsize=11)
                    fig.tight_layout()
                    with ph_wc:
                        # ⚠️ NO usar key aquí (evita FigureCanvasAgg.print_png(key=...))
                        st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                if vista_wc == "Por pilar → candidatos":
                    pilares_disp = sorted({k[1] for k in counters.keys()})
                    pil_sel = st.selectbox("Pilar", pilares_disp, index=0, key="wc_sel_pilar") if pilares_disp else None
                    if not pil_sel:
                        with ph_wc: st.warning("No hay pilares disponibles para esta vista.")
                    else:
                        keys = [k for k in counters.keys() if k[1] == pil_sel]
                        keys.sort(key=lambda k: support.get(k, 0.0), reverse=True)
                        _render_wcs(keys[:max_paneles], label_getter=lambda k: k[0])
                else:  # Por candidato → pilares
                    cands_disp = sorted({k[0] for k in counters.keys()})
                    cand_sel = st.selectbox("Candidato", cands_disp, index=0, key="wc_sel_cand") if cands_disp else None
                    if not cand_sel:
                        with ph_wc: st.warning("No hay candidatos disponibles para esta vista.")
                    else:
                        keys = [k for k in counters.keys() if k[0] == cand_sel]
                        keys.sort(key=lambda k: support.get(k, 0.0), reverse=True)
                        _render_wcs(keys[:max_paneles], label_getter=lambda k: k[1])
    except Exception as e:
        with ph_wc: st.error(f"Error generando nubes: {e}")

    st.caption(
        "📌 Para un candidato, todos sus *chunks* suman 100%. El *share* es el % de tokens asociados a cada pilar "
        "sobre el total, ponderado por *w_tokens* para evitar doble conteo cuando un chunk cubre varios subtemas."
    )
