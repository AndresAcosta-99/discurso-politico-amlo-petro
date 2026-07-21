"""
analisis_tematico.py — Fase 6: Topic Modeling con BERTopic
===========================================================
Ajustado para corpus pequeños (~50-70 documentos por candidato).
Cambios clave respecto a la versión original:
  - UMAP con n_neighbors=5 (corpus chico; valor original causaba colapso en tema -1)
  - HDBSCAN con min_cluster_size=5 (evita que todo quede como "ruido")
  - random_state=42 en UMAP para reproducibilidad (requisito metodológico de tesis)
  - nr_topics forzado a un máximo razonable para el tamaño del corpus
  - Análisis separado por candidato (no conjunto) para comparar temáticas
  - Exporta tabla de temas + gráfico de evolución temporal por candidato
  - Normaliza por día (CPM) para no inflar fechas con más discursos
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# PARÁMETROS — ajustá aquí sin tocar el código de análisis
# ---------------------------------------------------------------------------

CANDIDATO_A = "AMLO"
CANDIDATO_B = "Petro"

# Periodo de campaña electoral de cada candidato
# BERTopic se entrena solo con discursos de campaña, no con todo el corpus
PERIODOS_CAMPAÑA = {
    CANDIDATO_A: (2018, 2018),   # año inicio, año fin inclusive
    CANDIDATO_B: (2021, 2022),   # incluye campaña primera vuelta + ballottage
}

# Para corpus ~50-70 docs por candidato estos son los valores óptimos
NR_TOPICS = 8          # máximo de temas; BERTopic puede devolver menos
N_NEIGHBORS_UMAP = 5   # bajo para corpus chico (default 15 colapsa todo)
N_COMPONENTS_UMAP = 5
MIN_CLUSTER_SIZE = 5   # HDBSCAN; mínimo de docs para formar un tema
MIN_SAMPLES_HDBSCAN = 3
RANDOM_STATE = 42      # reproducibilidad

SPANISH_STOP_WORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes",
    "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "es", "ser",
    "son", "fue", "sido", "hacer", "tener", "decir", "haber", "poder",
    "mexico", "colombia", "pais", "gobierno", "presidente", "candidato",
    "ano", "año", "dia", "vez", "tiempo",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def filtrar_campaña(df, candidato):
    """Filtra por candidato y periodo de campaña definido en PERIODOS_CAMPAÑA."""
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    año_ini, año_fin = PERIODOS_CAMPAÑA[candidato]
    mask = (
        (df["candidato"] == candidato)
        & (df["fecha"].dt.year >= año_ini)
        & (df["fecha"].dt.year <= año_fin)
    )
    sub = df[mask].copy()
    sub["fecha_inicio"] = sub["fecha"].min()
    sub["dia_relativo"] = (sub["fecha"] - sub["fecha_inicio"]).dt.days
    return sub


def cargar_bertopic():
    """Carga BERTopic con manejo de import y parámetros para corpus chico."""
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        return BERTopic, UMAP, HDBSCAN, CountVectorizer
    except ImportError as e:
        print(
            f"Error de dependencias: {e}\n"
            "Instalá con: pip install bertopic umap-learn hdbscan",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# ANÁLISIS PRINCIPAL
# ---------------------------------------------------------------------------

def ejecutar_bertopic_por_candidato(df, candidato, carpeta_salida):
    """
    Entrena BERTopic con los discursos de campaña de un candidato.
    Exporta tabla de temas y gráfico de evolución temporal (CPM por día).
    """
    BERTopic, UMAP, HDBSCAN, CountVectorizer = cargar_bertopic()

    df_cand = filtrar_campaña(df, candidato)
    if df_cand.empty:
        print(f"  [!] Sin discursos de campaña para {candidato}. Saltando.")
        return

    # Un documento = un discurso completo (texto_limpio)
    textos = df_cand["texto_limpio"].dropna().astype(str).tolist()
    print(f"  {candidato}: {len(textos)} discursos de campaña para BERTopic")

    if len(textos) < 10:
        print(f"  [!] Menos de 10 documentos para {candidato}. BERTopic requiere más datos.")
        return

    # Modelos con parámetros para corpus chico
    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS_UMAP,
        n_components=N_COMPONENTS_UMAP,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,  # reproducibilidad
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES_HDBSCAN,
        prediction_data=True,
        gen_min_span_tree=True,
    )
    vectorizer_model = CountVectorizer(stop_words=SPANISH_STOP_WORDS, min_df=2)

    topic_model = BERTopic(
        language="spanish",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=NR_TOPICS,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(textos)
    df_cand = df_cand.iloc[: len(topics)].copy()
    df_cand["tema_id"] = topics

    # Tabla de temas
    info_temas = topic_model.get_topic_info()
    slug = candidato.lower().replace(" ", "_")
    info_temas.to_csv(
        os.path.join(carpeta_salida, f"bertopic_temas_{slug}.csv"), index=False
    )

    # Etiquetas legibles
    nombres_temas = {
        row["Topic"]: f"T{row['Topic']}: {row['Name']}"
        for _, row in info_temas.iterrows()
    }
    df_cand["tema_nombre"] = df_cand["tema_id"].map(nombres_temas)

    # Exportar asignación discurso → tema
    df_cand[["archivo_origen", "fecha", "dia_relativo", "tema_id", "tema_nombre"]].to_csv(
        os.path.join(carpeta_salida, f"bertopic_asignacion_{slug}.csv"), index=False
    )

    # --- Evolución temporal normalizada (CPM por día) ---
    # Usamos dia_relativo para que la comparación sea independiente del calendario
    total_por_dia = df_cand.groupby("dia_relativo").size().rename("total_dia")

    evolucion = (
        df_cand[df_cand["tema_id"] != -1]  # excluir documentos sin tema
        .groupby(["dia_relativo", "tema_nombre"])
        .size()
        .rename("count")
        .reset_index()
        .merge(total_por_dia, on="dia_relativo")
    )
    evolucion["cpm"] = (evolucion["count"] / evolucion["total_dia"]) * 1_000_000

    evolucion.to_csv(
        os.path.join(carpeta_salida, f"bertopic_evolucion_{slug}.csv"), index=False
    )

    # Gráfico de evolución temática
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot = evolucion.pivot(index="dia_relativo", columns="tema_nombre", values="cpm").fillna(0)
    pivot.plot(kind="area", stacked=True, alpha=0.7, ax=ax)
    ax.set_title(
        f"Evolución temática BERTopic — {candidato} (campaña electoral)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Día relativo de campaña (Día 0 = primer discurso)")
    ax.set_ylabel("Densidad de temas (CPM por día)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(carpeta_salida, f"bertopic_evolucion_{slug}.png"), dpi=150
    )
    plt.close()

    print(f"  -> BERTopic completado para {candidato}. "
          f"Temas encontrados: {info_temas['Topic'].nunique() - 1} "
          f"(+ clase -1 de ruido)")


def comparar_temas_entre_candidatos(carpeta_salida):
    """
    Lee las tablas de temas de ambos candidatos y genera una tabla comparativa
    con las palabras top de cada tema, para facilitar la lectura en el capítulo.
    """
    registros = []
    for cand in [CANDIDATO_A, CANDIDATO_B]:
        slug = cand.lower().replace(" ", "_")
        ruta = os.path.join(carpeta_salida, f"bertopic_temas_{slug}.csv")
        if not os.path.exists(ruta):
            continue
        df_t = pd.read_csv(ruta)
        df_t["candidato"] = cand
        registros.append(df_t)

    if not registros:
        return

    df_comp = pd.concat(registros, ignore_index=True)
    df_comp = df_comp[df_comp["Topic"] != -1]  # excluir clase de ruido
    df_comp.to_csv(
        os.path.join(carpeta_salida, "bertopic_comparacion_temas.csv"), index=False
    )
    print("  -> Tabla comparativa de temas exportada.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    CARPETA_TRABAJO = os.path.dirname(os.path.abspath(__file__))
    ruta_entrada = os.path.join(CARPETA_TRABAJO, "dataset_textos_limpios.csv")
    carpeta_salida = os.path.join(CARPETA_TRABAJO, "matrices_finales_tesis")
    os.makedirs(carpeta_salida, exist_ok=True)

    print("=" * 65)
    print("     FASE 6 — TOPIC MODELING CON BERTOPIC")
    print("=" * 65)

    if not os.path.exists(ruta_entrada):
        print(f"Error: No se encontró '{ruta_entrada}'. Ejecutá Fase 2 primero.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(ruta_entrada)

    for candidato in [CANDIDATO_A, CANDIDATO_B]:
        print(f"\n--- Procesando: {candidato} ---")
        try:
            ejecutar_bertopic_por_candidato(df, candidato, carpeta_salida)
        except Exception as e:
            print(f"  [!] Error en BERTopic para {candidato}: {e}", file=sys.stderr)
            # No abortamos: si falla uno, seguimos con el otro
            continue

    print("\n--- Generando tabla comparativa de temas ---")
    comparar_temas_entre_candidatos(carpeta_salida)

    print("\n" + "=" * 65)
    print(f" ¡FASE 6 COMPLETADA! Outputs en: {carpeta_salida}")
    print("=" * 65)


if __name__ == "__main__":
    main()
