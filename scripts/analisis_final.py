"""
analisis_final.py — Motor de Minería Politológica (Fase 5)
===========================================================
Todas las comparaciones entre candidatos usan CPM (cuentas por millón)
para controlar diferencias de longitud en el corpus.
Los nombres propios (PROPN) ya fueron excluidos en la Fase 3 (analisis_gramatical.py).

Dimensiones:
  1. Top lemas por CPM (sin TF-IDF: muy sensible a toponymia con PROPN filtrados)
  2. Asimetría léxica contrastiva (log-ratio normalizado, Hardie 2014)
  3. Construcción discursiva del adversario (campo semántico a priori + modificadores)
  4. Bloques teóricos (densidad CPM por categoría + evolución temporal por candidato)
  5. Evolución temporal comparada entre candidatos (día relativo, misma escala)
  6. Keyness G² (estadístico de asociación léxica entre candidatos)

Nota: BERTopic, bigramas y TF-IDF fueron evaluados y descartados durante el desarrollo
      de este pipeline (corpus insuficiente, errores de lematización y contaminación por
      toponimia residual, respectivamente). El razonamiento completo de esas decisiones
      está documentado en notebooks/analysis.ipynb, no en este script.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# CONFIGURACIÓN CENTRAL — editá aquí sin tocar el código de análisis
# ---------------------------------------------------------------------------

CANDIDATO_A = "AMLO"
CANDIDATO_B = "Petro"

# Ruido léxico: se filtra SOLO en los análisis de frecuencia, no en el corpus
RUIDO_UNIFICADO = {
    "mexico", "colombia", "nacional", "republica", "estado", "pais",
    "ciudad", "region", "departamento", "municipio", "federacion",
    "latinoamerica", "america", "mundo", "gobierno",
    "presidente", "gobernador", "senador", "contralor", "ministro",
    "secretario", "candidato", "dirigente", "lider", "compañero",
    "señor", "señora", "doctor", "licenciado",
    "ano", "año", "mes", "dia", "hoy", "ayer", "semana", "momento",
    "tiempo", "vez", "periodo", "epoca", "fecha",
    "entonces", "bueno", "ahora", "aqui", "ahi", "pues", "claro",
    "bien", "verdad", "decir", "hacer", "ir", "venir", "tener",
    "poder", "querer", "saber", "ver", "dar", "haber",
    "millon", "millones", "mil", "cien", "ciento", "peso",
    "moreira", "humberto", "bautista", "pan",
    "gente",
}

# Campos semánticos del adversario — asimétricos por candidato (no intercambiables)
# Fundamentados en la literatura de cada caso
CAMPOS_ADVERSARIO = {
    CANDIDATO_A: [  # AMLO — "Mafia del poder" + bloque conservador-neoliberal
        "mafia", "conservador", "oligarca", "corrupto", "corrupcion",
        "prian", "privilegio", "neoliberal", "tecnocrata", "cacique",
        "traidor", "impostor", "fifi", "bloque", "oligarquia",
        "imposicion", "fraude", "robo", "saqueo", "impunidad",
        "cartel", "plutocrata", "rapaz", "parasito", "cupula",
    ],
    CANDIDATO_B: [  # Petro — uribismo + élite extractivista + paramilitarismo
        "oligarquia", "uribismo", "paramilitar", "clan", "casta",
        "corrupto", "corrupcion", "elite", "establecimiento", "gamonal",
        "latifundio", "mafioso", "cacique", "privilegio", "extractivismo",
        "narco", "paraestado", "violencia", "despojo", "represion",
        "feudal", "rentista", "terrateniente", "milico", "clientelismo",
    ],
}

# Bloques teóricos operacionalizados — citable en capítulo metodológico
# Pitkin (1967): representación sustantiva
# Katz y Mair (2009): partidos cartel / captura del Estado
# O'Donnell (1996): accountability y crisis de representación
BLOQUES_TEORICOS = {
    CANDIDATO_A: {
        "Representacion_Popular": [
            "pueblo", "ciudadano", "pobre", "trabajador", "humilde",
            "marginado", "necesitado", "comunidad", "base", "mayoria",
            "abajo", "popular", "nacion", "patria", "hermano", "obrero",
            "campesino", "indigena", "mujer", "joven",
        ],
        "Critica_Al_Cartel": [
            "mafia", "corrupto", "corrupcion", "prian", "privilegio",
            "conservador", "oligarca", "fifi", "tecnocrata", "imposicion",
            "fraude", "robo", "saqueo", "impunidad", "traicion", "cacique",
            "nepotismo", "cartel", "captura", "oligarquia",
        ],
        "Ruptura_Con_Modelo": [
            "transformacion", "cambio", "historia", "cuarta", "regeneracion",
            "honestidad", "justicia", "austeridad", "soberania", "autonomia",
            "independencia", "liberacion", "rescate", "refundacion",
            "alternativa", "esperanza", "nuevo", "ruptura", "proyecto",
        ],
    },
    CANDIDATO_B: {
        "Representacion_Popular": [
            "pueblo", "ciudadano", "excluido", "joven", "pobreza",
            "comunidad", "humilde", "campesino", "indigena", "obrero",
            "trabajador", "mayoria", "hermano", "vida", "mujer",
            "victima", "pobre", "base", "popular", "periferia",
        ],
        "Critica_Al_Cartel": [
            "oligarquia", "uribismo", "casta", "clan", "paramilitar",
            "corrupto", "corrupcion", "elite", "establecimiento", "gamonal",
            "latifundio", "mafioso", "cacique", "privilegio", "extractivismo",
            "narco", "despojo", "represion", "clientelismo", "captura",
        ],
        "Ruptura_Con_Modelo": [
            "pacto", "cambio", "paz", "historico", "democracia", "potencia",
            "vida", "transicion", "justicia", "soberania", "reforma",
            "transformacion", "alternativa", "futuro", "esperanza",
            "posneoliberal", "nuevo", "ruptura", "proyecto", "refundacion",
        ],
    },
}

# ---------------------------------------------------------------------------
# UTILIDADES CENTRALES
# ---------------------------------------------------------------------------

def total_tokens_por_candidato(df):
    """Devuelve un dict {candidato: total_tokens} para normalización CPM."""
    return df.groupby("candidato").size().to_dict()


def filtrar_ruido(df):
    """Elimina tokens del RUIDO_UNIFICADO. Aplicar solo antes de análisis de frecuencia."""
    return df[~df["palabra_lema"].isin(RUIDO_UNIFICADO)].copy()


def top_cpm(df, candidato, totales, top_n=20, pos_filter=None):
    """
    Devuelve un DataFrame con los top_n lemas por CPM para un candidato.
    Usa CPM en lugar de frecuencia absoluta para comparaciones válidas.
    """
    sub = df[df["candidato"] == candidato].copy()
    if pos_filter:
        sub = sub[sub["categoria_gramatical"] == pos_filter]

    conteos = sub["palabra_lema"].value_counts().head(top_n).reset_index()
    conteos.columns = ["lema", "count"]
    conteos["cpm"] = (conteos["count"] / totales[candidato]) * 1_000_000
    return conteos


def guardar_csv(df, carpeta, nombre):
    ruta = os.path.join(carpeta, nombre)
    df.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


def grafico_barras_h(data, x, y, titulo, xlabel, ylabel, carpeta, nombre_archivo, palette="Blues_r"):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    ruta = os.path.join(carpeta, nombre_archivo)
    plt.savefig(ruta, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# DIMENSIÓN 1 — Top lemas por CPM + evolución temporal
# ---------------------------------------------------------------------------

def analizar_conteos_y_evolucion(df, carpeta, totales, top_n=20):
    print("-> Dimensión 1: Top lemas por CPM y evolución temporal...")
    df = filtrar_ruido(df)

    for cand in [CANDIDATO_A, CANDIDATO_B]:
        top = top_cpm(df, cand, totales, top_n=top_n)
        slug = cand.lower().replace(" ", "_")

        guardar_csv(top, carpeta, f"d1_top_lemas_cpm_{slug}.csv")
        grafico_barras_h(
            top, x="cpm", y="lema",
            titulo=f"Palabras más frecuentes (CPM) — {cand}",
            xlabel="Frecuencia por millón de tokens",
            ylabel="Lema",
            carpeta=carpeta,
            nombre_archivo=f"d1_top_lemas_cpm_{slug}.png",
        )

        # Evolución temporal de los top 5
        top5 = top["lema"].head(5).tolist()
        df_cand = df[df["candidato"] == cand].copy()
        df_cand["fecha"] = pd.to_datetime(df_cand["fecha"], errors="coerce")
        df_top5 = df_cand[df_cand["palabra_lema"].isin(top5)]

        if not df_top5.empty:
            # Normalizar por día para no inflar fechas con más discursos
            total_dia = df_cand.groupby("fecha").size().rename("total_dia")
            evolucion = (
                df_top5.groupby(["fecha", "palabra_lema"])
                .size()
                .rename("count")
                .reset_index()
                .merge(total_dia, on="fecha")
            )
            evolucion["cpm"] = (evolucion["count"] / evolucion["total_dia"]) * 1_000_000
            evolucion = evolucion.sort_values("fecha")

            guardar_csv(evolucion, carpeta, f"d1_evolucion_temporal_cpm_{slug}.csv")

            fig, ax = plt.subplots(figsize=(13, 6))
            sns.lineplot(
                data=evolucion, x="fecha", y="cpm",
                hue="palabra_lema", marker="o", linewidth=2, ax=ax,
            )
            ax.set_title(f"Evolución temporal de conceptos clave (CPM) — {cand}",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("CPM (por día)")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(carpeta, f"d1_evolucion_temporal_cpm_{slug}.png"), dpi=150)
            plt.close()


# ---------------------------------------------------------------------------
# DIMENSIÓN 3 — Asimetría léxica contrastiva (log-ratio normalizado)
# ---------------------------------------------------------------------------

def calcular_asimetria_contrastiva(df, carpeta, totales, top_n=15):
    """
    Log-ratio normalizado (Hardie, 2014) en lugar del ratio simple.
    Más robusto para corpus de tamaños distintos.
    """
    print("-> Dimensión 3: Asimetría léxica contrastiva (log-ratio)...")
    df = filtrar_ruido(df)

    pivot = (
        df.groupby(["palabra_lema", "candidato"])
        .size()
        .unstack(fill_value=0)
    )

    # Asegurarse de que ambas columnas existen
    for cand in [CANDIDATO_A, CANDIDATO_B]:
        if cand not in pivot.columns:
            pivot[cand] = 0

    n_a = totales[CANDIDATO_A]
    n_b = totales[CANDIDATO_B]

    # Frecuencia relativa con suavizado Laplace (+1) para evitar log(0)
    pivot["rel_a"] = (pivot[CANDIDATO_A] + 1) / n_a
    pivot["rel_b"] = (pivot[CANDIDATO_B] + 1) / n_b
    pivot["log_ratio"] = np.log2(pivot["rel_a"] / pivot["rel_b"])

    pivot = pivot.reset_index()

    # Más característico de A
    excl_a = pivot.nlargest(top_n, "log_ratio")[["palabra_lema", "log_ratio", CANDIDATO_A, CANDIDATO_B]]
    guardar_csv(excl_a, carpeta, f"d3_asimetria_exclusivos_{CANDIDATO_A.lower()}.csv")
    grafico_barras_h(
        excl_a, x="log_ratio", y="palabra_lema",
        titulo=f"Especificidad léxica: más característico de {CANDIDATO_A} vs {CANDIDATO_B}",
        xlabel="Log-ratio (base 2)",
        ylabel="Lema",
        carpeta=carpeta,
        nombre_archivo=f"d3_asimetria_exclusivos_{CANDIDATO_A.lower()}.png",
        palette="Greens_r",
    )

    # Más característico de B
    excl_b = pivot.nsmallest(top_n, "log_ratio")[["palabra_lema", "log_ratio", CANDIDATO_A, CANDIDATO_B]]
    guardar_csv(excl_b, carpeta, f"d3_asimetria_exclusivos_{CANDIDATO_B.lower()}.csv")
    grafico_barras_h(
        excl_b, x="log_ratio", y="palabra_lema",
        titulo=f"Especificidad léxica: más característico de {CANDIDATO_B} vs {CANDIDATO_A}",
        xlabel="Log-ratio (base 2)",
        ylabel="Lema",
        carpeta=carpeta,
        nombre_archivo=f"d3_asimetria_exclusivos_{CANDIDATO_B.lower()}.png",
        palette="Reds_r",
    )


# ---------------------------------------------------------------------------
# DIMENSIÓN 4 — Construcción discursiva del adversario
# ---------------------------------------------------------------------------

def analizar_construccion_adversario(df, carpeta, totales, top_n=15):
    """
    Lógica invertida respecto a la versión original:
    - Define los campos semánticos del adversario por candidato (CAMPOS_ADVERSARIO)
    - Calcula densidad CPM de cada campo
    - Analiza modificadores (ADJ/VERB) en ventana de ±4 tokens dentro del mismo documento
    - Analiza bigramas internos del campo (qué palabras del campo aparecen juntas)
    """
    print("-> Dimensión 4: Construcción discursiva del adversario...")

    resultados_densidad = []
    resultados_modificadores = []

    for cand, campo in CAMPOS_ADVERSARIO.items():
        df_cand = df[df["candidato"] == cand].copy().reset_index(drop=True)
        total = totales[cand]

        # --- 4a. Densidad CPM del campo completo ---
        en_campo = df_cand[df_cand["palabra_lema"].isin(campo)]
        densidad_cpm = (len(en_campo) / total) * 1_000_000
        resultados_densidad.append({
            "candidato": cand,
            "tokens_campo_adversario": len(en_campo),
            "cpm_campo_adversario": densidad_cpm,
        })

        # --- 4b. Frecuencia CPM de cada término del campo ---
        freq_terminos = (
            en_campo["palabra_lema"]
            .value_counts()
            .reset_index()
        )
        freq_terminos.columns = ["termino", "count"]
        freq_terminos["cpm"] = (freq_terminos["count"] / total) * 1_000_000
        slug = cand.lower().replace(" ", "_")
        guardar_csv(freq_terminos, carpeta, f"d4_terminos_adversario_{slug}.csv")
        grafico_barras_h(
            freq_terminos.head(top_n), x="cpm", y="termino",
            titulo=f"Campo semántico del adversario (CPM) — {cand}",
            xlabel="Frecuencia por millón de tokens",
            ylabel="Término",
            carpeta=carpeta,
            nombre_archivo=f"d4_terminos_adversario_cpm_{slug}.png",
            palette="Reds_r",
        )

        # --- 4c. Modificadores (ADJ/VERB) en ventana ±4 por documento ---
        for archivo, df_doc in df_cand.groupby("archivo_origen"):
            tokens = df_doc.reset_index(drop=True)
            posiciones_campo = tokens[tokens["palabra_lema"].isin(campo)].index

            for pos in posiciones_campo:
                inicio = max(0, pos - 4)
                fin = min(len(tokens) - 1, pos + 4)
                for i in range(inicio, fin + 1):
                    if i == pos:
                        continue
                    fila_ctx = tokens.loc[i]
                    if fila_ctx["categoria_gramatical"] in ("ADJ", "VERB"):
                        resultados_modificadores.append({
                            "candidato": cand,
                            "termino_campo": tokens.loc[pos, "palabra_lema"],
                            "modificador": fila_ctx["palabra_lema"],
                            "tipo": fila_ctx["categoria_gramatical"],
                        })

    # Guardar densidades comparativas
    df_densidad = pd.DataFrame(resultados_densidad)
    guardar_csv(df_densidad, carpeta, "d4_densidad_campo_adversario.csv")

    # Graficar densidad comparativa
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_densidad, x="candidato", y="cpm_campo_adversario",
                palette="Reds", ax=ax)
    ax.set_title("Densidad del campo semántico del adversario (CPM)", fontweight="bold")
    ax.set_xlabel("Candidato")
    ax.set_ylabel("CPM")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "d4_densidad_adversario_comparativa.png"), dpi=150)
    plt.close()

    # Modificadores por candidato y tipo
    if resultados_modificadores:
        df_mod = pd.DataFrame(resultados_modificadores)
        guardar_csv(df_mod, carpeta, "d4_modificadores_adversario_raw.csv")

        for cand in [CANDIDATO_A, CANDIDATO_B]:
            slug = cand.lower().replace(" ", "_")
            df_mod_cand = df_mod[df_mod["candidato"] == cand]
            for tipo in ("ADJ", "VERB"):
                res = (
                    df_mod_cand[df_mod_cand["tipo"] == tipo]["modificador"]
                    .value_counts()
                    .head(top_n)
                    .reset_index()
                )
                res.columns = ["modificador", "frecuencia"]
                guardar_csv(res, carpeta, f"d4_modificadores_{tipo.lower()}_{slug}.csv")
                grafico_barras_h(
                    res, x="frecuencia", y="modificador",
                    titulo=f"Modificadores ({tipo}) del campo adversario — {cand}",
                    xlabel="Frecuencia de coocurrencia",
                    ylabel="Lema",
                    carpeta=carpeta,
                    nombre_archivo=f"d4_modificadores_{tipo.lower()}_{slug}.png",
                    palette="Purples_r",
                )


# ---------------------------------------------------------------------------
# DIMENSIÓN 6 — Bloques teóricos: densidad CPM + evolución temporal
# ---------------------------------------------------------------------------

def analizar_bloques_teoricos(df, carpeta, totales):
    """
    Para cada candidato y categoría teórica:
    - Densidad CPM (comparable entre candidatos)
    - Evolución temporal de la densidad (por fecha normalizada)
    - Gráfico comparativo entre candidatos por categoría
    """
    print("-> Dimensión 6: Bloques teóricos (densidad CPM + evolución)...")

    resultados = []

    for cand, categorias in BLOQUES_TEORICOS.items():
        df_cand = df[df["candidato"] == cand].copy()
        total = totales[cand]

        for categoria, palabras in categorias.items():
            en_cat = df_cand[df_cand["palabra_lema"].isin(palabras)]
            count = len(en_cat)
            cpm = (count / total) * 1_000_000
            resultados.append({
                "candidato": cand,
                "categoria_teorica": categoria,
                "frecuencia_absoluta": count,
                "cpm": cpm,
            })

    df_bloques = pd.DataFrame(resultados)
    guardar_csv(df_bloques, carpeta, "d6_bloques_teoricos_cpm.csv")

    # Gráfico comparativo por categoría
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=df_bloques, x="categoria_teorica", y="cpm",
        hue="candidato", palette="muted", ax=ax,
    )
    ax.set_title(
        "Saturación de categorías teóricas en el discurso electoral (CPM)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Categoría analítica")
    ax.set_ylabel("CPM (tokens por millón)")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "d6_comparativa_bloques_teoricos_cpm.png"), dpi=150)
    plt.close()

    # --- Evolución temporal de cada bloque por candidato ---
    for cand, categorias in BLOQUES_TEORICOS.items():
        slug = cand.lower().replace(" ", "_")
        df_cand = df[df["candidato"] == cand].copy()
        df_cand["fecha"] = pd.to_datetime(df_cand["fecha"], errors="coerce")
        total_dia = df_cand.groupby("fecha").size().rename("total_dia")

        registros_evol = []
        for categoria, palabras in categorias.items():
            en_cat = df_cand[df_cand["palabra_lema"].isin(palabras)].copy()
            if en_cat.empty:
                continue
            por_dia = (
                en_cat.groupby("fecha").size()
                .rename("count")
                .reset_index()
                .merge(total_dia, on="fecha")
            )
            por_dia["cpm"] = (por_dia["count"] / por_dia["total_dia"]) * 1_000_000
            por_dia["categoria"] = categoria
            registros_evol.append(por_dia)

        if not registros_evol:
            continue

        df_evol = pd.concat(registros_evol).sort_values("fecha")
        guardar_csv(df_evol, carpeta, f"d6_evolucion_bloques_{slug}.csv")

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.lineplot(
            data=df_evol, x="fecha", y="cpm",
            hue="categoria", marker="o", linewidth=2, ax=ax,
        )
        ax.set_title(
            f"Evolución temporal de categorías teóricas (CPM) — {cand}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Fecha")
        ax.set_ylabel("CPM")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"d6_evolucion_bloques_{slug}.png"), dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# DIMENSIÓN 5 — Evolución temporal comparada entre candidatos
# ---------------------------------------------------------------------------

def analizar_evolucion_comparada(df, carpeta, totales):
    """
    Compara la evolución de los bloques teóricos entre candidatos usando
    día relativo (Día 0 = primer discurso de cada candidato), lo que hace
    comparables dos campañas en calendarios distintos.

    Para cada categoría teórica genera un gráfico con ambos candidatos
    en la misma escala CPM, permitiendo ver convergencias y divergencias
    en el timing discursivo.
    """
    print("-> Dim 5: Evolución temporal comparada (día relativo)...")

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Calcular día relativo por candidato (Día 0 = primer discurso)
    fecha_inicio = df.groupby("candidato")["fecha"].min().rename("fecha_inicio")
    df = df.merge(fecha_inicio, on="candidato")
    df["dia_relativo"] = (df["fecha"] - df["fecha_inicio"]).dt.days

    # Categorías compartidas entre ambos candidatos (intersección de claves)
    categorias_a = set(BLOQUES_TEORICOS.get(CANDIDATO_A, {}).keys())
    categorias_b = set(BLOQUES_TEORICOS.get(CANDIDATO_B, {}).keys())
    categorias_comunes = categorias_a & categorias_b

    registros = []
    for cand, categorias in BLOQUES_TEORICOS.items():
        df_cand = df[df["candidato"] == cand].copy()
        total = totales[cand]
        total_dia = df_cand.groupby("dia_relativo").size().rename("total_dia")

        for categoria, palabras in categorias.items():
            if categoria not in categorias_comunes:
                continue
            en_cat = df_cand[df_cand["palabra_lema"].isin(palabras)].copy()
            if en_cat.empty:
                continue
            por_dia = (
                en_cat.groupby("dia_relativo").size()
                .rename("count")
                .reset_index()
                .merge(total_dia, on="dia_relativo")
            )
            por_dia["cpm"] = (por_dia["count"] / por_dia["total_dia"]) * 1_000_000
            por_dia["categoria"] = categoria
            por_dia["candidato"] = cand
            registros.append(por_dia)

    if not registros:
        print("  [!] Sin datos para evolución comparada.")
        return

    df_comp = pd.concat(registros, ignore_index=True)
    guardar_csv(df_comp, carpeta, "d5_evolucion_comparada.csv")

    # Un gráfico por categoría teórica con ambos candidatos superpuestos
    colores = {CANDIDATO_A: "#2196F3", CANDIDATO_B: "#E64A19"}

    for categoria in categorias_comunes:
        sub = df_comp[df_comp["categoria"] == categoria]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(13, 6))
        for cand in [CANDIDATO_A, CANDIDATO_B]:
            data_cand = sub[sub["candidato"] == cand].sort_values("dia_relativo")
            if data_cand.empty:
                continue
            ax.plot(
                data_cand["dia_relativo"],
                data_cand["cpm"],
                label=cand,
                color=colores[cand],
                linewidth=2.5,
                marker="o",
                markersize=4,
                alpha=0.85,
            )
            # Línea de tendencia suavizada (media móvil 7 días)
            if len(data_cand) >= 7:
                data_cand = data_cand.set_index("dia_relativo")
                media_movil = data_cand["cpm"].rolling(window=7, center=True, min_periods=3).mean()
                ax.plot(
                    media_movil.index,
                    media_movil.values,
                    color=colores[cand],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.5,
                    label=f"{cand} (tendencia)",
                )

        ax.set_title(
            f"Evolución comparada — {categoria.replace('_', ' ')}\n"
            f"(Día 0 = primer discurso de campaña de cada candidato)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Día relativo de campaña")
        ax.set_ylabel("CPM (tokens por millón, normalizado por día)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        nombre_archivo = f"d5_evolucion_comparada_{categoria.lower()}.png"
        plt.savefig(os.path.join(carpeta, nombre_archivo), dpi=150)
        plt.close()

    # Gráfico resumen: densidad promedio por categoría y candidato (barras agrupadas)
    resumen = (
        df_comp.groupby(["candidato", "categoria"])["cpm"]
        .mean()
        .reset_index(name="cpm_promedio")
    )
    guardar_csv(resumen, carpeta, "d5_resumen_comparado_categorias.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=resumen, x="categoria", y="cpm_promedio",
        hue="candidato", palette=colores, ax=ax,
    )
    ax.set_title(
        f"Densidad promedio de categorías teóricas — {CANDIDATO_A} vs {CANDIDATO_B}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Categoría analítica")
    ax.set_ylabel("CPM promedio")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "d5_resumen_comparado_categorias.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# DIMENSIÓN 6 — Keyness G² + CPM global
# ---------------------------------------------------------------------------

def calcular_keyness_g2(df, carpeta, totales):
    print("-> Dimensión 7: Keyness G² y CPM global...")
    df = filtrar_ruido(df)

    freq = df.groupby(["candidato", "palabra_lema"]).size().reset_index(name="count")
    for cand in [CANDIDATO_A, CANDIDATO_B]:
        freq.loc[freq["candidato"] == cand, "total"] = totales[cand]
    freq["cpm"] = (freq["count"] / freq["total"]) * 1_000_000
    guardar_csv(freq, carpeta, "d7_cpm_global.csv")

    # G² entre los dos candidatos
    pivot = freq.pivot(index="palabra_lema", columns="candidato", values="count").fillna(0)
    for cand in [CANDIDATO_A, CANDIDATO_B]:
        if cand not in pivot.columns:
            pivot[cand] = 0

    n_a, n_b = totales[CANDIDATO_A], totales[CANDIDATO_B]
    pivot["total_obs"] = pivot[CANDIDATO_A] + pivot[CANDIDATO_B]
    total_general = n_a + n_b
    pivot["exp_a"] = (pivot["total_obs"] * n_a) / total_general
    pivot["exp_b"] = (pivot["total_obs"] * n_b) / total_general

    def log_lik(obs, exp):
        if obs == 0 or exp == 0:
            return 0.0
        return 2 * obs * np.log(obs / exp)

    pivot["g2_a"] = pivot.apply(lambda r: log_lik(r[CANDIDATO_A], r["exp_a"]), axis=1)
    pivot["g2_b"] = pivot.apply(lambda r: log_lik(r[CANDIDATO_B], r["exp_b"]), axis=1)
    pivot["keyness"] = np.where(
        pivot[CANDIDATO_A] > pivot["exp_a"],
        pivot["g2_a"],
        -pivot["g2_b"],
    )

    df_keyness = pivot[["keyness", "total_obs", CANDIDATO_A, CANDIDATO_B]].sort_values(
        "keyness", ascending=False
    )
    guardar_csv(df_keyness.reset_index(), carpeta, "d7_keyness_g2.csv")

    # Top keywords por candidato
    top_a = df_keyness.head(15).reset_index()
    top_b = df_keyness.tail(15).sort_values("keyness").reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=top_a, x="keyness", y="palabra_lema", palette="Blues_r", ax=axes[0])
    axes[0].set_title(f"Keywords G² — {CANDIDATO_A}", fontweight="bold")
    axes[0].set_xlabel("G² (keyness)")
    axes[0].set_ylabel("Lema")

    sns.barplot(data=top_b, x="keyness", y="palabra_lema", palette="Oranges_r", ax=axes[1])
    axes[1].set_title(f"Keywords G² — {CANDIDATO_B}", fontweight="bold")
    axes[1].set_xlabel("G² (keyness, negativo = más frecuente en Petro)")
    axes[1].set_ylabel("")

    plt.suptitle(f"Análisis de Keyness G²: {CANDIDATO_A} vs {CANDIDATO_B}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, "d7_keyness_g2_comparativo.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    CARPETA_TRABAJO = os.path.dirname(os.path.abspath(__file__))
    ruta_entrada = os.path.join(CARPETA_TRABAJO, "dataset_tokens_analizados.csv")
    carpeta_salidas = os.path.join(CARPETA_TRABAJO, "matrices_finales_tesis")
    os.makedirs(carpeta_salidas, exist_ok=True)

    print("=" * 65)
    print("     MOTOR DE MINERÍA POLITOLÓGICA — TESIS FSoc UBA")
    print("=" * 65)

    if not os.path.exists(ruta_entrada):
        print(f"Error: falta '{ruta_entrada}'. Ejecutá primero la Fase 3.", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(ruta_entrada)
        df["palabra_lema"] = df["palabra_lema"].astype(str).str.lower().str.strip()

        totales = total_tokens_por_candidato(df)
        print(f"\nTokens totales por candidato (base CPM):")
        for cand, n in totales.items():
            print(f"  {cand}: {n:,} tokens")

        print("\n-> Dim 1: Top lemas por CPM")
        analizar_conteos_y_evolucion(df, carpeta_salidas, totales)
        print("-> Dim 2: Asimetría léxica (log-ratio)")
        calcular_asimetria_contrastiva(df, carpeta_salidas, totales)
        print("-> Dim 3: Campo semántico del adversario")
        analizar_construccion_adversario(df, carpeta_salidas, totales)
        print("-> Dim 4: Bloques teóricos")
        analizar_bloques_teoricos(df, carpeta_salidas, totales)
        print("-> Dim 5: Evolución temporal comparada")
        analizar_evolucion_comparada(df, carpeta_salidas, totales)
        print("-> Dim 6: Keyness G²")
        calcular_keyness_g2(df, carpeta_salidas, totales)

        print("\n" + "=" * 65)
        print(f" ¡PROCESAMIENTO COMPLETO!")
        print(f" Outputs en: {carpeta_salidas}")
        print("=" * 65)

    except Exception:
        print(f"Fallo crítico:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
