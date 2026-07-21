import argparse
import os
import sys
import traceback
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Fase 4: Generador de Reportes. Extrae las palabras, sustantivos, verbos y adjetivos más frecuentes por actor político."
    )
    p.add_argument(
        "--input",
        "-i",
        help="Nombre del CSV de tokens de la Fase 3 (por defecto: dataset_tokens_analizados.csv)",
        default="dataset_tokens_analizados.csv",
    )
    p.add_argument(
        "--top",
        "-t",
        type=int,
        help="Cantidad de palabras a extraer por categoría (por defecto: 20)",
        default=20,
    )
    return p.parse_args()


def exportar_y_mostrar_top(df, candidato, pos_tag, nombre_pos, top_n, carpeta):
    """Filtra la matriz por candidato y categoría gramatical, cuenta frecuencias y exporta."""
    # Filtrar por candidato
    df_cand = df[df["candidato"] == candidato]

    # Filtrar por categoría gramatical si se especifica
    if pos_tag:
        df_filtrado = df_cand[df_cand["categoria_gramatical"] == pos_tag]
        titulo = f"TOP {top_n} {nombre_pos.upper()} - {candidato}"
        archivo_salida = f"top_{top_n}_{nombre_pos.lower()}_{candidato.lower()}.csv"
    else:
        df_filtrado = df_cand
        titulo = f"TOP {top_n} PALABRAS CLAVE GLOBALES - {candidato}"
        archivo_salida = f"top_{top_n}_global_{candidato.lower()}.csv"

    # Calcular frecuencias de los lemas
    frecuencias = (
        df_filtrado["palabra_lema"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    frecuencias.columns = ["lema", "frecuencia"]

    # Imprimir reporte en pantalla
    print("\n" + "=" * 45)
    print(f" {titulo}")
    print("=" * 45)
    for idx, fila in frecuencias.iterrows():
        print(f" {idx+1:>2}. {fila['lema']:<25} | {fila['frecuencia']:>5} veces")

    # Guardar en CSV
    ruta_exportacion = os.path.join(carpeta, archivo_salida)
    frecuencias.to_csv(ruta_exportacion, index=False, encoding="utf-8")


def main():
    ARGS = parse_args()
    CARPETA_TRABAJO = os.path.dirname(os.path.abspath(__file__))

    ruta_entrada = (
        ARGS.input
        if os.path.isabs(ARGS.input)
        else os.path.join(CARPETA_TRABAJO, ARGS.input)
    )

    # Crear una carpeta interna para no inundar la raíz con los reportes CSV
    carpeta_reportes = os.path.join(CARPETA_TRABAJO, "reportes_retorica")
    os.makedirs(carpeta_reportes, exist_ok=True)

    if not os.path.exists(ruta_entrada):
        print(
            f"Error: No se encontró la matriz de tokens en '{ruta_entrada}'. Ejecuta la Fase 3 primero.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        print(f"Leyendo matriz atómica de tokens desde: {ruta_entrada}...")
        df = pd.read_csv(ruta_entrada)

        # Identificar candidatos únicos en tu corpus (ej: ['Andrés Manuel López Obrador', 'Petro'])
        candidatos = df["candidato"].unique()

        # Categorías gramaticales que nos interesan metodológicamente
        categorias_analisis = [
            (None, "Global"),  # Sin filtro (vocabulario general)
            ("NOUN", "Sustantivos"),  # Agenda temática
            ("VERB", "Verbos"),  # Agenda de acción
            ("ADJ", "Adjetivos"),  # Encuadre ideológico
        ]

        # Ejecutar el bucle cruzado por cada candidato y cada categoría
        for candidato in candidatos:
            print(f"\nProcesando análisis retórico para: {candidato}")
            for pos_tag, nombre_pos in zip(*zip(*categorias_analisis)):
                exportar_y_mostrar_top(
                    df,
                    candidato,
                    pos_tag,
                    nombre_pos,
                    ARGS.top,
                    carpeta_reportes,
                )

        print("\n" + "=" * 60)
        print(" ¡PROCESO FINALIZADO CON ÉXITO!")
        print(f" Las tablas CSV han sido guardadas en la carpeta:\n --> {carpeta_reportes}")
        print("=" * 60)

    except Exception:
        tb = traceback.format_exc()
        print(
            f"Fallo catastrófico al generar los reportes:\n{tb}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()