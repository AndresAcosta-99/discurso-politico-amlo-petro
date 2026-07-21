import argparse
import csv
import logging
import os
import re
import sys
import traceback
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Fase 2 (Optimizado): Purifica el texto e integra fechas usando el log del downloader-transcriptor."
    )
    p.add_argument(
        "--input",
        "-i",
        help="CSV de la Fase 1 (por defecto: dataset_textos_extraidos.csv)",
        default="dataset_textos_extraidos.csv",
    )
    p.add_argument(
        "--metadata",
        "-m",
        help="Archivo log del transcriptor (ej: transcriptions_log_AMLO.csv)",
        required=True,  # Ahora es requerido para forzar la triangulación temporal
    )
    p.add_argument(
        "--output",
        "-o",
        help="CSV limpio resultante (por defecto: dataset_textos_limpios.csv)",
        default="dataset_textos_limpios.csv",
    )
    return p.parse_args()


def purificar_discurso(texto):
    """Remueve encabezados, marcas de tiempo, puntuación y normaliza a minúsculas sin tildes."""
    if not isinstance(texto, str) or not texto.strip():
        return ""

    # 1. Ruido del transcriptor
    texto = re.sub(r"(?i)transcripción de audio", " ", texto)
    texto = re.sub(r"\[\d{2}:\d{2}\s*(?:→|->|-)\s*\d{2}:\d{2}\]", " ", texto)

    # 2. Normalización básica
    texto = texto.lower()
    remplazo_tildes = str.maketrans("áéíóúü", "aeiouu")
    texto = texto.translate(remplazo_tildes)

    # 3. Puntuación y espacios
    texto = re.sub(r"[^\w\s]|_", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# REEMPLAZAR EN CLEAN.PY

def simplificar_nombre_archivo(nombre):
    if not isinstance(nombre, str):
        return ""
    
    # 1. Pasar a minúsculas
    nombre_limpio = nombre.lower()
    
    # 2. Quitar extensiones comunes (.docx, .doc, .csv) si vienen integradas
    if nombre_limpio.endswith(".docx"):
        nombre_limpio = nombre_limpio[:-5]
    elif nombre_limpio.endswith(".doc") or nombre_limpio.endswith(".csv"):
        nombre_limpio = nombre_limpio[:-4]
        
    # 3. Remover tildes de forma manual y robusta
    remplazo_tildes = str.maketrans("áéíóúüñ", "aeiouun")
    nombre_limpio = nombre_limpio.translate(remplazo_tildes)
    
    # 4. Eliminar CUALQUIER cosa que no sea una letra de la 'a' a la 'z' o un número (0-9)
    nombre_limpio = re.sub(r"[^a-z0-9]", "", nombre_limpio)
    
    return nombre_limpio.strip()

def main():
    ARGS = parse_args()
    CARPETA_TRABAJO = os.path.dirname(os.path.abspath(__file__))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info(
        "--- Iniciando Pipeline de Limpieza y Triangulación Temporal ---"
    )

    ruta_entrada = (
        ARGS.input
        if os.path.isabs(ARGS.input)
        else os.path.join(CARPETA_TRABAJO, ARGS.input)
    )
    ruta_metadata = (
        ARGS.metadata
        if os.path.isabs(ARGS.metadata)
        else os.path.join(CARPETA_TRABAJO, ARGS.metadata)
    )
    ruta_salida = (
        ARGS.output
        if os.path.isabs(ARGS.output)
        else os.path.join(CARPETA_TRABAJO, ARGS.output)
    )

    if not os.path.exists(ruta_entrada):
        logging.error(f"No se encontró el CSV de la Fase 1 en: {ruta_entrada}.")
        sys.exit(1)

    if not os.path.exists(ruta_metadata):
        logging.error(f"No se encontró el archivo de log en: {ruta_metadata}.")
        sys.exit(1)

    # 1. PROCESAMIENTO DEL LOG DEL TRANSCRIPTOR (HEADERLESS)
    dict_fechas = {}
    logging.info(f"Cargando log del transcriptor desde: {ruta_metadata}")
    try:
        # Forzamos header=None porque el archivo no tiene nombres de columnas nativos
        df_meta = pd.read_csv(ruta_metadata, header=None)

        # Columna 0 = Título/Archivo, Columna 1 = Fecha
        for _, fila in df_meta.iterrows():
            titulo_original = str(fila[0])
            fecha_valor = str(fila[1])

            clave_busqueda = simplificar_nombre_archivo(titulo_original)
            if clave_busqueda:
                dict_fechas[clave_busqueda] = fecha_valor

        logging.info(
            f"Diccionario indexado creado con {len(dict_fechas)} llaves cronológicas."
        )
    except Exception as em:
        logging.critical(f"Error crítico leyendo el log de fechas: {em}")
        sys.exit(1)

    # 2. PROCESAR DATASET DE DISCURSOS EXTRAÍDOS
    try:
        logging.info(f"Cargando extracto de la Fase 1: {ruta_entrada}")
        df = pd.read_csv(ruta_entrada)

        df_exitosos = df[df["estado_proceso"] == "Exitoso"].copy()
        if df_exitosos.empty:
            logging.warning("No hay discursos exitosos para limpiar.")
            sys.exit(0)

        # 3. COMPONENTE DE LIMPIEZA TEXTUAL
        logging.info("Purificando textos (removiendo marcas de tiempo y ruido)...")
        df_exitosos["texto_limpio"] = df_exitosos["texto_crudo"].apply(
            purificar_discurso
        )
        df_exitosos["conteo_palabras_limpias"] = df_exitosos["texto_limpio"].apply(
            lambda t: len(t.split()) if t else 0
        )

        # 4. TRIANGULACIÓN POR COINCIDENCIA DIFUSA
        logging.info("Cruzando archivos con sus fechas correspondientes...")
        fechas_asignadas = []
        no_encontrados = 0

        for idx, fila in df_exitosos.iterrows():
            archivo_word = str(fila["archivo_origen"])
            clave_word = simplificar_nombre_archivo(archivo_word)

            if clave_word in dict_fechas:
                fechas_asignadas.append(dict_fechas[clave_word])
            else:
                fechas_asignadas.append("Sin Fecha")
                no_encontrados += 1
                logging.warning(
                    f"⚠️ No se pudo triangular la fecha para el archivo: '{archivo_word}'"
                )

        df_exitosos["fecha"] = fechas_asignadas

        # 5. MATRIZ DE SALIDA PURA
        columnas_finales = [
            "archivo_origen",
            "candidato",
            "fecha",
            "texto_limpio",
            "conteo_palabras_limpias",
        ]
        df_final = df_exitosos[columnas_finales]

        df_final.to_csv(
            ruta_salida, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8"
        )

        # 6. MÉTRICAS DE CONTROL
        logging.info("=" * 50)
        logging.info("          REPORTE DE CALIDAD Y CRONOLOGÍA")
        logging.info("=" * 50)
        logging.info(f"Total discursos purificados: {len(df_final)}")
        logging.info(
            f"Fechas vinculadas con éxito: {len(df_final) - no_encontrados}"
        )
        if no_encontrados > 0:
            logging.warning(
                f"Archivos huérfanos (sin fecha): {no_encontrados}. Revisa si los nombres difieren drásticamente."
            )
        logging.info("=" * 50)
        logging.info(f"Matriz de la Fase 2 guardada en:\n--> {ruta_salida}")

    except Exception:
        tb = traceback.format_exc()
        logging.critical(f"Fallo catastrófico en la Fase 2:\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()