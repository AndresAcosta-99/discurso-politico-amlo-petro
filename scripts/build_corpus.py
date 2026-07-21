import argparse
import csv
import logging
import os
import traceback
import pandas as pd
from docx import Document


def parse_args():
    p = argparse.ArgumentParser(
        description="Extrae textos de .docx organizados por subcarpeta (candidato)"
    )
    p.add_argument(
        "--root",
        "-r",
        help="Carpeta raíz a procesar (por defecto: carpeta del script)",
    )
    p.add_argument(
        "--output",
        "-o",
        help="Nombre o ruta del CSV de salida",
        default="dataset_textos_extraidos.csv",
    )
    return p.parse_args()


# 1. DETECCIÓN DINÁMICA DEL ENTORNO
ARGS = parse_args()
CARPETA_RAIZ = ARGS.root or os.path.dirname(os.path.abspath(__file__))

# Configuración del Logging unificado
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)

logging.info("--- Iniciando Pipeline de Extracción ---")
logging.info(f"Carpeta base detectada: {CARPETA_RAIZ}")

datos_extraidos = []

# 2. ESCANEO JERÁRQUICO (Estructura de Carpetas)
if not os.path.exists(CARPETA_RAIZ):
    logging.error(f"La ruta especificada no existe: {CARPETA_RAIZ}")
    exit(1)

for item in os.listdir(CARPETA_RAIZ):
    ruta_subcarpeta = os.path.join(CARPETA_RAIZ, item)

    # El programa solo entra si el ítem es una carpeta (Representa al Candidato)
    if os.path.isdir(ruta_subcarpeta):
        candidato = item  # El nombre de la subcarpeta define la etiqueta del candidato
        logging.info(
            f"Escaneando documentos para el candidato: [{candidato}]"
        )

        # Escaneamos los archivos dentro de la subcarpeta del candidato
        for archivo in os.listdir(ruta_subcarpeta):
            # Filtro exclusivo para archivos Word válidos (evita archivos temporales de Office)
            if archivo.endswith(".docx") and not archivo.startswith("~$"):
                ruta_completa_docx = os.path.join(ruta_subcarpeta, archivo)

                # 3. PROCESAMIENTO Y EXTRACCIÓN CIEGA
                try:
                    doc = Document(ruta_completa_docx)
                    # Unimos los párrafos conservando los saltos de línea estructurales
                    texto_completo = "\n".join(
                        [
                            p.text
                            for p in doc.paragraphs
                            if p.text.strip() != ""
                        ]
                    )

                    # Registro de Éxito
                    datos_extraidos.append(
                        {
                            "archivo_origen": archivo,
                            "candidato": candidato,
                            "texto_crudo": texto_completo,
                            "estado_proceso": "Exitoso",
                            "auditoria_error": "",
                            "conteo_caracteres": len(texto_completo),
                        }
                    )

                except Exception as e:
                    # Registro de Fallo con Stack Trace completo
                    tb = traceback.format_exc()
                    logging.error(
                        f"Error procesando {ruta_completa_docx}: {e}"
                    )

                    datos_extraidos.append(
                        {
                            "archivo_origen": archivo,
                            "candidato": candidato,
                            "texto_crudo": "",
                            "estado_proceso": "Error",
                            "auditoria_error": tb,
                            "conteo_caracteres": 0,
                        }
                    )

# 4. CONSTRUCCIÓN DEL RECIBO TABULAR (CSV Unificado)
if datos_extraidos:
    df_resultados = pd.DataFrame(datos_extraidos)

    # Definir ruta de salida: si es un nombre plano, se guarda en la CARPETA_RAIZ
    if os.path.isabs(ARGS.output):
        ruta_salida_csv = ARGS.output
    else:
        ruta_salida_csv = os.path.join(CARPETA_RAIZ, ARGS.output)

    # Exportación robusta encapsulando todo en comillas dobles
    df_resultados.to_csv(
        ruta_salida_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8"
    )

    # 5. REPORTE GENERAL DE CONTROL (Vía Logging)
    logging.info("=" * 50)
    logging.info("         RECIBO DE CONTROL DE EXTRACCIÓN")
    logging.info("=" * 50)

    # Resumen de estados
    resumen_estados = df_resultados["estado_proceso"].value_counts()
    for estado, conteo in resumen_estados.items():
        logging.info(f"Estado {estado}: {conteo} archivos.")

    logging.info("-" * 50)
    logging.info(f"Total de discursos procesados: {len(df_resultados)}")

    # Desglose por candidato (solo exitosos)
    df_exitosos = df_resultados[df_resultados["estado_proceso"] == "Exitoso"]
    if not df_exitosos.empty:
        logging.info("Discursos exitosos por candidato:")
        resumen_cand = df_exitosos["candidato"].value_counts()
        for cand, conteo in resumen_cand.items():
            logging.info(f"  > {cand}: {conteo}")

    # Alerta de errores
    df_errores = df_resultados[df_resultados["estado_proceso"] == "Error"]
    if not df_errores.empty:
        logging.warning(
            f"❌ Se detectaron {len(df_errores)} errores durante el proceso. Revisar el CSV para ver los detalles del traceback."
        )

    logging.info("=" * 50)
    logging.info(f"Archivo de salida generado en: {ruta_salida_csv}")

else:
    logging.warning(
        "No se encontraron subcarpetas o archivos .docx válidos para procesar."
    )