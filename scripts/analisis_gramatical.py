import argparse
import csv
import logging
import os
import sys
import traceback
import pandas as pd
import spacy


# ---------------------------------------------------------------------------
# RUIDO ESTRUCTURAL — se aplica SOLO en los análisis de frecuencia (analisis_final.py),
# NO durante la tokenización. Aquí se define para importarlo desde otros módulos.
# En esta fase conservamos el corpus completo para no perder contexto semántico.
# ---------------------------------------------------------------------------
RUIDO_UNIFICADO = {
    # Geográfico-institucional
    "mexico", "colombia", "nacional", "republica", "estado", "pais",
    "ciudad", "region", "departamento", "municipio", "federacion",
    "latinoamerica", "america", "mundo", "gobierno",
    # Cargos y roles genéricos
    "presidente", "gobernador", "senador", "contralor", "ministro",
    "secretario", "candidato", "dirigente", "lider", "compañero",
    "señor", "señora", "doctor", "licenciado",
    # Temporalidad genérica
    "ano", "año", "mes", "dia", "hoy", "ayer", "semana", "momento",
    "tiempo", "vez", "periodo", "epoca", "fecha",
    # Conectores discursivos que spaCy no filtra bien en discurso oral
    "entonces", "bueno", "ahora", "aqui", "ahi", "pues", "claro",
    "bien", "verdad", "decir", "hacer", "ir", "venir", "tener",
    "poder", "querer", "saber", "ver", "dar", "haber",
    # Cuantificadores genéricos
    "millon", "millones", "mil", "cien", "ciento", "peso", "año",
    # Nombres propios filtrados manualmente (corpus específico)
    "moreira", "humberto", "bautista", "pan",
    # Palabras que spaCy lematiza mal en discurso político oral
    "gente",  # queda como ruido porque es comodín; se preserva solo en bloque teórico
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fase 3: Segmentación NLP. Desarma los discursos en una matriz "
                    "atómica de tokens con metadata temporal. Usa es_core_news_lg para "
                    "mejor lematización en discurso político."
    )
    p.add_argument(
        "--input", "-i",
        default="dataset_textos_limpios.csv",
        help="CSV limpio con fechas de la Fase 2 (por defecto: dataset_textos_limpios.csv)",
    )
    p.add_argument(
        "--output", "-o",
        default="dataset_tokens_analizados.csv",
        help="CSV de tokens resultante (por defecto: dataset_tokens_analizados.csv)",
    )
    p.add_argument(
        "--modelo", "-M",
        default="es_core_news_lg",
        help="Modelo spaCy a usar. lg recomendado; sm como fallback.",
    )
    return p.parse_args()


def main():
    ARGS = parse_args()
    CARPETA_TRABAJO = os.path.dirname(os.path.abspath(__file__))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("--- Iniciando Pipeline de Modelado Lingüístico (Fase 3) ---")

    ruta_entrada = (
        ARGS.input if os.path.isabs(ARGS.input)
        else os.path.join(CARPETA_TRABAJO, ARGS.input)
    )
    ruta_salida = (
        ARGS.output if os.path.isabs(ARGS.output)
        else os.path.join(CARPETA_TRABAJO, ARGS.output)
    )

    if not os.path.exists(ruta_entrada):
        logging.error(f"No se encontró el archivo de la Fase 2 en: {ruta_entrada}")
        sys.exit(1)

    try:
        # ------------------------------------------------------------------
        # 1. Cargar modelo spaCy — intenta lg, cae a md, luego a sm
        # ------------------------------------------------------------------
        modelos_fallback = [ARGS.modelo, "es_core_news_md", "es_core_news_sm"]
        nlp = None
        for modelo in modelos_fallback:
            try:
                nlp = spacy.load(modelo)
                logging.info(f"Modelo cargado: {modelo}")
                break
            except OSError:
                logging.warning(f"Modelo '{modelo}' no encontrado, intentando siguiente...")

        if nlp is None:
            logging.critical(
                "No se encontró ningún modelo spaCy. Instalá al menos uno con:\n"
                "  python -m spacy download es_core_news_lg"
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # 2. Cargar dataset
        # ------------------------------------------------------------------
        logging.info(f"Cargando dataset desde: {ruta_entrada}")
        df = pd.read_csv(ruta_entrada)
        df["texto_limpio"] = df["texto_limpio"].fillna("")
        df = df[df["texto_limpio"] != ""].copy()
        logging.info(f"Discursos listos para NLP: {len(df)}")

        # ------------------------------------------------------------------
        # 3. Tokenización — SIN filtro de ruido léxico (se aplica en analisis_final.py)
        #    Sí filtramos nombres propios (PROPN) aquí porque contaminan todos
        #    los análisis de frecuencia con toponymia y nombres de personas.
        #    Excepción: los términos del campo adversario y bloques teóricos
        #    se definen como lemas comunes (no PROPN), así que no se pierden.
        # ------------------------------------------------------------------
        logging.info("Extrayendo tokens con morfología y lemas...")
        logging.info("  [Filtro activo] Excluyendo nombres propios (PROPN) del corpus.")
        datos_tokens = []

        pipeline_disable = ["ner", "parser"]
        discursos_procesados = nlp.pipe(
            df["texto_limpio"],
            batch_size=15,
            disable=pipeline_disable,
        )

        for doc, (_, fila) in zip(discursos_procesados, df.iterrows()):
            for token in doc:
                if (
                    not token.is_stop
                    and not token.is_space
                    and not token.like_num
                    and len(token.text) > 1
                    and token.pos_ not in ("PUNCT", "SYM", "X", "NUM", "PROPN")
                ):
                    datos_tokens.append({
                        "archivo_origen":       fila["archivo_origen"],
                        "candidato":            fila["candidato"],
                        "fecha":                fila["fecha"],
                        "palabra_original":     token.text,
                        "palabra_lema":         token.lemma_.lower(),
                        "categoria_gramatical": token.pos_,
                    })

        # ------------------------------------------------------------------
        # 4. Construir y guardar la Matriz Atómica
        # ------------------------------------------------------------------
        df_tokens = pd.DataFrame(datos_tokens)

        if df_tokens.empty:
            logging.warning("No se extrajo ningún token válido. Revisá el corpus.")
            sys.exit(0)

        df_tokens.to_csv(ruta_salida, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

        # ------------------------------------------------------------------
        # 5. Reporte de métricas
        # ------------------------------------------------------------------
        logging.info("=" * 55)
        logging.info("      REPORTE DE TOKENIZACIÓN")
        logging.info("=" * 55)
        logging.info(f"Total tokens extraídos:  {len(df_tokens):,}")
        logging.info(f"Tokens únicos (lemas):   {df_tokens['palabra_lema'].nunique():,}")
        logging.info("Tokens por candidato:")
        for cand, conteo in df_tokens.groupby("candidato").size().items():
            logging.info(f"  {cand}: {conteo:,} tokens")
        logging.info(f"Matriz guardada en: {ruta_salida}")
        logging.info("=" * 55)

    except Exception:
        logging.critical(f"Fallo catastrófico en Fase 3:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
