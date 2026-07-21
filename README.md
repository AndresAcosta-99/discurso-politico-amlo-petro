# Análisis de discurso político computacional — AMLO vs. Petro

Pipeline de análisis de discurso computacional aplicado a discursos de campaña de Andrés Manuel López Obrador (México, 2018) y Gustavo Petro (Colombia, 2021–2022), desarrollado como parte de la tesis de maestría *"Neoliberalismo, desgaste de la representación y ascenso electoral de alternativas en México y Colombia (1990–2022)"* (FSoc-UBA, dir. Ana Belén Mercado).

**[→ Ver el notebook con la documentación metodológica completa y los resultados](notebooks/analysis.ipynb)** — GitHub lo renderiza directamente en el navegador, con gráficos incluidos.

## Qué hace este pipeline

Toma discursos transcritos (`.docx`, uno por discurso) y los procesa en cinco fases secuenciales hasta producir seis dimensiones de análisis de discurso: vocabulario más frecuente y su evolución temporal, asimetría léxica contrastiva (log-ratio), construcción discursiva del adversario político, densidad de categorías teóricas (representación popular / crítica al cartel / ruptura con el modelo neoliberal), evolución temporal comparada entre campañas de distinta duración y calendario, y keyness (G²).

## Estructura del repositorio

```
.
├── scripts/
│   ├── build_corpus.py        # Fase 1 — extrae texto de los .docx por candidato
│   ├── clean.py                # Fase 2 — limpieza textual y triangulación de fechas
│   ├── analisis_gramatical.py  # Fase 3 — tokenización y lematización (spaCy, es_core_news_lg)
│   ├── reporte_retorico.py     # Fase 4 — auditoría de vocabulario crudo por categoría gramatical
│   ├── analisis_final.py       # Fase 5 — motor de análisis: las 6 dimensiones
│   └── main.py                 # Orquestador: corre las fases 1–5 en cascada
├── notebooks/
│   └── analysis.ipynb          # Documentación del razonamiento metodológico + resultados
├── explorations/
│   └── analisis_tematico.py    # BERTopic (Fase 6) — explorado y descartado; ver notebook §2
├── requirements.txt
└── README.md
```

> **Nota:** este repositorio no incluye el transcriptor de audio (Whisper/yt-dlp) usado para generar los `.docx` de entrada. Ese componente vive en un repo aparte: [youtube-whisper-transcriber](https://github.com/AndresAcosta-99/youtube-whisper-transcriber).

## Cómo correrlo

```bash
# 1. Clonar y crear entorno virtual
git clone <url-del-repo>
cd <nombre-del-repo>
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
python -m spacy download es_core_news_lg

# 3. Organizar los discursos de entrada
#    discursos/
#    ├── AMLO/
#    │   ├── discurso_2018_01_15.docx
#    │   └── ...
#    └── Petro/
#        ├── discurso_2022_03_02.docx
#        └── ...

# 4. Correr el pipeline completo
python scripts/main.py --root discursos/ --metadata transcriptions_log_final.csv
```

Esto genera, en cascada: `dataset_textos_extraidos.csv` → `dataset_textos_limpios.csv` → `dataset_tokens_analizados.csv` → reportes retóricos y las 6 dimensiones de análisis (CSV + PNG) en `matrices_finales_tesis/`.

Si ya tenés `dataset_tokens_analizados.csv` (por ejemplo, para reproducir solo el análisis final sin repetir la tokenización), podés correr directamente:

```bash
python scripts/analisis_final.py
```

## Decisiones metodológicas — versión corta

- **CPM (cuentas por millón) en vez de frecuencias absolutas**, porque el corpus de cada candidato tiene un tamaño distinto (~17k tokens AMLO vs. ~71k Petro).
- **Log-ratio con suavizado de Laplace (Hardie, 2014)** para medir especificidad léxica sin que las palabras raras distorsionen el ranking.
- **Nombres propios excluidos de la tokenización** para que la toponimia no contamine los análisis de frecuencia.
- **TF-IDF, bigramas y BERTopic fueron evaluados y descartados** — quedó código en `explorations/` y el razonamiento completo documentado en el notebook, en vez de simplemente borrar el rastro.

El notebook (`notebooks/analysis.ipynb`) documenta esto con más detalle, incluyendo qué se intentó, por qué no funcionó, y qué se hizo en su lugar.

## Estado del proyecto

Es un análisis exploratorio en desarrollo activo, no un output cerrado. La sección "Próximos pasos" del notebook detalla las líneas de continuación previstas (cruce con datos de Latinobarómetro, examen cualitativo de plataformas electorales, mecanismos institucionales específicos por país).

## Contexto académico

Este pipeline es un insumo metodológico para la tesis de maestría de Andrés Acosta Gaviria (FSoc-UBA). El análisis cuantitativo complementa —no reemplaza— el análisis cualitativo e histórico-comparado que constituye el cuerpo principal del argumento.
