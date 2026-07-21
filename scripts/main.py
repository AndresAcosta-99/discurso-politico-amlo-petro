import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador Secuencial del Pipeline de Tesis."
    )
    parser.add_argument(
        "--metadata",
        "-m",
        help="Archivo log de fechas (por defecto: transcriptions_log_final.csv)",
        default="transcriptions_log_final.csv",
    )
    parser.add_argument(
        "--root",
        "-r",
        help="Carpeta raíz con los .docx de los candidatos (opcional para build_corpus.py)",
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        help="Cantidad de palabras para el reporte retórico (por defecto: 20)",
        default=20,
    )

    args = parser.parse_args()

    # Detectar la carpeta donde están guardados tus scripts
    dir_scripts = os.path.dirname(os.path.abspath(__file__))

    # Verificar que el archivo de fechas exista antes de iniciar todo el flujo
    ruta_metadata = (
        args.metadata
        if os.path.isabs(args.metadata)
        else os.path.join(dir_scripts, args.metadata)
    )
    if not os.path.exists(ruta_metadata):
        print(
            f"❌ Error: No se encontró el archivo de fechas en '{ruta_metadata}'.",
            file=sys.stderr,
        )
        print(
            "Asegúrate de que 'transcriptions_log_final.csv' esté en la misma carpeta o indícalo con --metadata.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Definir los comandos secuenciales respetando las CLI de tus archivos
    cmd_fase1 = [sys.executable, os.path.join(dir_scripts, "build_corpus.py")]
    if args.root:
        cmd_fase1.extend(["--root", args.root])

    cmd_fase2 = [
        sys.executable,
        os.path.join(dir_scripts, "clean.py"),
        "--metadata",
        args.metadata,
    ]

    cmd_fase3 = [
        sys.executable,
        os.path.join(dir_scripts, "analisis_gramatical.py"),
    ]

    # Actualizado con el guion bajo correspondiente
    cmd_fase4 = [
        sys.executable,
        os.path.join(dir_scripts, "reporte_retorico.py"),
        "--top",
        str(args.top),
    ]

    cmd_fase5 = [
        sys.executable,
        os.path.join(dir_scripts, "analisis_final.py"),
    ]

    # Pipeline en cascada 
    # Fase 6 (BERTopic) eliminada: corpus insuficiente (~33-51 docs por candidato).
    # Fase 4 (reporte_retorico) mantenida como auditoría de vocabulario crudo.
    pipeline = [
        ("Fase 1: Extracción del Corpus",          cmd_fase1),
        ("Fase 2: Limpieza y Fechas",              cmd_fase2),
        ("Fase 3: Modelado Lingüístico (NLP)",     cmd_fase3),
        ("Fase 4: Reporte Retórico (auditoría)",   cmd_fase4),
        ("Fase 5: Minería Politológica y Gráficos",cmd_fase5),
    ]

    print("=" * 65)
    print("     EJECUTANDO PIPELINE DE INVESTIGACIÓN EN CASCADA")
    print("=" * 65)
    print(f" Carpeta de trabajo: {dir_scripts}")
    print(f" Log de control:    {args.metadata}\n")

    for nombre_fase, comando in pipeline:
        print(f"\n>>> Iniciando: {nombre_fase}")
        print(f" Ejecutando comando: {' '.join(comando[1:])}")
        print("-" * 65)

        # Ejecutamos con cwd=dir_scripts para asegurar que las rutas
        # relativas internas de tus scripts sigan funcionando a la perfección
        resultado = subprocess.run(comando, cwd=dir_scripts)

        if resultado.returncode != 0:
            print(
                f"\n❌ Error crítico: La {nombre_fase} falló con código de salida {resultado.returncode}.",
                file=sys.stderr,
            )
            print(
                "El orquestador detuvo la ejecución para evitar cascadas de errores sobre archivos vacíos o corruptos.",
                file=sys.stderr,
            )
            sys.exit(resultado.returncode)

    print("\n" + "=" * 65)
    print(" 🎉 ¡ORQUESTACIÓN COMPLETADA!")
    print(" Todos tus scripts se ejecutaron en secuencia de manera exitosa.")
    print("=" * 65)


if __name__ == "__main__":
    main()