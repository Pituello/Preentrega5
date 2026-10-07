"""
Configuración central, leída de variables de entorno (.env).

Se carga el .env apenas se importa el paquete, así ningún módulo depende de
que el script que lo usa haya llamado antes a load_dotenv(). Los valores que
puede cambiar el usuario se leen con funciones (al momento de usarlos), no
como constantes al importar (Correccion de temperature de Preentrega 3).
"""
import os

from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")
TRAZAS_DIR = os.getenv("TRAZAS_DIR", "trazas")


def _entero(nombre: str, default: int, minimo: int = 1) -> int:
    valor = os.getenv(nombre, str(default))
    try:
        numero = int(valor)
    except ValueError:
        raise ValueError(f"{nombre}='{valor}' en el .env no es un número entero")
    if numero < minimo:
        raise ValueError(f"{nombre}={numero} tiene que ser >= {minimo}")
    return numero


def recursion_limit() -> int:
    """
    Techo de pasos del grafo por cada invocación (error a evitar: "Bucles
    Infinitos"). Cada vuelta modelo -> herramientas -> modelo consume pasos;
    10 alcanza para unas 4 llamadas a herramientas encadenadas.
    """
    return _entero("RECURSION_LIMIT", 10, minimo=2)


def max_mensajes_contexto() -> int:
    """
    Cuántos mensajes recientes ve el LLM en cada llamada (error a evitar:
    "Estado Sucio"). El historial completo sigue guardado en el checkpoint;
    solo se recorta lo que se le manda al modelo.
    """
    return _entero("MAX_MENSAJES_CONTEXTO", 20, minimo=4)
