"""
Ejecución de turnos con persistencia y armado de la traza ReAct.

Persistencia: AsyncSqliteSaver es la versión asíncrona del SqliteSaver que
pide la consigna (mismo paquete, langgraph-checkpoint-sqlite, mismo archivo
.sqlite); se usa la asíncrona porque todo el agente corre con asyncio.
Guarda un checkpoint del estado después de cada nodo. Con el mismo thread_id,
el grafo retoma el historial aunque el proceso se haya cerrado.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from . import config
from .grafo import construir_grafo
from .llm import get_llm


# Errores del proveedor del LLM que conviene mostrar con un mensaje claro en vez
# de un traceback: los de LangChain (credenciales, cuota, 503 tras reintentos) y
# los de Ollama, que no los envuelve LangChain (servidor apagado -> error de
# conexión de httpx; modelo no descargado -> ollama.ResponseError).
try:
    from ollama import ResponseError as _ErrorDeOllama
except ImportError:  # langchain-ollama no instalado
    _ErrorDeOllama = ModelError
ERRORES_DEL_PROVEEDOR: tuple[type[Exception], ...] = (ModelError, httpx.HTTPError, _ErrorDeOllama)


class PasoReAct(BaseModel):
    tipo: str  # "decision" | "observacion" | "respuesta"
    herramienta: str | None = None
    argumentos: dict[str, Any] | None = None
    contenido: str = ""
    es_error: bool = False


class Turno(BaseModel):
    thread_id: str
    pregunta: str
    pasos: list[PasoReAct]
    respuesta: str
    llamadas_a_herramientas: int
    limite_alcanzado: bool = False


@asynccontextmanager
async def abrir_agente(
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """
    Compila el grafo con persistencia. Por defecto: LLM según el .env y
    checkpoints en SQLite (CHECKPOINT_DB). Los tests pasan un LLM simulado
    y/o un checkpointer propio.
    """
    grafo = construir_grafo(llm or get_llm())
    if checkpointer is not None:
        yield grafo.compile(checkpointer=checkpointer)
        return
    async with AsyncSqliteSaver.from_conn_string(config.CHECKPOINT_DB) as saver:
        # Crea las tablas si el archivo es nuevo. Sin esto, operaciones como
        # borrar un thread fallan ("no such table") antes del primer guardado.
        await saver.setup()
        yield grafo.compile(checkpointer=saver)


def texto_de(mensaje: AnyMessage) -> str:
    """Normaliza el content (Gemini a veces devuelve una lista de bloques en vez de un string)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return str(contenido)


def pasos_de(mensajes: list[AnyMessage]) -> list[PasoReAct]:
    """Convierte los mensajes de un turno en pasos Reason -> Act -> Observe -> respuesta."""
    pasos: list[PasoReAct] = []
    for m in mensajes:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                pasos.append(PasoReAct(tipo="decision", herramienta=tc["name"], argumentos=tc["args"],
                                       contenido=texto_de(m)))
        elif isinstance(m, ToolMessage):
            es_error = m.status == "error" or texto_de(m).startswith("ERROR")
            pasos.append(PasoReAct(tipo="observacion", herramienta=m.name, contenido=texto_de(m),
                                   es_error=es_error))
        elif isinstance(m, AIMessage):
            pasos.append(PasoReAct(tipo="respuesta", contenido=texto_de(m)))
    return pasos


async def preguntar(app: CompiledStateGraph, pregunta: str, thread_id: str) -> Turno:
    """Ejecuta un turno del agente en el thread indicado y devuelve su traza."""
    config_turno = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.recursion_limit()}

    # Cuántos mensajes tenía el thread antes: la traza del turno son solo los nuevos.
    estado_previo = await app.aget_state(config_turno)
    previos = len(estado_previo.values.get("messages", [])) if estado_previo.values else 0

    limite_alcanzado = False
    try:
        await app.ainvoke({"messages": [HumanMessage(pregunta)]}, config=config_turno)
    except GraphRecursionError:
        # El techo de pasos cortó un posible bucle: lo hecho hasta acá quedó
        # guardado en el checkpoint, así que igual se puede armar la traza.
        limite_alcanzado = True

    estado = await app.aget_state(config_turno)
    nuevos = estado.values["messages"][previos + 1 :]  # +1: saltea la pregunta del usuario
    pasos = pasos_de(nuevos)

    if limite_alcanzado:
        respuesta = (
            f"(Se alcanzó el límite de {config.recursion_limit()} pasos sin una respuesta final: "
            "el agente se detuvo para evitar un bucle infinito.)"
        )
    else:
        respuesta = next((p.contenido for p in reversed(pasos) if p.tipo == "respuesta"), "")

    return Turno(
        thread_id=thread_id,
        pregunta=pregunta,
        pasos=pasos,
        respuesta=respuesta,
        llamadas_a_herramientas=sum(1 for p in pasos if p.tipo == "decision"),
        limite_alcanzado=limite_alcanzado,
    )


def formatear_traza(turno: Turno) -> str:
    """Traza legible, con el mismo formato que el ejemplo de la consigna."""
    lineas = [f'Usuario: "{turno.pregunta}"   [thread_id={turno.thread_id}]']
    for paso in turno.pasos:
        if paso.tipo == "decision":
            args = ", ".join(f"{k}={v!r}" for k, v in (paso.argumentos or {}).items())
            lineas.append(f"→ El agente decide usar la herramienta: {paso.herramienta}({args})")
        elif paso.tipo == "observacion":
            marca = " [ERROR]" if paso.es_error else ""
            lineas.append(f"→ La herramienta devuelve{marca}: {paso.contenido}")
    lineas.append(f'Respuesta: "{turno.respuesta}"')
    lineas.append(f"(llamadas a herramientas en este turno: {turno.llamadas_a_herramientas})")
    return "\n".join(lineas)


def avisar_error_del_llm(error: Exception) -> None:
    """Mensaje claro en vez de un traceback largo cuando el proveedor del LLM no responde."""
    print(f"\n❌ El proveedor del LLM rechazó el pedido: {type(error).__name__}")
    if isinstance(error, httpx.ConnectError):
        print("   No se pudo conectar. Si usás Ollama: abrilo (o corré 'ollama serve') y revisá OLLAMA_BASE_URL.")
    elif isinstance(error, _ErrorDeOllama) and "not found" in str(error).lower():
        print("   Ollama no tiene ese modelo descargado: corré 'ollama pull <modelo>' (el de OLLAMA_MODEL).")
    elif "perday" in str(error).lower():
        print("   Se agotó la cuota DIARIA del modelo (capa gratuita). Opciones:")
        print("   - esperar a que se reinicie (el mensaje de error de abajo dice cuánto falta),")
        print("   - usar otro modelo, que tiene su propia cuota (GEMINI_MODEL en el .env),")
        print("   - o usar otro proveedor (LLM_PROVIDER=openai u anthropic, con su API key).")
    print(f"\n   Detalle: {str(error)[:400]}")
