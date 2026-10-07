"""
El grafo ReAct del agente:

    START -> modelo --(tools_condition)--> herramientas --> modelo  (ciclo)
                    \\--(sin tool_calls)--> END

- El LLM decide SOLO si necesita una herramienta (tool_calls en su
  respuesta): no hay rutas manuales if/else. tools_condition mira el último
  mensaje y elige entre "tools" y END.
- La arista herramientas -> modelo cierra el ciclo: el resultado de la
  herramienta vuelve al LLM, que puede pedir otra (razonamiento multi-paso)
  o responder.
- El grafo no conoce al checkpointer: se compila afuera con uno u otro
  (SQLite en el uso normal, en memoria en los tests).
"""
from langchain_core.exceptions import ModelAPIError, ModelConnectionError, ModelRateLimitError, ModelTimeoutError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from . import config
from .catalogo import SUCURSALES
from .tools import HERRAMIENTAS

SYSTEM_PROMPT = f"""Sos el asesor virtual de una concesionaria Volkswagen. Respondés en español,
en forma breve y clara, sobre los modelos del catálogo, sus versiones y el stock disponible.

Cómo trabajar:
1. Para cualquier dato de versiones, motores o stock, usá las herramientas: nunca inventes
   datos ni respondas de memoria.
2. El stock se consulta por version_id: si no lo tenés, primero buscá las versiones del modelo.
3. Si una herramienta devuelve un ERROR por un problema temporal (por ejemplo, un timeout),
   reintentá la misma consulta una vez.
4. Si devuelve un ERROR porque el dato no existe (un modelo que no está en el catálogo, una
   sucursal inexistente), no reintentes igual: explicáselo al usuario y pedile una aclaración,
   mencionando las opciones válidas.
5. Si la pregunta es ambigua (por ejemplo, no dice la versión o la sucursal), podés consultar
   todas las opciones razonables o pedir que la precise.

Sucursales: {", ".join(SUCURSALES)}. El stock es de una concesionaria de demostración (ficticio)."""


# Errores TRANSITORIOS del proveedor del LLM (sobrecarga, rate limit, red):
# vale la pena reintentarlos. Quedan afuera a propósito los que no se arreglan
# reintentando: API key inválida, modelo inexistente, request mal armado.
ERRORES_TRANSITORIOS_LLM = (ModelAPIError, ModelRateLimitError, ModelTimeoutError, ModelConnectionError)

# Señales de que se agotó una cuota DIARIA (ej: la capa gratuita de Gemini,
# "GenerateRequestsPerDayPerProjectPerModel"): es un 429 igual que el límite
# por minuto, pero no se libera hasta dentro de horas, así que reintentar a
# los pocos segundos solo hace perder tiempo.
_CUOTA_DIARIA = ("perday", "per day", "daily")


def es_error_transitorio(error: Exception) -> bool:
    if not isinstance(error, ERRORES_TRANSITORIOS_LLM):
        return False
    mensaje = str(error).lower()
    return not any(senal in mensaje for senal in _CUOTA_DIARIA)


# Reintento del nodo modelo: hasta 4 intentos, esperando ~2s, 4s y 8s entre
# ellos. Es distinto del "Ciclo de Retorno" de la consigna (errores de las
# HERRAMIENTAS, que resuelve el propio LLM): esto cubre fallas del LLM mismo,
# como el 503 "high demand" que devuelve Gemini en horas pico.
REINTENTO_LLM = RetryPolicy(initial_interval=2.0, backoff_factor=2.0, max_attempts=4,
                            retry_on=es_error_transitorio)


class EstadoAgente(MessagesState):
    """
    Estado del agente. Hereda de MessagesState el campo
    messages: Annotated[list[AnyMessage], add_messages]: el reducer
    add_messages hace que cada nodo AGREGUE mensajes al historial en vez de
    reemplazarlo. No se guardan acá objetos de infraestructura (LLM, base de
    datos): el estado tiene solo los datos que tienen que persistir.
    """


def recortar_contexto(mensajes: list[AnyMessage], max_mensajes: int) -> list[AnyMessage]:
    """
    Error a evitar "Estado Sucio": con el checkpointer, el historial de un
    thread crece en cada turno. El estado completo se conserva, pero al LLM
    se le mandan solo los últimos max_mensajes. start_on="human" asegura que
    el recorte empiece en un mensaje del usuario, así nunca queda un resultado
    de herramienta "huérfano", separado de la llamada que lo pidió.
    """
    return trim_messages(
        mensajes,
        strategy="last",
        token_counter=len,  # cuenta mensajes, no tokens
        max_tokens=max_mensajes,
        start_on="human",
    )


def construir_grafo(llm: BaseChatModel) -> StateGraph:
    llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

    async def nodo_modelo(state: EstadoAgente) -> dict[str, list[AnyMessage]]:
        """Razonar: el LLM ve el historial reciente y decide si llama una herramienta o responde."""
        historial = recortar_contexto(state["messages"], config.max_mensajes_contexto())
        respuesta = await llm_con_herramientas.ainvoke([SystemMessage(SYSTEM_PROMPT), *historial])
        return {"messages": [respuesta]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("modelo", nodo_modelo, retry_policy=REINTENTO_LLM)
    # Actuar/Observar: ejecuta las tool_calls. handle_tool_errors=True convierte
    # las excepciones (como el timeout simulado) en un mensaje para el LLM.
    grafo.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))

    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges("modelo", tools_condition, {"tools": "herramientas", END: END})
    grafo.add_edge("herramientas", "modelo")  # el ciclo ReAct
    return grafo
