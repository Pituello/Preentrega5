"""
Tests sin API key: un LLM "guionado" devuelve respuestas fijas (incluyendo
tool_calls), así se verifica la MECÁNICA del agente (ciclo ReAct, reintento,
memoria, recursion_limit, recorte de contexto) de forma reproducible y gratis.
Que el LLM real razone bien se demuestra aparte, con la traza de demo.py.

Uso:
    python -m pytest -v
"""
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field

from agente import tools
from agente.grafo import recortar_contexto
from agente.sesion import abrir_agente, preguntar


class LLMGuionado(BaseChatModel):
    """Devuelve las respuestas en orden y registra qué mensajes recibió en cada llamada."""

    respuestas: list[AIMessage]
    errores: list[Exception] = Field(default_factory=list)  # se lanzan antes de responder
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    repetir_ultima: bool = False  # para simular un agente que nunca termina

    @property
    def _llm_type(self) -> str:
        return "guionado"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "LLMGuionado":
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.recibidos.append(list(messages))
        if self.errores:
            raise self.errores.pop(0)
        if self.repetir_ultima and len(self.respuestas) == 1:
            respuesta = self.respuestas[0].model_copy(update={"id": None})
        else:
            respuesta = self.respuestas.pop(0)
        return ChatResult(generations=[ChatGeneration(message=respuesta)])


def llamada(nombre: str, args: dict[str, Any], id_: str) -> dict[str, Any]:
    return {"name": nombre, "args": args, "id": id_, "type": "tool_call"}


@pytest.fixture(autouse=True)
def sistema_de_cordoba_limpio():
    """Cada test arranca con el 'sistema de Córdoba' sin consultas previas."""
    tools._SUCURSALES_CONSULTADAS.clear()


def test_razonamiento_multi_paso():
    """El agente encadena buscar_versiones -> consultar_stock (>= 2 llamadas) y responde."""
    llm = LLMGuionado(respuestas=[
        AIMessage("", tool_calls=[llamada("buscar_versiones", {"modelo": "Tera"}, "1")]),
        AIMessage("", tool_calls=[
            llamada("consultar_stock", {"version_id": "TERA-HIGH-AT6", "sucursal": "Palermo"}, "2"),
            llamada("consultar_stock", {"version_id": "TERA-OUTFIT-AT6", "sucursal": "Palermo"}, "3"),
        ]),
        AIMessage("En Palermo hay 2 Tera High y 1 Tera Outfit."),
    ])

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "¿Hay Tera automático en Palermo?", "t1")

    turno = asyncio.run(correr())
    assert turno.llamadas_a_herramientas == 3
    assert [p.tipo for p in turno.pasos].count("observacion") == 3
    assert not any(p.es_error for p in turno.pasos)
    stock = [p for p in turno.pasos if p.tipo == "observacion" and p.herramienta == "consultar_stock"]
    assert '"unidades": 2' in stock[0].contenido  # TERA-HIGH-AT6 en Palermo
    assert turno.respuesta == "En Palermo hay 2 Tera High y 1 Tera Outfit."


def test_reintento_ante_timeout():
    """La 1.ª consulta a Córdoba da timeout; ToolNode lo devuelve como error y el reintento funciona."""
    pedido = {"version_id": "SAVEIRO-EXTREME-CD", "sucursal": "Córdoba"}
    llm = LLMGuionado(respuestas=[
        AIMessage("", tool_calls=[llamada("consultar_stock", pedido, "1")]),
        AIMessage("", tool_calls=[llamada("consultar_stock", pedido, "2")]),
        AIMessage("Hay 2 Saveiro Extreme en Córdoba."),
    ])

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "¿Hay Saveiro Extreme en Córdoba?", "t2")

    turno = asyncio.run(correr())
    observaciones = [p for p in turno.pasos if p.tipo == "observacion"]
    assert observaciones[0].es_error and "TimeoutError" in observaciones[0].contenido
    assert not observaciones[1].es_error and '"unidades": 2' in observaciones[1].contenido
    # el LLM recibió el error como mensaje (no se cortó el programa)
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in llm.recibidos[1])


def test_dato_inexistente_devuelve_error_al_llm():
    """Un modelo fuera del catálogo vuelve como 'ERROR: ...' para que el agente pida aclaración."""
    llm = LLMGuionado(respuestas=[
        AIMessage("", tool_calls=[llamada("buscar_versiones", {"modelo": "Amarok"}, "1")]),
        AIMessage("La Amarok no está en nuestro catálogo. ¿Te interesa otro modelo?"),
    ])

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "¿Tienen Amarok?", "t3")

    turno = asyncio.run(correr())
    assert turno.pasos[1].es_error and "Modelos disponibles" in turno.pasos[1].contenido
    assert "Amarok" in turno.respuesta


def test_memoria_persiste_en_sqlite_tras_reiniciar(tmp_path: Path):
    """Con el mismo thread_id, una conexión NUEVA a SQLite recupera el historial."""
    db = str(tmp_path / "checkpoints.sqlite")
    llm_1 = LLMGuionado(respuestas=[AIMessage("¡Hola! ¿Qué modelo te interesa?")])
    llm_2 = LLMGuionado(respuestas=[AIMessage("Me preguntaste por el Tera.")])

    async def correr():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            async with abrir_agente(llm=llm_1, checkpointer=saver) as app:
                await preguntar(app, "Hola, me interesa el Tera", "sesion-x")
        # "reinicio": conexión nueva al mismo archivo
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            async with abrir_agente(llm=llm_2, checkpointer=saver) as app:
                await preguntar(app, "¿Por qué modelo te pregunté?", "sesion-x")
                await preguntar(app, "Pregunta en otro thread", "sesion-y")

    llm_2.respuestas.append(AIMessage("Thread nuevo, sin historial."))
    asyncio.run(correr())
    textos_turno_2 = [m.content for m in llm_2.recibidos[0] if isinstance(m, HumanMessage)]
    assert textos_turno_2 == ["Hola, me interesa el Tera", "¿Por qué modelo te pregunté?"]
    textos_otro_thread = [m.content for m in llm_2.recibidos[1] if isinstance(m, HumanMessage)]
    assert textos_otro_thread == ["Pregunta en otro thread"]  # los threads no se mezclan


def test_recursion_limit_corta_bucles(monkeypatch: pytest.MonkeyPatch):
    """Un agente que nunca deja de pedir herramientas se corta por recursion_limit, sin romper."""
    monkeypatch.setenv("RECURSION_LIMIT", "5")
    llm = LLMGuionado(
        respuestas=[AIMessage("", tool_calls=[llamada("buscar_versiones", {"modelo": "Polo"}, "x")])],
        repetir_ultima=True,
    )

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "Loop", "t5")

    turno = asyncio.run(correr())
    assert turno.limite_alcanzado
    assert "límite de 5 pasos" in turno.respuesta


def test_recorte_de_contexto():
    """Con historial largo, al LLM le llegan como máximo N mensajes, empezando por uno del usuario."""
    historial: list[BaseMessage] = []
    for i in range(10):
        historial += [HumanMessage(f"pregunta {i}"),
                      AIMessage("", tool_calls=[llamada("buscar_versiones", {"modelo": "Tera"}, f"c{i}")]),
                      ToolMessage("[]", tool_call_id=f"c{i}", name="buscar_versiones"),
                      AIMessage(f"respuesta {i}")]
    recortado = recortar_contexto(historial, 10)
    assert len(recortado) <= 10
    assert isinstance(recortado[0], HumanMessage)
    assert recortado[-1].content == "respuesta 9"


def test_herramientas_normalizan_entradas():
    resultado = asyncio.run(tools.buscar_versiones.ainvoke({"modelo": "VW tera"}))
    assert "TERA-HIGH-AT6" in resultado
    resultado = asyncio.run(tools.consultar_stock.ainvoke({"version_id": "tera-high-at6", "sucursal": "palermo"}))
    assert '"version_id": "TERA-HIGH-AT6"' in resultado and '"sucursal": "Palermo"' in resultado
    assert asyncio.run(tools.consultar_stock.ainvoke({"version_id": "NO-EXISTE"})).startswith("ERROR")


def test_sqlite_nuevo_permite_borrar_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Con un checkpoints.sqlite recién creado, borrar un thread no falla con 'no such table'."""
    from agente import config

    monkeypatch.setattr(config, "CHECKPOINT_DB", str(tmp_path / "nuevo.sqlite"))
    llm = LLMGuionado(respuestas=[])

    async def correr():
        async with abrir_agente(llm=llm) as app:  # usa SQLite (CHECKPOINT_DB), no un checkpointer de test
            await app.checkpointer.adelete_thread("cualquiera")

    asyncio.run(correr())


@pytest.fixture
def reintento_rapido(monkeypatch: pytest.MonkeyPatch):
    """Misma política de reintento del nodo modelo, pero sin esperar segundos entre intentos."""
    from langgraph.types import RetryPolicy

    from agente import grafo

    rapido = RetryPolicy(initial_interval=0.01, max_attempts=4, retry_on=grafo.es_error_transitorio)
    monkeypatch.setattr(grafo, "REINTENTO_LLM", rapido)


def test_llm_sobrecargado_se_reintenta(reintento_rapido):
    """Un 503 'high demand' del proveedor se reintenta y el turno termina bien."""
    from langchain_core.exceptions import ModelAPIError

    llm = LLMGuionado(
        respuestas=[AIMessage("Respuesta después del reintento.")],
        errores=[ModelAPIError("503 UNAVAILABLE (simulado)"), ModelAPIError("503 UNAVAILABLE (simulado)")],
    )

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "Hola", "t-503")

    turno = asyncio.run(correr())
    assert turno.respuesta == "Respuesta después del reintento."
    assert len(llm.recibidos) == 3  # 2 fallidos + 1 exitoso


def test_api_key_invalida_no_se_reintenta(reintento_rapido):
    """Un error que no se arregla reintentando (credenciales) falla de inmediato."""
    from langchain_core.exceptions import ModelAuthenticationError

    llm = LLMGuionado(respuestas=[AIMessage("no debería llegar")],
                      errores=[ModelAuthenticationError("API key inválida (simulado)")])

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "Hola", "t-auth")

    with pytest.raises(ModelAuthenticationError):
        asyncio.run(correr())
    assert len(llm.recibidos) == 1  # un solo intento


def test_cuota_diaria_agotada_no_se_reintenta(reintento_rapido):
    """Un 429 por cuota DIARIA (no por minuto) falla de inmediato: no se libera hasta dentro de horas."""
    from langchain_core.exceptions import ModelRateLimitError

    error = ModelRateLimitError(
        "429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier (simulado)"
    )
    llm = LLMGuionado(respuestas=[AIMessage("no debería llegar")], errores=[error])

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "Hola", "t-cuota")

    with pytest.raises(ModelRateLimitError):
        asyncio.run(correr())
    assert len(llm.recibidos) == 1


def test_rate_limit_por_minuto_si_se_reintenta(reintento_rapido):
    """Un 429 común (límite por minuto) sí se reintenta."""
    from langchain_core.exceptions import ModelRateLimitError

    llm = LLMGuionado(respuestas=[AIMessage("OK")],
                      errores=[ModelRateLimitError("429 Too Many Requests (simulado)")])

    async def correr():
        async with abrir_agente(llm=llm, checkpointer=MemorySaver()) as app:
            return await preguntar(app, "Hola", "t-rpm")

    assert asyncio.run(correr()).respuesta == "OK"
    assert len(llm.recibidos) == 2


@pytest.fixture
def sin_keys(monkeypatch: pytest.MonkeyPatch):
    """Aísla el test del .env real: sin keys ni LLM_PROVIDER cargados."""
    for variable in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                     "OPENAI_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "TEMPERATURE"):
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def test_ollama_nunca_se_elige_solo(sin_keys):
    """Sin keys y sin LLM_PROVIDER no se cae en Ollama: hay que pedirlo explícitamente."""
    from agente.llm import elegir_proveedor

    with pytest.raises(RuntimeError, match="LLM_PROVIDER=ollama"):
        elegir_proveedor()
    sin_keys.setenv("LLM_PROVIDER", "ollama")
    assert elegir_proveedor() == "ollama"


def test_ollama_y_openai_se_arman_con_herramientas(sin_keys):
    """Los dos proveedores que no se pueden probar sin servidor/crédito se construyen y aceptan bind_tools."""
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    from agente.llm import get_llm
    from agente.tools import HERRAMIENTAS

    sin_keys.setenv("OLLAMA_MODEL", "qwen2.5")
    ollama = get_llm("ollama")
    assert isinstance(ollama, ChatOllama) and ollama.model == "qwen2.5"
    ollama.bind_tools(HERRAMIENTAS)

    sin_keys.setenv("OPENAI_API_KEY", "sk-fake")
    sin_keys.setenv("LLM_PROVIDER", "openai")
    gpt = get_llm()
    assert isinstance(gpt, ChatOpenAI) and gpt.model_name == "gpt-4o-mini" and gpt.temperature == 0
    gpt.bind_tools(HERRAMIENTAS)


def test_ollama_apagado_falla_claro_y_sin_reintentos(sin_keys):
    """Con el servidor de Ollama apagado, el error se reconoce como del proveedor y no se reintenta."""
    from agente.llm import get_llm
    from agente.sesion import ERRORES_DEL_PROVEEDOR

    sin_keys.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")  # puerto sin nada escuchando

    async def correr():
        async with abrir_agente(llm=get_llm("ollama"), checkpointer=MemorySaver()) as app:
            return await preguntar(app, "Hola", "t-ollama")

    with pytest.raises(ERRORES_DEL_PROVEEDOR) as error:
        asyncio.run(correr())
    assert "ConnectError" in type(error.value).__name__
