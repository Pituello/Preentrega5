"""
Corrida de demostración: un escenario por cada criterio de aceptación de la
consigna. Genera la traza ReAct en trazas/traza_ejecucion.json y
trazas/traza_ejecucion.log.

Uso:
    python demo.py              # corre los 5 escenarios desde cero
    python demo.py --desde 3    # retoma desde el escenario 3

La traza se guarda después de CADA escenario. Si el proveedor del LLM corta a
mitad de camino (por ejemplo, se agota la cuota diaria), lo ya hecho no se
pierde: con --desde N se retoma más tarde. Funciona porque las conversaciones
de los escenarios anteriores quedaron guardadas en checkpoints.sqlite (el
escenario 5 necesita el historial de los escenarios 1 y 2).

La corrida completa hace unas 12-15 llamadas al LLM.
"""
import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agente import config
from agente.llm import describir_llm
from agente.sesion import ERRORES_DEL_PROVEEDOR, Turno, abrir_agente, avisar_error_del_llm, formatear_traza, preguntar

THREAD_TERA = "demo-stock-tera"
THREAD_SAVEIRO = "demo-saveiro-cordoba"
THREAD_ACLARACION = "demo-modelo-inexistente"


@dataclass(frozen=True)
class Escenario:
    numero: int
    nombre: str
    criterio: str
    pregunta: str
    thread_id: str
    conexion: int  # escenarios con distinta conexión = "reinicio" del programa entre medio


ESCENARIOS = [
    Escenario(1, "Razonamiento multi-paso",
              "Autonomía: el agente encadena herramientas solo (buscar_versiones -> consultar_stock)",
              "¿Hay stock del Tera con caja automática en la sucursal Palermo?", THREAD_TERA, 1),
    Escenario(2, "Memoria en la misma sesión",
              "Resiliencia de estado: '¿y en Belgrano?' solo tiene sentido recordando el turno anterior",
              "¿Y en Belgrano?", THREAD_TERA, 1),
    Escenario(3, "Ciclo de retorno ante un timeout",
              "El sistema de Córdoba falla la primera vez: el agente reintenta",
              "¿Tienen la Saveiro Extreme en la sucursal Córdoba?", THREAD_SAVEIRO, 1),
    Escenario(4, "Ciclo de retorno ante un dato inexistente",
              "La Amarok no está en el catálogo: el agente explica y pide aclaración, sin inventar",
              "¿Cuántas Amarok tienen en stock?", THREAD_ACLARACION, 1),
    Escenario(5, "Memoria después de reiniciar",
              "Persistencia en SQLite: nueva conexión, mismo thread_id, el agente retoma la conversación",
              "De las versiones automáticas que me dijiste, ¿qué motor tienen?", THREAD_TERA, 2),
]

ARCHIVO_JSON = Path(config.TRAZAS_DIR) / "traza_ejecucion.json"
ARCHIVO_LOG = Path(config.TRAZAS_DIR) / "traza_ejecucion.log"


def guardar_traza(resultados: list[dict], checkpoints: dict[str, int] | None = None) -> None:
    """Escribe el .json y el .log con los escenarios completados hasta ahora."""
    ARCHIVO_JSON.parent.mkdir(exist_ok=True)
    traza = {
        "actualizado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recursion_limit": config.recursion_limit(),
        "checkpoint_db": config.CHECKPOINT_DB,
        "escenarios_completados": f"{len(resultados)} de {len(ESCENARIOS)}",
        "checkpoints_por_thread": checkpoints,
        "escenarios": resultados,
    }
    ARCHIVO_JSON.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")

    log = [f"Traza ReAct — {traza['actualizado_en']} — recursion_limit={traza['recursion_limit']} — "
           f"escenarios completados: {traza['escenarios_completados']}\n"]
    for r in resultados:
        turno = Turno(**{k: v for k, v in r.items() if k in Turno.model_fields})
        log.append(f"### {r['numero']}. {r['escenario']} — {r['criterio']}\n"
                   f"[LLM: {r['llm']} — {r['ejecutado_en']}]\n{formatear_traza(turno)}\n")
    if checkpoints:
        log.append(f"Checkpoints guardados por thread: {checkpoints}")
    ARCHIVO_LOG.write_text("\n".join(log), encoding="utf-8")


def resultados_previos(desde: int) -> list[dict]:
    """Al retomar, recupera de la traza guardada los escenarios anteriores a 'desde'."""
    if desde == 1:
        # corrida desde cero: se descarta la traza de corridas anteriores
        ARCHIVO_JSON.unlink(missing_ok=True)
        ARCHIVO_LOG.unlink(missing_ok=True)
        return []
    if not ARCHIVO_JSON.exists():
        raise SystemExit(f"No hay traza previa en {ARCHIVO_JSON}: corré primero 'python demo.py' sin --desde.")
    previos = json.loads(ARCHIVO_JSON.read_text(encoding="utf-8"))["escenarios"]
    previos = [r for r in previos if r["numero"] < desde]
    faltantes = set(range(1, desde)) - {r["numero"] for r in previos}
    if faltantes:
        raise SystemExit(f"La traza guardada no tiene los escenarios {sorted(faltantes)}: "
                         f"retomá desde --desde {min(faltantes)}.")
    return previos


async def main(desde: int) -> None:
    resultados = resultados_previos(desde)
    pendientes = [e for e in ESCENARIOS if e.numero >= desde]
    llm = describir_llm()

    # Se borran solo los threads que arrancan de cero en esta corrida: los que
    # usan escenarios ya hechos (el del Tera, que retoma el 5) se conservan.
    threads_previos = {e.thread_id for e in ESCENARIOS if e.numero < desde}
    threads_a_borrar = {e.thread_id for e in pendientes} - threads_previos

    for conexion in sorted({e.conexion for e in pendientes}):
        # Cada "conexion" abre el checkpointer de nuevo: todo lo que el agente
        # sabe de las conversaciones previas sale del archivo SQLite.
        async with abrir_agente() as app:
            for thread in threads_a_borrar:
                await app.checkpointer.adelete_thread(thread)
            threads_a_borrar = set()

            for e in (e for e in pendientes if e.conexion == conexion):
                turno = await preguntar(app, e.pregunta, e.thread_id)
                resultados.append({
                    "numero": e.numero, "escenario": e.nombre, "criterio": e.criterio, "llm": llm,
                    "ejecutado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **turno.model_dump(),
                })
                guardar_traza(resultados)  # se guarda escenario por escenario
                print(f"\n{'=' * 90}\n{e.numero}. {e.nombre} — {e.criterio}\n{'=' * 90}")
                print(formatear_traza(turno))

    async with abrir_agente() as app:
        checkpoints = {}
        for thread in (THREAD_TERA, THREAD_SAVEIRO, THREAD_ACLARACION):
            historial = app.aget_state_history({"configurable": {"thread_id": thread}})
            checkpoints[thread] = len([c async for c in historial])
    guardar_traza(resultados, checkpoints)

    multi_paso = next(r["llamadas_a_herramientas"] for r in resultados if r["numero"] == 1)
    print(f"\n{'=' * 90}")
    print(f"Escenario 1: {multi_paso} llamadas a herramientas "
          f"({'✅ multi-paso' if multi_paso >= 2 else '⚠️ menos de 2: revisar'})")
    print(f"Checkpoints guardados por thread: {checkpoints}")
    print(f"✅ Traza completa ({len(resultados)} de {len(ESCENARIOS)} escenarios) en {ARCHIVO_JSON} y {ARCHIVO_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrida de demostración del agente")
    parser.add_argument("--desde", type=int, default=1, choices=range(1, len(ESCENARIOS) + 1),
                        help="número de escenario desde el cual retomar (default: 1, todo desde cero)")
    desde = parser.parse_args().desde
    try:
        asyncio.run(main(desde))
    except ERRORES_DEL_PROVEEDOR as error:  # credenciales, cuota, proveedor caído, Ollama apagado
        avisar_error_del_llm(error)
        if ARCHIVO_JSON.exists():
            hechos = json.loads(ARCHIVO_JSON.read_text(encoding="utf-8"))["escenarios"]
            siguiente = max((r["numero"] for r in hechos), default=0) + 1
            print(f"\n💾 La traza parcial quedó guardada ({len(hechos)} escenarios). "
                  f"Para retomar: python demo.py --desde {siguiente}")
