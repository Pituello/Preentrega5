"""
Chat interactivo con el asesor. Muestra la traza ReAct de cada respuesta.

Uso:
    python main.py                      # thread "sesion-usuario"
    python main.py --thread mi-sesion   # otro thread_id

Con el mismo --thread, la conversación continúa aunque cierres el programa:
el historial vive en checkpoints.sqlite.
"""
import argparse
import asyncio


from agente.sesion import ERRORES_DEL_PROVEEDOR, abrir_agente, avisar_error_del_llm, formatear_traza, preguntar


async def main(thread_id: str) -> None:
    async with abrir_agente() as app:
        estado = await app.aget_state({"configurable": {"thread_id": thread_id}})
        previos = len(estado.values.get("messages", [])) if estado.values else 0

        print("🚗 Asesor virtual Volkswagen (agente ReAct con memoria persistente)")
        print(f"   thread_id: {thread_id} | mensajes guardados: {previos} | 'salir' para terminar\n")

        while True:
            pregunta = input("🧑 Vos: ").strip()
            if pregunta.lower() in ("salir", "exit", "quit", ""):
                print("\n👋 Sesión guardada. Volvé con el mismo --thread para continuarla.")
                break
            turno = await preguntar(app, pregunta, thread_id)
            print("\n" + formatear_traza(turno) + "\n" + "-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asesor virtual Volkswagen")
    parser.add_argument("--thread", default="sesion-usuario", help="thread_id de la conversación")
    try:
        asyncio.run(main(parser.parse_args().thread))
    except ERRORES_DEL_PROVEEDOR as error:  # credenciales, cuota, proveedor caído, Ollama apagado
        avisar_error_del_llm(error)
