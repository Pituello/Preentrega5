"""
Herramientas del agente (el "contrato" de Tool Calling):

- Nombre y docstring: el LLM decide si usar una herramienta SOLO con eso.
  Por eso los docstrings dicen qué devuelve cada una, cuándo usarla y en
  qué orden ("Descripciones Vagas" es el primer error a evitar).
- Esquema de entrada con Pydantic (args_schema): valida los argumentos que
  manda el LLM antes de tocar la "base de datos".
- Least Privilege: las dos herramientas son de solo lectura. El agente no
  puede modificar el catálogo ni el stock.
- Errores como texto: un modelo inexistente o un ID mal escrito no rompen
  el programa; vuelven al LLM como "ERROR: ..." para que reintente o pida
  aclaración (el "Ciclo de Retorno" de la consigna).
"""
import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .catalogo import STOCK, SUCURSALES, VERSIONES, normalizar

# Sucursales cuyo "sistema de stock" ya respondió una vez en este proceso.
# Córdoba simula un sistema inestable: su PRIMERA consulta da timeout.
_SUCURSALES_CONSULTADAS: set[str] = set()
SUCURSAL_INESTABLE = "Córdoba"


class BuscarVersionesInput(BaseModel):
    modelo: str = Field(
        description="Nombre del modelo de Volkswagen, por ejemplo 'Tera', 'Polo' o 'Saveiro'. Sin la marca."
    )


class ConsultarStockInput(BaseModel):
    version_id: str = Field(
        description="ID exacto de la versión, tal como lo devuelve buscar_versiones (ej: 'TERA-HIGH-AT6')."
    )
    sucursal: str | None = Field(
        default=None,
        description=f"Sucursal a consultar: una de {SUCURSALES}. Si se omite, devuelve todas las sucursales.",
    )


@tool(args_schema=BuscarVersionesInput)
async def buscar_versiones(modelo: str) -> str:
    """Busca en el catálogo de la concesionaria las versiones disponibles de un modelo
    de Volkswagen (Tera, Polo, Virtus, Nivus, Taos, Vento o Saveiro).

    Devuelve, para cada versión: version_id, nombre de la versión, motor, potencia (CV),
    torque (Nm), transmisión y si es automática. Usala cuando el usuario pregunte por
    versiones, motores, potencia o cajas de un modelo, y SIEMPRE antes de consultar_stock,
    porque el stock se consulta por version_id y el usuario no lo conoce.

    Devuelve un texto que empieza con ERROR si el modelo no está en el catálogo."""
    clave = normalizar(modelo).removeprefix("volkswagen ").removeprefix("vw ")
    encontradas = [v for v in VERSIONES if normalizar(v.modelo) == clave]
    if not encontradas:
        modelos = sorted({v.modelo for v in VERSIONES})
        return (
            f"ERROR: el modelo '{modelo}' no está en el catálogo de la concesionaria. "
            f"Modelos disponibles: {', '.join(modelos)}."
        )
    return json.dumps([v.model_dump() for v in encontradas], ensure_ascii=False)


@tool(args_schema=ConsultarStockInput)
async def consultar_stock(version_id: str, sucursal: str | None = None) -> str:
    """Consulta cuántas unidades hay en stock de UNA versión, en una sucursal o en todas.

    Necesita el version_id exacto: si solo tenés el nombre del modelo, primero usá
    buscar_versiones para obtenerlo. Para comparar varias versiones (por ejemplo, todas
    las automáticas de un modelo), llamala una vez por cada version_id.

    Devuelve las unidades por sucursal (0 = sin stock). Devuelve un texto que empieza con
    ERROR si el version_id o la sucursal no existen. Si el sistema de una sucursal no
    responde (timeout), conviene reintentar la consulta una vez."""
    version_id = version_id.strip().upper()
    stock_version = STOCK.get(version_id)
    if stock_version is None:
        return (
            f"ERROR: no existe la versión '{version_id}'. "
            "Usá buscar_versiones para obtener los version_id válidos."
        )

    if sucursal is None:
        return json.dumps({"version_id": version_id, "stock": stock_version}, ensure_ascii=False)

    canonica = next((s for s in SUCURSALES if normalizar(s) == normalizar(sucursal)), None)
    if canonica is None:
        return f"ERROR: la sucursal '{sucursal}' no existe. Sucursales válidas: {', '.join(SUCURSALES)}."

    # Falla de infraestructura simulada: se LANZA una excepción (no se devuelve
    # texto), como haría un cliente HTTP real. ToolNode(handle_tool_errors=True)
    # la convierte en un mensaje para el LLM en vez de cortar la ejecución.
    if canonica == SUCURSAL_INESTABLE and canonica not in _SUCURSALES_CONSULTADAS:
        _SUCURSALES_CONSULTADAS.add(canonica)
        raise TimeoutError(
            f"El sistema de stock de la sucursal {canonica} no respondió a tiempo (timeout simulado)."
        )
    _SUCURSALES_CONSULTADAS.add(canonica)

    return json.dumps(
        {"version_id": version_id, "sucursal": canonica, "unidades": stock_version[canonica]},
        ensure_ascii=False,
    )


HERRAMIENTAS = [buscar_versiones, consultar_stock]
