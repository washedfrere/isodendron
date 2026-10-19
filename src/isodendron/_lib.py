"""
isodendron._lib — Localización y carga de la shared library de ISODENDRON.

Este módulo se encarga exclusivamente de encontrar libisodendron y cargarla
con ctypes. El resto del paquete nunca toca ctypes.CDLL directamente: solo usa
el objeto `lib` que este módulo expone.

Orden de búsqueda:

  1. Variable de entorno ISODENDRON_LIB, si apunta a un fichero existente.
  2. Recurso empaquetado `isodendron/<nombre>`, que es donde lo instala
     scikit-build-core en la wheel (y en la instalación editable).

Si no se encuentra, se lanza RuntimeError con un mensaje que indique cómo
instalar o compilar la biblioteca.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path


# ----------------------------------------------------------------------
# Nombre de la shared library según plataforma
# ----------------------------------------------------------------------

def _lib_filename() -> str:
    if sys.platform.startswith("win"):
        return "isodendron.dll"
    if sys.platform == "darwin":
        return "libisodendron.dylib"
    return "libisodendron.so"


# ----------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------

#: Mantiene vivo el fichero extraído por importlib.resources mientras el
#: proceso use la biblioteca. ctypes necesita que el fichero siga en disco.
_stack = ExitStack()
atexit.register(_stack.close)


def _load() -> tuple[ctypes.CDLL, Path]:
    """
    Carga la shared library desde ISODENDRON_LIB o desde el paquete.

    Devuelve (CDLL, ruta). Lanza RuntimeError si no encuentra el fichero.
    """
    env = os.environ.get("ISODENDRON_LIB")
    if env:
        path = Path(env)
        if not path.is_file():
            raise RuntimeError(
                f"ISODENDRON_LIB apunta a un fichero que no existe: {path}"
            )
        return ctypes.CDLL(str(path)), path

    resource = files("isodendron").joinpath(_lib_filename())
    try:
        path = _stack.enter_context(as_file(resource))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se encontró la shared library de ISODENDRON dentro del "
            "paquete.\n"
            "Instala una wheel para tu plataforma con `pip install isodendron` "
            "o compila el proyecto con `pip install .`.\n"
            "También puedes definir ISODENDRON_LIB apuntando al fichero "
            ".dll/.so/.dylib."
        ) from exc

    return ctypes.CDLL(str(path)), Path(path)


#: Objeto CDLL con la biblioteca cargada. Los módulos hermanos importan
#: esto y declaran las firmas de las funciones que usen.
lib, lib_path = _load()
