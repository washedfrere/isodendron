"""
isodendron — Árbol de decisión ISODENDRON.

ISODENDRON es un árbol de decisión binario y determinista para features y
target discretos. En cada nodo divide por la condición

    feature == valor   vs   feature != valor

eligiendo el valor que mejor equilibra el tamaño de las dos ramas. El
target no participa en la elección del split: solo se usa para resumir
cada nodo con su moda y la probabilidad asociada.

Uso básico:

    from isodendron import Isodendron

    data = [
        ([1, 2, 3], 0),
        ([1, 5, 3], 1),
        ([4, 2, 3], 0),
        ([4, 5, 3], 1),
    ]

    model = Isodendron().fit(data, max_deep=8, p_min=(1, 1))
    print(model.predict([[1, 2, 3], [4, 5, 3]]))

La biblioteca C subyacente se carga al importar el paquete. Se instala con
`pip install isodendron`; para trabajar desde el repositorio, `pip install .`
o `pip install -e .`. Si el fichero no se encuentra, el import falla con
RuntimeError; también puedes definir ISODENDRON_LIB apuntando al .dll/.so/.dylib.

API pública:

    Isodendron          Clase principal. Entrena, predice, serializa.
    load_csv       Carga un CSV y devuelve (X, y, n, f).
    to_arrays      Convierte lista de tuplas a (X, y, n, f).
    from_columns   Convierte listas paralelas a (X, y, n, f).

Excepciones:

    IsodendronError     Error devuelto por la biblioteca C.
    DatasetError   Error de formato o contenido del dataset.

Todo lo demás (ctypes, arena, blob, formato binario) queda encapsulado
en módulos privados que no forman parte de la API estable.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _package_version

from .dataset import (
    DatasetError,
    from_columns,
    load_csv,
    to_arrays,
    INT32_MAX,
    INT32_MIN,
)
from ._ffi import IsodendronError
from .tree import Isodendron

try:
    __version__ = _package_version("isodendron")
except PackageNotFoundError:  # árbol de fuentes sin instalar
    __version__ = "0.1.0"

__all__ = [
    # API principal
    "Isodendron",
    "load_csv",
    "to_arrays",
    "from_columns",
    # Excepciones
    "IsodendronError",
    "DatasetError",
    # Constantes útiles
    "INT32_MIN",
    "INT32_MAX",
    # Metadatos
    "__version__",
]
