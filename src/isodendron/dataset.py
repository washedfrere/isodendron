"""
isodendron.dataset — Carga y normalización del dataset.

ISODENDRON trabaja con features y target discretos, todos ellos enteros
de 32 bits con signo. Este módulo convierte distintas representaciones
de entrada (lista de tuplas, listas paralelas, CSV) a la forma que
_ffi.train necesita:

  - X: array.array('i') en row-major, longitud n * f
  - y: array.array('i'), longitud n

Reglas de validación:

  - Cada muestra debe ser una tupla (features, target).
  - Features debe ser una secuencia de enteros del mismo tamaño en
    todas las muestras.
  - Target debe ser un entero.
  - Todos los valores deben caber en int32_t (rango
    [-2^31, 2^31 - 1]). Los que no quepan provocan OverflowError.

El formato "lista de tuplas" es deliberadamente simple:

    data = [
        ([1, 2, 3], 0),
        ([1, 5, 3], 1),
        ([4, 2, 3], 0),
        ...
    ]

Por convención, la primera lista son features y el segundo valor es
el target.
"""

from __future__ import annotations

import array
import csv
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Reutilizamos el typecode que ya validó _ffi al importarse.
_I32 = "i"

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


# ----------------------------------------------------------------------
# Excepciones
# ----------------------------------------------------------------------

class DatasetError(ValueError):
    """Error de formato o contenido en el dataset de entrada."""


# ----------------------------------------------------------------------
# Conversión desde lista de tuplas
# ----------------------------------------------------------------------

def to_arrays(
    data: Iterable[tuple[Sequence[int], int]],
) -> tuple[array.array, array.array, int, int]:
    """
    Convierte una lista de tuplas (features, target) a los dos arrays
    que espera _ffi.train.

    Parámetros:
      data   Iterable de tuplas. Cada tupla es
             (secuencia_de_features, target).

    Devuelve:
      (X, y, n, f) donde X es array.array('i') row-major, y es
      array.array('i'), n es el número de muestras y f el número de
      features.

    Lanza:
      DatasetError  si el dataset está vacío, tiene formas
                    inconsistentes o contiene valores no enteros.
      OverflowError si algún valor no cabe en int32_t.

    Notas:
      - Aceptamos cualquier iterable (list, tuple, generador). Para
        datos muy grandes, pasar un generador evita materializar la
        lista completa en memoria, aunque el resultado (X, y) sí
        se materializa, porque es lo que cruza la frontera.
      - No ordenamos ni transformamos las features. Si el dataset
        viene con columnas en un orden concreto, ese orden se
        respeta en X.
    """
    X = array.array(_I32)
    y = array.array(_I32)

    n = 0
    f: int | None = None

    for i, sample in enumerate(data):
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise DatasetError(
                f"muestra {i}: se esperaba (features, target), "
                f"se recibió {type(sample).__name__}"
            )

        feats, target = sample

        if not isinstance(feats, (list, tuple, array.array)):
            raise DatasetError(
                f"muestra {i}: features debe ser secuencia, "
                f"se recibió {type(feats).__name__}"
            )

        if f is None:
            f = len(feats)
            if f == 0:
                raise DatasetError(
                    "las muestras no pueden tener 0 features"
                )
        elif len(feats) != f:
            raise DatasetError(
                f"muestra {i}: se esperaban {f} features, "
                f"se recibieron {len(feats)}"
            )

        _append_checked_int(X, feats, f"muestra {i} feature")
        _append_checked_int(y, [target], f"muestra {i} target")

        n += 1

    if n == 0:
        raise DatasetError("el dataset está vacío")
    assert f is not None

    return X, y, n, f


# ----------------------------------------------------------------------
# Conversión desde listas paralelas
# ----------------------------------------------------------------------

def from_columns(
    features: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> tuple[array.array, array.array, int, int]:
    """
    Convierte listas paralelas a los arrays que espera _ffi.train.

    Parámetros:
      features   Lista de filas. features[i] es la i-ésima muestra,
                 con sus f valores de feature.
      targets    Lista de targets, misma longitud que features.

    Devuelve:
      (X, y, n, f) igual que to_arrays.

    Lanza:
      DatasetError  si las longitudes no cuadran o hay formas
                    inconsistentes.
      OverflowError si algún valor no cabe en int32_t.

    Esta función es un atajo para quien ya tiene los datos en dos
    listas separadas en lugar de intercalados en tuplas.
    """
    if len(features) != len(targets):
        raise DatasetError(
            f"features tiene {len(features)} filas, "
            f"targets tiene {len(targets)} valores"
        )
    return to_arrays(zip(features, targets))


# ----------------------------------------------------------------------
# Carga desde CSV
# ----------------------------------------------------------------------

def load_csv(
    path: str | Path,
    target_col: int | str,
    feature_cols: Sequence[int | str] | None = None,
    *,
    has_header: bool = True,
    delimiter: str = ",",
) -> tuple[array.array, array.array, int, int]:
    """
    Carga un CSV y lo convierte a los arrays que espera _ffi.train.

    Parámetros:
      path          Ruta al fichero CSV.
      target_col    Índice (int) o nombre (str) de la columna target.
      feature_cols  Índices o nombres de columnas a usar como features.
                    Si es None, se usan todas las columnas excepto el
                    target, en el orden en que aparecen.
      has_header    Si True, la primera fila es cabecera y target_col
                    y feature_cols pueden ser nombres. Si False, deben
                    ser índices.
      delimiter     Separador de campos. Por defecto ','.

    Devuelve:
      (X, y, n, f) igual que to_arrays.

    Lanza:
      DatasetError  si hay problemas de formato, columnas inexistentes,
                    celdas no enteras o inconsistencias de tamaño.
      FileNotFoundError si el fichero no existe.

    Notas:
      - Todas las celdas deben ser enteros. Los que no lo sean se
        rechazan con DatasetError, indicando fila y columna.
      - El CSV se lee en modo texto con codificación UTF-8.
      - No hay inferencia de tipos: si una columna contiene "1.0"
        se rechaza, aunque sea un entero disfrazado. Es responsabilidad
        del usuario discretizar antes de llamar a ISODENDRON.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no existe el fichero: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)

        rows: Iterator[list[str]]
        header: list[str] | None = None

        if has_header:
            try:
                header = next(reader)
            except StopIteration:
                raise DatasetError(f"{path}: fichero vacío") from None

        # Resolver índices de columnas a posiciones concretas.
        tgt_idx = _resolve_col(target_col, header, "target_col")
        if feature_cols is None:
            # Todas menos el target, en orden de aparición.
            n_cols = len(header) if header is not None else None
            if n_cols is None:
                # Sin cabecera y sin feature_cols explícitas no podemos
                # saber cuántas columnas hay hasta leer la primera fila.
                # Leemos una fila para averiguarlo y luego la procesamos
                # junto al resto.
                try:
                    first = next(reader)
                except StopIteration:
                    raise DatasetError(
                        f"{path}: fichero sin filas de datos"
                    ) from None
                n_cols = len(first)
                feat_idxs = [i for i in range(n_cols) if i != tgt_idx]
                rows = _chain_one(first, reader)
            else:
                feat_idxs = [i for i in range(n_cols) if i != tgt_idx]
                rows = reader
        else:
            feat_idxs = [
                _resolve_col(c, header, f"feature_cols[{i}]")
                for i, c in enumerate(feature_cols)
            ]
            rows = reader

        if not feat_idxs:
            raise DatasetError(
                "no hay columnas de features seleccionadas"
            )

        X = array.array(_I32)
        y = array.array(_I32)
        n = 0
        f = len(feat_idxs)

        for lineno, row in enumerate(rows, start=(2 if has_header else 1)):
            if not row:
                # Líneas vacías: las saltamos silenciosamente. Es
                # habitual en CSV generados a mano con saltos extra.
                continue

            needed = max([tgt_idx] + feat_idxs) + 1
            if len(row) < needed:
                raise DatasetError(
                    f"{path}:{lineno}: se esperaban al menos "
                    f"{needed} columnas, se encontraron {len(row)}"
                )

            _append_checked_cell(
                y, row[tgt_idx], f"{path}:{lineno}: target"
            )
            for j, col in enumerate(feat_idxs):
                _append_checked_cell(
                    X, row[col],
                    f"{path}:{lineno}: feature[{j}]"
                )
            n += 1

    if n == 0:
        raise DatasetError(f"{path}: no se cargó ninguna fila")

    return X, y, n, f


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------

def _append_checked_int(
    dst: array.array,
    values: Iterable[int],
    context: str,
) -> None:
    """
    Añade valores a un array, verificando que son enteros y caben en
    int32_t. Acepta bool como no-entero (bool es subclase de int, y
    aquí no queremos silenciosamente tratarlo como 0/1).
    """
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise DatasetError(
                f"{context}: se esperaba int, "
                f"se recibió {type(v).__name__} ({v!r})"
            )
        if v < INT32_MIN or v > INT32_MAX:
            raise OverflowError(
                f"{context}: {v} no cabe en int32_t"
            )
        dst.append(v)


def _append_checked_cell(
    dst: array.array,
    raw: str,
    context: str,
) -> None:
    """
    Convierte una celda de CSV a int y la añade al array.
    Rechaza floats, notación científica, espacios alrededor y
    cualquier cosa que no sea un entero literal.
    """
    s = raw.strip()
    if not s:
        raise DatasetError(f"{context}: celda vacía")
    try:
        v = int(s, 10)
    except ValueError as exc:
        raise DatasetError(
            f"{context}: {raw!r} no es un entero"
        ) from exc
    if v < INT32_MIN or v > INT32_MAX:
        raise OverflowError(f"{context}: {v} no cabe en int32_t")
    dst.append(v)


def _resolve_col(
    col: int | str,
    header: list[str] | None,
    name: str,
) -> int:
    """
    Traduce una referencia de columna (índice o nombre) a un índice
    entero. Si es str y no hay header, es error. Si es str y no está
    en el header, es error.
    """
    if isinstance(col, bool):
        raise DatasetError(f"{name}: bool no es un índice válido")
    if isinstance(col, int):
        if col < 0:
            raise DatasetError(f"{name}: índice negativo ({col})")
        return col
    if isinstance(col, str):
        if header is None:
            raise DatasetError(
                f"{name}: se dio un nombre ({col!r}) pero el CSV no "
                f"tiene cabecera"
            )
        try:
            return header.index(col)
        except ValueError:
            raise DatasetError(
                f"{name}: columna {col!r} no encontrada en el header"
            ) from None
    raise DatasetError(
        f"{name}: se esperaba int o str, "
        f"se recibió {type(col).__name__}"
    )


def _chain_one(
    first: list[str],
    rest: Iterator[list[str]],
) -> Iterator[list[str]]:
    """
    Generador auxiliar que produce primero `first` y luego el resto.
    Se usa para el caso en el que hay que leer una fila del CSV para
    deducir el número de columnas y luego procesarla junto al resto.
    """
    yield first
    yield from rest


__all__ = [
    "DatasetError",
    "to_arrays",
    "from_columns",
    "load_csv",
    "INT32_MIN",
    "INT32_MAX",
]
