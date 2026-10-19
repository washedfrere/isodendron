"""
isodendron._ffi — Binding ctypes sobre libisodendron.

Este módulo declara las firmas de las funciones C y expone wrappers
que se comportan como Python normal:

  - Las entradas aceptan cualquier secuencia de ints y se normalizan
    a array.array('i') antes de cruzar la frontera.
  - Las salidas son bytes (el blob) o array.array('i') (predicciones),
    nunca punteros crudos.
  - Los códigos de error de C se traducen a excepciones Python.
  - La memoria asignada por C se copia a un objeto Python y se libera
    inmediatamente con isodendron_free.

Nada fuera de este módulo toca ctypes.CDLL ni conoce los códigos
ISODENDRON_OK/ISODENDRON_ERR_*.
"""

from __future__ import annotations

import array
import ctypes
from typing import Iterable, Sequence

from ._lib import lib, lib_path


# ----------------------------------------------------------------------
# Verificación del entorno
# ----------------------------------------------------------------------

#: Tipo de array que usamos para cruzar la frontera. Debe ser de 4 bytes
#: para coincidir con int32_t. En todas las plataformas modernas 'i' es
#: signed int de 4 bytes; si no lo fuera, el binding no funcionaría y
#: es mejor saberlo al importar que en el primer fit().
_I32 = "i"
if array.array(_I32).itemsize != 4:
    raise ImportError(
        "ISODENDRON requiere que array.array('i') sea de 4 bytes "
        f"(itemsize actual: {array.array(_I32).itemsize})"
    )


# ----------------------------------------------------------------------
# Excepción propia
# ----------------------------------------------------------------------

class IsodendronError(RuntimeError):
    """
    Error devuelto por la biblioteca C.

    Lleva el código numérico original en .code, por si el llamante
    quiere distinguir entre ISODENDRON_ERR_MEM y ISODENDRON_ERR_ARG sin parsear
    el mensaje.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


# Códigos espejo de isodendron.h. Se duplican aquí a propósito: el binding
# es el único punto del lado Python que debe conocerlos.
_ISODENDRON_OK = 0
_ISODENDRON_ERR_MEM = 1
_ISODENDRON_ERR_ARG = 2

_ERROR_NAMES = {
    _ISODENDRON_ERR_MEM: "ISODENDRON_ERR_MEM",
    _ISODENDRON_ERR_ARG: "ISODENDRON_ERR_ARG",
}


def _raise_on_error(code: int, context: str) -> None:
    if code == _ISODENDRON_OK:
        return
    name = _ERROR_NAMES.get(code, f"código {code}")
    raise IsodendronError(code, f"{context}: {name}")


# ----------------------------------------------------------------------
# Declaración de firmas
# ----------------------------------------------------------------------

_lib = lib

# const char *isodendron_version(void)
_lib.isodendron_version.argtypes = []
_lib.isodendron_version.restype = ctypes.c_char_p

# int isodendron_train(const int32_t *X, const int32_t *y,
#                 uint32_t n, uint32_t f,
#                 uint32_t max_deep,
#                 uint32_t p_min_num, uint32_t p_min_den,
#                 uint8_t **out_blob, size_t *out_len)
_lib.isodendron_train.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.POINTER(ctypes.c_int32),   # y
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.c_uint32,                  # max_deep
    ctypes.c_uint32,                  # p_min_num
    ctypes.c_uint32,                  # p_min_den
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # out_blob
    ctypes.POINTER(ctypes.c_size_t),                 # out_len
]
_lib.isodendron_train.restype = ctypes.c_int

# int isodendron_train_mt(const int32_t *X, const int32_t *y,
#                    uint32_t n, uint32_t f, uint32_t max_deep,
#                    uint32_t p_min_num, uint32_t p_min_den,
#                    uint32_t n_threads,
#                    uint8_t **out_blob, size_t *out_len)
_lib.isodendron_train_mt.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.POINTER(ctypes.c_int32),   # y
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.c_uint32,                  # max_deep
    ctypes.c_uint32,                  # p_min_num
    ctypes.c_uint32,                  # p_min_den
    ctypes.c_uint32,                  # n_threads
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # out_blob
    ctypes.POINTER(ctypes.c_size_t),                 # out_len
]
_lib.isodendron_train_mt.restype = ctypes.c_int

# int isodendron_predict(const uint8_t *blob, size_t blob_len,
#                   const int32_t *X, uint32_t n, uint32_t f,
#                   int32_t *out_pred)
_lib.isodendron_predict.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # blob
    ctypes.c_size_t,                  # blob_len
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.POINTER(ctypes.c_int32),   # out_pred
]
_lib.isodendron_predict.restype = ctypes.c_int

# int isodendron_predict_full(const uint8_t *blob, size_t blob_len,
#                        const int32_t *X, uint32_t n, uint32_t f,
#                        int32_t *out_pred, int32_t *out_n_mode,
#                        int32_t *out_leaf_n)
_lib.isodendron_predict_full.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # blob
    ctypes.c_size_t,                  # blob_len
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.POINTER(ctypes.c_int32),   # out_pred
    ctypes.POINTER(ctypes.c_int32),   # out_n_mode
    ctypes.POINTER(ctypes.c_int32),   # out_leaf_n
]
_lib.isodendron_predict_leaf.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # blob
    ctypes.c_size_t,                  # blob_len
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.POINTER(ctypes.c_int32),   # out_pred
    ctypes.POINTER(ctypes.c_uint32),  # out_leaf_idx
]
_lib.isodendron_predict_leaf.restype = ctypes.c_int

_lib.isodendron_train_mt_cap.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.POINTER(ctypes.c_int32),   # y
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.c_uint32,                  # max_deep
    ctypes.c_uint32,                  # p_min_num
    ctypes.c_uint32,                  # p_min_den
    ctypes.c_uint32,                  # max_targets
    ctypes.c_uint32,                  # n_threads
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # out_blob
    ctypes.POINTER(ctypes.c_size_t),                 # out_len
]
_lib.isodendron_train_mt_cap.restype = ctypes.c_int

class _HojaAnomala(ctypes.Structure):
    _fields_ = [
        ("nodo", ctypes.c_uint32),
        ("N_previo", ctypes.c_int32),
        ("payload_len", ctypes.c_uint32),
        ("resta", ctypes.c_uint32),
        ("N_final", ctypes.c_int32),
    ]


class _PartialDiag(ctypes.Structure):
    _fields_ = [
        ("puntos", ctypes.c_uint64),
        ("hojas", ctypes.c_uint32),
        ("decayadas", ctypes.c_uint32),
        ("vaciadas", ctypes.c_uint32),
        ("resta_maxima", ctypes.c_uint32),
        ("anomalas", ctypes.c_uint32),
        ("capacidad", ctypes.c_uint32),
        ("detalle", ctypes.POINTER(_HojaAnomala)),
    ]


_lib.isodendron_partial_fit_diag.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(_PartialDiag),
]
_lib.isodendron_partial_fit_diag.restype = ctypes.c_int

_lib.isodendron_partial_fit.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),   # blob_in
    ctypes.c_size_t,                  # blob_len
    ctypes.POINTER(ctypes.c_int32),   # X
    ctypes.POINTER(ctypes.c_int32),   # y
    ctypes.c_uint32,                  # n
    ctypes.c_uint32,                  # f
    ctypes.c_uint32,                  # max_targets
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # out_blob
    ctypes.POINTER(ctypes.c_size_t),                 # out_len
]
_lib.isodendron_partial_fit.restype = ctypes.c_int

_lib.isodendron_predict_full.restype = ctypes.c_int

# void isodendron_free(void *ptr)
_lib.isodendron_free.argtypes = [ctypes.c_void_p]
_lib.isodendron_free.restype = None


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------

def _as_i32_array(seq: Iterable[int], name: str) -> array.array:
    """
    Normaliza una secuencia de ints a array.array('i').

    Si ya es un array.array('i'), se devuelve tal cual para evitar
    copias. Si es otra secuencia, se crea un array nuevo. Los valores
    fuera del rango int32 se rechazan en la construcción del array
    (array.array lanza OverflowError).
    """
    if isinstance(seq, array.array):
        if seq.typecode != _I32:
            raise TypeError(
                f"{name}: se esperaba array.array('{_I32}'), "
                f"se recibió array.array('{seq.typecode}')"
            )
        return seq
    try:
        return array.array(_I32, seq)
    except OverflowError as exc:
        raise OverflowError(
            f"{name}: algún valor no cabe en int32_t"
        ) from exc


def _ptr_i32(arr: array.array) -> ctypes.POINTER(ctypes.c_int32):
    """
    Devuelve un puntero al buffer interno de un array.array('i').
    Usa buffer_info para obtener la dirección; el array debe seguir
    vivo mientras el puntero se use.
    """
    addr, _ = arr.buffer_info()
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_int32))


def _ptr_u8(data: bytes) -> ctypes.POINTER(ctypes.c_uint8):
    """
    Devuelve un puntero al buffer de un objeto bytes. Los bytes son
    inmutables, así que el buffer es de solo lectura y no necesita
    copia. C no modificará el blob.
    """
    return ctypes.cast(
        ctypes.c_char_p(data),
        ctypes.POINTER(ctypes.c_uint8),
    )


# ----------------------------------------------------------------------
# API pública del módulo
# ----------------------------------------------------------------------

def version() -> str:
    """
    Devuelve la versión de la biblioteca C como str.
    """
    raw = _lib.isodendron_version()
    if raw is None:
        return ""
    return raw.decode("ascii", errors="replace")


def train_mt(
    X: Sequence[int],
    y: Sequence[int],
    n: int,
    f: int,
    max_deep: int,
    p_min_num: int,
    p_min_den: int,
    hilos: int = 1,
) -> bytes:
    """
    Entrena un árbol ISODENDRON y devuelve el blob serializado como bytes.

    Es la variante multihilo: la construcción del árbol se reparte entre
    `hilos` hilos. El reparto es transparente, así que el blob es idéntico
    byte a byte al de un solo hilo; solo cambia el tiempo.

    Parámetros:
      X           Matriz row-major de features, longitud n * f.
                  Acepta list, tuple o array.array('i').
      y           Vector de targets, longitud n.
      n           Número de muestras.
      f           Número de features.
      max_deep    Profundidad máxima.
      p_min_num   Numerador de P_MIN.
      p_min_den   Denominador de P_MIN (debe ser > 0).
                  P_MIN = p_min_num / p_min_den es la pureza mínima de la
                  moda: un nodo se convierte en hoja si
                  n_mode / N >= P_MIN. Se rechaza p_min_num = 0 porque
                  dejaría el árbol en una sola hoja.
      hilos       Número de hilos (>= 1). 1 equivale a train().

    Devuelve:
      El blob como bytes. Independiente del buffer que C haya usado:
      se copia y se libera antes de volver.

    Lanza:
      TypeError      si X o y no son secuencias de ints compatibles.
      OverflowError  si algún valor no cabe en int32_t.
      IsodendronError     si C devuelve un código de error.
      ValueError     si n, f, max_deep, hilos o p_min_den son inválidos,
                     o si p_min_num es 0.
    """
    if n <= 0 or f <= 0 or max_deep <= 0 or p_min_den <= 0:
        raise ValueError(
            f"parámetros inválidos: n={n}, f={f}, "
            f"max_deep={max_deep}, p_min_den={p_min_den}"
        )
    if p_min_num == 0:
        raise ValueError(
            "p_min_num=0 deja el árbol reducido a una sola hoja; "
            "usa (1, 1) para exigir pureza total"
        )
    if hilos < 1:
        raise ValueError(f"hilos debe ser >= 1 (recibido: {hilos})")

    X_arr = _as_i32_array(X, "X")
    y_arr = _as_i32_array(y, "y")

    if len(X_arr) != n * f:
        raise ValueError(
            f"X tiene {len(X_arr)} elementos, se esperaban {n * f}"
        )
    if len(y_arr) != n:
        raise ValueError(
            f"y tiene {len(y_arr)} elementos, se esperaban {n}"
        )

    # Preparamos los punteros de salida. ctypes los inicializa a
    # NULL/0; la biblioteca C los rellena con malloc.
    out_blob = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t(0)

    code = _lib.isodendron_train_mt(
        _ptr_i32(X_arr),
        _ptr_i32(y_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        ctypes.c_uint32(max_deep),
        ctypes.c_uint32(p_min_num),
        ctypes.c_uint32(p_min_den),
        ctypes.c_uint32(hilos),
        ctypes.byref(out_blob),
        ctypes.byref(out_len),
    )
    _raise_on_error(code, "isodendron_train_mt")

    # Si por lo que sea C devolvió OK pero sin blob, es un error de
    # contrato. Lo tratamos como error duro.
    if not out_blob or out_len.value == 0:
        if out_blob:
            _lib.isodendron_free(out_blob)
        raise IsodendronError(
            _ISODENDRON_ERR_ARG,
            "isodendron_train_mt: ISODENDRON_OK pero blob vacío o nulo",
        )

    try:
        blob = ctypes.string_at(out_blob, out_len.value)
    finally:
        _lib.isodendron_free(out_blob)

    return blob


def train(
    X: Sequence[int],
    y: Sequence[int],
    n: int,
    f: int,
    max_deep: int,
    p_min_num: int,
    p_min_den: int,
) -> bytes:
    """
    Entrena un árbol ISODENDRON y devuelve el blob serializado como bytes.

    Variante de un solo hilo. Para repartir la construcción entre varios
    hilos, usa train_mt(); el blob resultante es el mismo.
    """
    return train_mt(X, y, n, f, max_deep, p_min_num, p_min_den, 1)


def predict(
    blob: bytes,
    X: Sequence[int],
    n: int,
    f: int,
) -> array.array:
    """
    Predice para un lote de muestras.

    Parámetros:
      blob   Blob devuelto por train().
      X      Matriz row-major de features, longitud n * f.
      n      Número de muestras.
      f      Número de features (debe coincidir con el del blob).

    Devuelve:
      array.array('i') de longitud n con las modas predichas.

    Lanza:
      TypeError      si X no es una secuencia de ints compatible.
      ValueError     si n o f son inválidos o los tamaños no cuadran.
      IsodendronError     si C devuelve un código de error (blob corrupto,
                     incompatible con f, etc.).
    """
    if n <= 0 or f <= 0:
        raise ValueError(f"parámetros inválidos: n={n}, f={f}")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(
            "blob debe ser bytes, bytearray o memoryview"
        )

    blob_bytes = bytes(blob)

    X_arr = _as_i32_array(X, "X")
    if len(X_arr) != n * f:
        raise ValueError(
            f"X tiene {len(X_arr)} elementos, se esperaban {n * f}"
        )

    # Reservamos el buffer de salida. array.array('i', [0]) * n crea
    # n ceros; from_buffer requiere un buffer escribible del tamaño
    # exacto, y eso es justo lo que tenemos.
    out = array.array(_I32, [0]) * n

    out_ptr = ctypes.cast(
        out.buffer_info()[0],
        ctypes.POINTER(ctypes.c_int32),
    )

    code = _lib.isodendron_predict(
        _ptr_u8(blob_bytes),
        ctypes.c_size_t(len(blob_bytes)),
        _ptr_i32(X_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        out_ptr,
    )
    _raise_on_error(code, "isodendron_predict")

    return out


def predict_full(
    blob: bytes,
    X: Sequence[int],
    n: int,
    f: int,
) -> tuple[array.array, array.array, array.array]:
    """
    Predice y devuelve además el resumen de la hoja alcanzada.

    Devuelve:
      (pred, n_mode, leaf_n), los tres array.array('i') de longitud n.
      La pureza de la hoja es P = n_mode / leaf_n: la exactitud del
      modelo dentro de esa región en el conjunto de entrenamiento.

    Lanza:
      TypeError    si X no es una secuencia de ints o blob no es bytes.
      ValueError   si n o f son inválidos o los tamaños no cuadran.
      IsodendronError   si C devuelve un código de error.
    """
    if n <= 0 or f <= 0:
        raise ValueError(f"parámetros inválidos: n={n}, f={f}")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob debe ser bytes, bytearray o memoryview")

    blob_bytes = bytes(blob)

    X_arr = _as_i32_array(X, "X")
    if len(X_arr) != n * f:
        raise ValueError(
            f"X tiene {len(X_arr)} elementos, se esperaban {n * f}"
        )

    pred = array.array(_I32, [0]) * n
    nmode = array.array(_I32, [0]) * n
    hoja = array.array(_I32, [0]) * n

    def _ptr(a: array.array) -> ctypes.POINTER(ctypes.c_int32):
        return ctypes.cast(a.buffer_info()[0], ctypes.POINTER(ctypes.c_int32))

    code = _lib.isodendron_predict_full(
        _ptr_u8(blob_bytes),
        ctypes.c_size_t(len(blob_bytes)),
        _ptr_i32(X_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        _ptr(pred),
        _ptr(nmode),
        _ptr(hoja),
    )
    _raise_on_error(code, "isodendron_predict_full")

    return pred, nmode, hoja


def predict_leaf(
    blob: bytes,
    X: Sequence[int],
    n: int,
    f: int,
) -> tuple[array.array, array.array]:
    """
    Predice y devuelve ademas el indice de nodo hoja de cada muestra.

    Devuelve (pred, hoja): `pred` es un array('i') con la moda de la hoja
    y `hoja` un array('I') con el indice del nodo en preorden, que es el
    que permite recuperar la distribucion de esa hoja.
    """
    if n <= 0 or f <= 0:
        raise ValueError(f"parametros invalidos: n={n}, f={f}")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob debe ser bytes, bytearray o memoryview")

    blob_bytes = bytes(blob)
    X_arr = _as_i32_array(X, "X")
    if len(X_arr) != n * f:
        raise ValueError(
            f"X tiene {len(X_arr)} elementos, se esperaban {n * f}"
        )

    pred = array.array(_I32, [0]) * n
    hojas = array.array("I", [0]) * n

    code = _lib.isodendron_predict_leaf(
        _ptr_u8(blob_bytes),
        ctypes.c_size_t(len(blob_bytes)),
        _ptr_i32(X_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        ctypes.cast(pred.buffer_info()[0], ctypes.POINTER(ctypes.c_int32)),
        ctypes.cast(hojas.buffer_info()[0], ctypes.POINTER(ctypes.c_uint32)),
    )
    _raise_on_error(code, "isodendron_predict_leaf")

    return pred, hojas


def train_mt_cap(
    X: Sequence[int],
    y: Sequence[int],
    n: int,
    f: int,
    max_deep: int,
    p_min_num: int,
    p_min_den: int,
    max_targets: int,
    hilos: int = 1,
) -> bytes:
    """
    Entrena igual que train_mt pero fija el **tope de entradas de
    distribución por hoja**.

    `max_targets = 0` desactiva el payload: las hojas quedan con
    payload_len = 0 y el blob no lleva distribuciones. En otro caso cada
    hoja guarda sus `max_targets` targets más frecuentes.
    """
    if n <= 0 or f <= 0 or max_deep <= 0 or p_min_den <= 0:
        raise ValueError(
            f"parámetros inválidos: n={n}, f={f}, max_deep={max_deep}, "
            f"p_min_den={p_min_den}"
        )
    if p_min_num == 0:
        raise ValueError("p_min_num=0 dejaría el árbol en una sola hoja")
    if max_targets < 0:
        raise ValueError(f"max_targets debe ser >= 0 (recibido: {max_targets})")
    if hilos < 1:
        raise ValueError(f"hilos debe ser >= 1 (recibido: {hilos})")

    X_arr = _as_i32_array(X, "X")
    y_arr = _as_i32_array(y, "y")
    if len(X_arr) != n * f:
        raise ValueError(
            f"X tiene {len(X_arr)} elementos, se esperaban {n * f}"
        )
    if len(y_arr) != n:
        raise ValueError(f"y tiene {len(y_arr)} elementos, se esperaban {n}")

    out_blob = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t(0)

    code = _lib.isodendron_train_mt_cap(
        _ptr_i32(X_arr),
        _ptr_i32(y_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        ctypes.c_uint32(max_deep),
        ctypes.c_uint32(p_min_num),
        ctypes.c_uint32(p_min_den),
        ctypes.c_uint32(max_targets),
        ctypes.c_uint32(hilos),
        ctypes.byref(out_blob),
        ctypes.byref(out_len),
    )
    _raise_on_error(code, "isodendron_train_mt_cap")

    if not out_blob or out_len.value == 0:
        if out_blob:
            _lib.isodendron_free(out_blob)
        raise IsodendronError(
            _ISODENDRON_ERR_ARG,
            "isodendron_train_mt_cap: ISODENDRON_OK pero blob vacío o nulo",
        )

    try:
        return ctypes.string_at(out_blob, out_len.value)
    finally:
        _lib.isodendron_free(out_blob)


def partial_fit(
    blob: bytes,
    X: Sequence[int],
    y: Sequence[int],
    n: int,
    f: int,
    max_targets: int,
) -> bytes:
    """
    Recalibra las hojas de `blob` con (X, y) **sin cambiar el árbol**.

    Los nodos internos quedan intactos; solo se reescriben N, K, mode,
    n_mode y la distribución de cada hoja. Devuelve un blob nuevo.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob debe ser bytes, bytearray o memoryview")
    if n <= 0 or f <= 0:
        raise ValueError(f"parámetros inválidos: n={n}, f={f}")
    if max_targets < 0:
        raise ValueError(f"max_targets debe ser >= 0 (recibido: {max_targets})")

    blob_bytes = bytes(blob)
    X_arr = _as_i32_array(X, "X")
    y_arr = _as_i32_array(y, "y")
    if len(X_arr) != n * f:
        raise ValueError(
            f"X tiene {len(X_arr)} elementos, se esperaban {n * f}"
        )
    if len(y_arr) != n:
        raise ValueError(f"y tiene {len(y_arr)} elementos, se esperaban {n}")

    out_blob = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t(0)

    code = _lib.isodendron_partial_fit(
        _ptr_u8(blob_bytes),
        ctypes.c_size_t(len(blob_bytes)),
        _ptr_i32(X_arr),
        _ptr_i32(y_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        ctypes.c_uint32(max_targets),
        ctypes.byref(out_blob),
        ctypes.byref(out_len),
    )
    _raise_on_error(code, "isodendron_partial_fit")

    if not out_blob or out_len.value == 0:
        if out_blob:
            _lib.isodendron_free(out_blob)
        raise IsodendronError(
            _ISODENDRON_ERR_ARG,
            "isodendron_partial_fit: ISODENDRON_OK pero blob vacío o nulo",
        )

    try:
        return ctypes.string_at(out_blob, out_len.value)
    finally:
        _lib.isodendron_free(out_blob)


def partial_fit_diag(
    blob: bytes,
    X: Sequence[int],
    y: Sequence[int],
    n: int,
    f: int,
    max_targets: int,
    capacidad: int = 64,
) -> tuple[bytes, dict]:
    """
    Igual que partial_fit, pero devuelve ademas un diagnostico.

    Devuelve (blob_nuevo, info). `info` lleva los contadores y hasta
    `capacidad` muestras de hojas anomalas, asi que con un dataset enorme
    basta con pedir las primeras: no se reserva memoria por hoja.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob debe ser bytes, bytearray o memoryview")
    if n <= 0 or f <= 0:
        raise ValueError(f"parametros invalidos: n={n}, f={f}")
    if max_targets < 0:
        raise ValueError(f"max_targets debe ser >= 0 (recibido: {max_targets})")
    if capacidad < 0:
        raise ValueError(f"capacidad debe ser >= 0 (recibido: {capacidad})")

    blob_bytes = bytes(blob)
    X_arr = _as_i32_array(X, "X")
    y_arr = _as_i32_array(y, "y")
    if len(X_arr) != n * f:
        raise ValueError(f"X tiene {len(X_arr)} elementos, se esperaban {n * f}")
    if len(y_arr) != n:
        raise ValueError(f"y tiene {len(y_arr)} elementos, se esperaban {n}")

    reserva = max(int(capacidad), 1)
    muestras = (_HojaAnomala * reserva)()
    diag = _PartialDiag()
    diag.capacidad = int(capacidad)
    diag.detalle = (ctypes.cast(muestras, ctypes.POINTER(_HojaAnomala))
                    if capacidad > 0 else None)

    out_blob = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t(0)

    code = _lib.isodendron_partial_fit_diag(
        _ptr_u8(blob_bytes),
        ctypes.c_size_t(len(blob_bytes)),
        _ptr_i32(X_arr),
        _ptr_i32(y_arr),
        ctypes.c_uint32(n),
        ctypes.c_uint32(f),
        ctypes.c_uint32(max_targets),
        ctypes.byref(out_blob),
        ctypes.byref(out_len),
        ctypes.byref(diag),
    )
    _raise_on_error(code, "isodendron_partial_fit_diag")

    if not out_blob or out_len.value == 0:
        if out_blob:
            _lib.isodendron_free(out_blob)
        raise IsodendronError(_ISODENDRON_ERR_ARG, "isodendron_partial_fit_diag: blob vacio")

    try:
        nuevo = ctypes.string_at(out_blob, out_len.value)
    finally:
        _lib.isodendron_free(out_blob)

    registradas = min(diag.anomalas, capacidad)
    info = {
        "puntos": diag.puntos,
        "hojas": diag.hojas,
        "decayadas": diag.decayadas,
        "vaciadas": diag.vaciadas,
        "resta_maxima": diag.resta_maxima,
        "anomalas": diag.anomalas,
        "detalle": [
            {
                "nodo": muestras[k].nodo,
                "N_previo": muestras[k].N_previo,
                "payload_len": muestras[k].payload_len,
                "resta": muestras[k].resta,
                "N_final": muestras[k].N_final,
            }
            for k in range(registradas)
        ],
    }
    return nuevo, info


def free(ptr: int) -> None:
    """
    Libera memoria asignada por la biblioteca. Solo para uso interno
    o para casos muy específicos; el binding normal nunca expone
    punteros al llamante.
    """
    _lib.isodendron_free(ctypes.c_void_p(ptr))


__all__ = [
    "IsodendronError",
    "lib_path",
    "version",
    "train",
    "train_mt",
    "train_mt_cap",
    "partial_fit",
    "partial_fit_diag",
    "predict",
    "predict_full",
    "predict_leaf",
    "free",
]



