"""
test_isodendron.py — Suite de pruebas del binding Python de ISODENDRON.

Cubre la API pública del paquete `isodendron`, que es lo que un usuario final
puede tocar:

    - localización y carga de la shared library
    - Isodendron.fit / predict / predict_one / dump
    - normalización de datasets (to_arrays, from_columns, load_csv)
    - validación de argumentos y errores
    - formato del blob y detección de blobs corruptos
    - serialización con pickle
    - determinismo y monotonía en la profundidad

Se puede ejecutar de dos maneras:

    pytest tests/test_isodendron.py         (si pytest está instalado)
    python tests/test_isodendron.py         (sin dependencias, sale 0 o 1)

No depende de ningún paquete de terceros: las comprobaciones usan
`assert` y el helper `raises` de este mismo módulo.
"""

from __future__ import annotations

import array
import contextlib
import os
import pickle
import random
import struct
import tempfile
import traceback
from pathlib import Path

# ----------------------------------------------------------------------
# Importación del paquete instalado
# ----------------------------------------------------------------------

try:
    from isodendron import (
        Isodendron,
        IsodendronError,
        DatasetError,
        to_arrays,
        from_columns,
        load_csv,
    )
    from isodendron import _ffi
    from isodendron._ffi import version as isodendron_version
    from isodendron._lib import lib_path
    from isodendron import __version__ as isodendron_py_version
except ImportError as exc:  # pragma: no cover - diagnóstico de entorno
    raise ImportError(
        "no se pudo importar el paquete isodendron. Instala el proyecto antes "
        "de ejecutar la suite:\n"
        "    python -m pip install .\n"
        "    # o, para desarrollo:\n"
        "    python -m pip install -e ."
    ) from exc


# ----------------------------------------------------------------------
# Constantes del formato del blob (deben coincidir con blob.h)
# ----------------------------------------------------------------------

_HEADER_FMT  = "<IIIIIIII"
_NODE_FMT    = "<IiiiiiiI"
_HEADER_SIZE = 32
_NODE_SIZE   = 32
_MAGIC       = 0x444F5349          # "HIDR" en LE


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------

@contextlib.contextmanager
def raises(exc_type):
    """
    Context manager que exige que el bloque lance `exc_type`.

    Equivalente a pytest.raises, pero sin depender de pytest, para que
    el fichero también se pueda ejecutar directamente.
    """
    try:
        yield
    except exc_type:
        return
    except BaseException as exc:  # noqa: BLE001 - queremos el mensaje
        raise AssertionError(
            f"se esperaba {exc_type.__name__} y llegó "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(
        f"se esperaba {exc_type.__name__} y no se lanzó nada"
    )


def modelo_pequeno() -> Isodendron:
    """
    Modelo mínimo y separable de forma exacta: f0 es el target y f1 es
    constante, así que un árbol de profundidad 1 lo acierta entero.
    """
    datos = [([d, 5], d) for d in (0, 1, 0, 1, 0, 1, 0, 1)]
    return Isodendron().fit(datos, max_deep=1, p_min=(1, 1))


def datos_sinteticos(n: int = 400, seed: int = 1234) -> list[tuple[list[int], int]]:
    """Dataset reproducible de 4 dígitos con target = f0*10 + f1."""
    rng = random.Random(seed)
    filas: list[tuple[list[int], int]] = []
    for _ in range(n):
        x = [rng.randrange(10) for _ in range(4)]
        filas.append((x, x[0] * 10 + x[1]))
    return filas


def modelo_sintetico(max_deep: int = 5) -> Isodendron:
    return Isodendron().fit(datos_sinteticos(), max_deep=max_deep, p_min=(1, 1))


# ======================================================================
# Entorno
# ======================================================================

def test_version_es_semver():
    v = isodendron_version()
    assert isinstance(v, str)
    assert v.strip(), "la versión no puede estar vacía"
    partes = v.split(".")
    assert len(partes) == 3, f"se esperaba major.minor.patch, llegó {v!r}"
    assert all(p.isdigit() for p in partes), f"versión no numérica: {v!r}"


def test_version_del_binding():
    assert isinstance(isodendron_py_version, str)
    assert isodendron_py_version.strip()


def test_la_shared_library_existe():
    assert lib_path.is_file(), f"no existe {lib_path}"


def test_las_firmas_ctypes_coinciden_con_la_version():
    """La versión que expone C debe ser la misma que se cargó."""
    assert isodendron_version() == _ffi.version()
    assert _ffi.lib_path == lib_path


# ======================================================================
# fit / predict
# ======================================================================

def test_fit_y_predict_exactos():
    model = modelo_pequeno()

    assert model.is_trained
    assert model.n_features == 2
    assert model.max_deep == 1
    assert model.n_nodes == 3            # raíz + dos hojas
    assert model.blob is not None

    assert model.predict([[0, 5], [1, 5], [1, 5], [0, 5]]) == [0, 1, 1, 0]

    # Acierta también las 8 filas de entrenamiento.
    x_all = [[d, 5] for d in (0, 1, 0, 1, 0, 1, 0, 1)]
    assert model.predict(x_all) == [0, 1, 0, 1, 0, 1, 0, 1]


def test_metadata_tras_fit():
    model = modelo_pequeno()
    meta = model.metadata

    assert meta["n_features"] == 2
    assert meta["max_deep"] == 1
    assert meta["p_min_num"] == 1
    assert meta["p_min_den"] == 1
    assert meta["version"] == isodendron_version()

    # metadata devuelve una copia: mutarla no afecta al modelo.
    meta["n_features"] = 999
    assert model.metadata["n_features"] == 2
    assert model.n_features == 2


def test_fit_devuelve_self_para_encadenar():
    datos = [([d, 5], d) for d in (0, 1, 0, 1)]
    assert Isodendron().fit(datos, max_deep=1, p_min=(1, 1)).predict([[0, 5]]) == [0]


def test_predict_one_coincide_con_predict():
    model = modelo_sintetico(max_deep=5)
    muestras = [x for x, _ in datos_sinteticos(n=40)]

    lote = model.predict(muestras)
    uno_a_uno = [model.predict_one(s) for s in muestras]
    assert lote == uno_a_uno


def test_rutas_c_y_python_coinciden():
    """
    Por encima de Isodendron._PYTHON_PREDICT_THRESHOLD predict() delega en C;
    por debajo deserializa el árbol y recorre en Python. Ambas rutas
    tienen que dar exactamente el mismo resultado.
    """
    model = modelo_sintetico(max_deep=6)
    umbral = Isodendron._PYTHON_PREDICT_THRESHOLD

    muestras = [x for x, _ in datos_sinteticos(n=umbral + 10)]
    assert len(muestras) > umbral, "el lote debe superar el umbral"

    por_c = model.predict(muestras)              # ruta C
    por_python = [model.predict_one(s) for s in muestras]
    assert por_c == por_python


def test_predict_acepta_una_sola_muestra_plana():
    model = modelo_pequeno()
    assert model.predict([0, 5]) == [0]
    assert model.predict([1, 5]) == [1]


def test_predict_lote_vacio():
    assert modelo_pequeno().predict([]) == []


def test_predict_full_coincide_con_predict():
    """predict_full usa siempre la ruta C; predict, la Python por debajo
    del umbral. Tienen que coincidir en la predicción."""
    model = modelo_sintetico(max_deep=5)
    muestras = [x for x, _ in datos_sinteticos(n=20)]

    completo = model.predict_full(muestras)
    assert len(completo) == len(muestras)
    assert [t[0] for t in completo] == model.predict(muestras)


def test_predict_full_lote_vacio():
    assert modelo_pequeno().predict_full([]) == []


def test_predict_full_hojas_puras():
    model = modelo_pequeno()
    for moda, p, n_mode, n_hoja in model.predict_full([[0, 5], [1, 5]]):
        assert n_mode == n_hoja
        assert p == 1.0
        assert moda in (0, 1)


def test_predict_full_hoja_impura():
    """Con max_deep=1 la hoja != tiene 48 filas, 16 de cada clase."""
    datos = [([i % 4, i], i % 4) for i in range(64)]
    model = Isodendron().fit(datos, max_deep=1, p_min=(1, 1))

    res = model.predict_full([[1, 5], [0, 5]])

    # f0 != 0 -> hoja derecha: empate a 16 entre 1, 2 y 3 -> intermedio
    # menor = 2.
    moda, p, n_mode, n_hoja = res[0]
    assert moda == 2
    assert n_mode == 16
    assert n_hoja == 48
    assert abs(p - 16.0 / 48.0) < 1e-12

    # f0 == 0 -> hoja izquierda pura.
    moda, p, n_mode, n_hoja = res[1]
    assert moda == 0
    assert n_mode == 16
    assert n_hoja == 16
    assert p == 1.0


def test_predict_full_sin_entrenar_falla():
    with raises(RuntimeError):
        Isodendron().predict_full([[0, 5]])
    with raises(ValueError):
        modelo_pequeno().predict_full([[0, 5, 1]])


def test_predict_sin_entrenar_falla():
    with raises(RuntimeError):
        Isodendron().predict([[0, 5]])
    with raises(RuntimeError):
        Isodendron().predict_one([0, 5])
    with raises(RuntimeError):
        Isodendron().n_nodes


def test_predict_numero_de_features_incorrecto():
    model = modelo_pequeno()
    with raises(ValueError):
        model.predict([[0, 5, 1]])
    with raises(ValueError):
        model.predict([[0, 5], [0]])
    with raises(ValueError):
        model.predict_one([0])


def test_target_constante():
    """Con target constante todo el árbol debe predecir ese valor."""
    datos = [([i % 5], 7) for i in range(100)]
    model = Isodendron().fit(datos, max_deep=4, p_min=(1, 1))
    preds = model.predict([x for x, _ in datos])
    assert set(preds) == {7}


def test_acepta_array_array_como_features():
    datos = [(array.array("i", [d, 5]), d) for d in (0, 1, 0, 1)]
    model = Isodendron().fit(datos, max_deep=1, p_min=(1, 1))
    assert model.predict([[0, 5], [1, 5]]) == [0, 1]


def test_repr():
    assert "no entrenado" in repr(Isodendron())
    r = repr(modelo_pequeno())
    assert "n_features=2" in r
    assert "max_deep=1" in r


# ======================================================================
# Parámetros de fit
# ======================================================================

def test_fit_parametros_invalidos():
    datos = [([0, 5], 0), ([1, 5], 1)]

    with raises(ValueError):
        Isodendron().fit(datos, max_deep=0)
    with raises(ValueError):
        Isodendron().fit(datos, max_deep=-1)
    with raises(ValueError):
        Isodendron().fit(datos, p_min=(1,))
    with raises(ValueError):
        Isodendron().fit(datos, p_min=(1, 0))
    with raises(ValueError):
        Isodendron().fit(datos, p_min=(-1, 10))
    with raises(ValueError):
        Isodendron().fit(datos, p_min="1/100")
    with raises(ValueError):
        Isodendron().fit(datos, p_min=(0, 1))
    with raises(ValueError):
        Isodendron().fit(datos, p_min=(0, 100))
    with raises(ValueError):
        Isodendron().fit(datos, hilos=0)
    with raises(ValueError):
        Isodendron().fit(datos, hilos=-1)
    with raises(ValueError):
        Isodendron().fit(datos, feature_names=["solo_uno"])
    with raises(ValueError):
        Isodendron().fit(datos, feature_names=[0, 1])


def test_fit_feature_names():
    model = Isodendron().fit(
        [([d, 5], d) for d in (0, 1, 0, 1)],
        max_deep=2,
        p_min=(1, 1),
        feature_names=["color", "tamano"],
    )
    assert model.feature_names == ["color", "tamano"]

    hojas = model.dump()
    assert hojas
    letras = [c["feature"] for h in hojas for c in h["conditions"]]
    assert letras, "el árbol debería tener al menos una condición"
    assert set(letras) <= {"color", "tamano"}


def test_sin_feature_names_usa_f0():
    hojas = modelo_sintetico(max_deep=3).dump()
    nombres = {c["feature"] for h in hojas for c in h["conditions"]}
    assert nombres
    assert all(n.startswith("f") for n in nombres), nombres


# ======================================================================
# dump
# ======================================================================

def test_dump_particiona_las_filas():
    datos = datos_sinteticos(n=400)
    model = Isodendron().fit(datos, max_deep=5, p_min=(1, 1))
    hojas = model.dump()

    assert hojas, "debe haber al menos una hoja"

    # Las hojas particionan exactamente el conjunto de entrenamiento.
    assert sum(h["N"] for h in hojas) == len(datos)

    for h in hojas:
        assert h["N"] >= 1
        assert h["K"] >= 1
        assert 1 <= h["n_mode"] <= h["N"]
        assert 0.0 <= h["P"] <= 1.0
        for c in h["conditions"]:
            assert c["op"] in ("==", "!=")
            assert isinstance(c["value"], int)


def test_dump_es_determinista():
    datos = datos_sinteticos(n=200)
    a = Isodendron().fit(datos, max_deep=4, p_min=(1, 1)).dump()
    b = Isodendron().fit(datos, max_deep=4, p_min=(1, 1)).dump()
    assert a == b


# ======================================================================
# Formato del blob
# ======================================================================

def test_formato_del_blob():
    model = modelo_pequeno()
    blob = model.blob
    assert isinstance(blob, bytes)

    magic, ver, n_feat, n_nodes, max_deep, p_num, p_den, resv = struct.unpack_from(
        _HEADER_FMT, blob, 0
    )

    assert magic == _MAGIC
    assert ver == 3
    assert n_feat == 2
    assert n_nodes == model.n_nodes == 3
    assert max_deep == 1
    assert p_num == 1 and p_den == 1
    assert resv == 16   # payload: 2 hojas puras x 1 entrada x 8 B
    assert len(blob) == _HEADER_SIZE + n_nodes * _NODE_SIZE + 16

    # La raíz es interna; las dos hojas llevan moda 0 y 1.
    is_leaf_raiz = struct.unpack_from(_NODE_FMT, blob, _HEADER_SIZE)[0]
    assert is_leaf_raiz == 0

    moda_izq = struct.unpack_from(_NODE_FMT, blob, _HEADER_SIZE + 32)[5]
    moda_der = struct.unpack_from(_NODE_FMT, blob, _HEADER_SIZE + 64)[5]
    assert (moda_izq, moda_der) == (0, 1)


def test_binding_predict_leaf():
    from isodendron import _ffi

    datos = [([i % 4, i], i % 4) for i in range(64)]
    X = [v for muestra, _t in datos for v in muestra]
    y = [t for _m, t in datos]

    blob = _ffi.train_mt_cap(X, y, 64, 2, 1, 1, 1, 10)
    pred, hojas = _ffi.predict_leaf(blob, X, 64, 2)

    assert len(pred) == 64 and len(hojas) == 64
    assert set(hojas) <= {1, 2}          # hojas del arbol de 3 nodos
    assert list(pred) == list(_ffi.predict(blob, X, 64, 2))


def test_binding_train_mt_cap_y_partial_fit():
    from isodendron import _ffi

    datos = [([i % 4, i], i % 4) for i in range(64)]
    X = [v for muestra, _t in datos for v in muestra]
    y = [t for _m, t in datos]

    b10 = _ffi.train_mt_cap(X, y, 64, 2, 1, 1, 1, 10)
    b0 = _ffi.train_mt_cap(X, y, 64, 2, 1, 1, 1, 0)

    assert struct.unpack_from("<I", b0, 28)[0] == 0     # sin payload
    assert len(b0) < len(b10)

    # Recalibrar con los mismos datos reproduce el blob.
    b2 = _ffi.partial_fit(b10, X, y, 64, 2, 10)
    assert b2 != b10          # partial_fit suma, no recalcula


def test_blob_corrupto_falla():
    blob = modelo_pequeno().blob

    # Demasiado corto para contener la cabecera.
    with raises(IsodendronError):
        Isodendron(blob=b"\x00" * 8)

    # Magic inválido.
    with raises(IsodendronError):
        Isodendron(blob=b"\x00" * _HEADER_SIZE)

    # n_features de la metadata incoherente con la del blob.
    with raises(IsodendronError):
        Isodendron(blob=blob, metadata={"n_features": 99})

    # Truncado: se detecta al deserializar el árbol para predecir.
    truncado = Isodendron(blob=blob[:-1])
    with raises(IsodendronError):
        truncado.predict([[0, 5]])

    # La ruta C (lote grande) también lo rechaza.
    with raises(IsodendronError):
        _ffi.predict(blob[:-1], [0, 5] * 64, 64, 2)


def test_isodendron_acepta_blob_y_metadata():
    original = modelo_pequeno()
    copia = Isodendron(blob=original.blob, metadata=original.metadata)
    assert copia.predict([[0, 5], [1, 5]]) == [0, 1]
    assert copia.blob == original.blob


# ======================================================================
# Serialización
# ======================================================================

def test_pickle_roundtrip():
    original = modelo_sintetico(max_deep=5)
    muestras = [x for x, _ in datos_sinteticos(n=50)]

    copia = pickle.loads(pickle.dumps(original))

    assert copia.blob == original.blob
    assert copia.predict(muestras) == original.predict(muestras)
    assert copia.predict_one(muestras[0]) == original.predict_one(muestras[0])


def test_pickle_sin_entrenar_falla():
    with raises(pickle.PicklingError):
        pickle.dumps(Isodendron())


# ======================================================================
# Determinismo y profundidad
# ======================================================================

def test_determinismo():
    datos = datos_sinteticos(n=300)
    a = Isodendron().fit(datos, max_deep=5, p_min=(1, 1))
    b = Isodendron().fit(datos, max_deep=5, p_min=(1, 1))
    assert a.blob == b.blob


def test_profundidad_mayor_no_da_menos_nodos():
    datos = datos_sinteticos(n=400)
    nodos = [
        Isodendron().fit(datos, max_deep=d, p_min=(1, 1)).n_nodes
        for d in (1, 2, 3, 5, 7)
    ]
    assert nodos == sorted(nodos), f"n_nodes no es monótono: {nodos}"
    assert nodos[0] >= 1


def test_max_deep_queda_en_la_metadata():
    for d in (1, 2, 5, 7):
        model = Isodendron().fit(datos_sinteticos(n=100), max_deep=d, p_min=(1, 1))
        assert model.max_deep == d


# ======================================================================
# Conversión de datasets
# ======================================================================

def test_to_arrays_forma():
    X, y, n, f = to_arrays([([1, 2, 3], 0), ([4, 5, 6], 1)])

    assert n == 2
    assert f == 3
    assert isinstance(X, array.array) and X.typecode == "i"
    assert isinstance(y, array.array) and y.typecode == "i"
    assert list(X) == [1, 2, 3, 4, 5, 6]       # row-major
    assert list(y) == [0, 1]


def test_to_arrays_acepta_generador():
    X, y, n, f = to_arrays((([i, i + 1], i) for i in range(3)))
    assert (n, f) == (3, 2)
    assert list(X) == [0, 1, 1, 2, 2, 3]
    assert list(y) == [0, 1, 2]


def test_to_arrays_errores():
    with raises(DatasetError):
        to_arrays([])

    with raises(DatasetError):                  # features de distinto tamaño
        to_arrays([([1, 2], 0), ([1], 1)])

    with raises(DatasetError):                  # features no es secuencia
        to_arrays([("ab", 0)])

    with raises(DatasetError):                  # muestra que no es par
        to_arrays([([1, 2], 0), ([1, 2, 3], 0, 0)])

    with raises(DatasetError):                  # target no entero
        to_arrays([([1, 2], "x")])

    with raises(DatasetError):                  # bool no cuenta como int
        to_arrays([([1, True], 0)])

    with raises(DatasetError):
        to_arrays([([1, 2], True)])

    with raises(DatasetError):                  # cero features
        to_arrays([([], 0)])


def test_to_arrays_con_una_sola_feature_es_valido():
    """Una feature es el mínimo válido; cero features se rechaza."""
    X, y, n, f = to_arrays([([7], 3)])
    assert (n, f) == (1, 1)
    assert list(X) == [7]
    assert list(y) == [3]


def test_to_arrays_overflow_de_int32():
    with raises(OverflowError):
        to_arrays([([2**31, 0], 0)])
    with raises(OverflowError):
        to_arrays([([0, 0], 2**31)])
    with raises(OverflowError):
        to_arrays([([-(2**31) - 1, 0], 0)])
    with raises(OverflowError):
        to_arrays([([0, 0], -(2**31) - 1)])

    # Los extremos exactos sí caben.
    X, y, n, f = to_arrays([([-(2**31), 2**31 - 1], -(2**31))])
    assert list(X) == [-(2**31), 2**31 - 1]
    assert list(y) == [-(2**31)]
    assert n == 1 and f == 2


def test_from_columns():
    X, y, n, f = from_columns([[1, 2], [3, 4]], [0, 1])

    assert (n, f) == (2, 2)
    assert list(X) == [1, 2, 3, 4]
    assert list(y) == [0, 1]

    # Y coincide con la vía de lista de tuplas.
    X2, y2, n2, f2 = to_arrays([([1, 2], 0), ([3, 4], 1)])
    assert (list(X), list(y)) == (list(X2), list(y2))
    assert (n, f) == (n2, f2)


def test_from_columns_longitudes_distintas():
    with raises(DatasetError):
        from_columns([[1, 2]], [0, 1])


# ======================================================================
# load_csv
# ======================================================================

def test_load_csv_con_cabecera():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "datos.csv"
        ruta.write_text("a,b,t\n1,2,3\n4,5,6\n", encoding="utf-8")

        X, y, n, f = load_csv(ruta, target_col="t")
        assert (n, f) == (2, 2)
        assert list(X) == [1, 2, 4, 5]
        assert list(y) == [3, 6]

        # El orden de las features es el que pide feature_cols.
        X2, y2, n2, f2 = load_csv(ruta, target_col="t", feature_cols=["b", "a"])
        assert f2 == 2
        assert list(X2) == [2, 1, 5, 4]

        # Subconjunto de columnas.
        X3, _, _, f3 = load_csv(ruta, target_col="t", feature_cols=["b"])
        assert f3 == 1
        assert list(X3) == [2, 5]

        # Por índice en lugar de por nombre.
        X4, y4, _, _ = load_csv(ruta, target_col=2, feature_cols=[0])
        assert list(X4) == [1, 4]
        assert list(y4) == [3, 6]


def test_load_csv_sin_cabecera():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "datos.csv"
        ruta.write_text("1,2,3\n4,5,6\n", encoding="utf-8")

        X, y, n, f = load_csv(ruta, target_col=2, has_header=False)
        assert (n, f) == (2, 2)
        assert list(X) == [1, 2, 4, 5]
        assert list(y) == [3, 6]


def test_load_csv_ignora_lineas_vacias():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "datos.csv"
        ruta.write_text("a,b\n1,2\n\n3,4\n", encoding="utf-8")

        X, y, n, f = load_csv(ruta, target_col="b", has_header=True)
        assert (n, f) == (2, 1)
        assert list(X) == [1, 3]


def test_load_csv_errores():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        with raises(FileNotFoundError):
            load_csv(tmp / "no_existe.csv", target_col=0)

        malo = tmp / "malo.csv"
        malo.write_text("a,b\n1,x\n", encoding="utf-8")

        with raises(DatasetError):              # celda no entera
            load_csv(malo, target_col="a", feature_cols=["b"])

        with raises(DatasetError):              # columna inexistente
            load_csv(malo, target_col="zzz")

        with raises(DatasetError):              # índice fuera de rango
            load_csv(malo, target_col="a", feature_cols=[7])

        vacio = tmp / "vacio.csv"
        vacio.write_text("", encoding="utf-8")
        with raises(DatasetError):
            load_csv(vacio, target_col=0)

        solo_cabecera = tmp / "cabecera.csv"
        solo_cabecera.write_text("a,b\n", encoding="utf-8")
        with raises(DatasetError):
            load_csv(solo_cabecera, target_col="a")

        corta = tmp / "corta.csv"
        corta.write_text("a,b\n1\n", encoding="utf-8")
        with raises(DatasetError):              # faltan columnas
            load_csv(corta, target_col="b")


def test_load_csv_alimenta_el_modelo():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "datos.csv"
        filas = ["f0,f1,t"]
        for d in (0, 1, 0, 1, 0, 1, 0, 1):
            filas.append(f"{d},5,{d}")
        ruta.write_text("\n".join(filas) + "\n", encoding="utf-8")

        X, y, n, f = load_csv(ruta, target_col="t", feature_cols=["f0", "f1"])
        datos = [(list(X[i * f:(i + 1) * f]), y[i]) for i in range(n)]

        model = Isodendron().fit(datos, max_deep=1, p_min=(1, 1))
        assert model.predict([[0, 5], [1, 5]]) == [0, 1]


# ======================================================================
# Runner sin dependencias
# ======================================================================

# ======================================================================
# Guardado y lectura del modelo (.isod)
# ======================================================================

def _datos_para_persistencia():
    return [([i % 4, (i * 7) % 5, i // 4], i % 4) for i in range(128)]


def test_guardado_y_lectura_roundtrip():
    import os
    import tempfile

    datos = _datos_para_persistencia()
    muestras = [x for x, _t in datos]
    original = Isodendron().fit(datos, max_deep=4, p_min=(1, 1),
                                feature_names=["a", "b", "c"])

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "modelo.isod")
        escrito = original.save(ruta)
        assert str(escrito) == ruta
        assert os.path.isfile(ruta)
        assert os.path.getsize(ruta) > len(original.blob)

        cargado = Isodendron.load(ruta)

    assert cargado.blob == original.blob
    assert cargado.dump() == original.dump()
    assert cargado.predict(muestras) == original.predict(muestras)
    assert cargado.predict_full(muestras) == original.predict_full(muestras)
    assert cargado.n_features == original.n_features
    assert cargado.max_deep == original.max_deep
    assert cargado.feature_names == original.feature_names


def test_guardado_no_sobrescribe_sin_permiso():
    import os
    import tempfile

    datos = _datos_para_persistencia()
    modelo = Isodendron().fit(datos, max_deep=4, p_min=(1, 1))

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "modelo.isod")
        modelo.save(ruta)

        with raises(FileExistsError):
            modelo.save(ruta)

        # Con overwrite=True si reemplaza, y el fichero sigue siendo valido.
        modelo.save(ruta, overwrite=True)
        assert Isodendron.load(ruta).blob == modelo.blob


def test_guardado_sin_entrenar_falla():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as carpeta:
        with raises(RuntimeError):
            Isodendron().save(os.path.join(carpeta, "modelo.isod"))


def test_lectura_fichero_inexistente():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as carpeta:
        with raises(FileNotFoundError):
            Isodendron.load(os.path.join(carpeta, "no_existe.isod"))


def test_lectura_rechaza_ficheros_invalidos():
    import os
    import tempfile

    modelo = Isodendron().fit(_datos_para_persistencia(), max_deep=4,
                              p_min=(1, 1))

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "modelo.isod")
        modelo.save(ruta)
        with open(ruta, "rb") as fh:
            bueno = fh.read()

        casos = (
            b"",                                     # vacio
            bueno[:8],                               # cabecera a medias
            bueno[:-16],                             # blob truncado
            b"OTROCOSA" + bueno[8:],                 # magic ajeno
            bueno[:8] + (999).to_bytes(4, "little") + bueno[12:],  # version
        )

        for contenido in casos:
            with open(ruta, "wb") as fh:
                fh.write(contenido)
            with raises(ValueError):
                Isodendron.load(ruta)


def test_max_targets_se_recuerda():
    import os
    import tempfile

    datos = _datos_para_persistencia()
    modelo = Isodendron().fit(datos, max_deep=4, p_min=(1, 1),
                              max_targets=5)

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "modelo.isod")
        modelo.save(ruta)
        cargado = Isodendron.load(ruta)

    assert cargado.max_targets == 5

    # El partial_fit hereda el tope: sin argumento y con 5 dan lo mismo.
    assert (cargado.partial_fit(datos).blob
            == cargado.partial_fit(datos, max_targets=5).blob)


def test_feature_names_none_tras_lectura():
    import os
    import tempfile

    datos = _datos_para_persistencia()
    modelo = Isodendron().fit(datos, max_deep=4, p_min=(1, 1))
    assert modelo.feature_names is None

    with tempfile.TemporaryDirectory() as carpeta:
        ruta = os.path.join(carpeta, "modelo.isod")
        modelo.save(ruta)
        cargado = Isodendron.load(ruta)

    assert cargado.feature_names is None
    assert cargado.dump() == modelo.dump()


def _main() -> int:
    pruebas = [
        (nombre, fn)
        for nombre, fn in sorted(globals().items())
        if nombre.startswith("test_") and callable(fn)
    ]

    print(f"isodendron-py {isodendron_py_version} (C {isodendron_version()}) — "
          f"{len(pruebas)} casos\n")
    print(f"shared library: {lib_path}\n")

    fallos = 0
    for nombre, fn in pruebas:
        try:
            fn()
        except BaseException:  # noqa: BLE001 - queremos el traceback
            fallos += 1
            print(f"  FALLA {nombre}")
            print("".join("      " + l for l in traceback.format_exc().splitlines(True)))
        else:
            print(f"  ok    {nombre}")

    print(f"\n{len(pruebas)} casos, {fallos} fallos")
    if fallos:
        print("RESULTADO: FALLOS")
        return 1
    print("RESULTADO: TODO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())




