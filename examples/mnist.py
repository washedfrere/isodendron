#!/usr/bin/env python3
"""
mnist.py — Prueba de exactitud y de estrés de ISODENDRON sobre MNIST.

Qué hace
--------
Carga MNIST en CSV (una fila por imagen: primera columna el dígito, las
784 siguientes los píxeles), binariza cada píxel a 0/1 con el umbral
indicado, entrena un árbol ISODENDRON por cada profundidad pedida y mide la
exactitud sobre un conjunto de test independiente.

Además de medir la exactitud, el script sirve como **prueba de estrés**
del clasificador: 784 features, decenas de miles de filas, árboles de
hasta ~120.000 nodos. Reporta tiempos, tamaño del árbol y del blob,
memoria pico del proceso, y comprueba que `isodendron_train` aguanta sin
devolver ISODENDRON_ERR_MEM ni corromper el blob.

Controles incluidos
-------------------
Sin ellos, un número de exactitud no significa nada:

  1. Línea base trivial: predecir siempre el dígito mayoritario del train.
  2. Control de etiquetas barajadas: misma X con las etiquetas permutadas.
     Si rinde igual o mejor que con las reales, queda demostrado que el
     árbol no está usando el target. ISODENDRON elige los splits solo por
     balance de tamaños (min(n_igual, n_distinto)), así que es lo que
     cabe esperar.
  3. Naive Bayes Bernoulli por píxel: un clasificador trivial pero que SÍ
     mira la etiqueta. Si acierta mucho y ISODENDRON poco, el problema no son
     los datos ni la binarización, sino el criterio de split.
  4. Entrenamientos concurrentes en hilos: `ctypes.CDLL` libera el GIL,
     así que varios `isodendron_train` simultáneos son paralelismo real. Se
     verifica que los blobs resultantes son idénticos al secuencial, que
     es lo que exige el contrato de thread-safety de isodendron.h.

Formato de los CSV
------------------
Cabecera con 785 nombres (`label,1x1,1x2,...,28x28`), terminador CRLF,
píxeles en 0..255. El script valida todo eso y falla con un mensaje claro
si no cuadra.

Uso
---
    python mnist.py
    python mnist.py --n-train 5000 --n-test 1000 --depths 4 8
    python mnist.py --downsample 2 --sin-barajado
    python mnist.py --train-mnist otra/ruta.csv --hilos 8

Solo usa la biblioteca estándar y el paquete isodendron.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import math
import os
import random
import struct
import sys
import threading
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Localización del proyecto
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

#: Píxeles de una imagen MNIST.
LADO = 28
PIXELES = LADO * LADO

#: Layout del blob, espejo de blob.h.
_HEADER_SIZE = 32
_NODE_SIZE = 32
_NODE_FMT = "<IiiiiiiI"

_N_CLASES = 10


def _mnist_default(filename: str) -> Path | None:
    """Ruta por defecto si el usuario define MNIST_DIR; si no, hay que pasar --train/--test."""
    base = os.environ.get("MNIST_DIR")
    return Path(base) / filename if base else None


try:
    from isodendron import Isodendron, IsodendronError
    from isodendron import __version__ as ISODENDRON_PY_VERSION
    from isodendron import _ffi
    from isodendron._ffi import version as isodendron_version
    from isodendron._lib import lib_path
except ImportError as exc:  # pragma: no cover - diagnóstico de entorno
    print("ERROR: no se pudo importar el paquete isodendron.", file=sys.stderr)
    print("  instala el proyecto con `pip install .` o `pip install -e .`",
          file=sys.stderr)
    print(f"  causa: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)


# ----------------------------------------------------------------------
# Memoria
# ----------------------------------------------------------------------

class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memoria_pico_mb() -> float | None:
    """Memoria pico del proceso en MB, o None si no se puede leer."""
    if sys.platform.startswith("win"):
        try:
            pmc = _ProcessMemoryCounters()
            pmc.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            for dll in ("psapi", "kernel32"):
                try:
                    fn = getattr(ctypes.windll, dll).GetProcessMemoryInfo
                    if fn(handle, ctypes.byref(pmc), pmc.cb):
                        return pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
                except Exception:
                    continue
        except Exception:
            pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux lo da en KB; macOS en bytes.
        return rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0 * 1024.0)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Carga de los CSV
# ----------------------------------------------------------------------

def _leer_cabecera_y_contar(ruta: Path) -> tuple[list[bytes], int]:
    with ruta.open("rb") as fh:
        cabecera = fh.readline()
        if not cabecera:
            raise SystemExit(f"ERROR: {ruta} está vacío")
        n = sum(1 for _ in fh)
    return cabecera.rstrip(b"\r\n").split(b","), n


def _reducir_fila(fila: bytes, factor: int) -> bytes:
    """
    Max-pooling (OR lógico, porque los valores ya son 0/1) de bloques
    factor x factor. Reduce 28x28 a (28/factor)^2 features.
    """
    if factor == 1:
        return fila
    lado = LADO // factor
    salida = bytearray(lado * lado)
    k = 0
    for i in range(lado):
        base = i * factor * LADO
        for j in range(lado):
            b = base + j * factor
            v = 0
            for di in range(factor):
                off = b + di * LADO
                for dj in range(factor):
                    if fila[off + dj]:
                        v = 1
                        break
                if v:
                    break
            salida[k] = v
            k += 1
    return bytes(salida)


def cargar_csv(
    ruta: Path,
    umbral: int,
    factor: int,
    n_max: int,
    seed: int,
    con_clases: bool,
    nombre: str,
) -> tuple[array.array, array.array, list[bytearray], int, int]:
    """
    Carga un CSV de MNIST.

    Devuelve (X, y, por_clase, n_features, n_filas):
      X          array('i') plano row-major con los píxeles binarizados
      y          array('i') con el dígito de cada fila
      por_clase  lista de 10 bytearray, cada uno con los píxeles de las
                 filas de esa clase concatenados (vacía si con_clases=False)
      n_features píxeles por fila tras el downsampling
    """
    if not ruta.is_file():
        raise SystemExit(f"ERROR: no existe {ruta}")

    t0 = time.perf_counter()
    columnas, total = _leer_cabecera_y_contar(ruta)

    if len(columnas) != PIXELES + 1:
        raise SystemExit(
            f"ERROR: {ruta} tiene {len(columnas)} columnas; se esperaban "
            f"{PIXELES + 1} (target + {PIXELES} píxeles)"
        )
    if columnas[0].strip().lower() not in (b"label", b"target", b"digito", b"class"):
        print(f"  aviso: la primera columna se llama {columnas[0]!r}; "
              f"se tratará como el target")

    f = PIXELES // (factor * factor)

    # Selección de filas (submuestreo reproducible).
    if n_max and n_max < total:
        seleccion: list[int] | None = sorted(
            random.Random(seed).sample(range(total), n_max)
        )
    else:
        seleccion = None

    X = array.array("i")
    y = array.array("i")
    por_clase = [bytearray() for _ in range(_N_CLASES)] if con_clases else []
    buffer = bytearray(4 * f)

    leidas = 0
    apuntador = 0
    with ruta.open("rb") as fh:
        fh.readline()  # cabecera
        for idx, linea in enumerate(fh):
            if seleccion is not None:
                if apuntador >= len(seleccion):
                    break
                if seleccion[apuntador] != idx:
                    continue
                apuntador += 1

            partes = linea.rstrip(b"\r\n").split(b",")
            if len(partes) != PIXELES + 1:
                raise SystemExit(
                    f"ERROR: {ruta}, fila {idx + 2}: {len(partes)} campos; "
                    f"se esperaban {PIXELES + 1}"
                )

            clase = int(partes[0])
            if not 0 <= clase < _N_CLASES:
                raise SystemExit(
                    f"ERROR: {ruta}, fila {idx + 2}: target {clase} fuera de 0..9"
                )
            y.append(clase)

            fila = bytes(1 if int(v) > umbral else 0 for v in partes[1:])
            if factor > 1:
                fila = _reducir_fila(fila, factor)
            if con_clases:
                por_clase[clase].extend(fila)

            # Empaquetado a int32 little-endian reutilizando el buffer:
            # los bytes 1..3 se quedan a 0 de una iteración a otra.
            buffer[0::4] = fila
            X.frombytes(buffer)

            leidas += 1
            if leidas % 20000 == 0:
                print(f"    {nombre}: {leidas} filas... "
                      f"({time.perf_counter() - t0:.0f} s)")

    if leidas == 0:
        raise SystemExit(f"ERROR: no se cargó ninguna fila de {ruta}")

    print(f"    {nombre}: {leidas} filas x {f} features "
          f"({time.perf_counter() - t0:.1f} s, {X.buffer_info()[1] * X.itemsize / 1048576:.0f} MB)")
    return X, y, por_clase, f, leidas


def dibujar(fila: bytes, factor: int) -> list[str]:
    """Dibuja una fila binarizada como arte ASCII."""
    lado = LADO // factor
    lineas = []
    for i in range(lado):
        lineas.append("".join(
            "#" if fila[i * lado + j] else "."
            for j in range(lado)
        ))
    return lineas


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------

def matriz_confusion(pred: list[int], real: list[int]) -> list[list[int]]:
    m = [[0] * _N_CLASES for _ in range(_N_CLASES)]
    for p, r in zip(pred, real):
        if 0 <= r < _N_CLASES and 0 <= p < _N_CLASES:
            m[r][p] += 1
    return m


def exactitud(pred: list[int], real: list[int]) -> float:
    if not real:
        return 0.0
    return sum(1 for p, r in zip(pred, real) if p == r) / len(real)


def imprimir_matriz(m: list[list[int]]) -> None:
    print("      pred:  " + "".join(f"{c:>7}" for c in range(_N_CLASES)))
    print("      " + "-" * (12 + 7 * _N_CLASES))
    for r in range(_N_CLASES):
        total = sum(m[r])
        celdas = "".join(
            f"{100.0 * m[r][c] / total:>6.1f}%" if total else f"{'-':>7}"
            for c in range(_N_CLASES)
        )
        print(f"      real {r}: {celdas}   (n={total})")


def estadisticas_arbol(blob: bytes) -> dict:
    """
    Recorre el blob en preorden contando nodos, hojas, profundidad
    alcanzada y N medio por hoja. No usa Isodendron.dump(), que materializa el
    camino de cada hoja y con ~100.000 nodos sería inasumible.
    """
    n_nodes = struct.unpack_from("<I", blob, 12)[0]
    hojas = 0
    suma_n = 0
    prof_max = 0
    pila = [(0, 0)]
    unpack = struct.Struct(_NODE_FMT).unpack_from
    while pila:
        idx, prof = pila.pop()
        is_leaf, _f, _v, n, _k, _med, _nmed, left = unpack(blob, _HEADER_SIZE + _NODE_SIZE * idx)
        if is_leaf:
            hojas += 1
            suma_n += n
            if prof > prof_max:
                prof_max = prof
        else:
            if prof + 1 > prof_max:
                prof_max = prof + 1
            pila.append((idx + 1, prof + 1))
            pila.append((idx + 1 + left, prof + 1))
    return {
        "n_nodes": n_nodes,
        "n_leaves": hojas,
        "profundidad": prof_max,
        "n_medio_hoja": (suma_n / hojas) if hojas else 0.0,
    }


# ----------------------------------------------------------------------
# Naive Bayes Bernoulli por píxel
# ----------------------------------------------------------------------

def naive_bayes(
    por_clase: list[bytearray],
    X_test: array.array,
    y_test: array.array,
    f: int,
    n_test: int,
) -> tuple[float, list[int], float]:
    """
    Clasificador trivial que SÍ usa la etiqueta: modelo Bernoulli
    independiente por píxel, con suavizado de Laplace.

        log P(x|c) = log P(c) + sum_p [ x_p*log p_pc + (1-x_p)*log(1-p_pc) ]

    Devuelve (exactitud, predicciones, segundos).
    """
    t0 = time.perf_counter()

    log_p = [[0.0] * f for _ in range(_N_CLASES)]   # log P(pixel=1 | clase)
    log_q = [[0.0] * f for _ in range(_N_CLASES)]   # log P(pixel=0 | clase)
    base = [0.0] * _N_CLASES                        # log P(c) + sum_p log q

    n_total = sum(len(buf) // f for buf in por_clase)
    for c, buf in enumerate(por_clase):
        nc = len(buf) // f
        if nc == 0:
            base[c] = float("-inf")
            continue
        suma_q = 0.0
        for p in range(f):
            unos = sum(buf[p::f])              # slice + sum en C
            pc = (unos + 1.0) / (nc + 2.0)     # Laplace
            lp = math.log(pc)
            lq = math.log(1.0 - pc)
            log_p[c][p] = lp
            log_q[c][p] = lq
            suma_q += lq
        base[c] = math.log(nc / n_total) + suma_q
        # Guardamos el ajuste (log p - log q) para no recalcularlo por fila.
        for p in range(f):
            log_p[c][p] = log_p[c][p] - log_q[c][p]

    pred: list[int] = []
    for i in range(n_test):
        fila = X_test[i * f:(i + 1) * f]
        encendidos = [p for p, v in enumerate(fila) if v]
        mejor_c = 0
        mejor_s = float("-inf")
        for c in range(_N_CLASES):
            if base[c] == float("-inf"):
                continue
            s = base[c]
            ajuste = log_p[c]
            for p in encendidos:
                s += ajuste[p]
            if s > mejor_s:
                mejor_s = s
                mejor_c = c
        pred.append(mejor_c)

    seg = time.perf_counter() - t0
    return exactitud(pred, list(y_test)), pred, seg


# ----------------------------------------------------------------------
# Prueba de concurrencia
# ----------------------------------------------------------------------

def prueba_concurrencia(
    X: array.array,
    y: array.array,
    n: int,
    f: int,
    hilos: int,
    max_deep: int,
    p_min: tuple[int, int],
) -> dict:
    """
    Entrena el mismo árbol `hilos` veces en paralelo y comprueba que
    todos los blobs son idénticos al secuencial de referencia.
    """
    print(f"  datos: {n} filas x {f} features, max_deep={max_deep}, "
          f"P_MIN={_fmt_p_min(p_min[0], p_min[1])}, {hilos} hilos")

    t0 = time.perf_counter()
    referencia = _ffi.train(X, y, n, f, max_deep, p_min[0], p_min[1])
    t_secuencial = time.perf_counter() - t0

    resultados: list[bytes | None] = [None] * hilos
    errores: list[BaseException | None] = [None] * hilos

    def trabajo(i: int) -> None:
        try:
            resultados[i] = _ffi.train(X, y, n, f, max_deep, p_min[0], p_min[1])
        except BaseException as exc:  # noqa: BLE001 - se reporta tal cual
            errores[i] = exc

    hilos_obj = [threading.Thread(target=trabajo, args=(i,)) for i in range(hilos)]
    t0 = time.perf_counter()
    for h in hilos_obj:
        h.start()
    for h in hilos_obj:
        h.join()
    t_paralelo = time.perf_counter() - t0

    fallos = [e for e in errores if e is not None]
    distintos = [
        i for i, b in enumerate(resultados)
        if b is None or b != referencia
    ]

    return {
        "hilos": hilos,
        "t_secuencial": t_secuencial,
        "t_paralelo": t_paralelo,
        "speedup": (t_secuencial * hilos / t_paralelo) if t_paralelo else 0.0,
        "errores": fallos,
        "distintos": distintos,
        "blob_bytes": len(referencia),
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def _fmt_p_min(num: int, den: int) -> str:
    """P_MIN como decimal, con al menos un decimal: 1/1 -> 1.0."""
    texto = f"{num / den:.4f}".rstrip("0")
    if texto.endswith("."):
        texto += "0"
    return texto


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exactitud y estrés de ISODENDRON sobre MNIST en CSV.",
    )
    parser.add_argument("--train-mnist", type=Path,
                        default=_mnist_default("mnist_train.csv"),
                        help="CSV de entrenamiento (o define MNIST_DIR)")
    parser.add_argument("--test-mnist", type=Path,
                        default=_mnist_default("mnist_test.csv"),
                        help="CSV de test (o define MNIST_DIR)")
    parser.add_argument("--umbral", type=int, default=64,
                        help="tinta si pixel > umbral (por defecto 64)")
    parser.add_argument("--downsample", type=int, default=1, choices=(1, 2, 4),
                        help="max-pooling factor x factor sobre 28x28")
    parser.add_argument("--n-train", type=int, default=0,
                        help="filas de train a usar (0 = todas)")
    parser.add_argument("--n-test", type=int, default=0,
                        help="filas de test a usar (0 = todas)")
    parser.add_argument("--eval-train", type=int, default=10000,
                        help="filas de train para medir la exactitud de train")
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12, 16],
                        help="profundidades a evaluar")
    parser.add_argument("--p-min", type=int, nargs=2, default=[1, 1],
                        metavar=("NUM", "DEN"),
                        help="pureza mínima de la moda, P_MIN = NUM/DEN "
                             "(por defecto 1 1: solo parar en nodos puros)")
    parser.add_argument("--seed", type=int, default=20240607)
    parser.add_argument("--csv", type=Path,
                        default=RESULTS_DIR / "mnist.csv")
    parser.add_argument("--dibujos", type=int, default=3,
                        help="imágenes de control a dibujar (0 = ninguna)")
    parser.add_argument("--sin-naive-bayes", action="store_true")
    parser.add_argument("--sin-barajado", action="store_true")
    parser.add_argument("--hilos", type=int, default=4,
                        help="hilos para la prueba de concurrencia (0 = omitir)")
    parser.add_argument("--hilos-n", type=int, default=10000,
                        help="filas para la prueba de concurrencia")
    parser.add_argument("--hilos-prof", type=int, default=8)
    args = parser.parse_args(argv)

    if args.train_mnist is None or args.test_mnist is None:
        parser.error("indica --train-mnist y --test-mnist, o define MNIST_DIR")

    if args.umbral < 0:
        parser.error("--umbral debe ser >= 0")
    depths = sorted(set(args.depths))
    p_min = (args.p_min[0], args.p_min[1])

    print("=" * 78)
    print("ISODENDRON SOBRE MNIST — EXACTITUD Y ESTRÉS")
    print("=" * 78)
    print(f"shared library : {lib_path}")
    print(f"isodendron C        : {isodendron_version()}   (binding isodendron {ISODENDRON_PY_VERSION})")
    print(f"binarización   : tinta si pixel > {args.umbral}")
    print(f"downsampling   : {args.downsample}x{args.downsample}")
    print(f"profundidades  : {depths}   "
          f"P_MIN = {_fmt_p_min(p_min[0], p_min[1])}")
    print()

    # ------------------------------------------------------------------
    # 1. Carga
    # ------------------------------------------------------------------
    print("CARGA DE DATOS")
    usar_nb = not args.sin_naive_bayes
    X_tr, y_tr, por_clase, f, n_tr = cargar_csv(
        args.train_mnist, args.umbral, args.downsample, args.n_train,
        args.seed, con_clases=usar_nb, nombre="train",
    )
    X_te, y_te, _, f_te, n_te = cargar_csv(
        args.test_mnist, args.umbral, args.downsample, args.n_test,
        args.seed, con_clases=False, nombre="test",
    )
    if f != f_te:
        raise SystemExit("ERROR: train y test no tienen el mismo número de features")

    reparto = [0] * _N_CLASES
    for c in y_tr:
        reparto[c] += 1
    print(f"  features: {f}   train: {n_tr}   test: {n_te}")
    print(f"  reparto de clases en train: {reparto}")
    print()

    # ------------------------------------------------------------------
    # 2. Control visual de la carga
    # ------------------------------------------------------------------
    if args.dibujos > 0:
        print("CONTROL VISUAL (primeras imágenes del CSV de test)")
        for k in range(min(args.dibujos, n_te)):
            fila = X_te[k * f:(k + 1) * f]
            print(f"  fila {k}  target={y_te[k]}  "
                  f"tinta={100.0 * sum(fila) / f:.0f}%")
            for linea in dibujar(fila, args.downsample):
                print("      " + linea)
        print()

    # ------------------------------------------------------------------
    # 3. Líneas base que no son ISODENDRON
    # ------------------------------------------------------------------
    print("=" * 78)
    print("LÍNEAS BASE")
    print("=" * 78)

    mayoritario = max(range(_N_CLASES), key=lambda c: reparto[c])
    acc_mayoritario = sum(1 for c in y_te if c == mayoritario) / n_te
    print(f"  trivial (siempre {mayoritario}, la clase más frecuente): "
          f"{acc_mayoritario:.2%}")

    pred_nb: list[int] | None = None
    acc_nb: float | None = None
    t_nb: float | None = None
    if usar_nb:
        acc_nb, pred_nb, t_nb = naive_bayes(por_clase, X_te, y_te, f, n_te)
        print(f"  naive Bayes Bernoulli por píxel ({f} features): "
              f"{acc_nb:.2%}   ({t_nb:.1f} s)")
        print()
        print("  Matriz de confusión del naive Bayes (filas = real):")
        imprimir_matriz(matriz_confusion(pred_nb, list(y_te)))
        del por_clase
    else:
        print("  naive Bayes: omitido (--sin-naive-bayes)")
    print()

    # ------------------------------------------------------------------
    # 4. ISODENDRON por profundidad
    # ------------------------------------------------------------------
    print("=" * 78)
    print("ISODENDRON")
    print("=" * 78)

    n_eval = min(args.eval_train, n_tr)
    idx_eval = sorted(random.Random(args.seed + 2).sample(range(n_tr), n_eval))

    def generador(X, y, n, perm=None):
        for i in range(n):
            yield X[i * f:(i + 1) * f], (y[i] if perm is None else y[perm[i]])

    perm_barajado: list[int] | None = None
    if not args.sin_barajado:
        perm_barajado = list(range(n_tr))
        random.Random(args.seed + 3).shuffle(perm_barajado)

    filas_test = [X_te[i * f:(i + 1) * f] for i in range(n_te)]

    resultados: list[dict] = []
    conf_mejor: list[list[int]] | None = None
    mejor_acc = -1.0

    for depth in depths:
        print("-" * 78)
        print(f"max_deep = {depth}")
        print("-" * 78)

        t0 = time.perf_counter()
        modelo = Isodendron().fit(generador(X_tr, y_tr, n_tr), max_deep=depth, p_min=p_min)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_test = modelo.predict(filas_test)
        t_pred = time.perf_counter() - t0

        pred_train = modelo.predict([X_tr[i * f:(i + 1) * f] for i in idx_eval])

        acc_test = exactitud(pred_test, list(y_te))
        acc_train = exactitud(pred_train, [y_tr[i] for i in idx_eval])

        st = estadisticas_arbol(modelo.blob)
        blob_mb = len(modelo.blob) / 1048576.0

        acc_barajado: float | None = None
        if perm_barajado is not None:
            modelo_b = Isodendron().fit(
                generador(X_tr, y_tr, n_tr, perm=perm_barajado),
                max_deep=depth, p_min=p_min,
            )
            acc_barajado = exactitud(modelo_b.predict(filas_test), list(y_te))
            del modelo_b

        print(f"  nodos={st['n_nodes']}  hojas={st['n_leaves']}  "
              f"prof.alc.={st['profundidad']}  N medio/hoja={st['n_medio_hoja']:.1f}")
        print(f"  blob={blob_mb:.2f} MB   fit={t_fit:.1f} s   predict={t_pred:.1f} s")
        print(f"  exactitud train={acc_train:.2%}   test={acc_test:.2%}"
              + (f"   (etiquetas barajadas: {acc_barajado:.2%})"
                 if acc_barajado is not None else ""))

        if acc_test > mejor_acc:
            mejor_acc = acc_test
            conf_mejor = matriz_confusion(pred_test, list(y_te))

        resultados.append({
            "depth": depth,
            "n_nodes": st["n_nodes"],
            "n_leaves": st["n_leaves"],
            "profundidad": st["profundidad"],
            "n_medio_hoja": st["n_medio_hoja"],
            "blob_mb": blob_mb,
            "fit_s": t_fit,
            "predict_s": t_pred,
            "train_exact": acc_train,
            "test_exact": acc_test,
            "barajado_exact": acc_barajado,
        })
        print()

    if conf_mejor is not None:
        print(f"Matriz de confusión del mejor modelo (test={mejor_acc:.2%}):")
        imprimir_matriz(conf_mejor)
        print()

    # ------------------------------------------------------------------
    # 5. Prueba de concurrencia
    # ------------------------------------------------------------------
    conc: dict | None = None
    if args.hilos > 0:
        print("=" * 78)
        print("PRUEBA DE CONCURRENCIA (isodendron.h declara isodendron_train thread-safe)")
        print("=" * 78)
        n_c = min(args.hilos_n, n_tr)
        idx_c = sorted(random.Random(args.seed + 5).sample(range(n_tr), n_c))
        Xc = array.array("i")
        yc = array.array("i")
        for i in idx_c:
            Xc.extend(X_tr[i * f:(i + 1) * f])
            yc.append(y_tr[i])
        conc = prueba_concurrencia(Xc, yc, n_c, f, args.hilos, args.hilos_prof, p_min)
        print(f"  secuencial 1 hilo : {conc['t_secuencial']:.2f} s")
        print(f"  {conc['hilos']} hilos en paralelo: {conc['t_paralelo']:.2f} s")
        print(f"  aceleración       : x{conc['speedup']:.2f}")
        print(f"  blobs idénticos   : {conc['hilos'] - len(conc['distintos'])}/{conc['hilos']}")
        if conc["errores"]:
            for e in conc["errores"]:
                print(f"  ERROR en un hilo: {type(e).__name__}: {e}")
        print()
        del Xc, yc

    # ------------------------------------------------------------------
    # 6. Comprobaciones
    # ------------------------------------------------------------------
    comprobaciones: list[tuple[str, bool, str]] = []
    ultimo = resultados[-1]

    if ultimo["barajado_exact"] is not None:
        comprobaciones.append((
            "el control de etiquetas barajadas no supera al real",
            ultimo["barajado_exact"] <= ultimo["test_exact"] + 0.05,
            f"real={ultimo['test_exact']:.2%} vs barajado={ultimo['barajado_exact']:.2%}",
        ))

    if acc_nb is not None:
        comprobaciones.append((
            "el naive Bayes demuestra que los datos son aprendibles",
            acc_nb > 0.50,
            f"naive Bayes={acc_nb:.2%}",
        ))
        comprobaciones.append((
            "ISODENDRON no alcanza al naive Bayes",
            ultimo["test_exact"] < acc_nb,
            f"ISODENDRON={ultimo['test_exact']:.2%} vs NB={acc_nb:.2%}",
        ))

    comprobaciones.append((
        "la exactitud de train no queda por debajo de la de test",
        ultimo["train_exact"] >= ultimo["test_exact"] - 0.02,
        f"train={ultimo['train_exact']:.2%} test={ultimo['test_exact']:.2%}",
    ))

    for r in resultados:
        # Tras los nodos va el payload de distribuciones, en pares
        # (target, count) de 8 bytes, asi que el exceso debe ser multiplo
        # de 8. Contar solo los nodos daba falso negativo desde el v3.
        n_esperado = _HEADER_SIZE + r["n_nodes"] * _NODE_SIZE
        exceso = r["blob_mb"] * 1048576 - n_esperado
        if exceso < -1 or (exceso % 8) > 1:
            comprobaciones.append((
                f"blob coherente con n_nodes en depth={r['depth']}", False,
                f"{r['blob_mb'] * 1048576:.0f} != {n_esperado}",
            ))
            break
    else:
        comprobaciones.append((
            "todos los blobs son coherentes con su n_nodes", True, ""
        ))

    if conc is not None:
        comprobaciones.append((
            "entrenamiento concurrente: sin errores y blobs idénticos",
            not conc["errores"] and not conc["distintos"],
            f"errores={len(conc['errores'])} distintos={len(conc['distintos'])}",
        ))
        comprobaciones.append((
            "el entrenamiento concurrente acelera (GIL liberado)",
            conc["speedup"] > 1.2,
            f"x{conc['speedup']:.2f}",
        ))

    print("=" * 78)
    print("COMPROBACIONES")
    print("=" * 78)
    fallos = 0
    for descripcion, ok, detalle in comprobaciones:
        marca = "ok   " if ok else "FALLA"
        extra = f"  [{detalle}]" if detalle else ""
        print(f"  {marca} {descripcion}{extra}")
        if not ok:
            fallos += 1
    print()

    # ------------------------------------------------------------------
    # 7. Resumen y CSV
    # ------------------------------------------------------------------
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(f"  trivial (clase mayoritaria) : {acc_mayoritario:.2%}")
    if acc_nb is not None:
        print(f"  naive Bayes Bernoulli       : {acc_nb:.2%}")
    for r in resultados:
        linea = (f"  ISODENDRON max_deep={r['depth']:<3}          "
                 f": {r['test_exact']:.2%}   "
                 f"(train {r['train_exact']:.2%}, "
                 f"{r['n_nodes']} nodos, {r['fit_s']:.1f} s)")
        if r["barajado_exact"] is not None:
            linea += f"   barajado {r['barajado_exact']:.2%}"
        print(linea)

    pico = memoria_pico_mb()
    if pico is not None:
        print(f"\n  memoria pico del proceso    : {pico:.0f} MB")
    print()

    if resultados:
        import csv as _csv
        campos = ["depth", "n_nodes", "n_leaves", "profundidad", "n_medio_hoja",
                  "blob_mb", "fit_s", "predict_s", "train_exact", "test_exact",
                  "barajado_exact"]
        try:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
                w.writeheader()
                for r in resultados:
                    w.writerow(r)
            print(f"resultados escritos en: {args.csv}")
        except OSError as exc:
            print(f"aviso: no se pudo escribir {args.csv}: {exc}")

    if fallos:
        print(f"\nRESULTADO: {fallos} comprobación(es) fallidas")
        return 3
    print("\nRESULTADO: todas las comprobaciones OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
