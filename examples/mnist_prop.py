#!/usr/bin/env python3
"""
mnist_prop.py — Barrido de P_MIN sobre MNIST, con 4 hilos.

Qué se prueba
-------------
P_MIN es el **segundo límite** de ISODENDRON, junto a max_deep, y mide la
**pureza del nodo**:

    un nodo deja de dividirse si  P = n_mode / N >= P_MIN

es decir, si la fracción de filas del nodo cuyo target vale su valor más
frecuente (la moda) alcanza el umbral. La condición es intrínseca al nodo
—no depende de un padre— y aplica también a la raíz, así que es un punto
de parada independiente de max_deep.

Este script barre max_deep = 20 (holgura de sobra: el árbol satura solo)
contra varios valores de P_MIN:

    P_MIN =  2/100  para solo si la moda ya cubre el 2% del nodo
    P_MIN = 10/100  para al 10% de pureza
    P_MIN = 20/100  para al 20% de pureza
    P_MIN =  1/1    exige pureza total: el árbol más profundo posible

Más P_MIN significa exigir más pureza, así que el número de nodos crece
de forma monótona. Es el comportamiento **contrario** al de la semántica
anterior, en la que P_MIN era una proporción de tamaño y un valor mayor
recortaba más el árbol.

El valor 0/1, que antes servía como referencia de "sin poda", ahora
significa "parar siempre" (n_mode >= 0 se cumple trivialmente) y deja
un árbol de un solo nodo. La referencia de árbol maximal es 1/1.

Hilos
-----
Cada prueba de P_MIN, por separado, se reparte entre 4 hilos: se llama a
`_ffi.train_mt` con `n_threads = --hilos`, que reparte la construcción
del árbol dentro de la biblioteca C. Las configuraciones se ejecutan una
tras otra.

Esto es lo adecuado aquí porque las configuraciones tienen costes muy
dispares: la de pureza total construye un árbol enorme y las demás paran
a las pocas divisiones, así que el tiempo total lo domina una sola. Si
en vez de repartir cada entrenamiento se lanzara una configuración por
hilo, tres núcleos se quedarían ociosos esperando a la pesada.

El reparto es transparente —las decisiones de split no dependen de qué
hilo construya cada subárbol— así que el blob sale idéntico al
secuencial, y el script lo comprueba comparando contra una tanda con
`--hilos 1`.

También se llama directamente a `_ffi` en vez de a `Isodendron.fit` para
mantener el GIL fuera del camino crítico: `to_arrays` valida elemento a
elemento en Python (unos 47 millones de iteraciones) y lo haría todo
secuencial.

Uso
---
    python mnist_prop.py
    python mnist_prop.py --n-train 3000 --n-test 800 --hilos 2
    python mnist_prop.py --p-min 0/1 20/100 10/100 2/100 1/100

Solo usa la biblioteca estándar y el paquete isodendron.
"""

from __future__ import annotations

import argparse
import array
import os
import random
import struct
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Localización del proyecto
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

LADO = 28
PIXELES = LADO * LADO
N_CLASES = 10

_HEADER_SIZE = 32
_NODE_SIZE = 32
_NODE_FMT = "<IiiiiiiI"
_MAGIC = 0x444F5349


def _mnist_default(filename: str) -> Path | None:
    """Ruta por defecto si el usuario define MNIST_DIR; si no, hay que pasar --train/--test."""
    base = os.environ.get("MNIST_DIR")
    return Path(base) / filename if base else None


try:
    from isodendron import IsodendronError
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
# Carga del CSV
# ----------------------------------------------------------------------

def cargar_csv(ruta: Path, umbral: int, n_max: int, seed: int,
               nombre: str) -> tuple[array.array, array.array, int, int]:
    """Devuelve (X, y, n_filas, n_features) con los píxeles binarizados."""
    if not ruta.is_file():
        raise SystemExit(f"ERROR: no existe {ruta}")

    t0 = time.perf_counter()
    with ruta.open("rb") as fh:
        cabecera = fh.readline()
        if not cabecera:
            raise SystemExit(f"ERROR: {ruta} está vacío")
        columnas = cabecera.rstrip(b"\r\n").split(b",")
        total = sum(1 for _ in fh)

    if len(columnas) != PIXELES + 1:
        raise SystemExit(
            f"ERROR: {ruta} tiene {len(columnas)} columnas; se esperaban "
            f"{PIXELES + 1} (target + {PIXELES} píxeles)"
        )

    if n_max and n_max < total:
        seleccion: list[int] | None = sorted(
            random.Random(seed).sample(range(total), n_max)
        )
    else:
        seleccion = None

    X = array.array("i")
    y = array.array("i")
    buffer = bytearray(4 * PIXELES)

    leidas = 0
    apuntador = 0
    with ruta.open("rb") as fh:
        fh.readline()
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
                    f"ERROR: {ruta}, fila {idx + 2}: {len(partes)} campos"
                )
            clase = int(partes[0])
            if not 0 <= clase < N_CLASES:
                raise SystemExit(
                    f"ERROR: {ruta}, fila {idx + 2}: target {clase} fuera de 0..9"
                )
            y.append(clase)

            # Bytes 1..3 del buffer quedan a 0 de una fila a la siguiente,
            # así que basta con escribir las posiciones 0 mod 4.
            buffer[0::4] = bytes(1 if int(v) > umbral else 0 for v in partes[1:])
            X.frombytes(buffer)
            leidas += 1

    if leidas == 0:
        raise SystemExit(f"ERROR: no se cargó ninguna fila de {ruta}")

    print(f"    {nombre}: {leidas} filas x {PIXELES} features "
          f"({time.perf_counter() - t0:.1f} s, "
          f"{X.buffer_info()[1] * X.itemsize / 1048576:.0f} MB)")
    return X, y, leidas, PIXELES


# ----------------------------------------------------------------------
# Trabajo por configuración (todo dentro del C, GIL liberado)
# ----------------------------------------------------------------------

def entrenar_y_predecir(
    X: array.array,
    y: array.array,
    n: int,
    X_te: array.array,
    y_te: array.array,
    n_te: int,
    f: int,
    max_deep: int,
    p_min: tuple[int, int],
    hilos: int = 1,
) -> dict:
    """
    Entrena un árbol repartiendo la construcción entre `hilos` hilos, y
    predice train y test. Todo el trabajo pesado ocurre dentro de la
    shared library, que libera el GIL.
    """
    pn, pd = p_min

    t0 = time.perf_counter()
    blob = _ffi.train_mt(X, y, n, f, max_deep, pn, pd, hilos)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_train = _ffi.predict(blob, X, n, f)
    pred_test = _ffi.predict(blob, X_te, n_te, f)
    t_pred = time.perf_counter() - t0

    return {
        "p_min": p_min,
        "blob": blob,
        "pred_train": list(pred_train),
        "pred_test": list(pred_test),
        "t_fit": t_fit,
        "t_pred": t_pred,
    }


def lanzar(
    configs: list[tuple[int, int]],
    X: array.array,
    y: array.array,
    n: int,
    X_te: array.array,
    y_te: array.array,
    n_te: int,
    f: int,
    max_deep: int,
    hilos: int,
) -> tuple[dict, float]:
    """Ejecuta todas las configuraciones y devuelve (resultados, segundos).

    Las configuraciones van en serie; lo que se reparte entre hilos es la
    construcción de cada árbol.
    """
    resultados: dict[tuple[int, int], dict] = {}
    t0 = time.perf_counter()

    for cfg in configs:
        resultados[cfg] = entrenar_y_predecir(
            X, y, n, X_te, y_te, n_te, f, max_deep, cfg, hilos
        )

    return resultados, time.perf_counter() - t0


# ----------------------------------------------------------------------
# Estadísticas del blob (solo con lo que hay dentro)
# ----------------------------------------------------------------------

def estadisticas(blob: bytes, n_raiz: int) -> dict:
    """Recorre el blob y resume el árbol. Sin datos ni deserialización."""
    if len(blob) < _HEADER_SIZE:
        raise IsodendronError(2, f"blob demasiado corto: {len(blob)} bytes")
    magic, version, n_feat, n_nodes, max_deep, pn, pd, _r = struct.unpack_from(
        "<IIIIIIII", blob, 0
    )
    if magic != _MAGIC:
        raise IsodendronError(2, f"magic inválido: 0x{magic:08x}")
    if len(blob) != _HEADER_SIZE + n_nodes * _NODE_SIZE + struct.unpack_from("<I", blob, 28)[0]:
        raise IsodendronError(2, "tamaño del blob incoherente con n_nodes")

    unpack = struct.Struct(_NODE_FMT).unpack_from
    hojas = 0
    hojas_n1 = 0
    suma_n = 0
    prof_max = 0
    pila = [(0, 0)]
    while pila:
        idx, prof = pila.pop()
        is_leaf, _feat, _val, nodo_n, _k, _med, _nmed, left = unpack(
            blob, _HEADER_SIZE + _NODE_SIZE * idx
        )
        if is_leaf:
            hojas += 1
            suma_n += nodo_n
            if nodo_n == 1:
                hojas_n1 += 1
            if prof > prof_max:
                prof_max = prof
        else:
            pila.append((idx + 1, prof + 1))
            pila.append((idx + 1 + left, prof + 1))

    return {
        "n_features": n_feat,
        "n_nodes": n_nodes,
        "max_deep": max_deep,
        "p_min": (pn, pd),
        "n_leaves": hojas,
        "n_leaves_n1": hojas_n1,
        "profundidad": prof_max,
        "n_medio_hoja": (suma_n / hojas) if hojas else 0.0,
        "blob_mb": len(blob) / 1048576.0,
        "n_raiz": n_raiz,
    }


def exactitud(pred: list[int], real: list[int]) -> float:
    if not real:
        return 0.0
    return sum(1 for p, r in zip(pred, real) if p == r) / len(real)


def _fmt_p_min(num: int, den: int) -> str:
    """P_MIN como decimal, con al menos un decimal: 1/1 -> 1.0."""
    texto = f"{num / den:.4f}".rstrip("0")
    if texto.endswith("."):
        texto += "0"
    return texto


def _etiqueta(p_min: tuple[int, int]) -> str:
    num, den = p_min
    if num == 0:
        return "0.0 (parar siempre)"
    return f"{_fmt_p_min(num, den)} ({100.0 * num / den:.0f}%)"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Barrido de P_MIN (pureza mínima de la moda) sobre MNIST.",
    )
    parser.add_argument("--train-mnist", type=Path,
                        default=_mnist_default("mnist_train.csv"),
                        help="CSV de entrenamiento (o define MNIST_DIR)")
    parser.add_argument("--test-mnist", type=Path,
                        default=_mnist_default("mnist_test.csv"),
                        help="CSV de test (o define MNIST_DIR)")
    parser.add_argument("--umbral", type=int, default=64,
                        help="tinta si pixel > umbral (por defecto 64)")
    parser.add_argument("--n-train", type=int, default=0, help="0 = todas")
    parser.add_argument("--n-test", type=int, default=0, help="0 = todas")
    parser.add_argument("--seed", type=int, default=20240607)
    parser.add_argument("--max-deep", type=int, default=20,
                        help="holgura de profundidad (por defecto 20)")
    parser.add_argument("--p-min", nargs="+",
                        default=["2/100", "10/100", "20/100", "1/1"],
                        metavar="NUM/DEN",
                        help="umbrales de pureza P_MIN a barrer")
    parser.add_argument("--hilos", type=int, default=4,
                        help="hilos por cada prueba de P_MIN (por defecto 4)")
    parser.add_argument("--sin-secuencial", action="store_true",
                        help="omite la tanda secuencial de referencia")
    parser.add_argument("--csv", type=Path,
                        default=RESULTS_DIR / "mnist_prop.csv")
    args = parser.parse_args(argv)

    if args.train_mnist is None or args.test_mnist is None:
        parser.error("indica --train-mnist y --test-mnist, o define MNIST_DIR")

    # --- parseo de las configuraciones de P_MIN ---
    configs: list[tuple[int, int]] = []
    for texto in args.p_min:
        if "/" not in texto:
            parser.error(f"P_MIN debe ser NUM/DEN (recibido: {texto!r})")
        a, b = texto.split("/", 1)
        try:
            num, den = int(a), int(b)
        except ValueError:
            parser.error(f"P_MIN no numérico: {texto!r}")
        if num < 0 or den <= 0:
            parser.error(f"P_MIN inválido: {texto!r} (num >= 0, den > 0)")
        if (num, den) not in configs:
            configs.append((num, den))

    if args.hilos < 1:
        parser.error("--hilos debe ser >= 1")
    if args.max_deep < 1:
        parser.error("--max-deep debe ser >= 1")

    print("=" * 78)
    print("BARRIDO DE P_MIN SOBRE MNIST (con hilos)")
    print("=" * 78)
    try:
        from isodendron._lib import lib_path  # noqa: E402
        print(f"shared library : {lib_path}")
    except Exception:
        print(f"shared library : {lib_path}")
    print(f"isodendron C        : {isodendron_version()}   (binding isodendron {ISODENDRON_PY_VERSION})")
    print(f"binarización   : tinta si pixel > {args.umbral}")
    print(f"max_deep       : {args.max_deep}")
    print(f"P_MIN          : " + ", ".join(_etiqueta(c) for c in configs))
    print(f"hilos          : {args.hilos}")
    print()
    print("Recordatorio: P_MIN es la pureza mínima de la moda "
          "(n_mode / N), no una proporción de tamaño respecto al padre.")
    print()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    print("CARGA DE DATOS")
    X, y, n, f = cargar_csv(args.train_mnist, args.umbral, args.n_train,
                            args.seed, "train")
    X_te, y_te, n_te, f_te = cargar_csv(args.test_mnist, args.umbral, args.n_test,
                                        args.seed, "test")
    if f != f_te:
        raise SystemExit("ERROR: train y test no coinciden en nº de features")
    y_te_lista = list(y_te)
    print(f"  {n} train / {n_te} test, {f} features "
          f"(cota de nodos del arena: {2 * n})")
    print()

    # ------------------------------------------------------------------
    # Tanda paralela
    # ------------------------------------------------------------------
    print("=" * 78)
    print(f"ENTRENAMIENTO CON {args.hilos} HILOS POR PRUEBA")
    print("=" * 78)
    try:
        res_par, t_par = lanzar(configs, X, y, n, X_te, y_te, n_te, f,
                                args.max_deep, args.hilos)
    except IsodendronError as exc:
        print(f"  ERROR de la biblioteca C: {exc} (código {exc.code})")
        return 2
    print(f"  tiempo de pared: {t_par:.1f} s para {len(configs)} configuraciones")
    for cfg in configs:
        r = res_par[cfg]
        print(f"    P_MIN={_etiqueta(cfg):<14} fit={r['t_fit']:6.1f} s   "
              f"predict={r['t_pred']:5.1f} s   blob={len(r['blob']) / 1048576:.2f} MB")
    print()

    # ------------------------------------------------------------------
    # Tanda secuencial (aceleración + determinismo)
    # ------------------------------------------------------------------
    res_seq: dict = {}
    t_seq: float | None = None
    if not args.sin_secuencial:
        print("=" * 78)
        print("REFERENCIA SECUENCIAL (1 hilo por prueba)")
        print("=" * 78)
        res_seq, t_seq = lanzar(configs, X, y, n, X_te, y_te, n_te, f,
                                args.max_deep, 1)
        print(f"  tiempo de pared: {t_seq:.1f} s")
        print()

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    print("=" * 78)
    print("RESULTADOS")
    print("=" * 78)
    cabecera = (f"  {'P_MIN':<16} {'nodos':>7} {'hojas':>7} {'N=1':>7} "
                f"{'prof':>5} {'N/hoja':>8} {'blob MB':>8} "
                f"{'fit s':>7} {'train':>8} {'test':>8}")
    print(cabecera)
    print("  " + "-" * (len(cabecera) - 2))

    filas: list[dict] = []
    for cfg in configs:
        r = res_par[cfg]
        st = estadisticas(r["blob"], n)
        acc_tr = exactitud(r["pred_train"], list(y))
        acc_te = exactitud(r["pred_test"], y_te_lista)
        fila = {
            "p_min_num": cfg[0],
            "p_min_den": cfg[1],
            "p_min_pct": 100.0 * cfg[0] / cfg[1],
            "n_nodes": st["n_nodes"],
            "n_leaves": st["n_leaves"],
            "n_leaves_n1": st["n_leaves_n1"],
            "profundidad": st["profundidad"],
            "n_medio_hoja": st["n_medio_hoja"],
            "blob_mb": st["blob_mb"],
            "fit_s": r["t_fit"],
            "predict_s": r["t_pred"],
            "train_exact": acc_tr,
            "test_exact": acc_te,
            "max_deep": args.max_deep,
            "n_train": n,
            "n_test": n_te,
            "n_features": f,
            "hilos": args.hilos,
            "t_paralelo": t_par,
            "t_secuencial": t_seq,
            "determinista": (cfg in res_seq and res_seq[cfg]["blob"] == r["blob"])
                              if res_seq else None,
        }
        filas.append(fila)
        print(f"  {_etiqueta(cfg):<16} {st['n_nodes']:>7} {st['n_leaves']:>7} "
              f"{st['n_leaves_n1']:>7} {st['profundidad']:>5} "
              f"{st['n_medio_hoja']:>8.1f} {st['blob_mb']:>8.2f} "
              f"{r['t_fit']:>7.1f} {acc_tr:>7.2%} {acc_te:>7.2%}")
    print()

    # ------------------------------------------------------------------
    # Aceleración
    # ------------------------------------------------------------------
    if t_seq is not None:
        print("-" * 78)
        print("ACELERACIÓN CON HILOS")
        print("-" * 78)
        suma_fit = sum(r["t_fit"] for r in res_par.values())
        print(f"  suma de fits (con hilos) : {suma_fit:6.1f} s "
              f"(cada prueba mide su propio tramo)")
        print(f"  pared con {args.hilos} hilos/prueba : {t_par:6.1f} s")
        print(f"  pared con 1 hilo/prueba  : {t_seq:6.1f} s")
        print(f"  aceleración real         : x{t_seq / t_par:.2f}")
        print(f"  eficiencia por hilo      : {100.0 * (t_seq / t_par) / args.hilos:.0f}%")
        print(f"  procesadores lógicos     : {os.cpu_count()}")
        print()

    # ------------------------------------------------------------------
    # Comprobaciones
    # ------------------------------------------------------------------
    comprobaciones: list[tuple[str, bool, str]] = []

    if res_seq:
        distintos = [c for c in configs if res_seq[c]["blob"] != res_par[c]["blob"]]
        comprobaciones.append((
            "los blobs paralelos son idénticos a los secuenciales",
            not distintos,
            f"{len(configs) - len(distintos)}/{len(configs)} coinciden",
        ))

    nodos_ordenados = [f_["n_nodes"] for f_ in filas]
    pmin_ordenados = [f_["p_min_pct"] for f_ in filas]
    if pmin_ordenados == sorted(pmin_ordenados):
        comprobaciones.append((
            "más pureza exigida implica más o igual nodos (monotonía)",
            all(a <= b for a, b in zip(nodos_ordenados, nodos_ordenados[1:])),
            " ".join(str(x) for x in nodos_ordenados),
        ))

    for f_ in filas:
        if f_["n_nodes"] > 2 * n:
            comprobaciones.append((
                f"P_MIN={_etiqueta((f_['p_min_num'], f_['p_min_den']))}: "
                f"nodos dentro de la cota del arena", False,
                f"{f_['n_nodes']} > {2 * n}",
            ))
            break
    else:
        comprobaciones.append((
            "todos los árboles caben en la cota de nodos del arena (2n)",
            True,
            f"máx {max(nodos_ordenados)} de {2 * n}",
        ))

    print("-" * 78)
    print("COMPROBACIONES")
    print("-" * 78)
    fallos = 0
    for descripcion, ok, detalle in comprobaciones:
        print(f"  {'ok   ' if ok else 'FALLA'} {descripcion}"
              + (f"  [{detalle}]" if detalle else ""))
        if not ok:
            fallos += 1
    print()

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    if len(filas) >= 2:
        # La referencia es el árbol más profundo, que corresponde a la
        # exigencia de pureza más alta (1/1 = pureza total).
        ref = filas[-1]
        print(f"  Árbol más profundo ({_etiqueta((ref['p_min_num'], ref['p_min_den']))}): "
              f"{ref['n_nodes']} nodos, {ref['n_leaves']} hojas, "
              f"{ref['n_leaves_n1']} hojas de tamaño 1, "
              f"test {ref['test_exact']:.2%}.")
        for f_ in filas[:-1]:
            dn = ref["n_nodes"] - f_["n_nodes"]
            pct = (100.0 * dn / ref["n_nodes"]) if ref["n_nodes"] else 0.0
            print(f"  {_etiqueta((f_['p_min_num'], f_['p_min_den'])):<16} "
                  f"ahorra {dn:>7} nodos ({pct:5.1f}%); "
                  f"test {ref['test_exact']:.2%} -> {f_['test_exact']:.2%}")
    print()
    print("  P_MIN es ahora un mando continuo: exigir más pureza hace crecer el")
    print("  árbol de forma monótona, y aflojar el umbral lo recorta. Ya no es")
    print("  una proporción de tamaño forzada al 50% por el criterio de balance,")
    print("  que era lo que lo dejaba sin recorrido útil.")
    print()

    if filas:
        import csv as _csv
        campos = list(filas[0].keys())
        try:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
                w.writeheader()
                w.writerows(filas)
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
