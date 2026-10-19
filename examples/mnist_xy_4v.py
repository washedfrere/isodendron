#!/usr/bin/env python3
"""
mnist_xy_4v.py - MNIST como nube de puntos (x, y, color).

Reformatea cada imagen como una nube de puntos: un punto por pixel de
tinta, con tres features y el digito de su imagen como target.

    x       columna, 0..27
    y       fila,    0..27
    color   nivel de gris, 0..3

Niveles del gris crudo:

    crudo <  32   -> nivel 0  (fondo: no se genera punto)
    crudo <  64   -> nivel 1
    crudo < 128   -> nivel 2
    crudo >= 128  -> nivel 3

Como un punto no puede reconocer un numero, la prediccion de una imagen
se obtiene agrupando las predicciones de sus puntos. La prediccion de una imagen se obtiene sumando los conteos de todas sus
hojas para cada target y eligiendo el de mayor total. Es aritmetica
entera, sin correcciones ni artefactos: ninguna probabilidad interviene
en la decision y el porcentaje de la matriz se calcula a posteriori sobre
los recuentos. P se conserva solo como indicador de pureza de la hoja.

Se prueban cuatro umbrales de pureza con profundidad ilimitada en la
practica (max_deep = 70, que el arbol no llega a agotar), y para cada uno
se muestra su matriz de confusion sobre test: cuatro pruebas.

La novena reutiliza el modelo de umbral 1.0 y lo refina con los datos de
test mediante isodendron_partial_fit: el arbol no cambia, solo se
recalibran las hojas. Despues se evalua sobre train y test juntos, con la
agregacion por hoja.

Aviso: en la novena prueba las hojas se estiman con test y se evalua
sobre test, asi que esa matriz sale optimista por construccion. Es una
demostracion del mecanismo de refinamiento, no una estimacion de
generalizacion.

Uso:
    python mnist_xy_4v.py
    python mnist_xy_4v.py --n-train-img 8000 --n-test-img 2000
"""

from __future__ import annotations

import argparse
import array
import os
import random
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------
# Localizacion del proyecto
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

LADO = 28
PIXELES = LADO * LADO
N_CLASES = 10
N_FEATURES = 3

_HEADER_SIZE = 32
_NODE_SIZE = 32
_NODE_FMT = "<IiiiiiiI"
_MAGIC = 0x444F5349

#: crudo 0..31 -> 0, 32..63 -> 1, 64..127 -> 2, >=128 -> 3
NIVEL = bytes([0] * 32 + [1] * 32 + [2] * 64 + [3] * 128)
POS_X = bytes(p % LADO for p in range(PIXELES))
POS_Y = bytes(p // LADO for p in range(PIXELES))

assert len(NIVEL) == 256

#: Umbrales de pureza que se prueban.
P_VALORES = ("0.1", "0.5", "0.75", "1.0")

#: Profundidad que en la practica no limita: el arbol se agota antes.
MAX_DEEP = 70

#: Tope de entradas de distribucion por hoja.
MAX_TARGETS = 10


def _mnist_default(filename: str) -> Path | None:
    """Ruta por defecto si el usuario define MNIST_DIR; si no, hay que pasar --train/--test."""
    base = os.environ.get("MNIST_DIR")
    return Path(base) / filename if base else None


try:
    from isodendron import IsodendronError
    from isodendron import _ffi
    from isodendron._lib import lib_path
except ImportError as exc:  # pragma: no cover - diagnóstico de entorno
    print("ERROR: no se pudo importar el paquete isodendron.", file=sys.stderr)
    print("  instala el proyecto con `pip install .` o `pip install -e .`",
          file=sys.stderr)
    print(f"  causa: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)


# ----------------------------------------------------------------------
# Carga del CSV como nube de puntos
# ----------------------------------------------------------------------

def cargar_csv_puntos(
    ruta: Path,
    n_img_max: int,
    seed: int,
    nombre: str,
) -> tuple[array.array, array.array, array.array, array.array, int, int]:
    """
    Convierte un CSV de MNIST en una nube de puntos.

    Devuelve (X, y, etiquetas, offsets, n_puntos, n_imagenes):
      X           array('i') con 3 valores por punto (x, y, nivel)
      y           array('i') con el digito de cada punto
      etiquetas   array('i') con el digito de cada imagen conservada
      offsets     array('i') de tamano n_imagenes+1; los puntos de la
                  imagen i son X[3*offsets[i] : 3*offsets[i+1]]
    """
    if not ruta.is_file():
        raise SystemExit(f"ERROR: no existe {ruta}")

    t0 = time.perf_counter()
    with ruta.open("rb") as fh:
        cabecera = fh.readline()
        if not cabecera:
            raise SystemExit(f"ERROR: {ruta} esta vacio")
        cols = cabecera.rstrip(b"\r\n").split(b",")
        total = sum(1 for _ in fh)

    if len(cols) != PIXELES + 1:
        raise SystemExit(
            f"ERROR: {ruta} tiene {len(cols)} columnas; se esperaban "
            f"{PIXELES + 1}"
        )

    if n_img_max and n_img_max < total:
        elegidas: list[int] | None = sorted(
            random.Random(seed).sample(range(total), n_img_max)
        )
    else:
        elegidas = None

    X = array.array("i")
    y = array.array("i")
    etiquetas = array.array("i")
    offsets = array.array("i", [0])

    total_puntos = 0
    procesadas = 0
    apuntador = 0

    with ruta.open("rb") as fh:
        fh.readline()
        for idx, linea in enumerate(fh):
            if elegidas is not None:
                if apuntador >= len(elegidas):
                    break
                if elegidas[apuntador] != idx:
                    continue
                apuntador += 1

            partes = linea.rstrip(b"\r\n").split(b",")
            if len(partes) != PIXELES + 1:
                raise SystemExit(
                    f"ERROR: {ruta}, fila {idx + 2}: {len(partes)} campos"
                )

            t = int(partes[0])
            if not 0 <= t < N_CLASES:
                raise SystemExit(
                    f"ERROR: {ruta}, fila {idx + 2}: target {t} fuera de 0..9"
                )

            fila: list[int] = []
            for p in range(PIXELES):
                nivel = NIVEL[int(partes[p + 1])]
                if nivel:
                    fila.append(POS_X[p])
                    fila.append(POS_Y[p])
                    fila.append(nivel)

            puntos = len(fila) // N_FEATURES
            if puntos == 0:
                continue

            X.extend(fila)
            y.extend([t] * puntos)
            etiquetas.append(t)
            total_puntos += puntos
            offsets.append(total_puntos)
            procesadas += 1

            if procesadas % 10000 == 0:
                print(f"    {nombre}: {procesadas} imagenes, "
                      f"{total_puntos} puntos... "
                      f"({time.perf_counter() - t0:.0f} s)")

    if procesadas == 0:
        raise SystemExit(f"ERROR: no se cargo ninguna imagen de {ruta}")

    print(f"    {nombre}: {procesadas} imagenes -> {total_puntos} puntos "
          f"({total_puntos / procesadas:.1f} por imagen) "
          f"en {time.perf_counter() - t0:.1f} s")
    return X, y, etiquetas, offsets, total_puntos, procesadas


# ----------------------------------------------------------------------
# Lectura del blob
# ----------------------------------------------------------------------

def estadisticas_arbol(blob: bytes) -> dict:
    """Nodos, hojas, profundidad maxima y tamano del payload."""
    magic, _ver, _nf, n_nodes, _md, _pn, _pd, payload = struct.unpack_from(
        "<IIIIIIII", blob, 0)
    if magic != _MAGIC:
        raise IsodendronError(2, f"magic invalido: 0x{magic:08x}")

    hojas = 0
    prof_max = 0
    pila = [(0, 0)]
    while pila:
        idx, prof = pila.pop()
        is_leaf, _f, _v, _N, _K, _mode, _nm, left = struct.unpack_from(
            _NODE_FMT, blob, _HEADER_SIZE + _NODE_SIZE * idx)
        if is_leaf:
            hojas += 1
            if prof > prof_max:
                prof_max = prof
        else:
            pila.append((idx + 1, prof + 1))
            pila.append((idx + 1 + left, prof + 1))

    return {"n_nodes": n_nodes, "n_leaves": hojas,
            "profundidad": prof_max, "payload_bytes": payload}


def tabla_de_hojas(blob: bytes) -> dict:
    """
    Devuelve {indice_nodo: (mode, n_mode, N, conteo)} para las hojas.

    `conteo` es la lista de 10 frecuencias absolutas. Las hojas sin datos
    (N = 0) se omiten: no aportan evidencia a ninguna imagen.
    """
    n_nodes = struct.unpack_from("<I", blob, 12)[0]
    tabla = {}

    for i in range(n_nodes):
        off = _HEADER_SIZE + i * _NODE_SIZE
        is_leaf, p_off, p_len, N, _K, mode, n_mode, _left = struct.unpack_from(
            _NODE_FMT, blob, off)
        if not is_leaf or N <= 0:
            continue

        conteo = [0] * N_CLASES
        for j in range(p_len):
            t, c = struct.unpack_from("<ii", blob, p_off + j * 8)
            if 0 <= t < N_CLASES:
                conteo[t] = c

        tabla[i] = (mode, n_mode, N, conteo)

    return tabla


# ----------------------------------------------------------------------
# Agregacion por imagen
# ----------------------------------------------------------------------

def argmax_con_desempate(fila: list) -> int:
    """
    Mayor valor de la fila. Si varios empatan gana el intermedio menor del
    conjunto empatado, igual que en la hoja.
    """
    tope = max(fila)
    empatados = [d for d in range(N_CLASES) if fila[d] == tope]
    return empatados[(len(empatados) - 1) // 2]


def agregar(
    hojas,
    tabla: dict,
    offsets: array.array,
    n_img: int,
) -> array.array:
    """
    Predice cada imagen sumando las frecuencias de sus hojas.

    Para cada target se acumulan los conteos de todas las hojas que
    activa la imagen y gana el de mayor total. Todo el calculo es entero:
    no interviene ninguna probabilidad, asi que no puede haber error de
    redondeo en la decision. El porcentaje de la matriz es un calculo a
    posteriori sobre los recuentos.

    Devuelve una prediccion por imagen; -1 si la imagen no activo ninguna
    hoja con datos.
    """
    salida = array.array("i")

    for i in range(n_img):
        hist: dict[int, int] = {}
        for k in range(offsets[i], offsets[i + 1]):
            lf = hojas[k]
            hist[lf] = hist.get(lf, 0) + 1

        fila = [0] * N_CLASES
        for lf, repes in hist.items():
            dato = tabla.get(lf)
            if dato is None:
                continue
            _mode, _n_mode, _N, conteo = dato
            for d in range(N_CLASES):
                fila[d] += repes * conteo[d]

        if not any(v > 0 for v in fila):
            salida.append(-1)
        else:
            salida.append(argmax_con_desempate(fila))

    return salida

def suma_de_frecuencias(hojas, tabla: dict, offsets, n_img: int,
                        real) -> list[list[int]]:
    """
    Suma, por digito real y por target, las frecuencias acumuladas.

    M[r][c] = suma sobre las imagenes de etiqueta r de la frecuencia total
    del target c. Cada imagen aporta el vector COMPLETO de conteos de sus
    hojas, no solo el ganador, y sin normalizar.
    """
    m = [[0] * N_CLASES for _ in range(N_CLASES)]

    for i in range(n_img):
        r = real[i]
        if not 0 <= r < N_CLASES:
            continue

        hist: dict[int, int] = {}
        for k in range(offsets[i], offsets[i + 1]):
            lf = hojas[k]
            hist[lf] = hist.get(lf, 0) + 1

        fila = m[r]
        for lf, repes in hist.items():
            dato = tabla.get(lf)
            if dato is None:
                continue
            _mode, _n_mode, _N, conteo = dato
            for d in range(N_CLASES):
                fila[d] += repes * conteo[d]

    return m


def dividir_por_minimo(m: list[list[int]]) -> list[list[int]]:
    """
    Divide cada fila por su minimo, con division entera.

    El minimo de la fila queda en 1 y el resto son cocientes enteros, asi
    que la reduccion es mucho mayor que restando: se pasa de miles de
    millones a unidades. Pierde resolucion fina, pero deja ver de un
    golpe que columna domina en cada fila.
    """
    salida = []
    for fila in m:
        minimo = min(fila)
        if minimo <= 0:
            minimo = 1
        salida.append([v // minimo for v in fila])
    return salida


def matriz_confusion(pred: array.array, real: array.array) -> list[list[int]]:
    m = [[0] * N_CLASES for _ in range(N_CLASES)]
    for p, r in zip(pred, real):
        if 0 <= r < N_CLASES and 0 <= p < N_CLASES:
            m[r][p] += 1
    return m


def acierto(m: list[list[int]]) -> float:
    total = sum(sum(f) for f in m)
    return (sum(m[c][c] for c in range(N_CLASES)) / total) if total else 0.0


def imprimir_matriz(m: list[list[int]], sangria: str = "    ") -> None:
    """
    Matriz de confusion en porcentajes, calculados a posteriori sobre los
    recuentos. La cabecera y la separacion usan el mismo prefijo de 8
    caracteres que las filas ("real N: ") para que todo quede alineado.
    """
    ancho = 7
    print(sangria + "pred:   "
          + "".join(f"{c:>{ancho}}" for c in range(N_CLASES)))
    print(sangria + "-" * (8 + ancho * N_CLASES))
    for r in range(N_CLASES):
        total = sum(m[r])
        celdas = "".join(
            f"{100.0 * m[r][c] / total:>6.1f}%" if total else f"{'-':>7}"
            for c in range(N_CLASES)
        )
        print(f"{sangria}real {r}: {celdas}   (n={total})")


def imprimir_matriz_frecuencias(m: list[list[int]],
                                 sangria: str = "    ",
                                 minimo: int = 6,
                                 con_n: bool = True) -> None:
    """
    Recuentos enteros. El ancho de columna se ajusta al numero mayor, con
    `minimo` como suelo, para que no se desalinee nunca.
    """
    tope = 0
    for fila in m:
        for v in fila:
            if v > tope:
                tope = v
    ancho = max(len(str(tope)), minimo)

    print(sangria + "pred:   "
          + " ".join(f"{c:>{ancho}}" for c in range(N_CLASES)))
    print(sangria + "-" * (8 + ancho * N_CLASES + N_CLASES - 1))
    for r in range(N_CLASES):
        total = sum(m[r])
        celdas = " ".join(f"{m[r][c]:>{ancho}}" for c in range(N_CLASES))
        cola = f"   (n={total})" if con_n else ""
        print(f"{sangria}real {r}: {celdas}{cola}")

def imprimir_prueba(titulo: str, cabecera: str, m: list[list[int]]) -> None:
    print("-" * 78)
    print(titulo)
    print(f"  {cabecera}")
    print(f"  acierto global: {acierto(m):.2%}")
    print()
    imprimir_matriz(m)
    print()
    print("  recuentos enteros (la decision real del agregado):")
    imprimir_matriz_frecuencias(m)
    print()

def fraccion(texto: str) -> tuple[int, int]:
    fr = Fraction(texto)
    return fr.numerator, fr.denominator


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def resumen_puntos(etiquetas, offsets, n_img: int, nombre: str) -> None:
    """
    Cuantos puntos aporta cada digito al pasar a (x, y, color).

    "con fondo" cuenta los 784 pixeles de cada imagen; "solo tinta" es lo
    que realmente se genera, descartando el nivel 0.
    """
    imagenes = [0] * N_CLASES
    puntos = [0] * N_CLASES
    for i in range(n_img):
        d = etiquetas[i]
        if 0 <= d < N_CLASES:
            imagenes[d] += 1
            puntos[d] += offsets[i + 1] - offsets[i]

    print(f"  {nombre}: puntos por digito")
    print(f"    {'dig':>3} {'imagenes':>9} {'con fondo':>12} "
          f"{'solo tinta':>12} {'% tinta':>8}")
    tot_i = tot_c = tot_s = 0
    for d in range(N_CLASES):
        con_fondo = imagenes[d] * PIXELES
        pct = (100.0 * puntos[d] / con_fondo) if con_fondo else 0.0
        print(f"    {d:>3} {imagenes[d]:>9} {con_fondo:>12} "
              f"{puntos[d]:>12} {pct:>7.1f}%")
        tot_i += imagenes[d]
        tot_c += con_fondo
        tot_s += puntos[d]
    pct = (100.0 * tot_s / tot_c) if tot_c else 0.0
    print(f"    {'tot':>3} {tot_i:>9} {tot_c:>12} {tot_s:>12} {pct:>7.1f}%")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MNIST como nube de puntos (x, y, color) con ISODENDRON.",
    )
    parser.add_argument("--train-mnist", type=Path,
                        default=_mnist_default("mnist_train.csv"),
                        help="CSV de entrenamiento (o define MNIST_DIR)")
    parser.add_argument("--test-mnist", type=Path,
                        default=_mnist_default("mnist_test.csv"),
                        help="CSV de test (o define MNIST_DIR)")
    parser.add_argument("--n-train-img", type=int, default=0,
                        help="imagenes de train a usar (0 = todas)")
    parser.add_argument("--n-test-img", type=int, default=0,
                        help="imagenes de test a usar (0 = todas)")
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20240607)
    args = parser.parse_args(argv)

    if args.train_mnist is None or args.test_mnist is None:
        parser.error("indica --train-mnist y --test-mnist, o define MNIST_DIR")

    if args.hilos < 1:
        parser.error("--hilos debe ser >= 1")

    print("=" * 78)
    print("ISODENDRON SOBRE MNIST COMO NUBE DE PUNTOS (X, Y, COLOR)")
    print("=" * 78)
    print(f"shared library : {lib_path}")
    print("features       : x 0..27, y 0..27, color 0..3")
    print("niveles        : crudo <32 -> 0 (fondo, fuera), <64 -> 1, "
          "<128 -> 2, >=128 -> 3")
    print(f"umbrales       : {', '.join(P_VALORES)}   max_deep={MAX_DEEP} "
          f"(no limita)")
    print("agregacion     : por hoja "
          "(suma de conteos)")
    print()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    print("CARGA DE DATOS (una fila por pixel de tinta)")
    t0 = time.perf_counter()
    X_tr, y_tr, et_tr, off_tr, n_pt_tr, n_im_tr = cargar_csv_puntos(
        args.train_mnist, args.n_train_img, args.seed, "train")
    X_te, y_te, et_te, off_te, n_pt_te, n_im_te = cargar_csv_puntos(
        args.test_mnist, args.n_test_img, args.seed, "test")
    print(f"  carga total: {time.perf_counter() - t0:.1f} s")
    print()

    if n_pt_tr == 0 or n_pt_te == 0:
        raise SystemExit("ERROR: alguna de las dos nubes de puntos esta vacia")

    # ------------------------------------------------------------------
    resumen_puntos(et_tr, off_tr, n_im_tr, "train")
    resumen_puntos(et_te, off_te, n_im_te, "test")
    # Cuatro pruebas: un umbral cada una, sobre test
    # ------------------------------------------------------------------
    modelos: dict[str, bytes] = {}
    prueba = 0

    for valor in P_VALORES:
        num, den = fraccion(valor)
        prueba += 1

        print("=" * 78)
        print(f"ENTRENAMIENTO con umbral {valor} (P_MIN = {num}/{den})")
        print("=" * 78)

        t0 = time.perf_counter()
        blob = _ffi.train_mt(X_tr, y_tr, n_pt_tr, N_FEATURES,
                             MAX_DEEP, num, den, args.hilos)
        t_fit = time.perf_counter() - t0

        modelos[valor] = blob
        st = estadisticas_arbol(blob)
        tabla = tabla_de_hojas(blob)

        t0 = time.perf_counter()
        _pred, hojas_te = _ffi.predict_leaf(blob, X_te, n_pt_te, N_FEATURES)
        t_pred = time.perf_counter() - t0

        cabecera_base = (
            f"nodos={st['n_nodes']}  hojas={st['n_leaves']}  "
            f"prof.alc.={st['profundidad']}  fit={t_fit:.1f} s  "
            f"predict={t_pred:.1f} s  hojas con datos={len(tabla)}"
        )
        print(f"  {cabecera_base}")
        print()

        titulo = (f"PRUEBA {prueba} - umbral {valor} - pronostico por hoja "
                  f"(suma de conteos) sobre test")
        pred = agregar(list(hojas_te), tabla, off_te, n_im_te)
        m = matriz_confusion(pred, et_te)
        imprimir_prueba(titulo, cabecera_base, m)

    # ------------------------------------------------------------------
    # Novena prueba: refinamiento con test y evaluacion train + test
    # ------------------------------------------------------------------
    print("=" * 78)
    print("PRUEBA 9 - umbral 1.0 acumulando TEST, evaluado en train+test")
    print("=" * 78)
    print("  El arbol NO se reentrena: isodendron_partial_fit ACUMULA los")
    print("  hojas con los puntos de test. Aviso: las hojas se estiman con")
    print("  el mismo test sobre el que se evalua, asi que la parte de test")
    print("  de esta matriz es optimista por construccion.")
    print()

    blob_base = modelos["1.0"]
    tabla_antes = tabla_de_hojas(blob_base)

    t0 = time.perf_counter()
    blob_ref, diag_ref = _ffi.partial_fit_diag(blob_base, X_te, y_te, n_pt_te,
                                   N_FEATURES, MAX_TARGETS, capacidad=16)
    t_ref = time.perf_counter() - t0

    st_ref = estadisticas_arbol(blob_ref)
    tabla_ref = tabla_de_hojas(blob_ref)

    print(f"  refinamiento: {t_ref:.1f} s")
    print(f"  nodos={st_ref['n_nodes']}  hojas={st_ref['n_leaves']}")
    print(f"  hojas={diag_ref['hojas']}  decayadas={diag_ref['decayadas']}  "
          f"sin datos={diag_ref['vaciadas']}  "
          f"resta_maxima={diag_ref['resta_maxima']}  "
          f"con datos despues={len(tabla_ref)}")
    if diag_ref["vaciadas"] > 0 and diag_ref["detalle"]:
        m = diag_ref["detalle"][0]
        print(f"  AVISO: {diag_ref['vaciadas']} hojas sin datos; primera -> "
              f"nodo={m['nodo']} N_previo={m['N_previo']} "
              f"payload_len={m['payload_len']} resta={m['resta']} "
              f"N_final={m['N_final']}")
    print()

    t0 = time.perf_counter()
    _p1, hojas_tr = _ffi.predict_leaf(blob_ref, X_tr, n_pt_tr, N_FEATURES)
    _p2, hojas_te_r = _ffi.predict_leaf(blob_ref, X_te, n_pt_te, N_FEATURES)
    t_pred_r = time.perf_counter() - t0

    off_todo = array.array(
        "i", list(off_tr[:-1]) + [v + n_pt_tr for v in off_te])
    et_todo = array.array("i", list(et_tr) + list(et_te))
    hojas_todo = array.array("I", list(hojas_tr) + list(hojas_te_r))
    n_im_todo = n_im_tr + n_im_te

    pred = agregar(list(hojas_todo), tabla_ref, off_todo, n_im_todo)
    m = matriz_confusion(pred, et_todo)

    sin_evidencia = sum(1 for p in pred if p < 0)
    cabecera = (
        f"imagenes={n_im_todo} ({n_im_tr} train + {n_im_te} test)  "
        f"puntos={n_pt_tr + n_pt_te}  predict={t_pred_r:.1f} s  "
        f"sin evidencia={sin_evidencia}"
    )
    imprimir_prueba(
        "PRUEBA 9 - umbral 1.0 acumulando test - pronostico por hoja "
        "sobre train+test",
        cabecera, m)

    sumas = suma_de_frecuencias(list(hojas_todo), tabla_ref, off_todo,
                                n_im_todo, et_todo)
    print("  suma de frecuencias por digito real y target, SIN normalizar:")
    print("  (cada imagen aporta el vector completo de conteos de sus hojas)")
    imprimir_matriz_frecuencias(sumas, minimo=8, con_n=False)
    print()
    print("  dividida: fila / minimo, division entera (el minimo queda en 1)")
    imprimir_matriz_frecuencias(dividir_por_minimo(sumas), minimo=8,
                                 con_n=False)
    print()
    print()

    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(f"  umbrales {', '.join(P_VALORES)} con max_deep={MAX_DEEP}")
    print("  agregacion por hoja : suma de conteos por target")
    print("  prueba 9            : refinada con test, evaluada en train+test")
    print()
    print("RESULTADO: ejecucion completada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
