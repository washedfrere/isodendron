#!/usr/bin/env python3
"""
synthetic.py — Prueba de exactitud de ISODENDRON sobre datasets sintéticos.

Objetivo
--------
Medir hasta dónde llega el clasificador ISODENDRON (paquete isodendron) sobre problemas
cuya función objetivo es *conocida y determinista*, de forma que cada
punto de exactitud perdido sea atribuible al modelo y no al ruido.

Escenarios
----------
Cuatro features por fila, cada una un dígito entre 0 y 9, y un target
que es la composición de esos dígitos:

  A) dígitos uniformes     target = concatenación decimal de las features
  B) 2 dígitos             target = f0*10 + f1                      0..99
  C) control 1 dígito      target = f0                              0..9
  D) desbalance            target = patrón de bits (ver abajo)      0..2^n-1

Por qué A no puede funcionar y D sí
-----------------------------------
El split de ISODENDRON se elige SOLO por balance de tamaños:

    score(f, v) = min(n_igual, n_distinto)

maximizado sobre todas las features y todos sus valores. El target no
participa en la selección: solo se usa después para resumir cada nodo
con su moda.

  - En A, B y C las features son uniformes sobre 0..9, así que todo par
    (feature, valor) puntúa igual (~N/10) y el empate lo rompe el ruido
    de muestreo. Los splits elegidos no significan nada y la exactitud
    se queda cerca del azar, por mucho que se suba la profundidad.

  - En D cada feature f_i tiene un dígito *mayoritario* (el propio i)
    que aparece en la mitad de las filas, y el resto de los dígitos se
    reparte entre la otra mitad. Eso crea un split (f_i == i) que parte
    cualquier nodo ~50/50, que es exactamente lo que el criterio premia.
    El target es el patrón de aciertos de esos cuatro dígitos
    mayoritarios, así que la partición que ISODENDRON elige *es* la que
    determina el target. Este es el caso para el que el algoritmo está
    diseñado, y sirve de contraste frente a A.

Profundidades
-------------
Se prueban max_deep = 2, 5 y 7 (configurable con --depths).

Métricas por configuración
--------------------------
  - nº de nodos y de hojas del árbol resultante
  - exactitud exacta (pred == target) en train y en test
  - MAE (error absoluto medio) en train y en test
  - acierto por dígito: fracción de posiciones correctas, en la base del
    escenario (10 para A/B/C, 2 para D)

Nota metodológica
-----------------
La partición train/test (70/30 por defecto, semilla fija) es necesaria
porque el target es determinista: evaluar sobre las mismas muestras de
entrenamiento mediría memorización, no generalización.

Uso
---
    python synthetic.py
    python synthetic.py --n 100000 --seed 7 --depths 2 5 7 12
    python synthetic.py --dump

Este script usa únicamente la biblioteca estándar y el paquete isodendron.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Localización del proyecto
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

try:
    from isodendron import Isodendron
    from isodendron import IsodendronError
    from isodendron import __version__ as ISODENDRON_PY_VERSION
    from isodendron._lib import lib_path
except ImportError as exc:  # pragma: no cover - diagnóstico de entorno
    print("ERROR: no se pudo importar el paquete isodendron.", file=sys.stderr)
    print("  instala el proyecto con `pip install .` o `pip install -e .`",
          file=sys.stderr)
    print(f"  causa: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)


# ----------------------------------------------------------------------
# Generadores de features
# ----------------------------------------------------------------------

def generar_digitos_uniformes(rng: random.Random, n_features: int) -> list[int]:
    """Cada feature es un dígito 0..9 equiprobable (escenarios A, B y C)."""
    return [rng.randrange(10) for _ in range(n_features)]


def generar_digitos_sesgados(rng: random.Random, n_features: int) -> list[int]:
    """
    Escenario D: cada feature f_i toma su dígito mayoritario (i) la mitad
    de las veces y, en la otra mitad, cualquiera de los otros nueve
    dígitos con la misma probabilidad.

    El resultado es que existe un split (f_i == i) que parte cualquier
    nodo aproximadamente al 50%, mientras que cualquier otro valor de la
    misma feature solo separa ~5% de las filas. Eso hace que el criterio
    de balance de ISODENDRON elija justo ese split, y no otro.
    """
    fila: list[int] = []
    for i in range(n_features):
        mayoritario = i % 10
        if rng.random() < 0.5:
            fila.append(mayoritario)
        else:
            # Uniforme sobre los otros nueve dígitos.
            v = rng.randrange(9)
            if v >= mayoritario:
                v += 1
            fila.append(v)
    return fila


def generate_features(n: int, n_features: int, seed: int, gen) -> list[list[int]]:
    """Genera n filas con el generador dado, de forma reproducible."""
    rng = random.Random(seed)
    return [gen(rng, n_features) for _ in range(n)]


# ----------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------

def target_all_digits(row: list[int]) -> int:
    """
    Concatenación decimal de todas las features. Con 4 dígitos da
    f0*1000 + f1*100 + f2*10 + f3, en el rango 0..9999.
    """
    v = 0
    for d in row:
        v = v * 10 + d
    return v


def target_two_digits(row: list[int]) -> int:
    """f0*10 + f1  ->  0..99"""
    return row[0] * 10 + row[1]


def target_control(row: list[int]) -> int:
    """f0  ->  0..9"""
    return row[0]


def target_desbalance(row: list[int]) -> int:
    """
    Patrón de bits: el bit i vale 1 si f_i toma su dígito mayoritario.

    Para 4 features el rango es 0..15. La clave es que el bit i se
    determina exactamente con el split (f_i == i), que es el que el
    criterio de balance selecciona.
    """
    v = 0
    for i, x in enumerate(row):
        v = (v << 1) | (1 if x == (i % 10) else 0)
    return v


# ----------------------------------------------------------------------
# Definición de los escenarios
# ----------------------------------------------------------------------

#: `width` es el número de posiciones del target que se comparan en la
#: métrica de acierto por dígito; None significa "tantas como features".
#: `base` es la base en la que se descompone el target para esa métrica.
#: `min_features` es el mínimo de features que el escenario necesita.
SCENARIOS = [
    {
        "key": "A_digitos",
        "title": "A) digitos uniformes  target = concatenacion decimal de las features",
        "gen": generar_digitos_uniformes,
        "fn": target_all_digits,
        "width": None,
        "base": 10,
        "min_features": 1,
    },
    {
        "key": "B_2digitos",
        "title": "B) 2 digitos uniformes  target = f0*10 + f1",
        "gen": generar_digitos_uniformes,
        "fn": target_two_digits,
        "width": 2,
        "base": 10,
        "min_features": 2,
    },
    {
        "key": "C_control",
        "title": "C) control 1 digito     target = f0",
        "gen": generar_digitos_uniformes,
        "fn": target_control,
        "width": 1,
        "base": 10,
        "min_features": 1,
    },
    {
        "key": "D_desbalance",
        "title": "D) target correlacionado con el desbalance  (patron en base 2 "
                 "de los digitos mayoritarios)",
        "nota": "cada f_i vale su digito mayoritario i en la mitad de las filas; "
                "el split (f_i == i) parte el nodo al 50% y es el que el criterio "
                "de balance elige",
        "gen": generar_digitos_sesgados,
        "fn": target_desbalance,
        "width": None,
        "base": 2,
        "min_features": 1,
    },
]


def split_indices(n: int, test_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Partición train/test reproducible."""
    rng = random.Random(seed + 1)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_frac)))
    n_test = min(n_test, n - 1) if n > 1 else 1
    return idx[n_test:], idx[:n_test]


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------

def digits_of(value: int, width: int, base: int) -> list[int]:
    """
    Descompone un entero en sus `width` dígitos en la base dada, del
    menos significativo al más. Los valores negativos se toman en valor
    absoluto: solo se usa para comparar posiciones.
    """
    v = abs(int(value))
    out = [0] * width
    for i in range(width):
        out[i] = v % base
        v //= base
    return out


def evaluate(
    X: list[list[int]],
    y: list[int],
    train_idx: list[int],
    test_idx: list[int],
    width: int,
    base: int,
    depths: list[int],
    p_min: tuple[int, int],
) -> list[dict]:
    """Entrena y evalúa un árbol por cada profundidad pedida."""
    rows: list[dict] = []

    train_data = [(X[i], y[i]) for i in train_idx]
    test_data = [(X[i], y[i]) for i in test_idx]

    for depth in depths:
        model = Isodendron()

        t0 = time.perf_counter()
        model.fit(train_data, max_deep=depth, p_min=p_min)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_train = model.predict([X[i] for i in train_idx])
        pred_test = model.predict([X[i] for i in test_idx])
        predict_s = time.perf_counter() - t0

        n_leaves = len(model.dump())

        row = {
            "depth": depth,
            "n_nodes": model.n_nodes,
            "n_leaves": n_leaves,
            "fit_s": fit_s,
            "predict_s": predict_s,
        }
        row.update(_accuracy(pred_train, [y[i] for i in train_idx], width, base, "train"))
        row.update(_accuracy(pred_test, [y[i] for i in test_idx], width, base, "test"))
        rows.append(row)

    return rows


def _accuracy(pred: list[int], truth: list[int], width: int, base: int, prefix: str) -> dict:
    """Exactitud exacta, MAE y acierto por dígito."""
    n = len(truth)
    if n == 0:
        return {
            f"{prefix}_exact": 0.0,
            f"{prefix}_mae": 0.0,
            f"{prefix}_digit_acc": 0.0,
        }

    exact = 0
    abs_err = 0
    digits_ok = 0
    for p, t in zip(pred, truth):
        if p == t:
            exact += 1
        abs_err += abs(p - t)
        dp, dt = digits_of(p, width, base), digits_of(t, width, base)
        digits_ok += sum(1 for a, b in zip(dp, dt) if a == b)

    return {
        f"{prefix}_exact": exact / n,
        f"{prefix}_mae": abs_err / n,
        f"{prefix}_digit_acc": digits_ok / (n * width),
    }


# ----------------------------------------------------------------------
# Presentación
# ----------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    header = (
        f"{'depth':>5} {'nodes':>6} {'leaves':>7} "
        f"{'acc_tr':>8} {'acc_te':>8} "
        f"{'mae_tr':>8} {'mae_te':>8} "
        f"{'dig_tr':>7} {'dig_te':>7} "
        f"{'fit_s':>8} {'pred_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['depth']:>5} {r['n_nodes']:>6} {r['n_leaves']:>7} "
            f"{r['train_exact']:>8.2%} {r['test_exact']:>8.2%} "
            f"{r['train_mae']:>8.2f} {r['test_mae']:>8.2f} "
            f"{r['train_digit_acc']:>7.2%} {r['test_digit_acc']:>7.2%} "
            f"{r['fit_s']:>8.3f} {r['predict_s']:>8.3f}"
        )


def dump_example(model: Isodendron, max_leaves: int = 12) -> None:
    """Imprime las primeras hojas del árbol, como control de sanidad."""
    leaves = model.dump()
    print(f"    hojas: {len(leaves)} (mostrando {min(len(leaves), max_leaves)})")
    for leaf in leaves[:max_leaves]:
        conds = " AND ".join(
            f"{c['feature']}{c['op']}{c['value']}" for c in leaf["conditions"]
        ) or "(raiz)"
        print(
            f"      {conds:<46} N={leaf['N']:<6} K={leaf['K']:<5} "
            f"mode={leaf['mode']:<8} P={leaf['P']:.3f}"
        )


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
        description="Prueba de exactitud de ISODENDRON sobre datasets sintéticos.",
    )
    parser.add_argument("--n", type=int, default=20000,
                        help="nº total de muestras (por defecto 20000)")
    parser.add_argument("--seed", type=int, default=20240607,
                        help="semilla del generador (por defecto 20240607)")
    parser.add_argument("--test-frac", type=float, default=0.30,
                        help="fracción de test (por defecto 0.30)")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 5, 7],
                        help="profundidades a evaluar (por defecto 2 5 7)")
    parser.add_argument("--p-min", type=int, nargs=2, default=[1, 1],
                        metavar=("NUM", "DEN"),
                        help="pureza mínima de la moda, P_MIN = NUM/DEN "
                             "(por defecto 1 1: solo parar en nodos puros)")
    parser.add_argument("--n-features", type=int, default=4,
                        help="nº de features (por defecto 4)")
    parser.add_argument("--csv", type=Path,
                        default=RESULTS_DIR / "synthetic.csv",
                        help="fichero CSV de resultados")
    parser.add_argument("--dump", action="store_true",
                        help="volcar las hojas del árbol de menor profundidad")
    args = parser.parse_args(argv)

    depths = sorted(set(args.depths))
    p_min = (args.p_min[0], args.p_min[1])

    if args.n < 2:
        parser.error("--n debe ser >= 2")
    if not (0.0 < args.test_frac < 1.0):
        parser.error("--test-frac debe estar estrictamente entre 0 y 1")
    if args.n_features < 1:
        parser.error("--n-features debe ser >= 1")

    print("=" * 78)
    print("PRUEBA DE EXACTITUD DE ISODENDRON")
    print("=" * 78)
    print(f"shared library : {lib_path}")
    from isodendron._ffi import version as _version  # noqa: E402
    print(f"isodendron C        : {_version()}   (binding isodendron {ISODENDRON_PY_VERSION})")
    print(f"python         : {sys.version.split()[0]} ({sys.executable})")
    print(f"muestras       : {args.n}  (train/test = "
          f"{1 - args.test_frac:.0%}/{args.test_frac:.0%}, semilla {args.seed})")
    print(f"features       : {args.n_features} digitos 0..9")
    print(f"profundidades  : {depths}   "
          f"P_MIN = {_fmt_p_min(p_min[0], p_min[1])}")
    print()

    train_idx, test_idx = split_indices(args.n, args.test_frac, args.seed)
    print(f"particion      : {len(train_idx)} train / {len(test_idx)} test")
    print()

    all_rows: list[dict] = []
    resultados: dict[tuple[str, int], dict] = {}

    escenarios = [s for s in SCENARIOS if args.n_features >= s["min_features"]]
    for scen in SCENARIOS:
        if scen not in escenarios:
            print(f"(escenario {scen['key']} omitido: necesita al menos "
                  f"{scen['min_features']} features y hay {args.n_features})")
            print()

    for scen in escenarios:
        width = scen["width"] if scen["width"] is not None else args.n_features
        base = scen["base"]

        X = generate_features(args.n, args.n_features, args.seed, scen["gen"])
        y = [scen["fn"](row) for row in X]

        y_min, y_max = min(y), max(y)
        print("-" * 78)
        print(scen["title"])
        print(f"   rango observado del target: {y_min}..{y_max}   "
              f"clases distintas: {len(set(y))}")
        print(f"   acierto por digito en base {base}, ancho {width}")
        if scen.get("nota"):
            print(f"   nota: {scen['nota']}")
        print("-" * 78)

        try:
            rows = evaluate(X, y, train_idx, test_idx, width, base, depths, p_min)
        except IsodendronError as exc:
            print(f"   ERROR de la biblioteca C: {exc} (codigo {exc.code})")
            return 2

        print_table(rows)

        if args.dump:
            best = Isodendron().fit(
                [(X[i], y[i]) for i in train_idx],
                max_deep=depths[0],
                p_min=p_min,
            )
            print(f"   arbol con max_deep={depths[0]}:")
            dump_example(best)

        print()
        for r in rows:
            r_out = {"scenario": scen["key"], "width": width, "base": base,
                     "n_train": len(train_idx), "n_test": len(test_idx),
                     "p_min_num": p_min[0], "p_min_den": p_min[1],
                     "n_features": args.n_features, "seed": args.seed}
            r_out.update(r)
            all_rows.append(r_out)
            resultados[(scen["key"], r["depth"])] = r

    # ------------------------------------------------------------------
    # Interpretación
    # ------------------------------------------------------------------
    print("=" * 78)
    print("LECTURA DE LOS RESULTADOS")
    print("=" * 78)
    nf = args.n_features
    lineas = [
        "A, B y C usan features uniformes: todos los pares (feature, valor)",
        "puntuan igual en el criterio de balance, asi que los splits los",
        "decide el ruido de muestreo y el arbol no aprende. Subir la",
        "profundidad no arregla nada porque el target nunca entra en la",
        "seleccion del split.",
        "",
        "D es el caso opuesto: el target esta definido precisamente por el",
        "split mas balanceado de cada feature, que es el que ISODENDRON elige.",
        f"Con {nf} features hacen falta {nf} divisiones para fijar los {nf} bits,",
        f"asi que a partir de max_deep >= {nf} la recuperacion es exacta.",
    ]
    if min(depths) < nf:
        lineas.append(
            f"Con max_deep = {min(depths)} solo se fijan {min(depths)} de los "
            f"{nf} bits, y de ahi la exactitud parcial del primer nivel."
        )
    lineas += [
        "",
        "Un detalle que se ve en la tabla: pasado ese umbral el arbol sigue",
        "creciendo (mas nodos, mas hojas) sin ganar ni un punto de exactitud.",
        "ISODENDRON no trata un nodo con target constante como hoja, asi que sigue",
        "dividiendo dentro de regiones que ya estan decididas.",
        "",
        "La comparacion A vs D aisla la propiedad que realmente gobierna el",
        "comportamiento: el target tiene que correlacionar con el balance de",
        "tamanos de un par (feature, valor).",
    ]
    print("\n".join(lineas))
    print()

    # ------------------------------------------------------------------
    # Comprobaciones de las expectativas estructurales
    # ------------------------------------------------------------------
    def _r(key: str, depth: int) -> dict | None:
        return resultados.get((key, depth))

    comprobaciones: list[tuple[str, bool, str]] = []

    # ------------------------------------------------------------------
    # Comprobaciones del binding de Python (capa Isodendron)
    # ------------------------------------------------------------------
    datos_bind = [([i % 4, (i * 7) % 5, i // 4], i % 4) for i in range(64)]
    m_bind = Isodendron().fit(datos_bind, max_deep=4, p_min=(1, 1))
    muestras = [muestra for muestra, _t in datos_bind]

    p1 = m_bind.predict(muestras)
    p2 = [m_bind.predict_one(s) for s in muestras]
    comprobaciones.append((
        "binding: predict_one coincide con predict",
        p1 == p2, f"{len(p1)} muestras",
    ))

    completo = m_bind.predict_full(muestras)
    ok_p = all(
        0.0 < p <= 1.0 and 1 <= n_mode <= n_hoja
        and abs(p - n_mode / n_hoja) < 1e-12
        for _m, p, n_mode, n_hoja in completo
    )
    comprobaciones.append((
        "binding: predict_full devuelve P = n_mode/N coherente",
        ok_p, f"{len(completo)} filas",
    ))
    comprobaciones.append((
        "binding: predict_full y predict dan la misma moda",
        [c[0] for c in completo] == p1, "",
    ))

    hojas = len(m_bind.dump())
    comprobaciones.append((
        "binding: n_nodes = 2*hojas - 1",
        m_bind.n_nodes == 2 * hojas - 1,
        f"nodos={m_bind.n_nodes} hojas={hojas}",
    ))

    try:
        Isodendron().predict(muestras)
        fallo_sin_fit = False
    except RuntimeError:
        fallo_sin_fit = True
    comprobaciones.append((
        "binding: predecir sin entrenar lanza RuntimeError", fallo_sin_fit, "",
    ))

    try:
        Isodendron().fit(datos_bind, max_deep=4, p_min=(0, 1))
        fallo_pmin = False
    except ValueError:
        fallo_pmin = True
    comprobaciones.append((
        "binding: p_min con numerador 0 lanza ValueError", fallo_pmin, "",
    ))

    try:
        Isodendron().fit(datos_bind, max_deep=4, p_min=(1, 0))
        fallo_den = False
    except ValueError:
        fallo_den = True
    comprobaciones.append((
        "binding: p_min con denominador 0 lanza ValueError", fallo_den, "",
    ))

    m_sin = Isodendron().fit(datos_bind, max_deep=4, p_min=(1, 1), max_targets=0)
    comprobaciones.append((
        "binding: fit(max_targets=0) desactiva el payload",
        len(m_sin.blob) < len(m_bind.blob),
        f"{len(m_sin.blob)} < {len(m_bind.blob)} bytes",
    ))

    pred_l, hojas_l = m_bind.predict_leaf(muestras)
    comprobaciones.append((
        "binding: predict_leaf da la misma moda y una hoja valida",
        pred_l == p1 and all(0 <= h < m_bind.n_nodes for h in hojas_l),
        f"{len(hojas_l)} hojas",
    ))

    m_acum = m_bind.partial_fit(datos_bind)
    comprobaciones.append((
        "binding: partial_fit devuelve un modelo nuevo con el mismo arbol",
        m_acum is not m_bind and m_acum.n_nodes == m_bind.n_nodes,
        f"nodos={m_acum.n_nodes}",
    ))

    _m_acum2, info = m_bind.partial_fit_diag(datos_bind, capacidad=8)
    comprobaciones.append((
        "binding: partial_fit_diag informa y no vacia hojas",
        info["vaciadas"] == 0 and info["hojas"] == hojas,
        f"hojas={info['hojas']} vaciadas={info['vaciadas']}",
    ))

    d_max = _r("D_desbalance", max(depths))
    if d_max is not None:
        if max(depths) >= args.n_features:
            comprobaciones.append((
                f"D con max_deep={max(depths)} >= {args.n_features} features "
                f"recupera el target exacto",
                d_max["test_exact"] >= 1.0,
                f"acc_te={d_max['test_exact']:.2%}",
            ))
        else:
            comprobaciones.append((
                f"D con max_deep={max(depths)} < {args.n_features} features "
                f"no puede ser exacto",
                d_max["test_exact"] < 1.0,
                f"acc_te={d_max['test_exact']:.2%}",
            ))

    d_min = _r("D_desbalance", min(depths))
    if d_min is not None and min(depths) < args.n_features:
        comprobaciones.append((
            f"D con max_deep={min(depths)} < {args.n_features} features "
            f"queda por debajo de la recuperacion total",
            d_min["test_exact"] < 1.0,
            f"acc_te={d_min['test_exact']:.2%}",
        ))

    # Los umbrales frente al azar solo se comprueban con muestra suficiente.
    a_max = _r("A_digitos", max(depths))
    if args.n >= 2000 and a_max is not None and d_max is not None:
        comprobaciones.append((
            "A y D se separan en el mismo presupuesto de profundidad",
            d_max["test_exact"] > a_max["test_exact"] + 0.10,
            f"A={a_max['test_exact']:.2%} vs D={d_max['test_exact']:.2%}",
        ))

    c_max = _r("C_control", max(depths))
    if args.n >= 2000 and c_max is not None:
        comprobaciones.append((
            "C supera el azar de 10 clases (> 10%)",
            c_max["test_exact"] > 0.10,
            f"acc_te={c_max['test_exact']:.2%}",
        ))

    print("-" * 78)
    print("COMPROBACIONES")
    print("-" * 78)
    fallos = 0
    for descripcion, ok, detalle in comprobaciones:
        if ok:
            print(f"  ok    {descripcion}  [{detalle}]")
        else:
            fallos += 1
            print(f"  FALLA {descripcion}  [{detalle}]")
    print()

    if all_rows:
        campos = [
            "scenario", "width", "base", "depth", "n_nodes", "n_leaves",
            "train_exact", "test_exact", "train_mae", "test_mae",
            "train_digit_acc", "test_digit_acc", "fit_s", "predict_s",
            "n_train", "n_test", "n_features", "p_min_num", "p_min_den",
            "seed",
        ]
        try:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"resultados escritos en: {args.csv}")
        except OSError as exc:
            print(f"aviso: no se pudo escribir {args.csv}: {exc}")

    if fallos:
        print(f"\nRESULTADO: {fallos} comprobacion(es) fallidas")
        return 3

    print("\nRESULTADO: todas las comprobaciones OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
