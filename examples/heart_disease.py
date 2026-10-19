#!/usr/bin/env python3
"""
heart_disease.py - Clasificacion binaria sobre el dataset de
enfermedad cardiaca, con el binding de Python de ISODENDRON.

Datos
-----
CSV separado por comas en `examples/data/heart.csv`, dentro del propio
repositorio, con 918 filas y 12 columnas: 11 predictores (categoricos y
numericos) y el target `HeartDisease` (1 = enfermedad, 0 = normal).

El dataset viaja con el repositorio para que la prueba sea autocontenida.
MNIST no lo hace por tamano (104 MB de entrenamiento), asi que sus scripts
lo leen de una ruta externa que se pasa por argumento.

Categorizacion
--------------
Las convenciones salen de `examples/data/metadata_heart_disease.txt`,
que viaja junto al dataset:

    Age            0..10 por decadas (28 -> 2, 77 -> 7); <0 -> 99
    Sex            1=M, 2=F, 0=otro
    ChestPainType  1=TA, 2=ATA, 3=NAP, 4=ASY, 0=otro
    RestingBP      1..5 en quintiles de min y max; 0 = nulo
    Cholesterol    1..5 en quintiles; 0 = nulo
    FastingBS      1 = 1, 0 = 0
    RestingECG     1=Normal, 2=ST, 3=ST-T, 4=T, 6=LVH, 0=otro
    MaxHR          0 = <60, 6..19 por decadas, 20 = >=200, 99 = otro
    ExerciseAngina 1=Y, 2=N, 0=otro
    Oldpeak        1..5 en quintiles; 0 = nulo
    ST_Slope       1=Up, 2=Flat, 3=Down, 0=otro
    HeartDisease   1 / 0

Tres decisiones que la metadata no fija y que hay que dejar escritas:

1. **Quintiles sobre train.** Los bordes de min y max se calculan con las
   filas de entrenamiento, no con el dataset completo, para no filtrar la
   distribucion de test. Los valores de test fuera de rango se recortan al
   primer y ultimo quintil.
2. **Ceros centinela.** En `RestingBP` y `Cholesterol` un 0 no es una
   medida: es el centinela de dato ausente de este dataset (172 casos en
   colesterol). Se mapean al codigo 0, que la metadata reserva para nulo.
   En `Oldpeak` un 0 si es una medida real (sin depresion del ST), asi que
   se quintiliza como cualquier otro valor.
3. **RestingECG tiene 2=ST y 5=ST en la metadata**, que es una errata del
   original. Los valores reales del CSV son Normal, ST y LVH, y se mapean
   a 1, 2 y 6.

Particion
---------
80% train y 20% test, aleatoria con semilla, sin solapamiento.

Uso
---
    python heart_disease.py
    python heart_disease.py --max-deep 10 --p-min 1/1 --seed 7
    python heart_disease.py --test-frac 0.3 --hilos 4
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------
# Localización del proyecto
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METADATA = DATA_DIR / "metadata_heart_disease.txt"

N_CLASES = 2

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
# Convenciones de categorizacion
# ----------------------------------------------------------------------

PREDICTORES = (
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak",
    "ST_Slope",
)
TARGET = "HeartDisease"

#: Columnas que se quintilizan.
QUINTILES = ("RestingBP", "Cholesterol", "Oldpeak")

#: Columnas en las que un 0 es el centinela de dato ausente.
CERO_ES_NULO = ("RestingBP", "Cholesterol")

SEXO = {"M": 1, "F": 2}
DOLOR = {"TA": 1, "ATA": 2, "NAP": 3, "ASY": 4}
ECG = {"Normal": 1, "ST": 2, "ST-T": 3, "T": 4, "LVH": 6}
ANGINA = {"Y": 1, "N": 2}
PENDIENTE = {"Up": 1, "Flat": 2, "Down": 3}


def _num(texto: str) -> float | None:
    """Convierte a float, o None si no es un numero."""
    try:
        return float(texto.strip())
    except (AttributeError, ValueError):
        return None


# ----------------------------------------------------------------------
# Lectura y categorizacion
# ----------------------------------------------------------------------

def leer_filas(ruta: Path) -> list[dict[str, str]]:
    if not ruta.is_file():
        raise SystemExit(f"ERROR: no existe {ruta}")
    with ruta.open(newline="", encoding="utf-8") as fh:
        lector = csv.DictReader(fh)
        cabecera = lector.fieldnames or []
        faltan = [c for c in PREDICTORES + (TARGET,) if c not in cabecera]
        if faltan:
            raise SystemExit(f"ERROR: faltan columnas en {ruta}: {faltan}")
        filas = list(lector)
    if not filas:
        raise SystemExit(f"ERROR: {ruta} no tiene filas")
    return filas


def bordes_quintiles(
    filas: list[dict[str, str]],
    indices: list[int],
    columna: str,
) -> tuple[float, float]:
    """Minimo y maximo de la columna en las filas indicadas (train)."""
    vals: list[float] = []
    for i in indices:
        v = _num(filas[i][columna])
        if v is None:
            continue
        if columna in CERO_ES_NULO and v == 0.0:
            continue
        vals.append(v)
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def categorizar(
    columna: str,
    texto: str,
    bordes: dict[str, tuple[float, float]],
) -> int:
    t = (texto or "").strip()

    if columna == "Age":
        v = _num(t)
        if v is None or v < 0:
            return 99
        return int(min(v // 10.0, 10.0))

    if columna == "Sex":
        return SEXO.get(t, 0)

    if columna == "ChestPainType":
        return DOLOR.get(t, 0)

    if columna == "FastingBS":
        return 1 if t == "1" else 0

    if columna == "RestingECG":
        return ECG.get(t, 0)

    if columna == "ExerciseAngina":
        return ANGINA.get(t, 0)

    if columna == "ST_Slope":
        return PENDIENTE.get(t, 0)

    if columna == "MaxHR":
        v = _num(t)
        if v is None:
            return 99
        if v < 60:
            return 0
        if v >= 200:
            return 20
        return int(v // 10.0)

    # Quintiles.
    v = _num(t)
    if v is None:
        return 0
    if columna in CERO_ES_NULO and v == 0.0:
        return 0
    lo, hi = bordes[columna]
    if v <= lo:
        return 1
    if v >= hi:
        return 5
    return 1 + int((v - lo) / (hi - lo) * 5.0)


def construir(filas, indices, bordes):
    """Devuelve (X, y) para las filas indicadas."""
    X: list[list[int]] = []
    y: list[int] = []
    for i in indices:
        fila = filas[i]
        X.append([categorizar(c, fila[c], bordes) for c in PREDICTORES])
        y.append(int(fila[TARGET]))
    return X, y


# ----------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------

def matriz_confusion(pred, real) -> list[list[int]]:
    m = [[0] * N_CLASES for _ in range(N_CLASES)]
    for p, r in zip(pred, real):
        if 0 <= r < N_CLASES and 0 <= p < N_CLASES:
            m[r][p] += 1
    return m


def exactitud(pred, real) -> float:
    if not real:
        return 0.0
    return sum(1 for p, r in zip(pred, real) if p == r) / len(real)


def balanced_accuracy(m) -> float:
    recalls = [m[c][c] / sum(m[c]) for c in range(N_CLASES) if sum(m[c])]
    return statistics.fmean(recalls) if recalls else 0.0


def kappa(m) -> float:
    n = sum(sum(f) for f in m)
    if n == 0:
        return 0.0
    po = sum(m[c][c] for c in range(N_CLASES)) / n
    filas = [sum(m[c]) for c in range(N_CLASES)]
    cols = [sum(m[r][c] for r in range(N_CLASES)) for c in range(N_CLASES)]
    pe = sum(filas[c] * cols[c] for c in range(N_CLASES)) / (n * n)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def por_clase(m) -> list[dict[str, float]]:
    salida = []
    for c in range(N_CLASES):
        tp = m[c][c]
        fp = sum(m[r][c] for r in range(N_CLASES)) - tp
        fn = sum(m[c]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
        salida.append({"precision": prec, "recall": rec, "f1": f1,
                       "soporte": sum(m[c])})
    return salida


def imprimir_matriz(m, titulo: str) -> None:
    print(f"  {titulo}")
    print(f"    {'real\\pred':>10} {'0':>8} {'1':>8}")
    for r in range(N_CLASES):
        celdas = "".join(f"{m[r][c]:>8}" for c in range(N_CLASES))
        print(f"    {r:>10}{celdas}   (n={sum(m[r])})")
    print()


def imprimir_matriz_pct(m, titulo: str) -> None:
    print(f"  {titulo}")
    print(f"    {'real\\pred':>10} {'0':>8} {'1':>8}")
    for r in range(N_CLASES):
        total = sum(m[r])
        celdas = "".join(
            f"{100.0 * m[r][c] / total:>7.1f}%" if total else f"{'-':>8}"
            for c in range(N_CLASES)
        )
        print(f"    {r:>10}{celdas}")
    print()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def _recorta(texto: str, ancho: int) -> str:
    t = (texto or "").strip()
    return t if len(t) <= ancho else t[:ancho]


def mostrar_categorizacion(filas, idx_train, bordes) -> None:
    """
    Muestra como se ha codificado cada columna, con los valores invalidos.

    Los recuentos son sobre las filas de entrenamiento. Los bordes de los
    quintiles se calculan con esas mismas filas, segun la metadata.
    """
    print("CATEGORIZACION")
    print("  convenciones de metadata_heart_disease.txt; recuentos sobre train")
    print()

    def contar(columna, fn) -> dict[int, int]:
        c: dict[int, int] = {}
        for i in idx_train:
            cod = fn(filas[i][columna])
            c[cod] = c.get(cod, 0) + 1
        return c

    print("  tabla directa:")
    for columna, mapa in (
        ("Sex", SEXO),
        ("ChestPainType", DOLOR),
        ("RestingECG", ECG),
        ("ExerciseAngina", ANGINA),
        ("ST_Slope", PENDIENTE),
    ):
        inv = ", ".join(f"{k}={v}" for k, v in
                        sorted(mapa.items(), key=lambda x: x[1]))
        c = contar(columna, lambda t, m=mapa: m.get((t or "").strip(), 0))
        print(f"    {columna:<15} {inv}")
        print(f"    {'':<15} codigo 0 (fuera de la tabla): {c.get(0, 0)}")

    c = contar("FastingBS", lambda t: 1 if (t or "").strip() == "1" else 0)
    print(f"    {'FastingBS':<15} 1=1, 0=0   "
          f"(codigo 1: {c.get(1, 0)}, codigo 0: {c.get(0, 0)})")
    print()

    print("  por decadas:")
    c = contar("Age", lambda t: categorizar("Age", t, bordes))
    print(f"    {'Age':<15} 0=[0,10) ... 10=[100,+inf), 99=otro")
    print(f"    {'':<15} " + ", ".join(f"{k}:{v}" for k, v in sorted(c.items())))

    c = contar("MaxHR", lambda t: categorizar("MaxHR", t, bordes))
    print(f"    {'MaxHR':<15} 0=[<60), 6=[60,70) ... 19=[190,200), "
          f"20=[200,+inf), 99=otro")
    print(f"    {'':<15} " + ", ".join(f"{k}:{v}" for k, v in sorted(c.items())))
    print()

    print("  por quintiles (bordes calculados con train):")
    for columna in QUINTILES:
        lo, hi = bordes[columna]
        ancho = (hi - lo) / 5.0
        tramos = ", ".join(
            f"{q + 1}=[{lo + q * ancho:g}, {lo + (q + 1) * ancho:g}]"
            for q in range(5)
        )
        c = contar(columna, lambda t, col=columna: categorizar(col, t, bordes))
        nota = ", 0 es centinela" if columna in CERO_ES_NULO else ""
        print(f"    {columna:<15} {tramos}")
        print(f"    {'':<15} codigo 0 (nulo{nota}): {c.get(0, 0)} "
              f"de {sum(c.values())}")
    print()


def imprimir_muestras(filas, idx_test, y_test, pred,
                      n_por_celda: int = 2, semilla: int = 0) -> None:
    """Muestra n predicciones de test por cada celda de la matriz."""
    azar = random.Random(semilla)
    cabecera = ("Age", "Sex", "CPType", "BP", "Chol", "FBS", "ECG",
                "MaxHR", "ExAng", "Oldpk", "Slope")
    anchos = (3, 3, 6, 4, 4, 3, 7, 5, 5, 6, 5)

    print("MUESTRAS DE TEST")
    print("  todas las columnas originales, con el target esperado y el "
          "obtenido")
    print()

    for esperado, obtenido, nombre in (
        (1, 1, "verdadero positivo"),
        (1, 0, "falso negativo"),
        (0, 0, "verdadero negativo"),
        (0, 1, "falso positivo"),
    ):
        candidatos = [k for k in range(len(idx_test))
                      if y_test[k] == esperado and pred[k] == obtenido]
        print(f"  esperado={esperado} obtenido={obtenido}  ({nombre})   "
              f"{len(candidatos)} casos en test")
        if not candidatos:
            print("    (ninguno)")
            print()
            continue

        elegidos = azar.sample(candidatos, min(n_por_celda, len(candidatos)))
        print("    " + " ".join(f"{c:>{a}}"
                                for c, a in zip(cabecera, anchos)))
        for k in sorted(elegidos):
            fila = filas[idx_test[k]]
            vals = [_recorta(fila[c], a)
                    for c, a in zip(PREDICTORES, anchos)]
            print("    " + " ".join(f"{v:>{a}}"
                                    for v, a in zip(vals, anchos)))
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clasificacion binaria de enfermedad cardiaca con "
                    "ISODENDRON.",
    )
    parser.add_argument("--csv", type=Path,
                        default=DATA_DIR / "heart.csv",
                        help="ruta del dataset")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="fraccion de test (por defecto 0.2)")
    parser.add_argument("--seed", type=int, default=19710624)
    parser.add_argument("--max-deep", type=int, default=9)
    parser.add_argument("--p-min", default="1/1", metavar="NUM/DEN")
    parser.add_argument("--hilos", type=int, default=1)
    parser.add_argument("--modelo", type=Path, default=None,
                        help="donde guardar el .isod (por defecto, un "
                             "temporal que se borra al terminar)")
    parser.add_argument("--overwrite", action="store_true",
                        help="permite reemplazar el fichero de modelo si "
                             "ya existe")
    args = parser.parse_args(argv)

    if not 0.0 < args.test_frac < 1.0:
        parser.error("--test-frac debe estar entre 0 y 1")
    if args.max_deep < 1:
        parser.error("--max-deep debe ser >= 1")
    if args.hilos < 1:
        parser.error("--hilos debe ser >= 1")

    try:
        num, den = (int(x) for x in args.p_min.split("/"))
    except (ValueError, TypeError):
        parser.error("--p-min debe tener la forma NUM/DEN (por ejemplo 1/1)")
    if num < 1 or den < 1:
        parser.error("--p-min requiere numerador y denominador >= 1")

    print("=" * 72)
    print("ISODENDRON - ENFERMEDAD CARDIACA")
    print("=" * 72)
    print(f"shared library : {lib_path}")
    print(f"binding        : isodendron {ISODENDRON_PY_VERSION}")
    print(f"dataset        : {args.csv}")
    print(f"metadata       : {METADATA}")
    print()

    # ------------------------------------------------------------------
    # Carga y particion
    # ------------------------------------------------------------------
    filas = leer_filas(args.csv)
    n = len(filas)

    azar = random.Random(args.seed)
    indices = list(range(n))
    azar.shuffle(indices)
    n_test = max(1, int(round(n * args.test_frac)))
    idx_test = sorted(indices[:n_test])
    idx_train = sorted(indices[n_test:])

    if set(idx_train) & set(idx_test):
        raise SystemExit("ERROR: train y test se solapan")
    if len(idx_train) + len(idx_test) != n:
        raise SystemExit("ERROR: train y test no cubren el dataset")

    # Bordes de quintiles calculados SOLO con train.
    bordes = {c: bordes_quintiles(filas, idx_train, c) for c in QUINTILES}

    X_train, y_train = construir(filas, idx_train, bordes)
    X_test, y_test = construir(filas, idx_test, bordes)

    print("PARTICION")
    print(f"  total          : {n} filas")
    print(f"  train          : {len(X_train)} ({len(X_train) / n:.1%})")
    print(f"  test           : {len(X_test)} ({len(X_test) / n:.1%})   "
          f"(semilla {args.seed})")
    print("  bordes de quintil (calculados con train):")
    for c in QUINTILES:
        lo, hi = bordes[c]
        print(f"    {c:<12} [{lo:g}, {hi:g}]")
    reparto = [sum(1 for v in y_train if v == c) for c in range(N_CLASES)]
    print(f"  reparto en train: clase 0 = {reparto[0]}, clase 1 = {reparto[1]}")
    print()

    mostrar_categorizacion(filas, idx_train, bordes)

    # ------------------------------------------------------------------
    # Entrenamiento y prediccion
    # ------------------------------------------------------------------
    print("ENTRENAMIENTO")
    datos_train = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    modelo = Isodendron()
    try:
        modelo.fit(datos_train, max_deep=args.max_deep, p_min=(num, den),
                   hilos=args.hilos, feature_names=PREDICTORES)
    except IsodendronError as exc:
        print(f"  ERROR de la biblioteca: {exc} (codigo {exc.code})")
        return 2

    hojas = modelo.dump()
    print(f"  nodos={modelo.n_nodes}  hojas={len(hojas)}  "
          f"max_deep={args.max_deep}  P_MIN={num}/{den}")
    print()

    pred = modelo.predict(X_test)
    pred_full = modelo.predict_full(X_test)
    m = matriz_confusion(pred, y_test)
    acc = exactitud(pred, y_test)

    print("PREDICCION")
    print(f"  muestras de test : {len(pred)}")
    print(f"  exactitud        : {acc:.2%}")
    print()

    # ------------------------------------------------------------------
    # Ocho muestras: dos por celda de la matriz
    # ------------------------------------------------------------------
    imprimir_muestras(filas, idx_test, y_test, pred, semilla=args.seed)

    # ------------------------------------------------------------------
    # Matriz de confusion con sus porcentajes
    # ------------------------------------------------------------------
    print("=" * 72)
    print("MATRIZ DE CONFUSION")
    print("=" * 72)
    imprimir_matriz(m, "Recuentos:")
    imprimir_matriz_pct(m, "Porcentaje por fila:")

    bal = balanced_accuracy(m)
    kap = kappa(m)
    clases = por_clase(m)

    print("=" * 72)
    print("METRICAS")
    print("=" * 72)
    print(f"  {'clase':<8} {'precision':>10} {'recall':>9} {'f1':>9} "
          f"{'soporte':>9}")
    print("  " + "-" * 48)
    for c in range(N_CLASES):
        d = clases[c]
        print(f"  {c:<8} {d['precision']:>10.2%} {d['recall']:>9.2%} "
              f"{d['f1']:>9.2%} {int(d['soporte']):>9}")
    macro_p = statistics.fmean(d["precision"] for d in clases)
    macro_r = statistics.fmean(d["recall"] for d in clases)
    macro_f = statistics.fmean(d["f1"] for d in clases)
    print("  " + "-" * 48)
    print(f"  {'macro':<8} {macro_p:>10.2%} {macro_r:>9.2%} "
          f"{macro_f:>9.2%} {len(y_test):>9}")
    print()

    print(f"  exactitud (accuracy)  : {acc:.2%}")
    print(f"  balanced accuracy     : {bal:.2%}")
    print(f"  kappa de Cohen        : {kap:.4f}")
    print()
    # ------------------------------------------------------------------
    # Comprobaciones
    # ------------------------------------------------------------------
    mayoritaria = max(range(N_CLASES),
                      key=lambda c: sum(1 for v in y_train if v == c))
    acc_trivial = sum(1 for v in y_test if v == mayoritaria) / len(y_test)
    etiquetas = sorted(set(y_test))
    conf_min = min(p for _m, p, _nm, _N in pred_full) if pred_full else 0.0
    conf_max = max(p for _m, p, _nm, _N in pred_full) if pred_full else 0.0
    coherencia = all(
        abs(p - nm / N) < 1e-12 and 1 <= nm <= N
        for _m, p, nm, N in pred_full
    )

    comprobaciones: list[tuple[str, bool, str]] = [
        ("train y test son disjuntos y cubren el dataset",
         not (set(idx_train) & set(idx_test)) and
         len(idx_train) + len(idx_test) == n,
         f"{len(idx_train)} + {len(idx_test)} = {n}"),
        ("el target solo toma los valores 0 y 1",
         etiquetas == [0, 1] or etiquetas in ([0], [1]),
         f"valores={etiquetas}"),
        ("todas las predicciones caen en las clases validas",
         all(0 <= p < N_CLASES for p in pred), f"{len(pred)} predicciones"),
        ("el arbol tiene mas de un nodo",
         modelo.n_nodes > 1 and len(hojas) > 1,
         f"nodos={modelo.n_nodes} hojas={len(hojas)}"),
        ("predict_full devuelve P = n_mode/N coherente",
         coherencia, f"P en [{conf_min:.3f}, {conf_max:.3f}]"),
    ]

    print("-" * 72)
    print("COMPROBACIONES")
    print("-" * 72)
    fallos = 0
    for descripcion, ok, detalle in comprobaciones:
        print(f"  {'ok   ' if ok else 'FALLA'} {descripcion}"
              + (f"  [{detalle}]" if detalle else ""))
        if not ok:
            fallos += 1
    print()

    if acc > acc_trivial:
        print(f"  el modelo supera al predictor trivial "
              f"({acc:.2%} > {acc_trivial:.2%})")
    else:
        print(f"  AVISO: el modelo no supera al predictor trivial "
              f"({acc:.2%} <= {acc_trivial:.2%}, clase {mayoritaria})")
    print()

    # ------------------------------------------------------------------
    # Guardado y lectura del modelo
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("GUARDADO Y LECTURA DEL MODELO (.isod)")
    print("=" * 72)

    temporal_dir = None
    if args.modelo is not None:
        ruta_modelo = Path(args.modelo)
    else:
        temporal_dir = Path(tempfile.mkdtemp(prefix="isodendron_"))
        ruta_modelo = temporal_dir / "modelo.isod"

    cargado = None
    persistencia: list[tuple[str, bool, str]] = []

    try:
        ruta_modelo = modelo.save(ruta_modelo, overwrite=args.overwrite)
        print(f"  guardado en    : {ruta_modelo}")
        print(f"  tamano         : {ruta_modelo.stat().st_size} bytes")
        print()

        cargado = Isodendron.load(ruta_modelo)

        datos_test = [(X_test[i], y_test[i]) for i in range(len(X_test))]
        persistencia = [
            ("el blob es identico byte a byte",
             cargado.blob == modelo.blob, f"{len(modelo.blob)} bytes"),
            ("dump() identico",
             cargado.dump() == modelo.dump(), f"{len(hojas)} hojas"),
            ("las predicciones son identicas",
             cargado.predict(X_test) == pred, f"{len(pred)} muestras"),
            ("predict_full identico componente a componente",
             cargado.predict_full(X_test) == pred_full, ""),
            ("max_targets recordado",
             cargado.max_targets == modelo.max_targets,
             f"{cargado.max_targets}"),
            ("n_features y max_deep se leen del blob",
             cargado.n_features == modelo.n_features
             and cargado.max_deep == modelo.max_deep,
             f"{cargado.n_features} features, max_deep={cargado.max_deep}"),
            ("feature_names conservados",
             cargado.feature_names == modelo.feature_names,
             str(cargado.feature_names)),
            ("partial_fit desde el cargado da el mismo blob",
             cargado.partial_fit(datos_test).blob
             == modelo.partial_fit(datos_test).blob, ""),
        ]

        print("  comprobaciones de ida y vuelta:")
        fallos_persistencia = 0
        for descripcion, ok, detalle in persistencia:
            print(f"  {'ok   ' if ok else 'FALLA'} {descripcion}"
                  + (f"  [{detalle}]" if detalle else ""))
            if not ok:
                fallos_persistencia += 1
        print()

        # Errores esperados del cargador.
        datos_ok = ruta_modelo.read_bytes()
        casos_error = (
            ("fichero vacio", b""),
            ("fichero truncado a 8 bytes", datos_ok[:8]),
            ("blob truncado", datos_ok[:-16]),
            ("magic de fichero ajeno", b"OTROCOSA" + datos_ok[8:]),
            ("version de formato alterada",
             datos_ok[:8] + (999).to_bytes(4, "little") + datos_ok[12:]),
        )

        print("  errores esperados:")
        for nombre, contenido in casos_error:
            ruta_modelo.write_bytes(contenido)
            try:
                Isodendron.load(ruta_modelo)
                ok = False
            except (ValueError, OSError):
                ok = True
            print(f"  {'ok   ' if ok else 'FALLA'} {nombre} se rechaza")
            if not ok:
                fallos_persistencia += 1

        try:
            modelo.save(ruta_modelo)
            ok = False
        except FileExistsError:
            ok = True
        print(f"  {'ok   ' if ok else 'FALLA'} save sin overwrite no "
              f"reemplaza")
        if not ok:
            fallos_persistencia += 1

        ruta_modelo.unlink()
        try:
            Isodendron.load(ruta_modelo)
            ok = False
        except FileNotFoundError:
            ok = True
        print(f"  {'ok   ' if ok else 'FALLA'} fichero inexistente se "
              f"rechaza")
        if not ok:
            fallos_persistencia += 1
        print()

        fallos += fallos_persistencia
    finally:
        if temporal_dir is not None and temporal_dir.exists():
            try:
                if ruta_modelo.exists():
                    ruta_modelo.unlink()
                temporal_dir.rmdir()
            except OSError:
                pass
    # ------------------------------------------------------------------
    # Re-entrenamiento con la parte de test
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("RE-ENTRENAMIENTO CON LA PARTE DE TEST")
    print("=" * 72)
    print("  Se acumulan las filas de test sobre el modelo CARGADO del")
    print("  fichero .isod, mediante partial_fit. El arbol NO cambia: solo se")
    print("  recalculan las hojas sumando la evidencia nueva, con un")
    print("  decaimiento previo para que la evidencia vieja no crezca sin")
    print("  limite.")
    print()
    print("  AVISO: se evalua sobre las filas sobre las que se acaba de")
    print("  acumular, asi que este resultado demuestra el mecanismo de")
    print("  actualizacion, NO es una estimacion de generalizacion.")
    print()

    datos_test = [(X_test[i], y_test[i]) for i in range(len(X_test))]
    modelo2, diag = cargado.partial_fit_diag(datos_test, capacidad=16)
    print(f"  hojas={diag['hojas']}  decayadas={diag['decayadas']}  "
          f"sin datos={diag['vaciadas']}  "
          f"resta maxima={diag['resta_maxima']}")
    print(f"  nodos={modelo2.n_nodes}  hojas={len(modelo2.dump())}  "
          f"(antes de acumular: nodos={modelo.n_nodes} hojas={len(hojas)})")
    print()

    pred2 = modelo2.predict(X_train)
    m2 = matriz_confusion(pred2, y_train)
    acc2 = exactitud(pred2, y_train)
    bal2 = balanced_accuracy(m2)
    kap2 = kappa(m2)
    clases2 = por_clase(m2)

    imprimir_matriz(m2, "Matriz de confusion tras el re-entrenamiento "
                        "(recuentos):")
    imprimir_matriz_pct(m2, "Porcentaje por fila:")

    print(f"  {'clase':<8} {'precision':>10} {'recall':>9} {'f1':>9} "
          f"{'soporte':>9}")
    print("  " + "-" * 48)
    for c in range(N_CLASES):
        d = clases2[c]
        print(f"  {c:<8} {d['precision']:>10.2%} {d['recall']:>9.2%} "
              f"{d['f1']:>9.2%} {int(d['soporte']):>9}")
    print()

    print("  comprobaciones del re-entrenamiento:")
    ok_arbol = modelo2.n_nodes == modelo.n_nodes
    ok_hojas = diag["vaciadas"] == 0
    print(f"  {'ok   ' if ok_arbol else 'FALLA'} el arbol no cambia: "
          f"{modelo2.n_nodes} nodos antes y despues")
    print(f"  {'ok   ' if ok_hojas else 'FALLA'} ninguna hoja se queda sin "
          f"datos  [{diag['vaciadas']}]")
    print()

    # ------------------------------------------------------------------
    # Comparacion de metricas
    # ------------------------------------------------------------------
    print("=" * 72)
    print("MEJORA / EMPEORAMIENTO TRAS EL RE-ENTRENAMIENTO")
    print("=" * 72)
    print(f"  {'metrica':<22} {'antes':>10} {'despues':>10} {'delta':>10}   "
          f"efecto")
    print("  " + "-" * 68)

    comparacion = (
        ("exactitud", acc, acc2),
        ("balanced accuracy", bal, bal2),
        ("precision clase 0", clases[0]["precision"], clases2[0]["precision"]),
        ("recall clase 0", clases[0]["recall"], clases2[0]["recall"]),
        ("f1 clase 0", clases[0]["f1"], clases2[0]["f1"]),
        ("precision clase 1", clases[1]["precision"], clases2[1]["precision"]),
        ("recall clase 1", clases[1]["recall"], clases2[1]["recall"]),
        ("f1 clase 1", clases[1]["f1"], clases2[1]["f1"]),
    )
    mejoran = empeoran = 0
    for nombre, antes, despues in comparacion:
        delta = despues - antes
        if delta > 1e-12:
            efecto = "mejora"
            mejoran += 1
        elif delta < -1e-12:
            efecto = "empeora"
            empeoran += 1
        else:
            efecto = "igual"
        print(f"  {nombre:<22} {antes:>10.2%} {despues:>10.2%} "
              f"{delta:>+9.2%}   {efecto}")

    dkap = kap2 - kap
    if dkap > 1e-12:
        efecto = "mejora"
        mejoran += 1
    elif dkap < -1e-12:
        efecto = "empeora"
        empeoran += 1
    else:
        efecto = "igual"
    print(f"  {'kappa de Cohen':<22} {kap:>10.4f} {kap2:>10.4f} "
          f"{dkap:>+10.4f}   {efecto}")
    print("  " + "-" * 68)
    print(f"  {mejoran} metricas mejoran, {empeoran} empeoran, "
          f"{len(comparacion) + 1 - mejoran - empeoran} iguales")
    print()

    aptitud = ("el modelo aprovecha la evidencia nueva"
               if acc2 > acc else
               "el modelo no gana con la evidencia nueva")
    print(f"  exactitud: {acc:.2%} -> {acc2:.2%}   ({aptitud})")
    print()
    if fallos:
        print(f"RESULTADO: {fallos} comprobacion(es) fallidas")
        return 3
    print("RESULTADO: todas las comprobaciones OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
