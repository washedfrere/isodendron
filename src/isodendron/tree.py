"""
isodendron.tree — Clase pública Isodendron.

Encapsula un árbol ISODENDRON entrenado. Es la única clase que el usuario
final necesita conocer.

Uso típico:

    from isodendron import Isodendron

    data = [
        ([1, 2, 3], 0),
        ([1, 5, 3], 1),
        ([4, 2, 3], 0),
    ]

    model = Isodendron().fit(data, max_deep=8, p_min=(1, 1))

    # Predicción por lote (usa el camino rápido en C).
    preds = model.predict([[1, 2, 3], [4, 2, 3]])

    # Predicción individual (deserializa y recorre en Python).
    p = model.predict_one([1, 2, 3])

    # Inspección del árbol.
    for leaf in model.dump():
        print(leaf)

    # Serialización.
    import pickle
    with open("model.pkl", "wb") as fh:
        pickle.dump(model, fh)

Diseño:

  - El blob (bytes) es el estado canónico del modelo. Todo lo demás
    (metadata, árbol deserializado) se deriva de él.
  - predict() usa C para lotes grandes y Python puro para lotes
    pequeños o muestras individuales. El umbral está en
    Isodendron._PYTHON_PREDICT_THRESHOLD.
  - El árbol deserializado se cachea perezosamente la primera vez que
    se necesita y no se invalida, porque el blob es inmutable.
"""

from __future__ import annotations

import json
import os

import pickle
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import _ffi
from .dataset import to_arrays


# Formatos de desempaquetado del blob. Deben coincidir con blob.h.
_HEADER_FMT = "<IIIIIIII"           # 32 bytes

#: Magic del blob, el mismo que BLOB_MAGIC en isodendron/src/blob.h.
_BLOB_MAGIC = 0x444F5349            # "ISOD" en little endian

#: Magic del fichero de modelo. Es distinto del del blob a proposito:
#: asi un .isod se distingue de un blob crudo mirando los primeros bytes.
_MODEL_MAGIC = b"ISODFILE"
_MODEL_HEADER_FMT = "<8sII"         # magic, version, json_len
_MODEL_HEADER_SIZE = struct.calcsize(_MODEL_HEADER_FMT)

#: Version del formato, la MISMA que BLOB_VERSION en isodendron/src/blob.h.
#: El blob lleva la suya en su header y el cargador exige que coincidan,
#: asi que una discrepancia se detecta en tiempo de ejecucion.
_FORMAT_VERSION = 3

#: Tope de entradas por hoja cuando el modelo no lo recuerda.
_DEFAULT_MAX_TARGETS = 10
_DIST_FMT = "<ii"                   # (target, count) del payload


def _leer_distribucion(blob: bytes, off: int, n: int) -> list[tuple[int, int]]:
    """
    Lee las `n` entradas (target, count) que una hoja tiene en el payload.

    `off` es el offset absoluto dentro del blob y `n` el número de
    entradas guardadas, que puede ser menor que el de targets distintos
    de la hoja si se alcanzó el tope.
    """
    if n <= 0:
        return []
    return [struct.unpack_from(_DIST_FMT, blob, off + i * 8)
            for i in range(n)]
_NODE_FMT = "<IiiiiiiI"             # 32 bytes

_HEADER_SIZE = 32
_NODE_SIZE = 32
_MAGIC = 0x444F5349                 # "HIDR" en LE


class Isodendron:
    """
    Árbol de decisión ISODENDRON.

    Un Isodendron sin entrenar tiene _blob = None y no puede predecir.
    Tras fit() el blob queda fijado y el modelo es inmutable.
    """

    #: A partir de este número de muestras, predict() delega en C.
    #: Por debajo, deserializa el árbol a Python y recorre en memoria.
    _PYTHON_PREDICT_THRESHOLD = 64

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def __init__(
        self,
        blob: bytes | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Crea una instancia. Normalmente se llama sin argumentos y se
        entrena con fit(). El constructor con blob y metadata existe
        para que pickle pueda reconstruir un modelo serializado.

        Parámetros:
          blob      Blob binario devuelto por _ffi.train. Si es None,
                    el modelo está sin entrenar.
          metadata  Diccionario con información auxiliar (n_features,
                    max_deep, p_min, feature_names...). No afecta al
                    comportamiento del modelo; solo se conserva para
                    introspección.
        """
        self._blob: bytes | None = blob
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self._header: tuple[int, ...] | None = None
        self._tree: list[tuple[int, int, int, int, int, int, int, int]] | None = None

        if blob is not None:
            self._parse_header()

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Iterable[tuple[Sequence[int], int]],
        max_deep: int = 8,
        p_min: tuple[int, int] = (1, 1),
        *,
        hilos: int = 1,
        max_targets: int = 10,
        feature_names: Sequence[str] | None = None,
    ) -> "Isodendron":
        """
        Entrena un árbol ISODENDRON.

        Parámetros:
          data           Lista (o iterable) de tuplas (features, target).
          max_deep       Profundidad máxima del árbol.
          p_min          Tupla (num, den) con la pureza mínima de la
                         moda para dejar de dividir un nodo. Un nodo
                         se convierte en hoja si n_mode / N >= num / den.
                         num = 0 se rechaza porque dejaría el árbol en
                         una sola hoja.
          hilos          Número de hilos para construir el árbol (>= 1).
                         El reparto es transparente: el blob resultante
                         es idéntico al de un solo hilo, así que esto
                         solo afecta al tiempo. Ojo: la conversión del
                         dataset (to_arrays) sigue siendo secuencial, así
                         que la aceleración se nota sobre todo con
                         datasets grandes.
          feature_names  Nombres opcionales de las features, para que
                         dump() y __repr__ sean más legibles.

        Devuelve:
          self, para permitir encadenar Isodendron().fit(...).predict(...).

        Lanza:
          DatasetError   si data está malformado.
          OverflowError  si algún valor no cabe en int32_t.
          IsodendronError     si la biblioteca C falla.
          ValueError     si max_deep <= 0 o p_min no es una tupla de dos
                         enteros con num >= 0 y den > 0. Ojo: num = 0
                         hace que la pureza exigida sea 0 y deja el árbol
                         reducido a una sola hoja.
        """
        if max_deep <= 0:
            raise ValueError(f"max_deep debe ser > 0 (recibido: {max_deep})")
        if (not isinstance(p_min, (tuple, list))
                or len(p_min) != 2):
            raise ValueError(
                f"p_min debe ser (num, den) (recibido: {p_min!r})"
            )
        p_min_num, p_min_den = int(p_min[0]), int(p_min[1])
        if p_min_num < 0 or p_min_den <= 0:
            raise ValueError(
                f"p_min inválido: num={p_min_num}, den={p_min_den}"
            )
        if p_min_num == 0:
            # P_MIN = 0 hace que la condición n_mode/N >= P_MIN se cumpla
            # siempre, así que el árbol quedaría en una sola hoja. No hay
            # ningún caso de uso legítimo para "no dividir nunca".
            raise ValueError(
                "p_min num=0 deja el árbol reducido a una sola hoja; "
                "usa (1, 1) para exigir pureza total"
            )

        X, y, n, f = to_arrays(data)

        if feature_names is not None:
            feature_names = list(feature_names)
            if len(feature_names) != f:
                raise ValueError(
                    f"feature_names tiene {len(feature_names)} nombres, "
                    f"se esperaban {f}"
                )
            if not all(isinstance(x, str) for x in feature_names):
                raise ValueError(
                    "todos los nombres de features deben ser str"
                )

        blob = _ffi.train_mt_cap(X, y, n, f, max_deep, p_min_num, p_min_den,
                              max_targets, hilos)

        self._blob = blob
        self._metadata = {
            "n_features": f,

            "max_deep": max_deep,
            "p_min_num": p_min_num,
            "p_min_den": p_min_den,
            "max_targets": max_targets,
            "version": _ffi.version(),
        }
        if feature_names is not None:
            self._metadata["feature_names"] = list(feature_names)

        self._header = None
        self._tree = None
        self._parse_header()
        return self

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, samples: Sequence[Sequence[int]]) -> list[int]:
        """
        Predice para un lote de muestras.

        Parámetros:
          samples   Lista de muestras, cada una una secuencia de f
                    enteros. El número de features debe coincidir con
                    el del entrenamiento.

        Devuelve:
          Lista de enteros, uno por muestra.

        Lanza:
          RuntimeError  si el modelo no está entrenado.
          ValueError    si las muestras tienen forma inconsistente.
          IsodendronError    si C detecta un blob corrupto o incompatible.

        Notas:
          - Para lotes grandes (>= Isodendron._PYTHON_PREDICT_THRESHOLD),
            delega en C.
          - Para lotes pequeños, usa el árbol deserializado en
            Python. Evita el coste de cruzar la frontera.
          - Para una sola muestra, usar predict_one().
        """
        self._require_trained()

        samples = list(samples)
        if not samples:
            return []

        f = self._metadata["n_features"]

        # Detectar formato: secuencia de secuencias, o secuencia plana
        # (una sola muestra).
        first = samples[0]
        if isinstance(first, int):
            # El usuario pasó una sola muestra en forma plana.
            samples = [samples]  # type: ignore[list-item]

        n = len(samples)
        for i, s in enumerate(samples):
            if len(s) != f:
                raise ValueError(
                    f"muestra {i}: se esperaban {f} features, "
                    f"se recibieron {len(s)}"
                )

        if n >= self._PYTHON_PREDICT_THRESHOLD:
            return self._predict_c(samples, n, f)

        return [self._predict_one_python(s) for s in samples]

    def predict_one(self, sample: Sequence[int]) -> int:
        """
        Predice para una sola muestra. Devuelve un int.

        Es ligeramente más rápido que predict([sample])[0] porque evita
        construir la lista externa, aunque la diferencia es marginal.
        """
        self._require_trained()

        f = self._metadata["n_features"]
        if len(sample) != f:
            raise ValueError(
                f"se esperaban {f} features, se recibieron {len(sample)}"
            )
        return self._predict_one_python(sample)

    def predict_full(
        self,
        samples: Sequence[Sequence[int]],
    ) -> list[tuple[int, float, int, int]]:
        """
        Predice y devuelve, por muestra, la tupla::

            (moda, P, n_mode, N)

        donde:

          moda       la predicción (el target más frecuente de la hoja)
          P          la pureza de esa hoja, P = n_mode / N
          n_mode     observaciones de la hoja cuyo target vale su moda
          N          observaciones totales de la hoja

        P es la exactitud del modelo dentro de esa región en el conjunto
        de **entrenamiento**, no un acierto esperado. Ojo: en árboles muy
        profundos casi todas las hojas tienen N = 1 y por tanto P = 1, así
        que ahí no discrimina nada. Queda a criterio del llamante ponderar
        por P o por n_mode.

        Lanza:
          RuntimeError  si el modelo no está entrenado.
          ValueError    si las muestras tienen forma inconsistente.
          IsodendronError    si C detecta un blob corrupto o incompatible.
        """
        self._require_trained()

        samples = list(samples)
        if not samples:
            return []

        f = self._metadata["n_features"]

        if isinstance(samples[0], int):
            samples = [samples]  # type: ignore[list-item]

        n = len(samples)
        for i, s in enumerate(samples):
            if len(s) != f:
                raise ValueError(
                    f"muestra {i}: se esperaban {f} features, "
                    f"se recibieron {len(s)}"
                )

        flat: list[int] = []
        for s in samples:
            flat.extend(s)

        pred, n_mode, leaf_n = _ffi.predict_full(self._blob, flat, n, f)
        return [
            (
                pred[i],
                (n_mode[i] / leaf_n[i]) if leaf_n[i] else 0.0,
                n_mode[i],
                leaf_n[i],
            )
            for i in range(n)
        ]

    def _predict_one_python(self, sample: Sequence[int]) -> int:
        tree = self._ensure_tree()
        idx = 0
        while True:
            is_leaf, feature_idx, value, _N, _K, mode, _nmode, left_size = tree[idx]
            if is_leaf:
                return mode
            if sample[feature_idx] == value:
                idx = idx + 1
            else:
                idx = idx + 1 + left_size

    def _predict_c(
        self,
        samples: Sequence[Sequence[int]],
        n: int,
        f: int,
    ) -> list[int]:
        # Aplanar a un único array row-major para cruzar la frontera
        # de una vez.
        flat: list[int] = []
        for s in samples:
            flat.extend(s)
        out = _ffi.predict(self._blob, flat, n, f)  # type: ignore[arg-type]
        return list(out)

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    def predict_leaf(
        self,
        samples: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[int]]:
        """
        Predice y devuelve ademas el indice de nodo hoja de cada muestra.

        Devuelve (predicciones, hojas). El indice es el del nodo en
        preorden, el mismo que usa el blob, y permite recuperar la
        distribucion de esa hoja.
        """
        self._require_trained()

        samples = list(samples)
        if not samples:
            return [], []

        f = self._metadata["n_features"]
        if isinstance(samples[0], int):
            samples = [samples]

        n = len(samples)
        for i, s in enumerate(samples):
            if len(s) != f:
                raise ValueError(
                    f"muestra {i}: se esperaban {f} features, "
                    f"se recibieron {len(s)}"
                )

        flat: list[int] = []
        for s in samples:
            flat.extend(s)

        pred, hojas = _ffi.predict_leaf(self._blob, flat, n, f)
        return list(pred), list(hojas)

    def partial_fit(
        self,
        data: Iterable[tuple[Sequence[int], int]],
        max_targets: int | None = None,
    ) -> "Isodendron":
        """
        ACUMULA un conjunto nuevo en las hojas y devuelve un modelo nuevo.

        El arbol no cambia: solo se recalculan las hojas, con un
        decaimiento previo para que la evidencia vieja no crezca sin
        limite. El modelo original queda intacto.
        """
        self._require_trained()
        tope = self.max_targets if max_targets is None else max_targets
        X, y, n, f = to_arrays(data)
        blob = _ffi.partial_fit(self._blob, X, y, n, f, tope)
        metadata = dict(self._metadata)
        metadata["max_targets"] = tope
        return self.__class__(blob, metadata)

    def partial_fit_diag(
        self,
        data: Iterable[tuple[Sequence[int], int]],
        max_targets: int | None = None,
        capacidad: int = 64,
    ) -> tuple["Isodendron", dict[str, Any]]:
        """
        Igual que partial_fit pero devuelve ademas el diagnostico.

        El diagnostico trae los contadores (hojas, decayadas, sin datos,
        resta maxima) y hasta `capacidad` muestras de hojas anomalas, para
        que con un dataset enorme no haya que reservar por hoja.
        """
        self._require_trained()
        tope = self.max_targets if max_targets is None else max_targets
        X, y, n, f = to_arrays(data)
        blob, info = _ffi.partial_fit_diag(self._blob, X, y, n, f,
                                           tope, capacidad)
        metadata = dict(self._metadata)
        metadata["max_targets"] = tope
        return self.__class__(blob, metadata), info

    @property
    def max_targets(self) -> int:
        """Tope de entradas por hoja con el que se construyo el modelo."""
        valor = self._metadata.get("max_targets")
        return _DEFAULT_MAX_TARGETS if valor is None else int(valor)

    def save(self, ruta: str | Path, overwrite: bool = False) -> Path:
        """
        Guarda el modelo en un fichero .isod.

        El fichero lleva todo junto: una cabecera con el magic y la version
        del formato, los metadatos en JSON y el blob del arbol. La version
        es la misma que la del blob, asi que cualquier cambio en cualquiera
        de las dos capas invalida el fichero y no se puede leer mal.

        Si `ruta` ya existe hay que pasar overwrite=True; asi no se pierde
        un modelo por un descuido. La escritura es atomica (a un temporal y
        luego os.replace), de modo que un fallo a mitad no deja el fichero
        corrupto.

        Devuelve la ruta escrita.
        """
        self._require_trained()

        destino = Path(ruta)
        if destino.exists() and not overwrite:
            raise FileExistsError(
                f"{destino} ya existe; pasa overwrite=True (o --overwrite) "
                f"si quieres reemplazarlo"
            )

        metadatos = {
            "max_targets": self.max_targets,
            "feature_names": self.feature_names,
            "biblioteca": _ffi.version(),
        }
        js = json.dumps(metadatos, separators=(",", ":"),
                        ensure_ascii=False).encode("utf-8")
        cabecera = struct.pack(_MODEL_HEADER_FMT, _MODEL_MAGIC,
                               _FORMAT_VERSION, len(js))

        temporal = destino.with_name(destino.name + ".tmp")
        try:
            with temporal.open("wb") as fh:
                fh.write(cabecera)
                fh.write(js)
                fh.write(self._blob)
            os.replace(temporal, destino)
        except BaseException:
            if temporal.exists():
                temporal.unlink()
            raise
        return destino

    @classmethod
    def load(cls, ruta: str | Path) -> "Isodendron":
        """
        Carga un modelo guardado con save().

        Todo lo que afecta a la inferencia (n_features, max_deep, p_min) se
        lee del header del blob, no del JSON: el blob es la unica fuente de
        verdad. Del JSON solo se toma lo que el blob no lleva: max_targets
        y feature_names.
        """
        origen = Path(ruta)
        if not origen.is_file():
            raise FileNotFoundError(f"no existe el fichero de modelo: {origen}")

        datos = origen.read_bytes()
        if len(datos) < _MODEL_HEADER_SIZE:
            raise ValueError(
                f"{origen} es demasiado corto para ser un modelo "
                f"({len(datos)} bytes)"
            )

        magic, version, json_len = struct.unpack_from(
            _MODEL_HEADER_FMT, datos, 0)
        if magic != _MODEL_MAGIC:
            raise ValueError(
                f"{origen} no es un fichero de modelo de isodendron "
                f"(magic {magic!r})"
            )
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{origen} es de la version de formato {version}; esta "
                f"biblioteca lee la {_FORMAT_VERSION}"
            )
        if json_len == 0 or _MODEL_HEADER_SIZE + json_len >= len(datos):
            raise ValueError(
                f"{origen}: longitud de metadatos incoherente ({json_len} "
                f"bytes sobre un fichero de {len(datos)})"
            )

        inicio = _MODEL_HEADER_SIZE + json_len
        try:
            metadatos = json.loads(datos[_MODEL_HEADER_SIZE:inicio]
                                   .decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{origen}: metadatos ilegibles ({exc})") from exc
        if not isinstance(metadatos, dict):
            raise ValueError(f"{origen}: los metadatos no son un diccionario")

        blob = datos[inicio:]
        if len(blob) < struct.calcsize(_HEADER_FMT):
            raise ValueError(f"{origen}: el modelo esta truncado")

        campos = struct.unpack_from(_HEADER_FMT, blob, 0)
        if campos[0] != _BLOB_MAGIC:
            raise ValueError(
                f"{origen}: el modelo no tiene el magic de isodendron "
                f"(0x{campos[0]:08x})"
            )
        if campos[1] != version:
            raise ValueError(
                f"{origen}: el modelo es de la version {campos[1]} pero la "
                f"cabecera del fichero dice {version}"
            )

        n_features = campos[2]
        nombres = metadatos.get("feature_names")
        if nombres is not None:
            if not isinstance(nombres, list) or len(nombres) != n_features:
                raise ValueError(
                    f"{origen}: feature_names no cuadra con las "
                    f"{n_features} features del modelo"
                )

        metadata = {
            "n_features": n_features,
            "max_deep": campos[4],
            "p_min_num": campos[5],
            "p_min_den": campos[6],
            "max_targets": int(metadatos.get("max_targets",
                                             _DEFAULT_MAX_TARGETS)),
            "version": _ffi.version(),
        }
        if nombres is not None:
            metadata["feature_names"] = [str(x) for x in nombres]

        try:
            modelo = cls(blob, metadata)
            modelo._ensure_tree()
        except Exception as exc:
            raise ValueError(
                f"{origen}: el modelo no es utilizable ({exc})"
            ) from exc
        return modelo

    def dump(self) -> list[dict[str, Any]]:
        """
        Devuelve una lista de hojas del árbol, cada una con el camino
        de condiciones que la alcanza y el resumen del target:

            [
                {
                    "conditions": [
                        {"feature": "color", "op": "==", "value": 3},
                        {"feature": "size",  "op": "!=", "value": 1},
                    ],
                    "N": 42,
                    "K": 2,
                    "mode": 5,
                    "n_mode": 28,
                    "P": 28 / 42,
                },
                ...
            ]

        El orden es el de preorden del árbol, así que las hojas
        aparecen en un orden determinista y reproducible.
        """
        self._require_trained()
        tree = self._ensure_tree()

        out: list[dict[str, Any]] = []

        # Recorrido iterativo con pila explícita para no anidar
        # recursión en Python.
        stack: list[tuple[int, list[dict[str, Any]]]] = [(0, [])]

        while stack:
            idx, path = stack.pop()
            node = tree[idx]
            is_leaf, feature_idx, value, N, K, mode, n_mode, left_size = node

            if is_leaf:
                out.append({
                    "conditions": list(path),
                    "N": N,
                    "K": K,
                    "mode": mode,
                    "n_mode": n_mode,
                    "P": (n_mode / N) if N > 0 else 0.0,
                    # En una hoja, feature_idx y value llevan el payload.
                    "distribucion": _leer_distribucion(self._blob, feature_idx,
                                                       value),
                    "total": sum(c for _t, c in
                                 _leer_distribucion(self._blob, feature_idx,
                                                    value)),
                })
                continue

            name = self._feature_name(feature_idx)

            # Preorden: primero el hijo ==, luego el !=. Como usamos
            # pila, apilamos en orden inverso para que el == salga
            # primero.
            right_path = list(path)
            right_path.append({"feature": name, "op": "!=", "value": value})
            stack.append((idx + 1 + left_size, right_path))

            left_path = list(path)
            left_path.append({"feature": name, "op": "==", "value": value})
            stack.append((idx + 1, left_path))

        # El recorrido con pila produce el orden inverso al preorden
        # deseado. Lo reordenamos por número de condiciones y luego
        # por orden de aparición, que para una pila bien apilada es
        # equivalente a invertir. Más simple: rehacer el recorrido
        # recursivo. Pero como ya tenemos la lista, la invertimos y
        # listo.
        out.reverse()
        return out

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._blob is not None

    @property
    def blob(self) -> bytes | None:
        return self._blob

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def n_features(self) -> int:
        self._require_trained()
        return self._metadata["n_features"]

    @property
    def n_nodes(self) -> int:
        self._require_trained()
        assert self._header is not None
        return self._header[3]

    @property
    def max_deep(self) -> int:
        self._require_trained()
        return self._metadata["max_deep"]

    @property
    def feature_names(self) -> list[str] | None:
        names = self._metadata.get("feature_names")
        if names is None:
            return None
        return list(names)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """
        Soporte para pickle. Reconstruye el objeto a través del
        constructor con (blob, metadata).

        Un Isodendron sin entrenar no es serializable: no hay nada que
        guardar. Lanzamos un error explícito en lugar de producir un
        pickle vacío que falle al cargarse.
        """
        if self._blob is None:
            raise pickle.PicklingError(
                "no se puede serializar un Isodendron sin entrenar"
            )
        return (self.__class__, (self._blob, self._metadata))

    def __repr__(self) -> str:
        if self._blob is None:
            return "Isodendron(no entrenado)"
        return (
            f"Isodendron(n_features={self._metadata.get('n_features')}, "
            f"n_nodes={self.n_nodes}, "
            f"max_deep={self._metadata.get('max_deep')}, "
            f"blob_size={len(self._blob)}B)"
        )

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if self._blob is None:
            raise RuntimeError(
                "el modelo no está entrenado; llama a fit() primero"
            )

    def _parse_header(self) -> None:
        if self._blob is None:
            self._header = None
            return
        if len(self._blob) < _HEADER_SIZE:
            raise _ffi.IsodendronError(
                _ffi._ISODENDRON_ERR_ARG,
                f"blob demasiado corto: {len(self._blob)} bytes",
            )
        fields = struct.unpack_from(_HEADER_FMT, self._blob, 0)
        magic = fields[0]
        if magic != _MAGIC:
            raise _ffi.IsodendronError(
                _ffi._ISODENDRON_ERR_ARG,
                f"magic inválido: 0x{magic:08x}, esperado 0x{_MAGIC:08x}",
            )
        self._header = fields

        # Coherencia interna entre metadata y blob.
        n_features = fields[2]
        if self._metadata.get("n_features") is None:
            self._metadata["n_features"] = n_features
        elif self._metadata["n_features"] != n_features:
            raise _ffi.IsodendronError(
                _ffi._ISODENDRON_ERR_ARG,
                f"n_features en metadata ({self._metadata['n_features']}) "
                f"no coincide con la del blob ({n_features})",
            )

    def _ensure_tree(self) -> list[tuple[int, int, int, int, int, int, int, int]]:
        """
        Deserializa el blob a una lista plana de nodos si no estaba
        ya en caché. Cada nodo es una tupla:

            (is_leaf, feature_idx, value, N, K, mode, n_mode, left_size)

        El orden es el del blob, es decir, preorden. Las posiciones
        de los hijos se calculan a partir de left_size, no se
        almacenan.
        """
        if self._tree is not None:
            return self._tree
        if self._blob is None:
            raise RuntimeError("no hay blob que deserializar")

        n_nodes = self._header[3] if self._header else None
        if n_nodes is None:
            self._parse_header()
            assert self._header is not None
            n_nodes = self._header[3]

        # El payload de distribuciones va detrás de los nodos.
        payload_bytes = self._header[7] if self._header else 0
        expected = _HEADER_SIZE + n_nodes * _NODE_SIZE + payload_bytes
        if len(self._blob) != expected:
            raise _ffi.IsodendronError(
                _ffi._ISODENDRON_ERR_ARG,
                f"tamaño del blob inconsistente: "
                f"{len(self._blob)} bytes, esperados {expected}",
            )

        nodes: list[tuple[int, int, int, int, int, int, int, int]] = []
        unpack = struct.Struct(_NODE_FMT).unpack_from
        for i in range(n_nodes):
            offset = _HEADER_SIZE + i * _NODE_SIZE
            nodes.append(unpack(self._blob, offset))  # type: ignore[arg-type]

        self._tree = nodes
        return nodes

    def _feature_name(self, idx: int) -> str:
        names = self._metadata.get("feature_names")
        if names is not None and 0 <= idx < len(names):
            return names[idx]
        return f"f{idx}"


__all__ = ["Isodendron"]
