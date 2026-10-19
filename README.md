# isodendron

[![CI](https://github.com/washedfrere/isodendron/actions/workflows/ci.yml/badge.svg)](https://github.com/washedfrere/isodendron/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/isodendron.svg)](https://pypi.org/project/isodendron/)
[![Python](https://img.shields.io/pypi/pyversions/isodendron.svg)](https://pypi.org/project/isodendron/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/washedfrere/isodendron/blob/main/LICENSE)

A deterministic decision-tree library where **the target never influences how
the tree is split**.

`isodendron` grows a tree by repeatedly asking a single question: *which
condition divides the samples most evenly?* Nothing else. The class labels are
only counted inside the leaves, once the geometry of the tree is already fixed.
That makes training fast, reproducible and free of the usual information-gain
machinery — and it produces trees that behave very differently from CART or
ID3, in ways this README tries to make explicit.

The name comes from the *isodieretic split* (Greek `isos`, equal, and
`dieretos`, divided): the process of finding the condition — a categorical
value or a numeric threshold — that maximises the equity in the number of
resulting samples between the branches.

The library is written in C11 with no dependencies beyond the platform
threading API, and ships with a thin Python binding built on `ctypes`.

---

## Contents

- [The isodieretic split](#the-isodieretic-split)
- [Installation](#installation)
- [Building from source](#building-from-source)
- [Quick start (Python)](#quick-start-python)
- [Python API](#python-api)
- [Persisting models](#persisting-models)
- [Incremental learning](#incremental-learning)
- [C API](#c-api)
- [The blob format](#the-blob-format)
- [Concurrency](#concurrency)
- [Tests](#tests)
- [What this library is not](#what-this-library-is-not)

---

## Installation

```bash
pip install isodendron
```

Wheels are published for Windows, Linux and macOS. They already contain the
compiled C library, so no compiler is needed. The package requires Python 3.11
or newer.

To work from a checkout:

```bash
git clone git@github.com:washedfrere/isodendron.git
cd isodendron
python -m pip install -e ".[dev]"
```

---

## The isodieretic split

For every candidate feature and every candidate value, the samples are split in
two: those equal to the value, and those different from it. The quality of that
split is

```
score = min(n_equal, n_distinct)
```

which is simply the size of the smaller branch — maximised means as balanced as
possible. For each feature the best value is kept, and among the features the
one with the highest score wins.

Three consequences follow, and they explain almost all the behaviour you will
observe:

1. **The target is never read during training.** Two datasets with identical
   features and completely different labels produce exactly the same tree.
2. **Training is deterministic.** There is no random seed, no shuffling, no
   tie-breaking by trial. The same input gives the same bytes, always.
3. **The tree learns the geometry of the data, not the labels.** If the labels
   happen to correlate with a balanced split, the tree will capture them. If
   they correlate with something else — a skewed split, a rare value, an
   imbalanced threshold — it will not, no matter how deep you grow it.

A leaf is closed when any of these holds:

| Condition | Meaning |
|---|---|
| `depth >= max_deep` | Tree budget exhausted. |
| `K <= 1` | Only one class remains. |
| `n_mode / N >= P_MIN` | The leaf is already pure enough. |

`P_MIN` is expressed as an exact fraction, not a float: the test is
`n_mode * p_min_den >= N * p_min_num`, so there is no floating-point rounding
anywhere in training.

Inside a leaf the **mode** is the most frequent class. Ties are broken by
taking the *lower median* of the tied classes — deterministic, and biased
towards the smaller label. **P** is `n_mode / N`, the purity of the leaf.

---

## Building from source

The C library is built with CMake and packaged with `scikit-build-core`. The
normal entry point is:

```bash
python -m pip install .
# or, for development
python -m pip install -e ".[dev]"
```

To build and test the C side on its own:

```bash
cmake -S . -B build -DISODENDRON_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The sources are plain C11 and compile without warnings under MSVC and gcc with
`-Wall -Wextra`. No compiler extensions are used.

Files:

```
csrc/src/isodendron.c   tree construction, prediction, partial_fit
csrc/src/arena.c        per-subtree bump allocator
csrc/src/counts.c       open-addressing counter with O(1) increments
csrc/src/blob.c         serialisation of the tree
csrc/src/blob.h         format constants
csrc/include/isodendron.h
csrc/tests/test_isodendron.c
```

---

## Quick start (Python)

```python
from isodendron import Isodendron

# Each sample is (features, label).
data = [
    ([0, 3, 1], 0),
    ([1, 3, 0], 1),
    ([0, 4, 0], 0),
    ([1, 4, 1], 1),
]

model = Isodendron().fit(data, max_deep=6, p_min=(1, 1))

model.predict([[0, 3, 1], [1, 4, 1]])   # -> [0, 1]
model.predict_one([0, 3, 1])            # -> 0
```

The binding looks for the shared library in this order:

1. the `ISODENDRON_LIB` environment variable, if it points at an existing file;
2. the `isodendron` package resource installed by the wheel.

If you get a "library not found" error, reinstall the package
(`python -m pip install --force-reinstall isodendron`) or build it from the
source tree with `python -m pip install .`.

---

## Python API

### `Isodendron`

```python
Isodendron(blob: bytes | None = None, metadata: dict | None = None)
```

An empty model, or one reconstructed from raw bytes. For files use
`Isodendron.load()` instead.

| Method | Returns | Description |
|---|---|---|
| `fit(data, max_deep=8, p_min=(1, 1), *, hilos=1, max_targets=10, feature_names=None)` | `self` | Train from `(features, label)` pairs. |
| `predict(samples)` | `list[int]` | One class per sample. |
| `predict_one(sample)` | `int` | Single sample. |
| `predict_full(samples)` | `list[tuple[int, float, int, int]]` | `(mode, P, n_mode, N)` per sample. |
| `predict_leaf(samples)` | `(list[int], list[int])` | Predictions and the preorder leaf index of each sample. |
| `partial_fit(data, max_targets=None)` | `Isodendron` | Accumulate samples into the existing leaves; returns a **new** model. |
| `partial_fit_diag(data, max_targets=None, capacidad=64)` | `(Isodendron, dict)` | As above, plus diagnostics. |
| `dump()` | `list[dict]` | Every leaf: path, counts, mode, P, distribution. |
| `save(path, overwrite=False)` | `Path` | Write a `.isod` file. |
| `Isodendron.load(path)` | `Isodendron` | Class method. Read a `.isod` file. |

| Property | Description |
|---|---|
| `blob` | Raw bytes of the tree, or `None`. |
| `metadata` | Copy of the model metadata. |
| `n_features` | Number of features. |
| `n_nodes` | Total node count. |
| `max_depth` | The `max_deep` the model was trained with. |
| `feature_names` | List of names, or `None`. |
| `max_targets` | Payload cap per leaf; inherited by `partial_fit`. |

### `IsodendronError`

Raised for library-level failures — a corrupt blob, an invalid argument, an
allocation failure. It carries a `code` attribute matching the C constants
(`ISODENDRON_OK`, `ISODENDRON_ERR_MEM`, `ISODENDRON_ERR_ARG`).

Contract errors are raised as ordinary Python exceptions instead:

| Situation | Exception |
|---|---|
| `predict` / `dump` / `save` before `fit` | `RuntimeError` |
| Wrong number of features in a sample | `ValueError` |
| `p_min` with a zero numerator or denominator | `ValueError` |
| `save` over an existing file without `overwrite` | `FileExistsError` |
| `load` on a missing file | `FileNotFoundError` |
| `load` on a foreign, truncated or version-mismatched file | `ValueError` |

### `dump()` in detail

`dump()` returns one dictionary per leaf, with the path that reaches it and the
full stored distribution:

```python
{
    "path": [("Age", 0, "=="), ("Cholesterol", 3, "!=")],
    "N": 120,
    "K": 3,
    "mode": 1,
    "n_mode": 96,
    "P": 0.8,
    "distribucion": [(1, 96), (0, 18), (2, 6)],
    "total": 120,
}
```

`distribucion` holds at most `max_targets` `(target, count)` pairs, ordered by
count descending and, on ties, by target ascending.

---

## Persisting models

A model and its metadata are stored in a single `.isod` file:

```
"ISODFILE"    8 bytes    file magic
uint32                   format version — the same number as the blob's
uint32                   length of the JSON metadata
JSON                     max_targets, feature_names, library version
bytes                    the blob, up to the end of the file
```

```python
model.save("heart.isod")                    # refuses to overwrite
loaded = Isodendron.load("heart.isod")

loaded.predict(X) == model.predict(X)       # True, byte for byte
```

Design notes:

- **One version number.** The container and the blob share it, so any change in
  either layer invalidates the file rather than risking a misread. It is
  checked twice: the container's against the library, and the blob's against
  the container's.
- **The blob is the only source of truth.** `n_features`, `max_deep` and
  `p_min` are read from the blob header. The JSON carries only what the blob
  does not: `max_targets`, `feature_names` and the library version at save
  time.
- **No pickle.** Nothing in the file is executable; loading is a `struct` and
  `json` parse.
- **Atomic writes.** The file is written to a sibling `.tmp` and then
  `os.replace`d, so an interrupted save never leaves a corrupt model.
- `max_targets` is remembered, so `partial_fit` uses it without being told.

---

## Incremental learning

`partial_fit` adds new samples to an existing tree **without rebuilding it**.
The structure is fixed; only the leaf distributions change.

```python
model = Isodendron().fit(train_data)
model = model.partial_fit(new_data)              # accumulate
model, info = model.partial_fit_diag(new_data)   # with diagnostics
```

Semantics, precisely:

1. For every leaf the old distribution is **decayed** by subtracting
   `min_count - 1` from every entry, which leaves the minimum at 1 and
   preserves the differences between classes exactly.
2. The new samples that fall in that leaf are added on top.
3. Leaves with no old distribution and no new samples keep their `N`. Nothing
   is ever emptied.

The decay is what keeps old evidence from dominating forever. It never changes
the ordering inside a leaf, and it never makes two distinct counts equal.

`partial_fit_diag` returns counters plus up to `capacidad` samples of anomalous
leaves, so a huge dataset does not need any per-leaf reservation:

```python
{
    "puntos": 918,
    "hojas": 265,
    "decayadas": 123,
    "vaciadas": 0,          # leaves left without data — should be 0
    "resta_maxima": 36,
    "anomalas": 0,
    "detalle": [],
}
```

**Important.** `partial_fit` is not a way to avoid re-training on new data and
then score it honestly. If you accumulate the test set and evaluate on the same
rows, you are measuring memorisation. The mechanism is sound; the estimator is
not.

---

## C API

Everything is declared in `include/isodendron.h` and exported from the shared
library. All arrays are `int32_t`, all logical sizes are `uint32_t`.

```c
const char *isodendron_version(void);

int isodendron_train(const int32_t *X, const int32_t *y,
                     uint32_t n, uint32_t f,
                     uint32_t max_deep,
                     uint32_t p_min_num, uint32_t p_min_den,
                     uint8_t **out_blob, size_t *out_len);

int isodendron_train_mt(const int32_t *X, const int32_t *y,
                        uint32_t n, uint32_t f,
                        uint32_t max_deep,
                        uint32_t p_min_num, uint32_t p_min_den,
                        uint32_t n_threads,
                        uint8_t **out_blob, size_t *out_len);

int isodendron_train_mt_cap(const int32_t *X, const int32_t *y,
                            uint32_t n, uint32_t f,
                            uint32_t max_deep,
                            uint32_t p_min_num, uint32_t p_min_den,
                            uint32_t max_targets,
                            uint32_t n_threads,
                            uint8_t **out_blob, size_t *out_len);

int isodendron_predict(const uint8_t *blob, size_t blob_len,
                       const int32_t *X, uint32_t n, uint32_t f,
                       int32_t *out_pred);
```

`X` is row-major: sample `i`, feature `j` is `X[i * f + j]`. Every `out_blob`
is heap allocated and belongs to the caller — free it with
`isodendron_free`, never with `free`.

Return codes are `ISODENDRON_OK` (0), `ISODENDRON_ERR_MEM` (1) and
`ISODENDRON_ERR_ARG` (2). On any non-zero return no output buffer is
allocated.

The remaining entry points — `predict_full`, `predict_leaf`,
`partial_fit`, `partial_fit_diag` and `free` — follow the same conventions;
their exact prototypes are in the header.

`predict_leaf` reports the **preorder index** of the leaf node, which is the
index used inside the blob. It is not a compact leaf number.

---

## The blob format

The blob is a self-describing, little-endian structure. Version 3.

```
header   32 bytes
nodes    32 bytes each, in preorder
payload  (target int32, count int32) pairs
```

Header, all `uint32`:

| Offset | Field |
|---|---|
| 0 | magic `0x444F5349` (`"ISOD"`) |
| 4 | version (`3`) |
| 8 | `n_features` |
| 12 | `n_nodes` |
| 16 | `max_deep` |
| 20 | `p_min_num` |
| 24 | `p_min_den` |
| 28 | `payload_bytes` |

Total size is `32 + n_nodes * 32 + payload_bytes`.

Node, all `int32`:

| Offset | Internal node | Leaf |
|---|---|---|
| 0 | `is_leaf` = 0 | `is_leaf` = 1 |
| 4 | `feature_idx` | `payload_off` |
| 8 | `value` | `payload_len` |
| 12 | `N` | `N` |
| 16 | `K` | `K` |
| 20 | `mode` | `mode` |
| 24 | `n_mode` | `n_mode` |
| 28 | `left_subtree_size` | 0 |

The split is `feature == value` on the left, `!=` on the right. `N` is the
number of samples in the subtree, `K` the number of distinct classes. The
payload holds `payload_len` pairs of 8 bytes each.

Leaves always carry a payload — including the ones closed by depth, purity or
lack of a usable split — so every leaf is readable by the same code path.

---

## Concurrency

`isodendron_train_mt` and `isodendron_train_mt_cap` explore disjoint subtrees
on separate threads using a worker pool. Training is **deterministic across
thread counts**: the same input produces the same blob with one thread or with
sixteen, byte for byte. The test suite asserts this explicitly.

`n_threads = 1` takes the sequential path, which is not merely the parallel
path with one worker: it uses a single arena and skips the fork logic.

---

## Tests

```bash
# Python
python -m pytest

# C
cmake -S . -B build -DISODENDRON_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Current status: the C suite runs **48 cases / 363 checks**, the Python suite
**55 cases**, both with zero failures.

The Python suite covers the public API, the error contract, the blob layout,
pickling, and the `.isod` round trip including every documented failure mode.

The scripts under `examples/` are experiments rather than tests. They
illustrate the consequences of the isodieretic split on synthetic data, on
MNIST pixels, on an `(x, y, colour)` cloud of the same digits, and on a
tabular heart-disease dataset. Each prints an honest confusion matrix and, in
the heart case, a save/load round trip and an incremental update.

---

## What this library is not

It is worth being blunt about the boundaries, because they follow from the
central design decision rather than from missing features.

**It ignores the target when splitting.** If a feature separates the classes
but does so unevenly, the tree will not use it. Accuracy on ordinary datasets
is therefore far from what a CART would give; the heart-disease example sits
around 79%, where gradient-boosted trees reach 88%.

**The leaf payload is capped.** With more than `max_targets` classes in one
leaf, the rarest ones are dropped from the distribution. The default of 10 fits
MNIST exactly; raise it for wider problems.

**Deep trees on high-dimensional data are not compact.** Balanced splits waste
capacity: the MNIST point-cloud experiment needed roughly 4000 nodes to reach
68% on three features. The tree is not the bottleneck there; the
representation is.

**There is no pruning, no feature importance and no missing-value handling.**
Categorical inputs must be encoded by the caller, thresholds included — see
`examples/data/metadata_heart_disease.txt` for a worked example.

**The mode tie-break favours the smaller label.** It is deterministic and
documented, but it is a bias, not a neutral choice.

---

## Repository layout

```
src/isodendron/            Python package
  __init__.py
  _ffi.py                  ctypes declarations and wrappers
  _lib.py                   library loading via importlib.resources
  dataset.py                CSV and Python-sequence normalisation
  tree.py                   Isodendron model, persistence and partial_fit
  py.typed
csrc/                      C library
  include/isodendron.h
  src/                     sources
  tests/test_isodendron.c
tests/test_isodendron.py   Python test suite
examples/                  experiment scripts and small datasets
pyproject.toml             packaging metadata and scikit-build-core config
CMakeLists.txt             C build and CTest
```

---

## License

MIT. See [LICENSE](https://github.com/washedfrere/isodendron/blob/main/LICENSE).

This means you can use, modify, redistribute and sell this code, including in
closed-source products, as long as the copyright notice and the permission
notice travel with it. There is no warranty of any kind.
