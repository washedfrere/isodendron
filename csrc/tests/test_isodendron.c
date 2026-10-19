/*
 * test_isodendron.c — Suite de pruebas de ISODENDRON.
 *
 * Cubre los cuatro módulos de la biblioteca:
 *
 *   - arena.c    asignador por regiones (alineación, agotamiento, rewind)
 *   - counts.c   tabla hash int32_t -> uint32_t (crecimiento, reset, iteración)
 *   - blob.c     serialización binaria (ida y vuelta, validación, errores)
 *   - isodendron.c    API pública (train, predict, version, free)
 *
 * No depende de ningún framework externo: se compila y se ejecuta con
 * `nmake /f Makefile.msvc test` o `make test`. El proceso devuelve 0 si
 * todo pasa y 1 si algo falla, de forma que sirve en integración continua.
 *
 * Los módulos internos (arena, counts, blob) se enlazan directamente
 * contra sus objetos, no contra la DLL: sus símbolos no cruzan la
 * frontera pública y no se exportan.
 */

#include "isodendron.h"

#include "arena.h"
#include "counts.h"
#include "blob.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Framework mínimo                                                    */
/* ------------------------------------------------------------------ */

static int         g_checks   = 0;
static int         g_failures = 0;
static const char *g_current  = "";

#define CHECK(cond)                                                     \
    do {                                                                \
        g_checks++;                                                     \
        if (!(cond)) {                                                  \
            g_failures++;                                               \
            printf("    FALLO %s:%d [%s] %s\n",                         \
                   __FILE__, __LINE__, g_current, #cond);               \
        }                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                  \
    do {                                                                \
        long long _va = (long long)(a);                                 \
        long long _vb = (long long)(b);                                 \
        g_checks++;                                                     \
        if (_va != _vb) {                                               \
            g_failures++;                                               \
            printf("    FALLO %s:%d [%s] %s == %s (%lld != %lld)\n",    \
                   __FILE__, __LINE__, g_current, #a, #b, _va, _vb);    \
        }                                                               \
    } while (0)

/* Lectura little-endian explícita, independiente de la máquina. */
static uint32_t rd_u32_le(const uint8_t *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint32_t blob_field(const uint8_t *blob, uint32_t offset)
{
    return rd_u32_le(blob + offset);
}

/* Offset del byte inicial del nodo `idx`. */
static uint32_t node_offset(uint32_t idx)
{
    return ISODENDRON_BLOB_HEADER_SIZE + idx * ISODENDRON_BLOB_NODE_SIZE;
}

/* ================================================================== */
/* arena                                                               */
/* ================================================================== */

static void test_arena_init_destroy(void)
{
    Arena a;

    CHECK_EQ(arena_init(&a, 0), -1);          /* capacidad 0 es error */

    CHECK_EQ(arena_init(&a, 64), 0);
    CHECK(!arena_failed(&a));
    CHECK_EQ(arena_used(&a), 0);
    CHECK_EQ(arena_remaining(&a), 64);
    CHECK_EQ(arena_mark(&a), 0);

    arena_destroy(&a);
    CHECK(arena_failed(&a));                  /* destruido == no usable */
    CHECK_EQ(arena_used(&a), 0);
    CHECK_EQ(arena_remaining(&a), 0);

    /* Destruir dos veces, o un arena que nunca se inicializó. */
    arena_destroy(&a);
    memset(&a, 0, sizeof(a));
    arena_destroy(&a);
    CHECK(arena_failed(&a));
}

static void test_arena_alloc_alignment(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 1024), 0);

    void *p1 = arena_alloc(&a, 1, 8);
    void *p2 = arena_alloc(&a, 1, 8);
    void *p3 = arena_alloc(&a, 4, 16);

    CHECK(p1 != NULL);
    CHECK(p2 != NULL);
    CHECK(p3 != NULL);
    CHECK_EQ((uintptr_t)p1 % 8, 0);
    CHECK_EQ((uintptr_t)p2 % 8, 0);
    CHECK_EQ((uintptr_t)p3 % 16, 0);
    CHECK_EQ((uintptr_t)p2 - (uintptr_t)p1, 8);

    /* Tras p1 (1 byte) y p2 (1 byte) el cursor está en 9; p3 alinea a 16. */
    CHECK_EQ(arena_used(&a), 16 + 4);
    CHECK_EQ((uintptr_t)p3 - (uintptr_t)p1, 16);

    arena_destroy(&a);
}

static void test_arena_alloc_align_no_potencia_de_dos(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 256), 0);

    void *q1 = arena_alloc(&a, 1, 1);
    void *q2 = arena_alloc(&a, 1, 3);        /* 3 se redondea a 4 */

    CHECK(q1 != NULL);
    CHECK(q2 != NULL);
    CHECK_EQ((uintptr_t)q2 % 4, 0);
    CHECK_EQ(arena_used(&a), 5);             /* 1 + (alinear 1 a 4) + 1 */

    arena_destroy(&a);
}

static void test_arena_alloc_zero(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 64), 0);

    size_t before = arena_used(&a);
    void  *z      = arena_alloc(&a, 0, 8);

    CHECK(z != NULL);                        /* puntero válido... */
    CHECK_EQ(arena_used(&a), before);        /* ...pero sin consumo */

    arena_destroy(&a);
}

static void test_arena_exhaustion(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 16), 0);

    CHECK(arena_alloc(&a, 32, 1) == NULL);
    CHECK(arena_failed(&a));                 /* queda marcado */
    CHECK(arena_alloc(&a, 1, 1) == NULL);    /* y ya no sirve nada */

    arena_destroy(&a);
}

static void test_arena_alloc_array(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 4096), 0);

    int32_t *v = (int32_t *)arena_alloc_array(&a, 100, sizeof(int32_t));
    CHECK(v != NULL);
    CHECK_EQ((uintptr_t)v % sizeof(int32_t), 0);
    CHECK_EQ(arena_used(&a), 400);

    /* count == 0: puntero válido sin consumo. */
    size_t before = arena_used(&a);
    void  *z      = arena_alloc_array(&a, 0, 4);
    CHECK(z != NULL);
    CHECK_EQ(arena_used(&a), before);

    /* Overflow de count * elem_size: ni se intenta. */
    Arena b;
    CHECK_EQ(arena_init(&b, 128), 0);
    CHECK(arena_alloc_array(&b, SIZE_MAX, 8) == NULL);
    CHECK(arena_failed(&b));
    arena_destroy(&b);

    arena_destroy(&a);
}

static void test_arena_rewind(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 256), 0);

    size_t mark = arena_mark(&a);
    arena_alloc(&a, 64, 1);
    arena_alloc(&a, 32, 1);
    CHECK_EQ(arena_used(&a), 96);

    arena_rewind(&a, mark);
    CHECK_EQ(arena_used(&a), 0);

    /* Rebobinar más allá del cursor no hace nada. */
    arena_alloc(&a, 10, 1);
    arena_rewind(&a, 999999);
    CHECK_EQ(arena_used(&a), 10);

    arena_destroy(&a);
}

static void test_arena_rewind_tras_fallo(void)
{
    Arena a;
    CHECK_EQ(arena_init(&a, 16), 0);

    arena_alloc(&a, 64, 1);
    CHECK(arena_failed(&a));

    arena_rewind(&a, 0);
    CHECK(arena_failed(&a));                 /* el fallo no se deshace */

    arena_destroy(&a);
}

/* ================================================================== */
/* counts                                                              */
/* ================================================================== */

static uint32_t g_iter_total;
static uint32_t g_iter_calls;

static int iter_sum(int32_t key, uint32_t value, void *user)
{
    (void)key;
    (void)user;
    g_iter_total += value;
    g_iter_calls++;
    return 1;
}

static int iter_stop(int32_t key, uint32_t value, void *user)
{
    (void)key;
    (void)value;
    (void)user;
    g_iter_calls++;
    return 0;
}

static void test_counts_basico(void)
{
    Arena   a;
    CHECK_EQ(arena_init(&a, 4096), 0);

    Counts *c = counts_new(&a, 0);
    CHECK(c != NULL);
    CHECK_EQ(counts_capacity(c), COUNTS_MIN_CAPACITY);
    CHECK_EQ(counts_size(c), 0);
    CHECK(!counts_failed(c));

    counts_inc(c, 5);
    counts_inc(c, 5);
    counts_inc(c, 5);

    CHECK_EQ(counts_get(c, 5), 3);
    CHECK_EQ(counts_size(c), 1);
    CHECK(counts_has(c, 5));
    CHECK_EQ(counts_get(c, 6), 0);
    CHECK(!counts_has(c, 6));

    counts_add(c, 7, 4);
    CHECK_EQ(counts_get(c, 7), 4);
    CHECK_EQ(counts_size(c), 2);

    counts_add(c, 7, 0);                     /* delta 0 no hace nada */
    CHECK_EQ(counts_get(c, 7), 4);

    arena_destroy(&a);
}

static void test_counts_claves_extremas(void)
{
    Arena   a;
    CHECK_EQ(arena_init(&a, 4096), 0);

    Counts *c = counts_new(&a, 16);

    counts_inc(c, INT32_MIN);
    counts_inc(c, INT32_MAX);
    counts_inc(c, -1);
    counts_inc(c, 0);

    CHECK_EQ(counts_size(c), 4);
    CHECK_EQ(counts_get(c, INT32_MIN), 1);
    CHECK_EQ(counts_get(c, INT32_MAX), 1);
    CHECK_EQ(counts_get(c, -1), 1);
    CHECK_EQ(counts_get(c, 0), 1);

    counts_inc(c, INT32_MIN);
    CHECK_EQ(counts_get(c, INT32_MIN), 2);
    CHECK_EQ(counts_size(c), 4);             /* sigue habiendo 4 claves */

    arena_destroy(&a);
}

static void test_counts_crecimiento(void)
{
    Arena   a;
    CHECK_EQ(arena_init(&a, 1u << 20), 0);

    Counts *c = counts_new(&a, 16);
    CHECK(c != NULL);

    const int32_t n = 1000;
    for (int32_t i = 0; i < n; i++) {
        counts_inc(c, i);
    }

    CHECK_EQ(counts_size(c), (uint32_t)n);
    CHECK(counts_capacity(c) > COUNTS_MIN_CAPACITY);
    CHECK_EQ(counts_capacity(c) & (counts_capacity(c) - 1), 0); /* potencia de 2 */
    CHECK(!counts_failed(c));

    /* Todas las claves siguen localizables tras los rehash. */
    int missing = 0;
    for (int32_t i = 0; i < n; i++) {
        if (counts_get(c, i) != 1) {
            missing++;
        }
    }
    CHECK_EQ(missing, 0);

    /* Y ninguna clave ajena aparece. */
    CHECK_EQ(counts_get(c, n + 1), 0);
    CHECK_EQ(counts_get(c, -99999), 0);

    arena_destroy(&a);
}

static void test_counts_reset(void)
{
    Arena   a;
    CHECK_EQ(arena_init(&a, 4096), 0);

    Counts *c = counts_new(&a, 16);
    for (int32_t i = 0; i < 10; i++) {
        counts_inc(c, i);
        counts_inc(c, i);
    }
    uint32_t cap = counts_capacity(c);
    CHECK_EQ(counts_size(c), 10);

    counts_reset(c);
    CHECK_EQ(counts_size(c), 0);
    CHECK_EQ(counts_capacity(c), cap);       /* la capacidad se conserva */
    CHECK_EQ(counts_get(c, 3), 0);

    counts_inc(c, 3);
    CHECK_EQ(counts_get(c, 3), 1);

    arena_destroy(&a);
}

static void test_counts_iterate(void)
{
    Arena   a;
    CHECK_EQ(arena_init(&a, 4096), 0);

    Counts *c = counts_new(&a, 16);
    for (int32_t i = 0; i < 20; i++) {
        counts_add(c, i, (uint32_t)(i + 1));
    }

    g_iter_total = 0;
    g_iter_calls = 0;
    uint32_t visited = counts_iterate(c, iter_sum, NULL);
    CHECK_EQ(visited, 20);
    CHECK_EQ(g_iter_calls, 20);
    CHECK_EQ(g_iter_total, 210u);            /* 1+2+...+20 */

    /* Parar en la primera entrada visitada. */
    g_iter_calls = 0;
    visited = counts_iterate(c, iter_stop, NULL);
    CHECK_EQ(g_iter_calls, 1);
    CHECK_EQ(visited, 1);

    /* Sin callback no se recorre nada. */
    CHECK_EQ(counts_iterate(c, NULL, NULL), 0);

    arena_destroy(&a);
}

static void test_counts_max_key(void)
{
    Arena   a;
    CHECK_EQ(arena_init(&a, 4096), 0);

    Counts *c = counts_new(&a, 16);
    counts_add(c, 4, 3);
    counts_add(c, 7, 9);
    counts_add(c, 9, 1);

    uint32_t v = 0;
    CHECK_EQ(counts_max_key_value(c, &v), 7);
    CHECK_EQ(v, 9);
    CHECK_EQ(counts_max_key(c), 7);

    /* out_value NULL es válido. */
    CHECK_EQ(counts_max_key_value(c, NULL), 7);

    arena_destroy(&a);
}

static void test_counts_nulos(void)
{
    CHECK(counts_failed(NULL) == 1);
    CHECK_EQ(counts_size(NULL), 0);
    CHECK_EQ(counts_capacity(NULL), 0);
    CHECK_EQ(counts_get(NULL, 1), 0);
    CHECK(!counts_has(NULL, 1));
    CHECK_EQ(counts_iterate(NULL, iter_sum, NULL), 0);
    CHECK_EQ(counts_max_key_value(NULL, NULL), 0);

    CHECK(counts_new(NULL, 16) == NULL);

    /* Las operaciones sobre NULL no deben reventar. */
    counts_inc(NULL, 1);
    counts_add(NULL, 1, 1);
    counts_reset(NULL);
}

/* ================================================================== */
/* blob                                                                */
/* ================================================================== */

static void test_blob_total_size(void)
{
    CHECK_EQ(blob_total_size(0, 0u), 32u);
    CHECK_EQ(blob_total_size(1, 0u), 64u);
    CHECK_EQ(blob_total_size(3, 0u), 128u);
    CHECK_EQ(blob_total_size(100, 0u), 32u + 100u * 32u);
}

static void test_blob_roundtrip(void)
{
    uint8_t buf[32 + 2 * 32];
    memset(buf, 0xAA, sizeof(buf));

    blob_write_header(buf, 4, 2, 7, 1, 100, 0u);

    BlobHeader h;
    CHECK_EQ(blob_read_header(buf, sizeof(buf), &h), BLOB_OK);
    CHECK_EQ(h.magic, BLOB_MAGIC);
    CHECK_EQ(h.version, BLOB_VERSION);
    CHECK_EQ(h.n_features, 4);
    CHECK_EQ(h.n_nodes, 2);
    CHECK_EQ(h.max_deep, 7);
    CHECK_EQ(h.p_min_num, 1);
    CHECK_EQ(h.p_min_den, 100);
    CHECK_EQ(h.payload_bytes, 0);

    BlobNode in;
    in.is_leaf          = 0;
    in.feature_idx      = 3;
    in.value            = -42;               /* valores negativos deben sobrevivir */
    in.N                = 1000;
    in.K                = 17;
    in.mode             = -7;
    in.n_mode           = 500;
    in.left_subtree_size = 1;
    blob_write_node(buf, 0, &in);

    BlobNode out;
    CHECK_EQ(blob_read_node(buf, sizeof(buf), 0, &out), BLOB_OK);
    CHECK_EQ(out.is_leaf, 0);
    CHECK_EQ(out.feature_idx, 3);
    CHECK_EQ(out.value, -42);
    CHECK_EQ(out.N, 1000);
    CHECK_EQ(out.K, 17);
    CHECK_EQ(out.mode, -7);
    CHECK_EQ(out.n_mode, 500);
    CHECK_EQ(out.left_subtree_size, 1);
}

static void test_blob_hoja_normalizada(void)
{
    uint8_t buf[32 + 32];
    blob_write_header(buf, 2, 1, 3, 1, 100, 0u);

    BlobNode n;
    n.is_leaf           = 1;
    n.feature_idx       = 9;                 /* basura: debe leerse como 0 */
    n.value             = 77;
    n.N                 = 10;
    n.K                 = 1;
    n.mode              = 4;
    n.n_mode            = 10;
    n.left_subtree_size = 5;
    blob_write_node(buf, 0, &n);

    BlobNode out;
    CHECK_EQ(blob_read_node(buf, sizeof(buf), 0, &out), BLOB_OK);
    CHECK_EQ(out.is_leaf, 1);
    /* En una hoja feature_idx y value llevan el payload: NO se normalizan. */
    CHECK_EQ(out.feature_idx, 9);
    CHECK_EQ(out.value, 77);
    CHECK_EQ(out.left_subtree_size, 0);
    CHECK_EQ(out.mode, 4);
    CHECK_EQ(out.N, 10);
}

static void test_blob_header_errores(void)
{
    uint8_t buf[32 + 2 * 32];
    BlobHeader h;

    CHECK_EQ(blob_read_header(NULL, sizeof(buf), &h), BLOB_ERR_NULL);
    CHECK_EQ(blob_read_header(buf, sizeof(buf), NULL), BLOB_ERR_NULL);
    CHECK_EQ(blob_read_header(buf, 16, &h), BLOB_ERR_TOO_SHORT);
    CHECK_EQ(blob_read_header(buf, 31, &h), BLOB_ERR_TOO_SHORT);

    blob_write_header(buf, 2, 2, 3, 1, 100, 0u);
    CHECK_EQ(blob_read_header(buf, sizeof(buf), &h), BLOB_OK);

    /* Magic corrupto. */
    uint8_t saved = buf[0];
    buf[0] = (uint8_t)(saved ^ 0xFFu);
    CHECK_EQ(blob_read_header(buf, sizeof(buf), &h), BLOB_ERR_MAGIC);
    buf[0] = saved;

    /* Versión desconocida. */
    buf[BLOB_OFF_VERSION] = (uint8_t)(BLOB_VERSION + 1u);
    CHECK_EQ(blob_read_header(buf, sizeof(buf), &h), BLOB_ERR_VERSION);
    buf[BLOB_OFF_VERSION] = (uint8_t)BLOB_VERSION;

    /* n_nodes que no cuadra con el tamaño. */
    CHECK_EQ(blob_read_header(buf, sizeof(buf) - 1, &h), BLOB_ERR_SIZE);
    CHECK_EQ(blob_read_header(buf, sizeof(buf) + 32, &h), BLOB_ERR_SIZE);

    /* Y el caso válido sigue siéndolo. */
    CHECK_EQ(blob_read_header(buf, sizeof(buf), &h), BLOB_OK);
}

static void test_blob_node_limites(void)
{
    uint8_t buf[32 + 2 * 32];

    blob_write_header(buf, 2, 2, 3, 1, 100, 0u);

    BlobNode n;
    memset(&n, 0, sizeof(n));
    blob_write_node(buf, 0, &n);
    blob_write_node(buf, 1, &n);

    BlobNode node;
    CHECK_EQ(blob_read_node(buf, sizeof(buf), 0, &node), BLOB_OK);
    CHECK_EQ(blob_read_node(buf, sizeof(buf), 1, &node), BLOB_OK);
    CHECK_EQ(blob_read_node(buf, sizeof(buf), 2, &node), BLOB_ERR_SIZE);
    CHECK_EQ(blob_read_node(buf, sizeof(buf) - 1, 1, &node), BLOB_ERR_SIZE);
    CHECK_EQ(blob_read_node(NULL, sizeof(buf), 0, &node), BLOB_ERR_NULL);
    CHECK_EQ(blob_read_node(buf, sizeof(buf), 0, NULL), BLOB_ERR_NULL);
}

static void test_blob_node_ptr(void)
{
    uint8_t buf[32 + 2 * 32];
    blob_write_header(buf, 2, 2, 3, 1, 100, 0u);

    const uint8_t *p0 = blob_node_ptr(buf, sizeof(buf), 0);
    const uint8_t *p1 = blob_node_ptr(buf, sizeof(buf), 1);

    CHECK(p0 == buf + 32);
    CHECK(p1 == buf + 64);
    CHECK_EQ(p1 - p0, 32);

    CHECK(blob_node_ptr(buf, sizeof(buf), 2) == NULL);
    CHECK(blob_node_ptr(buf, sizeof(buf) - 1, 1) == NULL);
    CHECK(blob_node_ptr(NULL, sizeof(buf), 0) == NULL);
    CHECK(blob_node_ptr(buf, 31, 0) == NULL);
}

static void test_blob_check_compat(void)
{
    uint8_t    buf[32 + 32];
    BlobHeader h;

    blob_write_header(buf, 4, 1, 3, 1, 100, 0u);
    CHECK_EQ(blob_read_header(buf, sizeof(buf), &h), BLOB_OK);

    CHECK_EQ(blob_check_compat(&h, 4), BLOB_OK);
    CHECK_EQ(blob_check_compat(&h, 5), BLOB_ERR_SIZE);
    CHECK_EQ(blob_check_compat(&h, 1), BLOB_ERR_SIZE);
    CHECK_EQ(blob_check_compat(NULL, 4), BLOB_ERR_NULL);
}

static void test_blob_escritura_nula(void)
{
    BlobNode n;
    memset(&n, 0, sizeof(n));

    /* Ninguna debe reventar. */
    blob_write_header(NULL, 1, 1, 1, 1, 1, 0u);
    blob_write_node(NULL, 0, &n);

    uint8_t buf[64];
    blob_write_node(buf, 0, NULL);
}

/* ================================================================== */
/* API pública: isodendron.h                                                */
/* ================================================================== */

#define SEP_N 8

/* Dataset separable de forma exacta: f0 es el target y f1 es constante. */
static void build_separable(int32_t *X, int32_t *y, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
        int32_t d = (int32_t)(i % 2u);
        X[i * 2u + 0u] = d;
        X[i * 2u + 1u] = 5;
        y[i] = d;
    }
}

static void test_version(void)
{
    const char *v = isodendron_version();
    CHECK(v != NULL);
    CHECK(v[0] != '\0');
    CHECK(v[0] >= '0' && v[0] <= '9');
}

static void test_train_argumentos_invalidos(void)
{
    int32_t  X[SEP_N * 2];
    int32_t  y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *blob = NULL;
    size_t   len  = 0;

    CHECK_EQ(isodendron_train(NULL, y, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_train(X, NULL, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, NULL, &len), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &blob, NULL), ISODENDRON_ERR_ARG);

    CHECK_EQ(isodendron_train(X, y, 0, 2, 1, 1, 1, &blob, &len), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_train(X, y, SEP_N, 0, 1, 1, 1, &blob, &len), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 0, 1, 1, &blob, &len), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 0, &blob, &len), ISODENDRON_ERR_ARG);

    /* En caso de error los parámetros de salida no se tocan. */
    uint8_t *sentinel_blob = (uint8_t *)(uintptr_t)0x1;
    size_t   sentinel_len  = 12345;
    CHECK_EQ(isodendron_train(X, y, 0, 2, 1, 1, 1, &sentinel_blob, &sentinel_len),
             ISODENDRON_ERR_ARG);
    CHECK(sentinel_blob == (uint8_t *)(uintptr_t)0x1);
    CHECK_EQ(sentinel_len, 12345);
}

static void test_train_predict_exacto(void)
{
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *blob = NULL;
    size_t   len  = 0;

    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);
    CHECK(blob != NULL);
    CHECK_EQ(len, ISODENDRON_BLOB_HEADER_SIZE + 3u * ISODENDRON_BLOB_NODE_SIZE
                   + blob_field(blob, 28));

    /* Cabecera. */
    CHECK_EQ(blob_field(blob, 0), ISODENDRON_BLOB_MAGIC);
    CHECK_EQ(blob_field(blob, 4), ISODENDRON_BLOB_VERSION);
    CHECK_EQ(blob_field(blob, 8), 2);        /* n_features */
    CHECK_EQ(blob_field(blob, 12), 3);       /* n_nodes    */
    CHECK_EQ(blob_field(blob, 16), 1);       /* max_deep   */
    CHECK_EQ(blob_field(blob, 20), 1);       /* p_min_num  */
    CHECK_EQ(blob_field(blob, 24), 1);       /* p_min_den  */
    CHECK_EQ(blob_field(blob, 28), 16);      /* payload_bytes */

    /* Raíz: interna, sobre f0, valor 0. */
    uint32_t r = node_offset(0);
    CHECK_EQ(blob_field(blob, r + 0), 0);            /* is_leaf      */
    CHECK_EQ(blob_field(blob, r + 4), 0);            /* feature_idx  */
    CHECK_EQ(blob_field(blob, r + 8), 0);            /* value        */
    CHECK_EQ(blob_field(blob, r + 12), SEP_N);       /* N            */
    CHECK_EQ(blob_field(blob, r + 16), 2);           /* K            */
    CHECK_EQ(blob_field(blob, r + 24), SEP_N / 2);   /* n_mode       */
    CHECK_EQ(blob_field(blob, r + 28), 1);           /* left_subtree */

    /*
     * Guarda de longitud: si la condición de parada dejara el árbol en la
     * raíz, leer los nodos 1 y 2 sería una lectura fuera del blob.
     */
    if (blob_field(blob, 12) < 3u) {
        isodendron_free(blob);
        return;
    }

    /* Hijo ==: f0 == 0, target constante 0. */
    uint32_t l = node_offset(1);
    CHECK_EQ(blob_field(blob, l + 0), 1);
    CHECK_EQ(blob_field(blob, l + 4), 128);          /* payload_off */
    CHECK_EQ(blob_field(blob, l + 8), 1);            /* payload_len */
    CHECK_EQ(blob_field(blob, l + 12), SEP_N / 2);
    CHECK_EQ(blob_field(blob, l + 16), 1);
    CHECK_EQ(blob_field(blob, l + 20), 0);           /* mode */
    CHECK_EQ(blob_field(blob, l + 28), 0);

    /* Hijo !=: f0 != 0, target constante 1. */
    uint32_t g = node_offset(2);
    CHECK_EQ(blob_field(blob, g + 0), 1);
    CHECK_EQ(blob_field(blob, g + 12), SEP_N / 2);
    CHECK_EQ(blob_field(blob, g + 16), 1);
    CHECK_EQ(blob_field(blob, g + 20), 1);           /* mode */
    CHECK_EQ(blob_field(blob, g + 28), 0);

    /* Predicción: exacta sobre las 8 filas de entrenamiento. */
    int32_t pred[SEP_N];
    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 2, pred), ISODENDRON_OK);
    int wrong = 0;
    for (uint32_t i = 0; i < SEP_N; i++) {
        if (pred[i] != y[i]) {
            wrong++;
        }
    }
    CHECK_EQ(wrong, 0);

    isodendron_free(blob);
}

static void test_train_determinista(void)
{
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *b1 = NULL;
    uint8_t *b2 = NULL;
    size_t   l1 = 0;
    size_t   l2 = 0;

    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 3, 1, 1, &b1, &l1), ISODENDRON_OK);
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 3, 1, 1, &b2, &l2), ISODENDRON_OK);

    CHECK_EQ(l1, l2);
    CHECK(b1 != NULL && b2 != NULL);
    CHECK(memcmp(b1, b2, l1) == 0);

    isodendron_free(b1);
    isodendron_free(b2);
}

static void test_train_multihilo_identico(void)
{
    /*
     * La construcción en paralelo reparte subárboles entre hilos, pero
     * las decisiones de split no dependen del reparto, así que el blob
     * tiene que salir byte a byte idéntico al secuencial.
     */
    const uint32_t n = 4000;
    const uint32_t f = 8;

    int32_t *X = (int32_t *)malloc((size_t)n * f * sizeof(int32_t));
    int32_t *y = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    CHECK(X != NULL && y != NULL);
    if (X == NULL || y == NULL) {
        free(X);
        free(y);
        return;
    }

    uint32_t state = 987654321u;
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < f; j++) {
            state = state * 1103515245u + 12345u;
            X[i * f + j] = (int32_t)((state >> 16) % 5u);
        }
        y[i] = (int32_t)((state >> 8) % 7u);
    }

    uint8_t *b1 = NULL;
    uint8_t *b4 = NULL;
    size_t   l1 = 0;
    size_t   l4 = 0;

    CHECK_EQ(isodendron_train_mt(X, y, n, f, 12, 1, 1, 1, &b1, &l1), ISODENDRON_OK);
    CHECK_EQ(isodendron_train_mt(X, y, n, f, 12, 1, 1, 4, &b4, &l4), ISODENDRON_OK);

    CHECK(b1 != NULL && b4 != NULL);
    CHECK_EQ(l1, l4);
    if (b1 != NULL && b4 != NULL && l1 == l4 && l1 > 0) {
        CHECK(memcmp(b1, b4, l1) == 0);
    }

    /* n_threads = 0 se trata como 1. */
    uint8_t *b0 = NULL;
    size_t   l0 = 0;
    CHECK_EQ(isodendron_train_mt(X, y, n, f, 12, 1, 1, 0, &b0, &l0), ISODENDRON_OK);
    CHECK_EQ(l0, l1);
    if (b0 != NULL && b1 != NULL && l0 == l1 && l0 > 0) {
        CHECK(memcmp(b0, b1, l0) == 0);
    }

    isodendron_free(b0);
    isodendron_free(b1);
    isodendron_free(b4);
    free(X);
    free(y);
}

static void test_multihilo_max_deep_alto(void)
{
    /*
     * Con max_deep alto, dimensionar el arena de cada hilo por el árbol
     * completo pediría 2^(max_deep+1) nodos POR HILO: con max_deep = 23
     * son 805 MB por hilo, inviable. El arena debe calcularse con las
     * filas reales del subárbol, así que esto tiene que caber y dar el
     * mismo blob con 1 y con 4 hilos.
     */
    const uint32_t n = 3000;
    const uint32_t f = 3;

    int32_t *X = (int32_t *)malloc((size_t)n * f * sizeof(int32_t));
    int32_t *y = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    CHECK(X != NULL && y != NULL);
    if (X == NULL || y == NULL) {
        free(X);
        free(y);
        return;
    }

    uint32_t state = 424242u;
    for (uint32_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        X[i * f + 0u] = (int32_t)((state >> 16) % 28u);
        X[i * f + 1u] = (int32_t)((state >> 8) % 28u);
        X[i * f + 2u] = (int32_t)((state >> 4) % 4u);
        y[i] = (int32_t)((state >> 2) % 10u);
    }

    uint8_t *b1 = NULL;
    uint8_t *b4 = NULL;
    size_t   l1 = 0;
    size_t   l4 = 0;

    CHECK_EQ(isodendron_train_mt(X, y, n, f, 23, 1, 1, 1, &b1, &l1), ISODENDRON_OK);
    CHECK_EQ(isodendron_train_mt(X, y, n, f, 23, 1, 1, 4, &b4, &l4), ISODENDRON_OK);

    CHECK(b1 != NULL && b4 != NULL);
    CHECK_EQ(l1, l4);
    if (b1 != NULL && b4 != NULL && l1 == l4 && l1 > 0) {
        CHECK(memcmp(b1, b4, l1) == 0);
    }

    isodendron_free(b1);
    isodendron_free(b4);
    free(X);
    free(y);
}

static void test_target_constante(void)
{
    /*
     * Con target constante la raíz es 100% pura, así que la parada por
     * pureza la convierte directamente en hoja. Todas las predicciones
     * deben valer ese valor.
     */
    const uint32_t n = 100;
    int32_t        X[100];
    int32_t        y[100];
    for (uint32_t i = 0; i < n; i++) {
        X[i] = (int32_t)i;
        y[i] = 7;
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, n, 1, 3, 1, 1, &blob, &len), ISODENDRON_OK);
    CHECK_EQ(blob_field(blob, 12), 1);       /* una sola hoja */

    int32_t pred[100];
    CHECK_EQ(isodendron_predict(blob, len, X, n, 1, pred), ISODENDRON_OK);

    int wrong = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (pred[i] != 7) {
            wrong++;
        }
    }
    CHECK_EQ(wrong, 0);

    isodendron_free(blob);
}

static void test_p_min_pureza_del_nodo(void)
{
    /*
     * P_MIN es la pureza mínima de la moda: un nodo se convierte en
     * hoja si n_mode / N >= P_MIN.
     *
     * Con estos datos la raíz tiene N = 64 y n_mode = 16 (empate a 16
     * entre 0, 1, 2 y 3, gana el intermedio menor, 1), es decir una
     * pureza exacta del 25%. Eso permite comprobar los tres lados del
     * umbral con aritmética exacta, incluido el caso de igualdad.
     */
    const uint32_t n = 64;
    static int32_t X[64 * 2];
    static int32_t y[64];
    for (uint32_t i = 0; i < n; i++) {
        X[i * 2u + 0u] = (int32_t)(i % 4u);
        X[i * 2u + 1u] = (int32_t)i;
        y[i] = (int32_t)(i % 4u);
    }

    uint8_t *b_igual = NULL;   /* P_MIN == pureza de la raíz */
    uint8_t *b_encima = NULL;  /* P_MIN por encima           */
    uint8_t *b_total = NULL;   /* P_MIN = 100%               */
    size_t   l_igual = 0, l_encima = 0, l_total = 0;

    /* 1/4 == 25%: la igualdad cuenta como suficientemente puro. */
    CHECK_EQ(isodendron_train(X, y, n, 2, 20, 1, 4, &b_igual, &l_igual), ISODENDRON_OK);
    /* 26/100: por encima de la pureza de la raíz, así que divide. */
    CHECK_EQ(isodendron_train(X, y, n, 2, 20, 26, 100, &b_encima, &l_encima), ISODENDRON_OK);
    /* 1/1: exige pureza total. */
    CHECK_EQ(isodendron_train(X, y, n, 2, 20, 1, 1, &b_total, &l_total), ISODENDRON_OK);

    /* La raíz se queda como hoja única, con su propia moda. */
    CHECK_EQ(blob_field(b_igual, 12), 1);                    /* n_nodes  */
    CHECK_EQ(l_igual, ISODENDRON_BLOB_HEADER_SIZE + ISODENDRON_BLOB_NODE_SIZE
                      + blob_field(b_igual, 28));
    CHECK_EQ(blob_field(b_igual, node_offset(0) + 0), 1);    /* is_leaf  */
    CHECK_EQ(blob_field(b_igual, node_offset(0) + 20), 1);   /* mode     */
    CHECK_EQ(blob_field(b_igual, node_offset(0) + 24), 16);  /* n_mode   */

    /*
     * Con 26/100 la raíz divide; sus dos hijos ya superan el umbral
     * (16/16 = 100% y 16/48 = 33%), así que el árbol queda en 3 nodos.
     */
    CHECK_EQ(blob_field(b_encima, 12), 3);
    CHECK_EQ(blob_field(b_encima, node_offset(0) + 0), 0);   /* raíz interna */

    /* Exigir pureza total obliga a bajar más. */
    CHECK(blob_field(b_total, 12) > 3u);

    /* Más exigencia nunca puede dar menos nodos. */
    CHECK(blob_field(b_igual, 12) <= blob_field(b_encima, 12));
    CHECK(blob_field(b_encima, 12) <= blob_field(b_total, 12));

    isodendron_free(b_igual);
    isodendron_free(b_encima);
    isodendron_free(b_total);
}

static void test_p_min_cero_para_siempre(void)
{
    /*
     * Con numerador 0 la condición n_mode*den >= N*0 se cumple siempre
     * (n_mode >= 0), así que la raíz se queda como hoja y el árbol es
     * un único nodo.
     *
     * Antes de este cambio, 0 significaba "no podar nunca": la semántica
     * se ha invertido, así que 0/1 ya no sirve como referencia de "sin
     * poda". Esa referencia es ahora 1/1.
     */
    const uint32_t n = 64;
    static int32_t X[64 * 2];
    static int32_t y[64];
    for (uint32_t i = 0; i < n; i++) {
        X[i * 2u + 0u] = (int32_t)(i % 4u);
        X[i * 2u + 1u] = (int32_t)i;
        y[i] = (int32_t)(i % 4u);
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, n, 2, 20, 0, 1, &blob, &len), ISODENDRON_OK);

    CHECK_EQ(blob_field(blob, 12), 1);
    CHECK_EQ(len, ISODENDRON_BLOB_HEADER_SIZE + ISODENDRON_BLOB_NODE_SIZE
                   + blob_field(blob, 28));
    CHECK_EQ(blob_field(blob, node_offset(0) + 0), 1);    /* is_leaf */
    CHECK_EQ(blob_field(blob, node_offset(0) + 20), 1);   /* mode  */

    /* Todas las filas reciben la moda de la raíz: empate a 16 entre
     * 0, 1, 2 y 3 -> intermedio menor = 1. */
    int32_t pred[64];
    CHECK_EQ(isodendron_predict(blob, len, X, n, 2, pred), ISODENDRON_OK);

    int distintas = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (pred[i] != 1) {
            distintas++;
        }
    }
    CHECK_EQ(distintas, 0);

    isodendron_free(blob);
}

static void test_p_min_independiente_de_max_deep(void)
{
    /*
     * La parada por pureza es intrínseca al nodo: no depende de la
     * profundidad ni de un padre. Si P_MIN muerde antes de que lo haga
     * max_deep, subir max_deep no puede cambiar el árbol resultante.
     */
    const uint32_t n = 64;
    static int32_t X[64 * 2];
    static int32_t y[64];
    for (uint32_t i = 0; i < n; i++) {
        X[i * 2u + 0u] = (int32_t)(i % 4u);
        X[i * 2u + 1u] = (int32_t)i;
        y[i] = (int32_t)(i % 4u);
    }

    const uint32_t profundidades[3] = { 4u, 12u, 20u };
    uint32_t       nodos[3];

    for (int k = 0; k < 3; k++) {
        uint8_t *blob = NULL;
        size_t   len  = 0;
        CHECK_EQ(isodendron_train(X, y, n, 2, profundidades[k], 26, 100,
                             &blob, &len), ISODENDRON_OK);
        nodos[k] = blob_field(blob, 12);
        isodendron_free(blob);
    }

    CHECK_EQ(nodos[0], 3u);
    CHECK_EQ(nodos[0], nodos[1]);
    CHECK_EQ(nodos[1], nodos[2]);
}

static void test_predict_argumentos_invalidos(void)
{
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);

    int32_t pred[SEP_N];

    CHECK_EQ(isodendron_predict(NULL, len, X, SEP_N, 2, pred), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_predict(blob, len, NULL, SEP_N, 2, pred), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 2, NULL), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_predict(blob, len, X, 0, 2, pred), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 0, pred), ISODENDRON_ERR_ARG);

    /* Blob truncado o incompleto. */
    CHECK_EQ(isodendron_predict(blob, 16, X, SEP_N, 2, pred), ISODENDRON_ERR_ARG);
    CHECK_EQ(isodendron_predict(blob, len - 1, X, SEP_N, 2, pred), ISODENDRON_ERR_ARG);

    /* Número de features incompatible con el del modelo. */
    int32_t X3[SEP_N * 3];
    for (uint32_t i = 0; i < SEP_N; i++) {
        X3[i * 3u + 0u] = X[i * 2u + 0u];
        X3[i * 3u + 1u] = 0;
        X3[i * 3u + 2u] = 0;
    }
    CHECK_EQ(isodendron_predict(blob, len, X3, SEP_N, 3, pred), ISODENDRON_ERR_ARG);

    /* Magic corrupto. */
    uint8_t saved = blob[0];
    blob[0] = (uint8_t)(saved ^ 0xFFu);
    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 2, pred), ISODENDRON_ERR_ARG);
    blob[0] = saved;
    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 2, pred), ISODENDRON_OK);

    isodendron_free(blob);
}

static void test_predict_blob_sin_nodos(void)
{
    /* Cabecera válida pero n_nodes == 0: no hay árbol que recorrer. */
    uint8_t buf[32];
    blob_write_header(buf, 2, 0, 1, 1, 100, 0u);

    int32_t X[2]    = { 0, 5 };
    int32_t pred[1] = { -1 };

    CHECK_EQ(isodendron_predict(buf, sizeof(buf), X, 1, 2, pred), ISODENDRON_ERR_ARG);
    CHECK_EQ(pred[0], -1);                   /* no se escribió nada */
}

static void test_predict_full_hojas_puras(void)
{
    /*
     * Con el dataset separable y max_deep = 1 las dos hojas son puras,
     * así que P = n_mode / N = 1 en todas las predicciones.
     */
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);

    int32_t pred[SEP_N];
    int32_t nmed[SEP_N];
    int32_t hoja_n[SEP_N];

    CHECK_EQ(isodendron_predict_full(blob, len, X, SEP_N, 2, pred, nmed, hoja_n),
             ISODENDRON_OK);

    int malos = 0;
    for (uint32_t i = 0; i < SEP_N; i++) {
        if (pred[i] != y[i]) {
            malos++;
        }
        if (nmed[i] != SEP_N / 2) {
            malos++;
        }
        if (hoja_n[i] != SEP_N / 2) {
            malos++;
        }
        if (nmed[i] != hoja_n[i]) {      /* P == 1 */
            malos++;
        }
    }
    CHECK_EQ(malos, 0);

    isodendron_free(blob);
}

static void test_predict_full_hoja_impura(void)
{
    /*
     * La raíz parte por f0 == 0. El hijo == tiene 16 filas de target 0
     * (puro); el hijo != tiene 48 filas con targets 1, 2 y 3 a partes
     * iguales: empate a 16, intermedio menor = 2; n_mode = 16 y P = 1/3.
     */
    const uint32_t n = 64;
    static int32_t X[64 * 2];
    static int32_t y[64];
    for (uint32_t i = 0; i < n; i++) {
        X[i * 2u + 0u] = (int32_t)(i % 4u);
        X[i * 2u + 1u] = (int32_t)i;
        y[i] = (int32_t)(i % 4u);
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, n, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);

    /* Dos muestras: una cae en la hoja pura y otra en la impura. */
    int32_t muestra[4] = { 0, 5, 1, 5 };
    int32_t pred[2], nmed[2], hoja_n[2];

    CHECK_EQ(isodendron_predict_full(blob, len, muestra, 2, 2, pred, nmed, hoja_n),
             ISODENDRON_OK);

    /* f0 == 0 -> hoja izquierda: 16 filas, todas de target 0. */
    CHECK_EQ(pred[0], 0);
    CHECK_EQ(nmed[0], 16);
    CHECK_EQ(hoja_n[0], 16);

    /* f0 != 0 -> hoja derecha: 48 filas, con 16 unos, 16 doses y 16 treses.
     * Empate a 16 -> intermedio menor = 2. */
    CHECK_EQ(pred[1], 2);
    CHECK_EQ(nmed[1], 16);
    CHECK_EQ(hoja_n[1], 48);
    /* P = 16/48 = 1/3, comprobado con aritmética entera. */
    CHECK_EQ(nmed[1] * 3, hoja_n[1]);

    isodendron_free(blob);
}

static void test_predict_full_extras_opcionales(void)
{
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);

    int32_t a[SEP_N], b[SEP_N], c[SEP_N], d[SEP_N];

    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 2, a), ISODENDRON_OK);
    CHECK_EQ(isodendron_predict_full(blob, len, X, SEP_N, 2, b, NULL, NULL), ISODENDRON_OK);
    CHECK_EQ(isodendron_predict_full(blob, len, X, SEP_N, 2, c, d, NULL), ISODENDRON_OK);

    int malos = 0;
    for (uint32_t i = 0; i < SEP_N; i++) {
        if (a[i] != b[i]) {
            malos++;   /* con los extras a NULL equivale a isodendron_predict */
        }
        if (a[i] != c[i]) {
            malos++;   /* pedir solo n_mode tampoco cambia la predicción */
        }
    }
    CHECK_EQ(malos, 0);

    /* out_pred es obligatorio. */
    CHECK_EQ(isodendron_predict_full(blob, len, X, SEP_N, 2, NULL, d, d),
             ISODENDRON_ERR_ARG);

    isodendron_free(blob);
}

static void test_distribucion_contenido(void)
{
    /* La raiz parte por f0 == 0. La hoja == tiene solo target 0; la
     * hoja != tiene 1, 2 y 3 con igual conteo, ordenados por target. */
    const uint32_t n = 64;
    static int32_t X[64 * 2];
    static int32_t y[64];
    for (uint32_t i = 0; i < n; i++) {
        X[i * 2u + 0u] = (int32_t)(i % 4u);
        X[i * 2u + 1u] = (int32_t)i;
        y[i] = (int32_t)(i % 4u);
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, n, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);

    int32_t t, c;
    uint32_t off = blob_field(blob, node_offset(1) + 4);
    uint32_t cnt = blob_field(blob, node_offset(1) + 8);
    CHECK_EQ(cnt, 1);
    blob_read_dist(blob, off, &t, &c);
    CHECK_EQ(t, 0);
    CHECK_EQ(c, 16);

    off = blob_field(blob, node_offset(2) + 4);
    cnt = blob_field(blob, node_offset(2) + 8);
    CHECK_EQ(cnt, 3);
    int32_t suma = 0;
    for (uint32_t i = 0; i < cnt; i++) {
        blob_read_dist(blob, off + i * 8u, &t, &c);
        CHECK_EQ(t, (int32_t)(i + 1u));
        CHECK_EQ(c, 16);
        suma += c;
    }
    CHECK_EQ(suma, 48);

    isodendron_free(blob);
}

static void test_distribucion_tope(void)
{
    const uint32_t n = 64;
    static int32_t X[64 * 2];
    static int32_t y[64];
    for (uint32_t i = 0; i < n; i++) {
        X[i * 2u + 0u] = (int32_t)(i % 4u);
        X[i * 2u + 1u] = (int32_t)i;
        y[i] = (int32_t)(i % 4u);
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;

    CHECK_EQ(isodendron_train_mt_cap(X, y, n, 2, 1, 1, 1, 2, 1, &blob, &len),
             ISODENDRON_OK);

    uint32_t off = blob_field(blob, node_offset(2) + 4);
    uint32_t cnt = blob_field(blob, node_offset(2) + 8);
    CHECK_EQ(cnt, 2);

    int32_t t, c, suma = 0;
    for (uint32_t i = 0; i < cnt; i++) {
        blob_read_dist(blob, off + i * 8u, &t, &c);
        CHECK_EQ(t, (int32_t)(i + 1u));
        suma += c;
    }
    CHECK_EQ(suma, 32);

    isodendron_free(blob);
    blob = NULL;

    CHECK_EQ(isodendron_train_mt_cap(X, y, n, 2, 1, 1, 1, 0, 1, &blob, &len),
             ISODENDRON_OK);
    CHECK_EQ(blob_field(blob, 28), 0u);
    CHECK_EQ(blob_field(blob, node_offset(1) + 8), 0u);

    isodendron_free(blob);
}

static void test_refresh_identidad(void)
{
    /* Recalibrar con los mismos datos debe dejar el blob identico. */
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *a = NULL, *b = NULL;
    size_t   la = 0, lb = 0;

    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &a, &la), ISODENDRON_OK);
    CHECK_EQ(isodendron_partial_fit(a, la, X, y, SEP_N, 2, 10, &b, &lb), ISODENDRON_OK);
    CHECK_EQ(la, lb);
    CHECK(b != NULL);
    if (a != NULL && b != NULL && la == lb) {
        /* Acumula: la hoja pasa de 4 a 5 (1 tras decaer + 4 nuevos). */
        int32_t t;
        int32_t c;
        uint32_t off = blob_field(b, node_offset(1) + 4);
        blob_read_dist(b, off, &t, &c);
        CHECK_EQ(t, 0);
        CHECK_EQ(c, 5);
        CHECK_EQ(blob_field(b, node_offset(1) + 12), 5);
    }

    isodendron_free(a);
    isodendron_free(b);
}

static void test_refresh_conserva_estructura(void)
{
    /* Con las etiquetas invertidas el arbol no cambia, pero las hojas si. */
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *a = NULL, *b = NULL;
    size_t   la = 0, lb = 0;
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &a, &la), ISODENDRON_OK);

    for (uint32_t i = 0; i < SEP_N; i++) {
        y[i] = 1 - y[i];
    }
    CHECK_EQ(isodendron_partial_fit(a, la, X, y, SEP_N, 2, 10, &b, &lb), ISODENDRON_OK);

    CHECK_EQ(blob_field(b, 12), blob_field(a, 12));
    CHECK_EQ(blob_field(b, node_offset(0) + 4), blob_field(a, node_offset(0) + 4));
    CHECK_EQ(blob_field(b, node_offset(0) + 8), blob_field(a, node_offset(0) + 8));
    CHECK_EQ(blob_field(b, node_offset(1) + 20), 1);

    uint8_t *malo = NULL;
    size_t   lmalo = 0;
    CHECK_EQ(isodendron_partial_fit(a, la, X, y, SEP_N, 3, 10, &malo, &lmalo),
             ISODENDRON_ERR_ARG);

    uint8_t *c = NULL;
    size_t   lc = 0;
    CHECK_EQ(isodendron_partial_fit(a, la, X, y, SEP_N, 2, 0, &c, &lc), ISODENDRON_OK);
    CHECK_EQ(blob_field(c, 28), 0u);
    isodendron_free(c);

    isodendron_free(a);
    isodendron_free(b);
}

static void test_predict_leaf_indices(void)
{
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &blob, &len), ISODENDRON_OK);

    int32_t  pred[SEP_N];
    int32_t  pred2[SEP_N];
    uint32_t hoja[SEP_N];

    CHECK_EQ(isodendron_predict_leaf(blob, len, X, SEP_N, 2, pred, hoja), ISODENDRON_OK);
    CHECK_EQ(isodendron_predict(blob, len, X, SEP_N, 2, pred2), ISODENDRON_OK);

    int malos = 0;
    for (uint32_t i = 0; i < SEP_N; i++) {
        if (pred[i] != pred2[i]) {
            malos++;
        }
        if (pred[i] != y[i]) {
            malos++;
        }
        /* El nodo referenciado es hoja y su mode es la prediccion. */
        if (blob_field(blob, node_offset(hoja[i])) != 1u) {
            malos++;
        }
        if (blob_field(blob, node_offset(hoja[i]) + 20u) != (uint32_t)pred[i]) {
            malos++;
        }
    }
    CHECK_EQ(malos, 0);

    /* out_leaf_idx es obligatorio. */
    CHECK_EQ(isodendron_predict_leaf(blob, len, X, SEP_N, 2, pred, NULL),
             ISODENDRON_ERR_ARG);

    isodendron_free(blob);
}

static void test_partial_fit_diag(void)
{
    int32_t X[SEP_N * 2];
    int32_t y[SEP_N];
    build_separable(X, y, SEP_N);

    uint8_t *a = NULL, *b = NULL, *c = NULL;
    size_t   la = 0, lb = 0, lc = 0;
    CHECK_EQ(isodendron_train(X, y, SEP_N, 2, 1, 1, 1, &a, &la), ISODENDRON_OK);

    CHECK_EQ(isodendron_partial_fit(a, la, X, y, SEP_N, 2, 10, &b, &lb), ISODENDRON_OK);

    IsodendronHojaAnomala muestras[4];
    IsodendronPartialDiag d;
    memset(&d, 0, sizeof(d));
    d.capacidad = 4;
    d.detalle   = muestras;
    CHECK_EQ(isodendron_partial_fit_diag(a, la, X, y, SEP_N, 2, 10, &c, &lc, &d),
             ISODENDRON_OK);

    /* Con diag, el blob debe ser identico al de sin diag. */
    CHECK_EQ(lb, lc);
    CHECK(b != NULL && c != NULL);
    if (b != NULL && c != NULL && lb == lc) {
        CHECK(memcmp(b, c, lb) == 0);
    }

    /* Dos hojas puras de 4: decaen a 1 y suman otras 4. */
    CHECK_EQ(d.hojas, 2);
    CHECK_EQ(d.decayadas, 2);
    CHECK_EQ(d.vaciadas, 0);
    CHECK_EQ(d.resta_maxima, 3);
    CHECK_EQ(d.anomalas, 0);

    isodendron_free(a);
    isodendron_free(b);
    isodendron_free(c);
}

static void test_train_predict_grande(void){
    /*
     * Ejercita el crecimiento de la tabla de conteos y un árbol de
     * varios cientos de nodos. Comprueba invariantes que deben cumplirse
     * siempre, no valores concretos del árbol.
     */
    const uint32_t n = 5000;
    const uint32_t f = 4;

    int32_t *X = (int32_t *)malloc((size_t)n * f * sizeof(int32_t));
    int32_t *y = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    CHECK(X != NULL);
    CHECK(y != NULL);
    if (X == NULL || y == NULL) {
        free(X);
        free(y);
        return;
    }

    uint32_t state = 12345u;
    int32_t  ymin  = INT32_MAX;
    int32_t  ymax  = INT32_MIN;
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < f; j++) {
            state = state * 1103515245u + 12345u;
            X[i * f + j] = (int32_t)((state >> 16) % 10u);
        }
        y[i] = X[i * f + 0u] * 1000 + X[i * f + 1u] * 100 +
               X[i * f + 2u] * 10 + X[i * f + 3u];
        if (y[i] < ymin) {
            ymin = y[i];
        }
        if (y[i] > ymax) {
            ymax = y[i];
        }
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, n, f, 7, 1, 1, &blob, &len), ISODENDRON_OK);
    CHECK(blob != NULL);
    if (blob == NULL) {
        /* Sin blob no hay nada más que comprobar, y seguir sería un
         * fallo de acceso que ocultaría el error real. */
        free(X);
        free(y);
        return;
    }

    uint32_t n_nodes = blob_field(blob, 12);
    CHECK(n_nodes >= 1u);
    CHECK(n_nodes <= 2u * n);
    CHECK_EQ(len, ISODENDRON_BLOB_HEADER_SIZE + (size_t)n_nodes * ISODENDRON_BLOB_NODE_SIZE
                   + blob_field(blob, 28));
    CHECK_EQ(blob_field(blob, 8), f);
    CHECK_EQ(blob_field(blob, 16), 7);

    /* left_subtree_size debe ser coherente: raíz + subárbol == n_nodes. */
    uint32_t left = blob_field(blob, node_offset(0) + 28);
    uint32_t is_leaf_root = blob_field(blob, node_offset(0) + 0);
    if (is_leaf_root) {
        CHECK_EQ(n_nodes, 1u);
        CHECK_EQ(left, 0u);
    } else {
        CHECK(left >= 1u);
        CHECK(1u + left < n_nodes);
    }

    int32_t *pred = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    CHECK(pred != NULL);
    if (pred != NULL) {
        CHECK_EQ(isodendron_predict(blob, len, X, n, f, pred), ISODENDRON_OK);

        /*
         * Toda predicción es la moda de una hoja, es decir, un valor
         * de y observado: debe caer dentro del rango de entrenamiento.
         */
        int outside = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (pred[i] < ymin || pred[i] > ymax) {
                outside++;
            }
        }
        CHECK_EQ(outside, 0);
    }
    free(pred);

    /* Un solo elemento también debe funcionar. */
    int32_t one = 0;
    CHECK_EQ(isodendron_predict(blob, len, X, 1, f, &one), ISODENDRON_OK);
    CHECK(one >= ymin && one <= ymax);

    isodendron_free(blob);
    free(X);
    free(y);
}

static void test_train_f_uno(void)
{
    /* f == 1 es el caso mínimo: el árbol solo puede dividir por esa feature. */
    const uint32_t n = 50;
    int32_t        X[50];
    int32_t        y[50];
    for (uint32_t i = 0; i < n; i++) {
        X[i] = (int32_t)(i / 5);             /* 10 valores, 5 filas cada uno */
        y[i] = (int32_t)(i / 5);
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    /*
     * Aislar los 10 valores requiere 9 divisiones encadenadas sobre la
     * rama !=, así que hace falta profundidad suficiente para llegar a
     * todas ellas.
     */
    CHECK_EQ(isodendron_train(X, y, n, 1, 12, 1, 1, &blob, &len), ISODENDRON_OK);

    int32_t pred[50];
    CHECK_EQ(isodendron_predict(blob, len, X, n, 1, pred), ISODENDRON_OK);

    int wrong = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (pred[i] != y[i]) {
            wrong++;
        }
    }
    CHECK_EQ(wrong, 0);                      /* cada valor de f0 es una clase pura */

    isodendron_free(blob);
}

static void test_free_nulo(void)
{
    isodendron_free(NULL);                        /* documentado como seguro */
}

static void test_train_max_deep_alto(void)
{
    /*
     * max_deep grande pero n pequeño: la cota de nodos por filas debe
     * evitar que el arena se quede corto y que el árbol crezca sin fin.
     */
    const uint32_t n = 200;
    static int32_t X[200];
    static int32_t y[200];
    for (uint32_t i = 0; i < n; i++) {
        X[i] = (int32_t)(i % 7u);
        y[i] = (int32_t)(i % 3u);
    }

    uint8_t *blob = NULL;
    size_t   len  = 0;
    CHECK_EQ(isodendron_train(X, y, n, 1, 30, 1, 1, &blob, &len), ISODENDRON_OK);

    uint32_t n_nodes = blob_field(blob, 12);
    CHECK(n_nodes <= 2u * n);
    CHECK_EQ(len, ISODENDRON_BLOB_HEADER_SIZE + (size_t)n_nodes * ISODENDRON_BLOB_NODE_SIZE
                   + blob_field(blob, 28));

    int32_t pred[200];
    CHECK_EQ(isodendron_predict(blob, len, X, n, 1, pred), ISODENDRON_OK);

    isodendron_free(blob);
}

/* ================================================================== */
/* Runner                                                              */
/* ================================================================== */

typedef struct {
    const char *name;
    void      (*fn)(void);
} TestCase;

static const TestCase k_tests[] = {
    /* arena */
    { "arena/init_y_destroy",            test_arena_init_destroy },
    { "arena/alineacion",                test_arena_alloc_alignment },
    { "arena/alineacion_no_potencia_2",  test_arena_alloc_align_no_potencia_de_dos },
    { "arena/alloc_cero",                test_arena_alloc_zero },
    { "arena/agotamiento",               test_arena_exhaustion },
    { "arena/alloc_array",               test_arena_alloc_array },
    { "arena/rewind",                    test_arena_rewind },
    { "arena/rewind_tras_fallo",         test_arena_rewind_tras_fallo },

    /* counts */
    { "counts/basico",                   test_counts_basico },
    { "counts/claves_extremas",          test_counts_claves_extremas },
    { "counts/crecimiento",              test_counts_crecimiento },
    { "counts/reset",                    test_counts_reset },
    { "counts/iterate",                  test_counts_iterate },
    { "counts/max_key",                  test_counts_max_key },
    { "counts/nulos",                    test_counts_nulos },

    /* blob */
    { "blob/total_size",                 test_blob_total_size },
    { "blob/roundtrip",                  test_blob_roundtrip },
    { "blob/hoja_normalizada",           test_blob_hoja_normalizada },
    { "blob/header_errores",             test_blob_header_errores },
    { "blob/node_limites",               test_blob_node_limites },
    { "blob/node_ptr",                   test_blob_node_ptr },
    { "blob/check_compat",               test_blob_check_compat },
    { "blob/escritura_nula",             test_blob_escritura_nula },

    /* API pública */
    { "isodendron/version",                   test_version },
    { "isodendron/train_argumentos",          test_train_argumentos_invalidos },
    { "isodendron/train_predict_exacto",      test_train_predict_exacto },
    { "isodendron/train_determinista",        test_train_determinista },
    { "isodendron/train_multihilo_identico",  test_train_multihilo_identico },
    { "isodendron/multihilo_max_deep_alto",   test_multihilo_max_deep_alto },
    { "isodendron/target_constante",          test_target_constante },
    { "isodendron/p_min_pureza_del_nodo",     test_p_min_pureza_del_nodo },
    { "isodendron/p_min_cero_para_siempre",   test_p_min_cero_para_siempre },
    { "isodendron/p_min_independiente_deep",  test_p_min_independiente_de_max_deep },
    { "isodendron/predict_argumentos",        test_predict_argumentos_invalidos },
    { "isodendron/predict_blob_sin_nodos",    test_predict_blob_sin_nodos },
    { "isodendron/predict_full_puras",        test_predict_full_hojas_puras },
    { "isodendron/predict_full_impura",       test_predict_full_hoja_impura },
    { "isodendron/predict_full_extras",       test_predict_full_extras_opcionales },
    { "isodendron/distribucion_contenido", test_distribucion_contenido },
    { "isodendron/distribucion_tope",      test_distribucion_tope },
    { "isodendron/refresh_identidad",        test_refresh_identidad },
    { "isodendron/refresh_estructura",       test_refresh_conserva_estructura },
    { "isodendron/partial_fit_diag",  test_partial_fit_diag },
    { "isodendron/predict_leaf",             test_predict_leaf_indices },
    { "isodendron/train_predict_grande",      test_train_predict_grande },
    { "isodendron/train_f_uno",               test_train_f_uno },
    { "isodendron/train_max_deep_alto",       test_train_max_deep_alto },
    { "isodendron/free_nulo",                 test_free_nulo },
};

int main(void)
{
    const size_t total = sizeof(k_tests) / sizeof(k_tests[0]);
    size_t       failed_tests = 0;

    /*
     * Sin búfer: si un caso provoca un fallo de acceso, el volcado
     * anterior no se pierde y se ve exactamente dónde murió.
     */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("ISODENDRON %s - suite de pruebas (%u casos)\n\n",
           isodendron_version(), (unsigned)total);

    for (size_t i = 0; i < total; i++) {
        g_current   = k_tests[i].name;
        int before  = g_failures;

        k_tests[i].fn();

        if (g_failures == before) {
            printf("  ok    %s\n", k_tests[i].name);
        } else {
            printf("  FALLA %s (%d comprobaciones fallidas)\n",
                   k_tests[i].name, g_failures - before);
            failed_tests++;
        }
    }

    printf("\n%u casos, %d comprobaciones, %d fallos\n",
           (unsigned)total, g_checks, g_failures);

    if (failed_tests == 0) {
        printf("RESULTADO: TODO OK\n");
        return 0;
    }

    printf("RESULTADO: %u caso(s) con fallos\n", (unsigned)failed_tests);
    return 1;
}








