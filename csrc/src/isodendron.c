/*
 * isodendron.c — Implementación del algoritmo ISODENDRON.
 *
 * Une las tres piezas anteriores:
 *   - arena:  toda la memoria del entrenamiento sale de un solo bloque.
 *   - counts: tabla hash reutilizable para contar valores por feature.
 *   - blob:   serialización del árbol construido.
 *
 * El algoritmo es un árbol binario determinista. En cada nodo busca el
 * split (feature == valor) que mejor equilibra el tamaño de las dos
 * ramas, con desempate por número de valores únicos y orden de aparición.
 * El target solo se usa para resumir cada nodo al final.
 *
 * Ver isodendron.h para el contrato público.
 */
#include "isodendron.h"
#include "arena.h"
#include "counts.h"
#include "blob.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Hilos                                                               */
/* ------------------------------------------------------------------ */

/*
 * Abstracción mínima de hilos y exclusión mutua: solo hace falta crear
 * un hilo, esperarlo y proteger un contador, así que la superficie es
 * deliberadamente pequeña. En Windows se usa la API nativa; en el resto,
 * pthreads.
 */
#ifdef _WIN32
#  include <windows.h>
typedef HANDLE           IsodendronHilo;
typedef CRITICAL_SECTION IsodendronMutex;
static void hm_init(IsodendronMutex *m)    { InitializeCriticalSection(m); }
static void hm_lock(IsodendronMutex *m)    { EnterCriticalSection(m); }
static void hm_unlock(IsodendronMutex *m)  { LeaveCriticalSection(m); }
static void hm_destroy(IsodendronMutex *m) { DeleteCriticalSection(m); }
#else
#  include <pthread.h>
typedef pthread_t       IsodendronHilo;
typedef pthread_mutex_t IsodendronMutex;
static void hm_init(IsodendronMutex *m)    { (void)pthread_mutex_init(m, NULL); }
static void hm_lock(IsodendronMutex *m)    { (void)pthread_mutex_lock(m); }
static void hm_unlock(IsodendronMutex *m)  { (void)pthread_mutex_unlock(m); }
static void hm_destroy(IsodendronMutex *m) { (void)pthread_mutex_destroy(m); }
#endif

/* Número máximo de subtareas simultáneas y tamaño mínimo de un nodo
 * para que compense bifurcarlo. */
#define ISODENDRON_MAX_TAREAS     16u
#define ISODENDRON_MIN_FILAS_FORK 1024u

/* ------------------------------------------------------------------ */
/* Tipos internos                                                      */
/* ------------------------------------------------------------------ */

/*
 * Nodo del árbol en construcción. Vive en el arena y solo existe
 * durante isodendron_train. La serialización a blob se hace al final.
 */
/*
 * Entrada de la distribución del target en una hoja: un valor y cuántas
 * observaciones lo tienen.
 */
typedef struct {
    int32_t target;
    int32_t count;
} DistEntry;

typedef struct TreeNode {
    int      is_leaf;
    int32_t  feature_idx;
    int32_t  value;
    int32_t  N;
    int32_t  K;
    int32_t  mode;
    int32_t  n_mode;
    uint32_t left_subtree_size;   /* relleno en compute_sizes */
    DistEntry *dist;              /* en el arena; NULL si interno */
    uint32_t dist_len;            /* entradas guardadas (<= max_targets) */
    struct TreeNode *left;        /* rama == */
    struct TreeNode *right;       /* rama != */
} TreeNode;

/*
 * Estado del entrenamiento. Se pasa por puntero a todas las funciones
 * internas para no inflar las firmas.
 */
typedef struct {
    const int32_t *X;             /* matriz row-major n*f */
    const int32_t *y;             /* targets, tamaño n */
    uint32_t       n;
    uint32_t       f;
    uint32_t       max_deep;
    uint32_t       p_min_num;
    uint32_t       p_min_den;
    uint32_t       max_targets;   /* tope de entradas por hoja */

    DistEntry     *dist_buf;      /* scratch de tamaño max_targets */

    int32_t       *idx;           /* array global de índices, tamaño n */
    int32_t       *scratch_keys;  /* buffer reutilizable, tamaño n */
    uint64_t      *features_used; /* bitmask, (f + 63) / 64 words */

    Counts        *counts;        /* tabla reutilizable */
    Arena         *arena;
    uint32_t       node_count;

    struct ParCtx *par;           /* NULL = construcción secuencial */
} BuildCtx;

/*
 * Espacio de trabajo de un hilo: su propio contexto (con scratch, tabla
 * de conteos y arena propios) más el arena en sí. Comparte con el resto
 * X, y, y el array idx, que es de solo lectura por rangos disjuntos.
 */
typedef struct {
    BuildCtx ctx;
    Arena    almacen;
} IsodendronWorker;

typedef struct ParCtx ParCtx;

/*
 * Una subtarea: la construcción de un subárbol concreto en otro hilo.
 */
typedef struct {
    BuildCtx *ctx;        /* espacio de trabajo del hilo */
    uint32_t  lo, hi;     /* rango de idx que le toca */
    uint32_t  depth;
    TreeNode *resultado;
    IsodendronHilo hilo;
} IsodendronTarea;

/*
 * Estado compartido del entrenamiento en paralelo. `tareas_usadas` va
 * repartiendo los espacios de trabajo; como el árbol de bifurcación
 * tiene como mucho 2^fork_depth - 1 nodos, reservar esa cantidad basta
 * y no hace falta devolverlos.
 */
struct ParCtx {
    unsigned    fork_depth;
    size_t      words;
    uint32_t    f;
    uint32_t    max_targets;
    unsigned    n_tareas;
    unsigned    tareas_usadas;
    IsodendronMutex  mutex;
    IsodendronWorker *workers;
    IsodendronTarea  *tareas;
};

/* ------------------------------------------------------------------ */
/* Utilidades                                                          */
/* ------------------------------------------------------------------ */

static int cmp_i32(const void *a, const void *b)
{
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

/* ------------------------------------------------------------------ */
/* Resumen de un nodo                                                  */
/* ------------------------------------------------------------------ */

/*
 * Calcula N, K, mode y n_mode del subconjunto idx[lo..hi).
 * Usa ctx->counts y ctx->scratch_keys como scratch, y los deja
 * en un estado no especificado al salir.
 *
 * mode es el valor del target MÁS FRECUENTE del nodo, no la mediana de
 * rango. Para un target categórico la moda es la clase más probable. Si
 * varios valores empatan en el máximo gana el intermedio menor del
 * conjunto empatado; el valor menor solo decide si empatan dos.
 */
static int summarize(BuildCtx *ctx, uint32_t lo, uint32_t hi,
                     int32_t *out_N, int32_t *out_K,
                     int32_t *out_mode, int32_t *out_n_mode,
                     uint32_t *out_dist_len)
{
    uint32_t n = hi - lo;
    *out_N = (int32_t)n;
    if (n == 0) {
        *out_K = 0;
        *out_mode = 0;
        *out_n_mode = 0;
        return 0;
    }

    counts_reset(ctx->counts);
    uint32_t k = 0;
    for (uint32_t i = lo; i < hi; i++) {
        int32_t t = ctx->y[ctx->idx[i]];
        uint32_t before = counts_size(ctx->counts);
        counts_inc(ctx->counts, t);
        if (counts_size(ctx->counts) > before) {
            ctx->scratch_keys[k++] = t;
        }
    }
    if (counts_failed(ctx->counts)) {
        return -1;
    }

    *out_K = (int32_t)k;

    qsort(ctx->scratch_keys, k, sizeof(int32_t), cmp_i32);

    /*
     * Moda: el valor con más observaciones. Si varios empatan en el
     * máximo, gana el INTERMEDIO MENOR del conjunto empatado, de forma
     * que el valor menor solo decide cuando empatan exactamente dos.
     *
     * Así se evita que todos los empates caigan siempre al valor mínimo,
     * que convertiría al árbol en un predictor sesgado hacia esa clase.
     * El precio es que el desempate depende del orden numérico de las
     * etiquetas: el modelo deja de ser indiferente a su renumeración.
     */
    uint32_t best = 0;
    for (uint32_t j = 0; j < k; j++) {
        uint32_t c = counts_get(ctx->counts, ctx->scratch_keys[j]);
        if (c > best) {
            best = c;
        }
    }

    uint32_t first  = 0;
    uint32_t n_tied = 0;
    int      visto  = 0;
    for (uint32_t j = 0; j < k; j++) {
        if (counts_get(ctx->counts, ctx->scratch_keys[j]) == best) {
            if (!visto) {
                first = j;
                visto = 1;
            }
            n_tied++;
        }
    }

    int32_t mode = ctx->scratch_keys[first + (n_tied - 1u) / 2u];

    *out_mode = mode;
    *out_n_mode = (int32_t)best;

    /*
     * Distribución: las `max_targets` entradas más frecuentes, en orden
     * de conteo descendente y, a igualdad, de target ascendente. Se
     * mantiene ordenada por inserción en un buffer de tamaño fijo, así
     * que el coste es O(entradas_distintas * max_targets).
     */
    DistEntry *dst = ctx->dist_buf;
    uint32_t   cap = ctx->max_targets;
    uint32_t   len = 0;

    for (uint32_t j = 0; j < k && cap > 0; j++) {
        int32_t  t = ctx->scratch_keys[j];
        int32_t  c = (int32_t)counts_get(ctx->counts, t);
        uint32_t p;

        if (len < cap) {
            p = len;
            len++;
        } else if (c > dst[cap - 1u].count
                   || (c == dst[cap - 1u].count && t < dst[cap - 1u].target)) {
            p = cap - 1u;   /* desplaza y descarta la última */
        } else {
            continue;
        }

        while (p > 0 && (dst[p - 1u].count < c
                         || (dst[p - 1u].count == c
                             && dst[p - 1u].target > t))) {
            dst[p] = dst[p - 1u];
            p--;
        }
        dst[p].target = t;
        dst[p].count  = c;
    }

    *out_dist_len = len;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Búsqueda del mejor split                                            */
/* ------------------------------------------------------------------ */

/*
 * Busca el mejor split (feature == valor) dentro de idx[lo..hi).
 *
 * Criterio: maximizar min(n_igual, n_distinto).
 * Desempate: mayor número de valores únicos en la feature, y en caso
 * de empate total, el primero en el orden de features (índice menor).
 *
 * Devuelve 1 si encontró un split válido, 0 en caso contrario.
 */
static int find_best_split(BuildCtx *ctx, uint32_t lo, uint32_t hi,
                           int32_t *out_feature, int32_t *out_value)
{
    uint32_t N = hi - lo;
    int      has_best = 0;
    uint32_t best_feature = 0;
    int32_t  best_value = 0;
    uint32_t best_score = 0;
    uint32_t best_unicos = 0;

    for (uint32_t f = 0; f < ctx->f; f++) {
        uint32_t word = f / 64u;
        uint64_t mask = (uint64_t)1 << (f % 64u);
        if (ctx->features_used[word] & mask) {
            continue;
        }

        counts_reset(ctx->counts);
        uint32_t k = 0;
        for (uint32_t i = lo; i < hi; i++) {
            int32_t v = ctx->X[(size_t)ctx->idx[i] * ctx->f + f];
            uint32_t before = counts_size(ctx->counts);
            counts_inc(ctx->counts, v);
            if (counts_size(ctx->counts) > before) {
                ctx->scratch_keys[k++] = v;
            }
        }
        if (counts_failed(ctx->counts)) {
            return 0;
        }

        uint32_t unicos = k;
        for (uint32_t j = 0; j < k; j++) {
            int32_t v = ctx->scratch_keys[j];
            uint32_t c = counts_get(ctx->counts, v);
            uint32_t c_der = N - c;
            uint32_t score = (c < c_der) ? c : c_der;
            if (!has_best
                || score > best_score
                || (score == best_score && unicos > best_unicos)) {
                has_best      = 1;
                best_feature  = f;
                best_value    = v;
                best_score    = score;
                best_unicos   = unicos;
            }
        }
    }

    if (!has_best || best_score == 0) {
        return 0;
    }
    *out_feature = (int32_t)best_feature;
    *out_value   = best_value;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Partición in-place                                                  */
/* ------------------------------------------------------------------ */

/*
 * Reordena idx[lo..hi) dejando primero las filas con feature == value
 * y después las que tienen feature != value. Devuelve el índice del
 * primer elemento de la segunda zona (el "punto de corte").
 */
static uint32_t partition(BuildCtx *ctx, uint32_t lo, uint32_t hi,
                          int32_t feature, int32_t value)
{
    uint32_t write = lo;
    for (uint32_t read = lo; read < hi; read++) {
        int32_t v = ctx->X[(size_t)ctx->idx[read] * ctx->f + feature];
        if (v == value) {
            int32_t tmp   = ctx->idx[write];
            ctx->idx[write] = ctx->idx[read];
            ctx->idx[read]  = tmp;
            write++;
        }
    }
    return write;
}

/* ------------------------------------------------------------------ */
/* Soporte de construcción en paralelo                                 */
/* ------------------------------------------------------------------ */

static TreeNode *build_tree(BuildCtx *ctx, uint32_t lo, uint32_t hi,
                            uint32_t depth);

static size_t compute_arena_size(uint32_t n, uint32_t f, uint32_t max_deep,
                                 uint32_t max_targets, int con_idx);

static void hilo_cuerpo(IsodendronTarea *t)
{
    t->resultado = build_tree(t->ctx, t->lo, t->hi, t->depth);
}

#ifdef _WIN32
static DWORD WINAPI hilo_entry(LPVOID arg)
{
    hilo_cuerpo((IsodendronTarea *)arg);
    return 0;
}
#else
static void *hilo_entry(void *arg)
{
    hilo_cuerpo((IsodendronTarea *)arg);
    return NULL;
}
#endif

static int hilo_lanzar(IsodendronHilo *h, IsodendronTarea *t)
{
#ifdef _WIN32
    *h = CreateThread(NULL, 0, hilo_entry, (LPVOID)t, 0, NULL);
    return (*h != NULL) ? 0 : -1;
#else
    return (pthread_create(h, NULL, hilo_entry, t) == 0) ? 0 : -1;
#endif
}

static void hilo_unir(IsodendronHilo h)
{
#ifdef _WIN32
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
#else
    (void)pthread_join(h, NULL);
#endif
}

/*
 * Reserva espacio de trabajo para un subárbol concreto.
 *
 * El arena se dimensiona por las filas REALES del subárbol (m) y por la
 * profundidad que le queda, no por el árbol completo. Así el total entre
 * todos los hilos ronda el de la versión secuencial en vez de
 * multiplicarlo por el número de hilos, que era lo que hacía inviable
 * entrenar con miles de filas y max_deep alto.
 *
 * Coste: un solo arena_init por bifurcación —como mucho 2^fork_depth-1
 * en todo el entrenamiento, es decir 1 con 2 hilos y 15 con 16—, no una
 * asignación por nodo.
 */
static IsodendronTarea *par_reservar(ParCtx *par, uint32_t m,
                                uint32_t profundidad_restante)
{
    IsodendronTarea *t = NULL;
    unsigned    i = 0;

    hm_lock(&par->mutex);
    if (par->tareas_usadas < par->n_tareas) {
        i = par->tareas_usadas++;
        t = &par->tareas[i];
    }
    hm_unlock(&par->mutex);

    if (t == NULL) {
        return NULL;
    }

    IsodendronWorker *w  = &par->workers[i];
    size_t       sz = compute_arena_size(m, par->f, profundidad_restante,
                                         par->max_targets, 0);

    if (arena_init(&w->almacen, sz) != 0) {
        t->ctx = NULL;
        return NULL;
    }

    BuildCtx *c = &w->ctx;

    c->arena         = &w->almacen;
    c->scratch_keys  = (int32_t *)arena_alloc_array(&w->almacen, m,
                                                    sizeof(int32_t));
    c->features_used = (uint64_t *)arena_alloc_array(&w->almacen, par->words,
                                                     sizeof(uint64_t));
    c->counts        = counts_new(&w->almacen, 16);
    c->max_targets   = par->max_targets;
    c->dist_buf      = (par->max_targets == 0)
                       ? NULL
                       : (DistEntry *)arena_alloc_array(&w->almacen,
                                                        par->max_targets,
                                                        sizeof(DistEntry));

    if (c->scratch_keys == NULL || c->features_used == NULL
        || c->counts == NULL
        || (par->max_targets > 0 && c->dist_buf == NULL)) {
        t->ctx = NULL;
        return NULL;
    }
    memset(c->features_used, 0, par->words * sizeof(uint64_t));

    t->ctx = c;
    return t;
}

/*
 * Prepara la subtarea del hijo !=. Su features_used es exactamente el
 * actual del padre, porque en la versión secuencial el veto de la
 * feature se levanta justo antes de construir esa rama.
 */
static void par_preparar_hijo(IsodendronTarea *t, const BuildCtx *padre,
                              uint32_t lo, uint32_t hi, uint32_t depth)
{
    BuildCtx *hijo = t->ctx;

    t->lo        = lo;
    t->hi        = hi;
    t->depth     = depth;
    t->resultado = NULL;

    hijo->X          = padre->X;
    hijo->y          = padre->y;
    hijo->n          = padre->n;
    hijo->f          = padre->f;
    hijo->max_deep   = padre->max_deep;
    hijo->p_min_num  = padre->p_min_num;
    hijo->p_min_den  = padre->p_min_den;
    hijo->idx        = padre->idx;    /* COMPARTIDO, rangos disjuntos */
    hijo->par        = padre->par;
    hijo->node_count = 0;
    memcpy(hijo->features_used, padre->features_used,
           ((ParCtx *)padre->par)->words * sizeof(uint64_t));
}

/*
 * Libera los espacios de trabajo. Debe llamarse solo después de
 * serializar el árbol, porque los nodos viven en esos arenas.
 */
static void par_liberar(ParCtx *par)
{
    if (par->workers == NULL) {
        return;
    }
    hm_destroy(&par->mutex);
    for (unsigned i = 0; i < par->n_tareas; i++) {
        arena_destroy(&par->workers[i].almacen);
    }
    free(par->workers);
    free(par->tareas);
    par->workers  = NULL;
    par->tareas   = NULL;
    par->n_tareas = 0;
}

/* ------------------------------------------------------------------ */
/* Construcción recursiva del árbol                                    */
/* ------------------------------------------------------------------ */

/*
 * Copia la distribución calculada en el scratch al nodo hoja. Devuelve
 * el nodo, o NULL si no hay arena para ella.
 */
static TreeNode *hoja_con_distribucion(BuildCtx *ctx, TreeNode *node,
                                       uint32_t len)
{
    node->dist     = NULL;
    node->dist_len = 0;
    if (len == 0) {
        return node;
    }

    DistEntry *d = (DistEntry *)arena_alloc(ctx->arena,
                                            (size_t)len * sizeof(DistEntry), 4);
    if (d == NULL) {
        return NULL;
    }
    memcpy(d, ctx->dist_buf, (size_t)len * sizeof(DistEntry));

    node->dist     = d;
    node->dist_len = len;
    return node;
}

static TreeNode *build_tree(BuildCtx *ctx, uint32_t lo, uint32_t hi,
                            uint32_t depth)
{
    uint32_t N = hi - lo;

    int32_t  sN, sK, sMode, sNMode;
    uint32_t sDistLen;
    if (summarize(ctx, lo, hi, &sN, &sK, &sMode, &sNMode, &sDistLen) != 0) {
        return NULL;
    }

    TreeNode *node = (TreeNode *)arena_alloc(ctx->arena,
                                             sizeof(TreeNode), 8);
    if (node == NULL) {
        return NULL;
    }
    ctx->node_count++;

    node->is_leaf          = 1;
    node->feature_idx      = 0;
    node->value            = 0;
    node->N                = sN;
    node->K                = sK;
    node->mode             = sMode;
    node->n_mode           = sNMode;
    node->left_subtree_size = 0;
    node->dist             = NULL;
    node->dist_len         = 0;
    node->left             = NULL;
    node->right            = NULL;

    /* Condición de parada por profundidad. */
    if (depth >= ctx->max_deep) {
        return hoja_con_distribucion(ctx, node, sDistLen);
    }

    /*
     * Condición de parada por pureza, SIEMPRE activa e independiente de
     * los parámetros: si todos los targets del nodo son iguales (K == 1)
     * no hay nada que ganar dividiéndolo, porque los dos hijos heredarían
     * la misma clase. Antes esto solo ocurría con P_MIN = 1/1; ahora vale
     * también con P_MIN menor, y evita gastar nodos en regiones ya puras
     * sin cambiar ninguna predicción.
     */
    if (sK <= 1) {
        return hoja_con_distribucion(ctx, node, sDistLen);
    }

    /* Condición de parada por pureza de la moda. P_MIN es la
     * probabilidad mínima del valor más frecuente, P = n_mode / N. Si el
     * nodo ya es suficientemente puro, se queda como hoja. La condición
     * es intrínseca al nodo, así que aplica también a la raíz. */
    if ((uint64_t)sNMode * ctx->p_min_den >=
        (uint64_t)N * ctx->p_min_num) {
        return hoja_con_distribucion(ctx, node, sDistLen);
    }

    /*
     * Los tres caminos siguientes dejan el nodo como hoja sin dividirlo.
     * TODOS deben pasar por hoja_con_distribucion: antes devolvian el
     * nodo tal cual y la hoja se quedaba con N y K correctos pero con la
     * distribucion vacia (payload_len = 0), que es lo que hacia
     * desaparecer hojas en partial_fit.
     */

    /* Un nodo con 0 o 1 filas no puede dividirse. */
    if (N <= 1) {
        return hoja_con_distribucion(ctx, node, sDistLen);
    }

    int32_t feature, value;
    if (!find_best_split(ctx, lo, hi, &feature, &value)) {
        return hoja_con_distribucion(ctx, node, sDistLen);
    }

    uint32_t mid = partition(ctx, lo, hi, feature, value);
    if (mid == lo || mid == hi) {
        return hoja_con_distribucion(ctx, node, sDistLen);
    }

    /* Vetar la feature en la rama == durante su construcción. */
    uint32_t word = (uint32_t)feature / 64u;
    uint64_t mask = (uint64_t)1 << ((uint32_t)feature % 64u);

    /*
     * Bifurcación: el hijo != se construye en otro hilo mientras este
     * sigue con el hijo ==. El veto de la feature se aplica dentro de la
     * rama ==, así que el hijo != arranca con el features_used actual,
     * que es justo lo que copia par_preparar_hijo.
     */
    IsodendronTarea *tarea = NULL;
    if (ctx->par != NULL
        && depth < ctx->par->fork_depth
        && (hi - lo) >= ISODENDRON_MIN_FILAS_FORK) {
        uint32_t restante = (ctx->max_deep > depth + 1)
                              ? (ctx->max_deep - (depth + 1)) : 0u;
        tarea = par_reservar(ctx->par, hi - mid, restante);
        if (tarea != NULL) {
            par_preparar_hijo(tarea, ctx, mid, hi, depth + 1);
            if (hilo_lanzar(&tarea->hilo, tarea) != 0) {
                tarea = NULL;   /* no se pudo crear: se hace secuencial */
            }
        }
    }

    ctx->features_used[word] |= mask;
    TreeNode *left = build_tree(ctx, lo, mid, depth + 1);
    ctx->features_used[word] &= ~mask;

    TreeNode *right = NULL;
    if (tarea != NULL) {
        hilo_unir(tarea->hilo);
        right = tarea->resultado;
    }

    if (left == NULL) {
        return NULL;
    }
    if (tarea == NULL) {
        right = build_tree(ctx, mid, hi, depth + 1);
    }
    if (right == NULL) {
        return NULL;
    }

    node->is_leaf     = 0;
    node->feature_idx = feature;
    node->value       = value;
    node->left        = left;
    node->right       = right;
    return node;
}

/* ------------------------------------------------------------------ */
/* Tamaños de subárbol                                                 */
/* ------------------------------------------------------------------ */

/*
 * Recorre el árbol en postorden calculando left_subtree_size. Devuelve
 * el tamaño total del subárbol cuya raíz es `n` (número de nodos).
 */
static uint32_t compute_sizes(TreeNode *n)
{
    if (n == NULL) {
        return 0;
    }
    if (n->is_leaf) {
        n->left_subtree_size = 0;
        return 1;
    }
    uint32_t l = compute_sizes(n->left);
    uint32_t r = compute_sizes(n->right);
    n->left_subtree_size = l;
    return 1 + l + r;
}

/* ------------------------------------------------------------------ */
/* Serialización                                                       */
/* ------------------------------------------------------------------ */

/*
 * Serializa el árbol en preorden. Los nodos van delante y el payload de
 * distribuciones detrás; a cada hoja se le anota el offset absoluto de su
 * bloque de entradas.
 */
static void serialize(const TreeNode *n, uint8_t *buf, uint32_t *pos,
                      uint32_t payload_base, uint32_t *payload_pos)
{
    BlobNode bn;
    bn.is_leaf           = (uint32_t)n->is_leaf;
    bn.N                 = n->N;
    bn.K                 = n->K;
    bn.mode              = n->mode;
    bn.n_mode            = n->n_mode;
    bn.left_subtree_size = n->left_subtree_size;

    if (n->is_leaf) {
        /* En una hoja estos dos campos llevan el payload, no feature ni
         * umbral. */
        bn.feature_idx       = (int32_t)(payload_base + *payload_pos);
        bn.value             = (int32_t)n->dist_len;
        bn.left_subtree_size = 0;

        for (uint32_t i = 0; i < n->dist_len; i++) {
            blob_write_dist(buf, payload_base + *payload_pos,
                            n->dist[i].target, n->dist[i].count);
            *payload_pos += BLOB_DIST_ENTRY_SIZE;
        }
    } else {
        bn.feature_idx = n->feature_idx;
        bn.value       = n->value;
    }

    blob_write_node(buf, *pos, &bn);
    (*pos)++;

    if (!n->is_leaf) {
        serialize(n->left, buf, pos, payload_base, payload_pos);
        serialize(n->right, buf, pos, payload_base, payload_pos);
    }
}

/* Bytes que ocupará el payload de todo el árbol. */
static uint32_t payload_bytes_de(const TreeNode *n)
{
    if (n == NULL) {
        return 0;
    }
    if (n->is_leaf) {
        return (uint32_t)n->dist_len * BLOB_DIST_ENTRY_SIZE;
    }
    return payload_bytes_de(n->left) + payload_bytes_de(n->right);
}

/* ------------------------------------------------------------------ */
/* Dimensionado del arena                                              */
/* ------------------------------------------------------------------ */

/*
 * Calcula una cota superior holgada para el arena. Cada componente
 * tiene su propio margen. Preferimos sobreestimar que reventar a
 * mitad de entrenamiento, porque el arena es un único malloc y el
 * desperdicio se libera al terminar.
 *
 * Componentes:
 *   - Nodos del árbol (cota por profundidad y por número de filas).
 *   - idx global (n * 4).
 *   - scratch_keys (n * 4).
 *   - features_used ((f + 63) / 64 * 8).
 *   - counts, contando resizes (~24n bytes de margen).
 */
static size_t compute_arena_size(uint32_t n, uint32_t f, uint32_t max_deep,
                                 uint32_t max_targets, int con_idx)
{
    size_t max_by_deep;
    if (max_deep >= 31u) {
        max_by_deep = (size_t)2 * n;   /* saturación: cota por filas */
    } else {
        max_by_deep = ((size_t)1 << (max_deep + 1)) - 1;
    }
    size_t max_by_rows = (size_t)2 * n;
    size_t max_nodes = (max_by_deep < max_by_rows) ? max_by_deep : max_by_rows;

    size_t node_bytes    = max_nodes * sizeof(TreeNode);
    /* Los hilos trabajadores no alojan idx: es compartido y vive en el
     * arena principal. */
    size_t idx_bytes     = con_idx ? (size_t)n * sizeof(int32_t) : 0;
    size_t scratch_bytes = (size_t)n * sizeof(int32_t);
    size_t words         = ((size_t)f + 63u) / 64u;
    size_t bits_bytes    = words * sizeof(uint64_t);

    /*
     * La tabla de conteos puede llegar a tener hasta n claves distintas
     * (caso peor: un target distinto por fila), y cada resize deja
     * huérfano dentro del arena el array de entradas anterior, porque no
     * hay free individual.
     *
     * Con crecimiento al 75% de carga y duplicación de capacidad, si k
     * es el número de claves distintas la capacidad final cumple
     * cap >= 4k/3 y cap < 8k/3, y el total de bytes de todos los arrays
     * encadenados es 8 * (2*cap - 16) < 128k/3, o sea menos de 42.7
     * bytes por fila. Se redondea a 48 para dejar margen.
     *
     * Subestimar esta componente no da un árbol peor: hace que
     * isodendron_train devuelva ISODENDRON_ERR_MEM con datasets que sí caben.
     */
    size_t counts_bytes  = (size_t)n * 48u + 1024u;

    /*
     * Distribuciones: como mucho hay una hoja por fila, y cada una
     * guarda hasta ISODENDRON_MAX_TARGETS entradas de 8 bytes.
     */
    size_t dist_bytes    = (size_t)n * (size_t)max_targets
                         * BLOB_DIST_ENTRY_SIZE;

    size_t total = node_bytes + idx_bytes + scratch_bytes
                 + bits_bytes + counts_bytes + dist_bytes + 1024u;
    return total;
}

/* ------------------------------------------------------------------ */
/* API pública: entrenamiento                                          */
/* ------------------------------------------------------------------ */

int isodendron_train_mt_cap(const int32_t *X,
                   const int32_t *y,
                   uint32_t       n,
                   uint32_t       f,
                   uint32_t       max_deep,
                   uint32_t       p_min_num,
                   uint32_t       p_min_den,
                   uint32_t       max_targets,
                   uint32_t       n_threads,
                   uint8_t      **out_blob,
                   size_t        *out_len)
{
    if (X == NULL || y == NULL || out_blob == NULL || out_len == NULL) {
        return ISODENDRON_ERR_ARG;
    }
    if (n == 0 || f == 0 || max_deep == 0 || p_min_den == 0) {
        return ISODENDRON_ERR_ARG;
    }
    if (n_threads == 0) {
        n_threads = 1;
    }

    size_t arena_size = compute_arena_size(n, f, max_deep, max_targets, 1);

    Arena arena;
    if (arena_init(&arena, arena_size) != 0) {
        return ISODENDRON_ERR_MEM;
    }

    BuildCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.X          = X;
    ctx.y          = y;
    ctx.n          = n;
    ctx.f          = f;
    ctx.max_deep   = max_deep;
    ctx.p_min_num  = p_min_num;
    ctx.p_min_den  = p_min_den;
    ctx.arena      = &arena;
    ctx.node_count = 0;
    ctx.max_targets = max_targets;

    ctx.dist_buf = (max_targets == 0)
                   ? NULL
                   : (DistEntry *)arena_alloc_array(&arena, max_targets,
                                                    sizeof(DistEntry));
    if (max_targets > 0 && ctx.dist_buf == NULL) {
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }

    ctx.idx = (int32_t *)arena_alloc_array(&arena, n, sizeof(int32_t));
    if (ctx.idx == NULL) {
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }
    for (uint32_t i = 0; i < n; i++) {
        ctx.idx[i] = (int32_t)i;
    }

    ctx.scratch_keys = (int32_t *)arena_alloc_array(&arena, n, sizeof(int32_t));
    if (ctx.scratch_keys == NULL) {
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }

    size_t words = ((size_t)f + 63u) / 64u;
    ctx.features_used = (uint64_t *)arena_alloc_array(&arena, words, sizeof(uint64_t));
    if (ctx.features_used == NULL) {
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }
    memset(ctx.features_used, 0, words * sizeof(uint64_t));

    ctx.counts = counts_new(&arena, 16);
    if (ctx.counts == NULL) {
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }

    /* --- espacios de trabajo para construir el árbol en paralelo --- */
    ParCtx   par;
    unsigned fork_depth = 0;
    unsigned n_tareas   = 0;

    memset(&par, 0, sizeof(par));
    while ((1u << fork_depth) < n_threads && fork_depth < 8u) {
        fork_depth++;
    }
    if (fork_depth > 0) {
        n_tareas = (1u << fork_depth) - 1u;
        if (n_tareas > ISODENDRON_MAX_TAREAS) {
            n_tareas = ISODENDRON_MAX_TAREAS;
        }
    }

    if (n_tareas > 0) {
        par.workers = (IsodendronWorker *)calloc(n_tareas, sizeof(IsodendronWorker));
        par.tareas  = (IsodendronTarea *)calloc(n_tareas, sizeof(IsodendronTarea));

        if (par.workers == NULL || par.tareas == NULL) {
            free(par.workers);
            free(par.tareas);
            arena_destroy(&arena);
            return ISODENDRON_ERR_MEM;
        }

        hm_init(&par.mutex);
        par.words         = words;
        par.f             = f;
        par.max_targets   = max_targets;
        par.fork_depth    = fork_depth;
        par.n_tareas      = n_tareas;
        par.tareas_usadas = 0;
        ctx.par           = &par;
    }

    TreeNode *root = build_tree(&ctx, 0, n, 0);

    int fallo = (root == NULL) || arena_failed(&arena);
    for (unsigned i = 0; i < par.tareas_usadas && !fallo; i++) {
        const BuildCtx *w = par.tareas[i].ctx;
        if (w != NULL && arena_failed(w->arena)) {
            fallo = 1;
        }
    }
    if (fallo) {
        par_liberar(&par);
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }

    (void)compute_sizes(root);

    uint32_t n_nodes = ctx.node_count;
    for (unsigned i = 0; i < par.tareas_usadas; i++) {
        if (par.tareas[i].ctx != NULL) {
            n_nodes += par.tareas[i].ctx->node_count;
        }
    }
    size_t   blob_len = blob_total_size(n_nodes, payload_bytes_de(root));
    uint8_t *blob     = (uint8_t *)malloc(blob_len);
    if (blob == NULL) {
        arena_destroy(&arena);
        return ISODENDRON_ERR_MEM;
    }

    blob_write_header(blob, f, n_nodes, max_deep, p_min_num, p_min_den,
                      payload_bytes_de(root));

    uint32_t pos        = 0;
    uint32_t payload_pos = 0;
    uint32_t payload_base = (uint32_t)(BLOB_HEADER_SIZE
                                       + (size_t)n_nodes * BLOB_NODE_SIZE);
    serialize(root, blob, &pos, payload_base, &payload_pos);

    par_liberar(&par);
    arena_destroy(&arena);

    *out_blob = blob;
    *out_len  = blob_len;
    return ISODENDRON_OK;
}

/*
 * Entrenamiento multihilo con el tope de distribución por defecto.
 */
int isodendron_train_mt(const int32_t *X,
                   const int32_t *y,
                   uint32_t       n,
                   uint32_t       f,
                   uint32_t       max_deep,
                   uint32_t       p_min_num,
                   uint32_t       p_min_den,
                   uint32_t       n_threads,
                   uint8_t      **out_blob,
                   size_t        *out_len)
{
    return isodendron_train_mt_cap(X, y, n, f, max_deep, p_min_num, p_min_den,
                              ISODENDRON_MAX_TARGETS, n_threads, out_blob, out_len);
}

/*
 * Entrenamiento secuencial. Se mantiene como función propia para no
 * cambiar la firma pública; equivale a isodendron_train_mt con un solo hilo.
 */
int isodendron_train(const int32_t *X,
                const int32_t *y,
                uint32_t       n,
                uint32_t       f,
                uint32_t       max_deep,
                uint32_t       p_min_num,
                uint32_t       p_min_den,
                uint8_t      **out_blob,
                size_t        *out_len)
{
    return isodendron_train_mt(X, y, n, f, max_deep, p_min_num, p_min_den, 1,
                          out_blob, out_len);
}

/* ------------------------------------------------------------------ */
/* API pública: predicción                                             */
/* ------------------------------------------------------------------ */

static int predict_impl(const uint8_t *blob,
                       size_t         blob_len,
                       const int32_t *X,
                       uint32_t       n,
                       uint32_t       f,
                       int32_t       *out_pred,
                       int32_t       *out_n_mode,
                       int32_t       *out_leaf_n,
                       uint32_t      *out_leaf_idx)
{
    if (blob == NULL || X == NULL || out_pred == NULL) {
        return ISODENDRON_ERR_ARG;
    }
    if (n == 0 || f == 0) {
        return ISODENDRON_ERR_ARG;
    }

    BlobHeader h;
    if (blob_read_header(blob, blob_len, &h) != BLOB_OK) {
        return ISODENDRON_ERR_ARG;
    }
    if (blob_check_compat(&h, f) != BLOB_OK) {
        return ISODENDRON_ERR_ARG;
    }
    if (h.n_nodes == 0) {
        return ISODENDRON_ERR_ARG;
    }

    for (uint32_t i = 0; i < n; i++) {
        const int32_t *x = X + (size_t)i * f;
        uint32_t node_idx = 0;

        for (;;) {
            BlobNode node;
            if (blob_read_node(blob, blob_len, node_idx, &node) != BLOB_OK) {
                return ISODENDRON_ERR_ARG;
            }
            if (node.is_leaf) {
                out_pred[i] = node.mode;
                if (out_n_mode != NULL) {
                    out_n_mode[i] = node.n_mode;
                }
                if (out_leaf_n != NULL) {
                    out_leaf_n[i] = node.N;
                }
                if (out_leaf_idx != NULL) {
                    out_leaf_idx[i] = node_idx;
                }
                break;
            }
            int32_t v = x[node.feature_idx];
            if (v == node.value) {
                node_idx = node_idx + 1;
            } else {
                node_idx = node_idx + 1 + node.left_subtree_size;
            }
            if (node_idx >= h.n_nodes) {
                return ISODENDRON_ERR_ARG;
            }
        }
    }
    return ISODENDRON_OK;
}

int isodendron_predict_full(const uint8_t *blob,
                       size_t         blob_len,
                       const int32_t *X,
                       uint32_t       n,
                       uint32_t       f,
                       int32_t       *out_pred,
                       int32_t       *out_n_mode,
                       int32_t       *out_leaf_n)
{
    return predict_impl(blob, blob_len, X, n, f, out_pred, out_n_mode,
                        out_leaf_n, NULL);
}

int isodendron_predict_leaf(const uint8_t *blob,
                       size_t         blob_len,
                       const int32_t *X,
                       uint32_t       n,
                       uint32_t       f,
                       int32_t       *out_pred,
                       uint32_t      *out_leaf_idx)
{
    if (out_leaf_idx == NULL) {
        return ISODENDRON_ERR_ARG;
    }
    return predict_impl(blob, blob_len, X, n, f, out_pred, NULL, NULL,
                        out_leaf_idx);
}

int isodendron_predict(const uint8_t *blob,
                  size_t         blob_len,
                  const int32_t *X,
                  uint32_t       n,
                  uint32_t       f,
                  int32_t       *out_pred)
{
    return isodendron_predict_full(blob, blob_len, X, n, f, out_pred, NULL, NULL);
}

/* ------------------------------------------------------------------ */
/* Recalibración de hojas                                              */
/* ------------------------------------------------------------------ */

/*
 * Acumula un conjunto nuevo en las hojas (ver isodendron.h). SUMA, no
 * recalcula, y aplica un decaimiento previo a todas las hojas.
 */
static int resumir_hoja(const uint8_t *blob, const int32_t *y,
                        const int32_t *orden, uint32_t lo, uint32_t hi,
                        Counts *cnts, int32_t *keys, uint32_t max_targets,
                        uint32_t old_off, uint32_t old_len, int32_t old_N,
                        int32_t *out_N, int32_t *out_K,
                        int32_t *out_mode, int32_t *out_n_mode,
                        DistEntry *dist, uint32_t *out_len)
{
    uint32_t n = hi - lo;
    *out_len = 0;

    counts_reset(cnts);
    uint32_t k = 0;

    /*
     * Decaimiento de lo ya acumulado: se resta (minimo - 1) a todos los
     * conteos guardados, de modo que el minimo de la hoja queda en 1. Se
     * aplica a TODAS las hojas, incluidas las que el conjunto nuevo no
     * visita, y N baja en la misma cantidad para conservar el residuo
     * N - suma(conteos).
     */
    uint32_t resta = 0;
    if (old_len > 0) {
        int32_t minimo = -1;
        for (uint32_t j = 0; j < old_len; j++) {
            int32_t t, c;
            blob_read_dist(blob, old_off + j * BLOB_DIST_ENTRY_SIZE, &t, &c);
            if (minimo < 0 || c < minimo) {
                minimo = c;
            }
        }
        if (minimo > 1) {
            resta = (uint32_t)(minimo - 1);
        }
        for (uint32_t j = 0; j < old_len; j++) {
            int32_t t, c;
            blob_read_dist(blob, old_off + j * BLOB_DIST_ENTRY_SIZE, &t, &c);
            int32_t nuevo = c - (int32_t)resta;
            if (nuevo < 1) {
                nuevo = 1;
            }
            counts_add(cnts, t, (uint32_t)nuevo);
            keys[k++] = t;
        }
    }

    *out_N = old_N - (int32_t)(old_len * resta) + (int32_t)n;

    for (uint32_t j = lo; j < hi; j++) {
        int32_t t = y[orden[j]];
        uint32_t antes = counts_size(cnts);
        counts_inc(cnts, t);
        if (counts_size(cnts) > antes) {
            keys[k++] = t;
        }
    }
    if (counts_failed(cnts)) {
        return -1;
    }
    /*
     * Sin entradas no hay moda, pero N NO se toca: el llamante ya lo
     * calculo como old_N - old_len*resta + puntos. Poniendolo a cero
     * aqui se vaciaban las hojas que no tenian distribucion y no
     * recibian puntos nuevos.
     */
    if (k == 0) {
        *out_K = 0;
        *out_mode = 0;
        *out_n_mode = 0;
        return 0;
    }
    *out_K = (int32_t)k;

    qsort(keys, k, sizeof(int32_t), cmp_i32);

    uint32_t best = 0;
    for (uint32_t j = 0; j < k; j++) {
        uint32_t c = counts_get(cnts, keys[j]);
        if (c > best) {
            best = c;
        }
    }

    uint32_t first = 0, n_tied = 0;
    int visto = 0;
    for (uint32_t j = 0; j < k; j++) {
        if (counts_get(cnts, keys[j]) == best) {
            if (!visto) {
                first = j;
                visto = 1;
            }
            n_tied++;
        }
    }
    *out_mode = keys[first + (n_tied - 1u) / 2u];
    *out_n_mode = (int32_t)best;

    if (dist == NULL || max_targets == 0) {
        return 0;
    }

    uint32_t len = 0;
    for (uint32_t j = 0; j < k; j++) {
        int32_t  t = keys[j];
        int32_t  c = (int32_t)counts_get(cnts, t);
        uint32_t p;

        if (len < max_targets) {
            p = len;
            len++;
        } else if (c > dist[max_targets - 1u].count
                   || (c == dist[max_targets - 1u].count
                       && t < dist[max_targets - 1u].target)) {
            p = max_targets - 1u;
        } else {
            continue;
        }

        while (p > 0 && (dist[p - 1u].count < c
                         || (dist[p - 1u].count == c
                             && dist[p - 1u].target > t))) {
            dist[p] = dist[p - 1u];
            p--;
        }
        dist[p].target = t;
        dist[p].count  = c;
    }

    *out_len = len;
    return 0;
}

static int partial_fit_impl(const uint8_t *blob_in, size_t blob_len,
                         const int32_t *X, const int32_t *y,
                         uint32_t n, uint32_t f, uint32_t max_targets,
                         uint8_t **out_blob, size_t *out_len,
                         IsodendronPartialDiag *diag)
{
    if (diag != NULL) {
        diag->puntos       = n;
        diag->hojas        = 0;
        diag->decayadas    = 0;
        diag->vaciadas     = 0;
        diag->resta_maxima = 0;
        diag->anomalas     = 0;
    }
    if (blob_in == NULL || X == NULL || y == NULL
        || out_blob == NULL || out_len == NULL) {
        return ISODENDRON_ERR_ARG;
    }
    if (n == 0 || f == 0) {
        return ISODENDRON_ERR_ARG;
    }

    BlobHeader h;
    if (blob_read_header(blob_in, blob_len, &h) != BLOB_OK) {
        return ISODENDRON_ERR_ARG;
    }
    if (blob_check_compat(&h, f) != BLOB_OK) {
        return ISODENDRON_ERR_ARG;
    }
    if (h.n_nodes == 0) {
        return ISODENDRON_ERR_ARG;
    }

    size_t    nn    = (size_t)h.n_nodes;
    BlobNode *nodos = (BlobNode *)calloc(nn, sizeof(BlobNode));
    int32_t  *hoja  = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t  *orden = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    /*
     * `keys` guarda una clave por cada entrada antigua de la hoja (hasta
     * max_targets) mas una por cada clave nueva, asi que su capacidad
     * tiene que ser n + max_targets. Reservar solo n desbordaba el buffer
     * en cuanto una hoja recibia todos los puntos del conjunto.
     */
    int32_t  *keys  = (int32_t *)malloc((size_t)(n + max_targets)
                                        * sizeof(int32_t));
    uint32_t *ini   = (uint32_t *)calloc(nn + 1u, sizeof(uint32_t));
    uint32_t *cur   = (uint32_t *)calloc(nn + 1u, sizeof(uint32_t));
    DistEntry *dist = (max_targets == 0)
                      ? NULL
                      : (DistEntry *)malloc((size_t)max_targets
                                            * sizeof(DistEntry));

    uint8_t *buf      = NULL;
    uint32_t nuevo_len = 0;
    int      rc        = ISODENDRON_ERR_MEM;
    Arena    arena;
    int      arena_ok  = 0;

    if (nodos == NULL || hoja == NULL || orden == NULL || keys == NULL
        || ini == NULL || cur == NULL
        || (max_targets > 0 && dist == NULL)) {
        goto fin;
    }

    for (uint32_t i = 0; i < h.n_nodes; i++) {
        if (blob_read_node(blob_in, blob_len, i, &nodos[i]) != BLOB_OK) {
            rc = ISODENDRON_ERR_ARG;
            goto fin;
        }
    }

    /* 1. Enrutar cada fila hasta su hoja y contar filas por nodo. */
    for (uint32_t i = 0; i < n; i++) {
        const int32_t *x = X + (size_t)i * f;
        uint32_t idx = 0;
        for (;;) {
            if (nodos[idx].is_leaf) {
                hoja[i] = (int32_t)idx;
                ini[idx]++;
                break;
            }
            int32_t v = x[nodos[idx].feature_idx];
            idx = (v == nodos[idx].value)
                  ? idx + 1u
                  : idx + 1u + nodos[idx].left_subtree_size;
            if (idx >= h.n_nodes) {
                rc = ISODENDRON_ERR_ARG;
                goto fin;
            }
        }
    }

    /* 2. Prefijos y colocación: agrupa las filas de cada hoja. */
    uint32_t acc = 0;
    for (size_t i = 0; i < nn; i++) {
        uint32_t c = ini[i];
        ini[i] = acc;
        acc += c;
    }
    ini[nn] = acc;
    for (size_t i = 0; i < nn; i++) {
        cur[i] = ini[i];
    }
    for (uint32_t i = 0; i < n; i++) {
        orden[cur[hoja[i]]++] = (int32_t)i;
    }

    if (arena_init(&arena, (size_t)n * 8u + 4096u) != 0) {
        goto fin;
    }
    arena_ok = 1;

    Counts *cnts = counts_new(&arena, 16);
    if (cnts == NULL) {
        goto fin;
    }

    /* 3. Primera pasada: resumen y tamaño del payload. */
    uint32_t payload_bytes = 0;
    for (size_t i = 0; i < nn; i++) {
        if (!nodos[i].is_leaf) {
            continue;
        }
        BlobNode viejo;
        if (blob_read_node(blob_in, blob_len, (uint32_t)i, &viejo) != BLOB_OK) {
            rc = ISODENDRON_ERR_ARG;
            goto fin;
        }
        int32_t  N, K, mode, n_mode;
        uint32_t len = 0;
        if (resumir_hoja(blob_in, y, orden, ini[i], ini[i + 1u], cnts, keys,
                         max_targets,
                         (uint32_t)viejo.feature_idx, (uint32_t)viejo.value,
                         viejo.N,
                         &N, &K, &mode, &n_mode, dist,
                         &len) != 0) {
            rc = ISODENDRON_ERR_MEM;
            goto fin;
        }
        nodos[i].N      = N;
        nodos[i].K      = K;
        nodos[i].mode   = mode;
        nodos[i].n_mode = n_mode;
        nodos[i].value  = (int32_t)len;   /* payload_len */
        payload_bytes  += len * BLOB_DIST_ENTRY_SIZE;

        if (diag != NULL) {
            uint32_t resta = 0;
            if (viejo.value > 0) {
                int32_t minimo = -1;
                for (uint32_t j = 0; j < (uint32_t)viejo.value; j++) {
                    int32_t tt, cc;
                    blob_read_dist(blob_in,
                                   (uint32_t)viejo.feature_idx
                                       + j * BLOB_DIST_ENTRY_SIZE,
                                   &tt, &cc);
                    if (minimo < 0 || cc < minimo) {
                        minimo = cc;
                    }
                }
                if (minimo > 1) {
                    resta = (uint32_t)(minimo - 1);
                }
            }

            diag->hojas++;
            if (resta > 0) {
                diag->decayadas++;
            }
            if (resta > diag->resta_maxima) {
                diag->resta_maxima = resta;
            }
            if (N <= 0 || (len == 0 && viejo.N > 0)) {
                diag->vaciadas++;
                if (diag->detalle != NULL
                    && diag->anomalas < diag->capacidad) {
                    IsodendronHojaAnomala *a = &diag->detalle[diag->anomalas++];
                    a->nodo        = (uint32_t)i;
                    a->N_previo    = viejo.N;
                    a->payload_len = (uint32_t)viejo.value;
                    a->resta       = resta;
                    a->N_final     = N;
                }
            }
        }
    }

    /* 4. Blob nuevo: nodos iguales salvo las hojas, payload al final. */
    size_t total = blob_total_size(h.n_nodes, payload_bytes);
    buf = (uint8_t *)calloc(total, 1);
    if (buf == NULL) {
        goto fin;
    }
    blob_write_header(buf, h.n_features, h.n_nodes, h.max_deep,
                      h.p_min_num, h.p_min_den, payload_bytes);

    uint32_t payload_base = (uint32_t)(BLOB_HEADER_SIZE
                                       + nn * BLOB_NODE_SIZE);
    uint32_t payload_pos  = 0;

    for (size_t i = 0; i < nn; i++) {
        if (nodos[i].is_leaf) {
            BlobNode viejo;
            if (blob_read_node(blob_in, blob_len, (uint32_t)i, &viejo)
                != BLOB_OK) {
                rc = ISODENDRON_ERR_ARG;
                goto fin;
            }
            int32_t  N, K, mode, n_mode;
            uint32_t len = 0;
            if (resumir_hoja(blob_in, y, orden, ini[i], ini[i + 1u], cnts,
                             keys, max_targets,
                             (uint32_t)viejo.feature_idx,
                             (uint32_t)viejo.value, viejo.N,
                             &N, &K, &mode, &n_mode, dist,
                             &len) != 0) {
                rc = ISODENDRON_ERR_MEM;
                goto fin;
            }
            nodos[i].feature_idx = (int32_t)(payload_base + payload_pos);
            nodos[i].value       = (int32_t)len;

            for (uint32_t j = 0; j < len; j++) {
                blob_write_dist(buf, payload_base + payload_pos,
                                dist[j].target, dist[j].count);
                payload_pos += BLOB_DIST_ENTRY_SIZE;
            }
        }
        blob_write_node(buf, (uint32_t)i, &nodos[i]);
    }

    nuevo_len = (uint32_t)total;
    rc = ISODENDRON_OK;

fin:
    if (arena_ok) {
        arena_destroy(&arena);
    }
    if (rc != ISODENDRON_OK) {
        free(buf);
        buf = NULL;
    }
    free(nodos);
    free(hoja);
    free(orden);
    free(keys);
    free(ini);
    free(cur);
    free(dist);

    if (rc != ISODENDRON_OK) {
        return rc;
    }
    *out_blob = buf;
    *out_len  = nuevo_len;
    return ISODENDRON_OK;
}

int isodendron_partial_fit(const uint8_t *blob_in, size_t blob_len,
                      const int32_t *X, const int32_t *y,
                      uint32_t n, uint32_t f, uint32_t max_targets,
                      uint8_t **out_blob, size_t *out_len)
{
    return partial_fit_impl(blob_in, blob_len, X, y, n, f, max_targets,
                            out_blob, out_len, NULL);
}

int isodendron_partial_fit_diag(const uint8_t *blob_in, size_t blob_len,
                           const int32_t *X, const int32_t *y,
                           uint32_t n, uint32_t f, uint32_t max_targets,
                           uint8_t **out_blob, size_t *out_len,
                           IsodendronPartialDiag *diag)
{
    return partial_fit_impl(blob_in, blob_len, X, y, n, f, max_targets,
                            out_blob, out_len, diag);
}

/* ------------------------------------------------------------------ */
/* API pública: liberación                                             */
/* ------------------------------------------------------------------ */

void isodendron_free(void *ptr)
{
    free(ptr);
}

/* ------------------------------------------------------------------ */
/* API pública: versión                                                */
/* ------------------------------------------------------------------ */

#ifndef ISODENDRON_VERSION_STR
#define ISODENDRON_VERSION_STR "0.1.0"
#endif

const char *isodendron_version(void)
{
    return ISODENDRON_VERSION_STR;
}
