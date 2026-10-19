/*
 * counts.c — Implementación de la tabla hash int32_t -> uint32_t.
 *
 * Ver counts.h para el contrato. En resumen:
 *
 *   - Open addressing con linear probing.
 *   - Capacidad potencia de dos, índice con máscara.
 *   - Entrada vacía = value == 0.
 *   - Crecimiento al 75% de carga.
 *   - Sin borrado individual: solo counts_reset.
 *   - Toda la memoria sale del arena del llamante.
 */
#include "counts.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Utilidades internas                                                 */
/* ------------------------------------------------------------------ */

/*
 * Redondea `x` a la siguiente potencia de dos >= x. Si x es 0, devuelve 1.
 * Asume que x no es tan grande como para desbordar size_t al redondear.
 */
static uint32_t round_up_pow2_u32(uint32_t x)
{
    if (x <= 1) {
        return 1;
    }
    if ((x & (x - 1)) == 0) {
        return x;
    }
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

/*
 * Finalizador de enteros de 32 bits. Mezcla los bits de la clave para
 * que claves correlativas (que es lo típico en features discretizadas
 * con códigos 0, 1, 2, ...) se distribuyan por todos los slots.
 *
 * Es un mixer tipo splitmix truncado a 32 bits. Tiene buena avalancha
 * y es barato (3 shifts, 2 multiplicaciones, 3 xors).
 */
static uint32_t hash_key(int32_t key)
{
    uint32_t x = (uint32_t)key;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* Alineación natural para todos los tipos que usamos. */
#define ISODENDRON_ALIGN sizeof(void *)

/* ------------------------------------------------------------------ */
/* Crecimiento                                                         */
/* ------------------------------------------------------------------ */

/*
 * Sustituye el array de entradas por uno del doble de tamaño y rehashea
 * las claves existentes.
 *
 * El array antiguo queda huérfano dentro del arena. Es aceptable: la
 * tabla crece pocas veces (log2 de la capacidad final) y el espacio
 * perdido es pequeño comparado con el resto del árbol.
 *
 * No modifica `size`: el número de claves no cambia.
 *
 * Devuelve 0 si tuvo éxito, -1 si no hay memoria. En caso de fallo, la
 * tabla queda con el array antiguo intacto pero marcada como failed.
 */
static int counts_resize(Counts *c, uint32_t new_capacity)
{
    CountsEntry *old_entries = c->entries;
    uint32_t     old_cap     = c->capacity;

    CountsEntry *new_entries = (CountsEntry *)
        arena_alloc_array(c->arena, new_capacity, sizeof(CountsEntry));
    if (new_entries == NULL) {
        c->failed = 1;
        return -1;
    }

    memset(new_entries, 0, (size_t)new_capacity * sizeof(CountsEntry));

    uint32_t mask = new_capacity - 1;

    for (uint32_t i = 0; i < old_cap; i++) {
        if (old_entries[i].value == 0) {
            continue;
        }
        int32_t key = old_entries[i].key;
        uint32_t slot = hash_key(key) & mask;
        while (new_entries[slot].value != 0) {
            slot = (slot + 1) & mask;
        }
        new_entries[slot] = old_entries[i];
    }

    c->entries  = new_entries;
    c->capacity = new_capacity;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Ciclo de vida                                                       */
/* ------------------------------------------------------------------ */

Counts *counts_new(Arena *a, uint32_t initial_cap)
{
    if (a == NULL) {
        return NULL;
    }

    uint32_t cap = initial_cap == 0 ? COUNTS_MIN_CAPACITY : initial_cap;
    cap = round_up_pow2_u32(cap);
    if (cap < COUNTS_MIN_CAPACITY) {
        cap = COUNTS_MIN_CAPACITY;
    }

    Counts *c = (Counts *)arena_alloc(a, sizeof(Counts), ISODENDRON_ALIGN);
    if (c == NULL) {
        return NULL;
    }

    CountsEntry *entries = (CountsEntry *)
        arena_alloc_array(a, cap, sizeof(CountsEntry));
    if (entries == NULL) {
        return NULL;
    }

    memset(entries, 0, (size_t)cap * sizeof(CountsEntry));

    c->entries  = entries;
    c->capacity = cap;
    c->size     = 0;
    c->arena    = a;
    c->failed   = 0;

    return c;
}

void counts_reset(Counts *c)
{
    if (c == NULL || c->entries == NULL) {
        return;
    }
    memset(c->entries, 0, (size_t)c->capacity * sizeof(CountsEntry));
    c->size = 0;
    /* Conservamos `failed` intencionadamente. */
}

/* ------------------------------------------------------------------ */
/* Inserción                                                           */
/* ------------------------------------------------------------------ */

void counts_add(Counts *c, int32_t key, uint32_t delta)
{
    if (c == NULL || c->failed || c->entries == NULL || delta == 0) {
        return;
    }

    uint32_t mask = c->capacity - 1;
    uint32_t slot = hash_key(key) & mask;

    /* Buscar la clave. Si existe, incrementar in situ. */
    while (c->entries[slot].value != 0) {
        if (c->entries[slot].key == key) {
            c->entries[slot].value += delta;
            return;
        }
        slot = (slot + 1) & mask;
    }

    /*
     * Clave nueva. Comprobar si al insertarla superaríamos el umbral
     * de carga. La comparación es (size + 1) / capacity > 3/4,
     * expresada en enteros para evitar división.
     */
    if ((c->size + 1u) * COUNTS_LOAD_DEN > c->capacity * COUNTS_LOAD_NUM) {
        uint32_t new_cap = c->capacity * 2u;
        /* Protección contra desbordamiento en capacity * 2. */
        if (new_cap < c->capacity) {
            c->failed = 1;
            return;
        }
        if (counts_resize(c, new_cap) != 0) {
            return;
        }
        /* Recalcular el slot tras el rehash. */
        mask = c->capacity - 1;
        slot = hash_key(key) & mask;
        while (c->entries[slot].value != 0) {
            slot = (slot + 1) & mask;
        }
    }

    c->entries[slot].key   = key;
    c->entries[slot].value = delta;
    c->size++;
}

void counts_inc(Counts *c, int32_t key)
{
    counts_add(c, key, 1u);
}

/* ------------------------------------------------------------------ */
/* Consulta                                                            */
/* ------------------------------------------------------------------ */

uint32_t counts_get(const Counts *c, int32_t key)
{
    if (c == NULL || c->entries == NULL || c->capacity == 0) {
        return 0;
    }

    uint32_t mask = c->capacity - 1;
    uint32_t slot = hash_key(key) & mask;

    while (c->entries[slot].value != 0) {
        if (c->entries[slot].key == key) {
            return c->entries[slot].value;
        }
        slot = (slot + 1) & mask;
    }
    return 0;
}

int counts_has(const Counts *c, int32_t key)
{
    return counts_get(c, key) != 0;
}

uint32_t counts_size(const Counts *c)
{
    if (c == NULL || c->failed) {
        return 0;
    }
    return c->size;
}

uint32_t counts_capacity(const Counts *c)
{
    if (c == NULL) {
        return 0;
    }
    return c->capacity;
}

int counts_failed(const Counts *c)
{
    if (c == NULL) {
        return 1;
    }
    return c->failed;
}

/* ------------------------------------------------------------------ */
/* Iteración                                                           */
/* ------------------------------------------------------------------ */

uint32_t counts_iterate(const Counts *c, counts_iter_fn fn, void *user)
{
    if (c == NULL || c->entries == NULL || fn == NULL) {
        return 0;
    }

    uint32_t visited = 0;
    for (uint32_t i = 0; i < c->capacity; i++) {
        if (c->entries[i].value == 0) {
            continue;
        }
        visited++;
        if (!fn(c->entries[i].key, c->entries[i].value, user)) {
            break;
        }
    }
    return visited;
}

/* ------------------------------------------------------------------ */
/* Búsqueda del máximo                                                 */
/* ------------------------------------------------------------------ */

int32_t counts_max_key_value(const Counts *c, uint32_t *out_value)
{
    /*
     * Contrato: la tabla no está vacía. Aun así, si lo estuviera,
     * devolvemos (0, 0) sin tocar memoria fuera de rango.
     */
    uint32_t best     = 0;
    int32_t  best_key = 0;

    if (c == NULL || c->entries == NULL) {
        if (out_value != NULL) {
            *out_value = 0;
        }
        return 0;
    }

    for (uint32_t i = 0; i < c->capacity; i++) {
        if (c->entries[i].value > best) {
            best     = c->entries[i].value;
            best_key = c->entries[i].key;
        }
    }

    if (out_value != NULL) {
        *out_value = best;
    }
    return best_key;
}

int32_t counts_max_key(const Counts *c)
{
    return counts_max_key_value(c, NULL);
}
