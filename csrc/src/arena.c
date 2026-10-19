/*
 * arena.c — Implementación del bump allocator de una sola región.
 *
 * Ver arena.h para el contrato completo. En resumen:
 *
 *   - arena_init reserva un bloque con malloc.
 *   - arena_alloc avanza un cursor alineado y devuelve punteros.
 *   - arena_destroy libera el bloque entero.
 *
 * No hay free individual. La bandera `failed` permite encadenar
 * asignaciones sin comprobar cada retorno.
 */
#include "arena.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Utilidades internas                                                 */
/* ------------------------------------------------------------------ */

/*
 * Redondea `align` a la siguiente potencia de dos >= 1.
 * Si align es 0, 1 o ya es potencia de dos, se devuelve tal cual
 * (salvo el caso 0, que se eleva a 1).
 */
static size_t round_up_pow2(size_t align)
{
    if (align <= 1) {
        return 1;
    }
    /* Si ya es potencia de dos, devolverla sin cambios. */
    if ((align & (align - 1)) == 0) {
        return align;
    }
    /* Propagar el bit más alto hacia abajo y sumar 1. */
    align--;
    align |= align >> 1;
    align |= align >> 2;
    align |= align >> 4;
    align |= align >> 8;
    align |= align >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    align |= align >> 32;
#endif
    align++;
    return align;
}

/*
 * Alinea `offset` hacia arriba al múltiplo de `align` (potencia de dos).
 */
static size_t align_up(size_t offset, size_t align)
{
    return (offset + align - 1) & ~(align - 1);
}

/* ------------------------------------------------------------------ */
/* Ciclo de vida                                                       */
/* ------------------------------------------------------------------ */

int arena_init(Arena *a, size_t capacity)
{
    if (a == NULL || capacity == 0) {
        return -1;
    }

    a->base     = (uint8_t *)malloc(capacity);
    a->offset   = 0;
    a->capacity = capacity;
    a->failed   = 0;

    if (a->base == NULL) {
        a->capacity = 0;
        a->failed   = 1;
        return -1;
    }

    return 0;
}

void arena_destroy(Arena *a)
{
    if (a == NULL) {
        return;
    }

    if (a->base != NULL) {
        free(a->base);
    }

    a->base     = NULL;
    a->offset   = 0;
    a->capacity = 0;
    a->failed   = 0;
}

/* ------------------------------------------------------------------ */
/* Asignación                                                          */
/* ------------------------------------------------------------------ */

void *arena_alloc(Arena *a, size_t size, size_t align)
{
    if (a == NULL || a->base == NULL || a->failed) {
        return NULL;
    }

    align = round_up_pow2(align);

    /* Caso especial: size == 0. Devolvemos un puntero válido (al base
     * más el offset actual) sin avanzar el cursor, de forma que dos
     * llamadas seguidas devuelvan el mismo puntero. Es un comportamiento
     * análogo a malloc(0), con la diferencia de que no hay asignación
     * real y el puntero no debe usarse para escribir. */
    if (size == 0) {
        return a->base + a->offset;
    }

    size_t aligned_offset = align_up(a->offset, align);

    /* Comprobar desbordamiento y capacidad. */
    if (aligned_offset > a->capacity ||
        size > a->capacity - aligned_offset) {
        a->failed = 1;
        return NULL;
    }

    uint8_t *ptr = a->base + aligned_offset;
    a->offset = aligned_offset + size;

    return ptr;
}

void *arena_alloc_array(Arena *a, size_t count, size_t elem_size)
{
    if (a == NULL) {
        return NULL;
    }

    /* Si count o elem_size son cero, delegamos en arena_alloc con size 0
     * para mantener un comportamiento uniforme (puntero válido, sin
     * avanzar el cursor). */
    if (count == 0 || elem_size == 0) {
        return arena_alloc(a, 0, elem_size);
    }

    /* Comprobar desbordamiento de count * elem_size. */
    if (count > SIZE_MAX / elem_size) {
        a->failed = 1;
        return NULL;
    }

    size_t total = count * elem_size;

    /* La alineación natural de un array de elementos de tamaño
     * elem_size es min(elem_size, sizeof(void*)) redondeado a potencia
     * de dos. En la práctica, alinear al propio elem_size redondeado es
     * suficiente y más conservador. */
    return arena_alloc(a, total, round_up_pow2(elem_size));
}

/* ------------------------------------------------------------------ */
/* Consulta                                                            */
/* ------------------------------------------------------------------ */

int arena_failed(const Arena *a)
{
    if (a == NULL || a->base == NULL) {
        return 1;
    }
    return a->failed;
}

size_t arena_used(const Arena *a)
{
    if (a == NULL || a->base == NULL) {
        return 0;
    }
    return a->offset;
}

size_t arena_remaining(const Arena *a)
{
    if (a == NULL || a->base == NULL) {
        return 0;
    }
    return a->capacity - a->offset;
}

size_t arena_mark(const Arena *a)
{
    if (a == NULL || a->base == NULL) {
        return 0;
    }
    return a->offset;
}

void arena_rewind(Arena *a, size_t mark)
{
    if (a == NULL || a->base == NULL) {
        return;
    }
    if (mark > a->offset) {
        return;
    }
    a->offset = mark;
    /* No tocamos `failed`: si el arena ya había fallado, sigue fallado
     * aunque rebobinemos. La capacidad no se recupera mágicamente. */
}
