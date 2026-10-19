/*
 * arena.h — Bump allocator de una sola región
 *
 * Un arena es un bloque de memoria contiguo del que se van sirviendo
 * asignaciones incrementales. Cada asignación avanza un offset interno
 * alineado al tamaño pedido. No hay free individual: toda la región se
 * libera de una vez con arena_destroy.
 *
 * Este patrón es el adecuado para ISODENDRON porque:
 *
 *   - El árbol tiene tamaño acotado (max_deep y p_min limitan el número
 *     de nodos), así que se puede dimensionar el arena de antemano.
 *   - La vida útil de todos los objetos internos coincide: nacen y
 *     mueren con la llamada a isodendron_train.
 *   - Elimina la posibilidad de fugas y de dobles liberaciones: hay un
 *     único free.
 *   - Es O(1) por asignación, sin búsqueda de huecos ni fragmentación.
 *
 * El arena no es thread-safe. Cada hilo que entrene un árbol debe usar
 * su propio arena. Esto es coherente con el contrato de isodendron_train,
 * que declara ser seguro para llamadas concurrentes.
 */
#ifndef ISODENDRON_ARENA_H
#define ISODENDRON_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Estructura                                                          */
/* ------------------------------------------------------------------ */

/*
 * Arena de una sola región.
 *
 * Campos:
 *   base      Puntero al inicio del bloque reservado con malloc.
 *             Es NULL si el arena no se ha inicializado o ya se ha
 *             destruido.
 *   offset    Bytes ya consumidos desde base, siempre múltiplo de la
 *             última alineación pedida. Es el "cursor" del bump.
 *   capacity  Tamaño total del bloque en bytes.
 *   failed    Bandera que se activa si alguna asignación no cupo.
 *             Una vez activada, todas las asignaciones siguientes
 *             devuelven NULL sin tocar la memoria, y el llamante puede
 *             comprobar el estado al final con arena_failed en lugar de
 *             verificar cada retorno.
 */
typedef struct {
    uint8_t *base;
    size_t   offset;
    size_t   capacity;
    int      failed;
} Arena;

/* ------------------------------------------------------------------ */
/* Ciclo de vida                                                       */
/* ------------------------------------------------------------------ */

/*
 * Inicializa un arena reservando `capacity` bytes con malloc.
 *
 * Devuelve:
 *   0   éxito
 *  -1   fallo de asignación (capacity == 0 también se considera error)
 *
 * La capacidad debe ser al menos 1. Para tamaños muy grandes (> SIZE_MAX/2)
 * el comportamiento es indefinido; el llamante debe validar antes.
 */
int arena_init(Arena *a, size_t capacity);

/*
 * Libera la memoria del arena y lo deja en estado "no inicializado".
 * Es seguro llamarla dos veces seguidas, o sobre un arena que nunca se
 * inicializó (con base == NULL).
 */
void arena_destroy(Arena *a);

/* ------------------------------------------------------------------ */
/* Asignación                                                          */
/* ------------------------------------------------------------------ */

/*
 * Reserva `size` bytes alineados a `align` desde el cursor actual.
 *
 * Parámetros:
 *   size    Número de bytes a reservar. Si es 0, devuelve un puntero no
 *           NULL único pero no avanza el cursor (comportamiento análogo
 *           a malloc(0) salvo que no asigna memoria real).
 *   align   Alineación en bytes. Debe ser potencia de dos. Si no lo es,
 *           se redondea hacia arriba a la siguiente potencia de dos.
 *           Valores típicos: 1, 2, 4, 8, 16.
 *
 * Devuelve:
 *   Puntero al bloque reservado, o NULL si no cabe.
 *
 * Notas:
 *   - El bloque devuelto no está inicializado a cero. Si el llamante
 *     necesita ceros, debe hacer memset por su cuenta.
 *   - Si el arena ya está en estado failed, devuelve NULL sin tocar
 *     memoria.
 *   - El puntero devuelto permanece válido hasta arena_destroy.
 */
void *arena_alloc(Arena *a, size_t size, size_t align);

/*
 * Variante que reserva espacio para `count` elementos de `size` bytes
 * cada uno, alineados al tamaño de `elem_size` redondeado a potencia
 * de dos. Pensada para arrays de structs: reserva count * size bytes
 * y garantiza que el primer byte está alineado a un múltiplo natural
 * para el tipo.
 *
 * Ejemplo:
 *   int32_t *v = arena_alloc_array(a, 100, sizeof(int32_t));
 *
 * Devuelve NULL si no cabe o si count * size desborda size_t.
 */
void *arena_alloc_array(Arena *a, size_t count, size_t elem_size);

/* ------------------------------------------------------------------ */
/* Consulta                                                            */
/* ------------------------------------------------------------------ */

/*
 * Devuelve 1 si el arena ha sufrido algún fallo de asignación, 0 en
 * caso contrario. Un arena no inicializado se considera fallido.
 */
int arena_failed(const Arena *a);

/*
 * Devuelve el número de bytes consumidos desde el inicio.
 * Para un arena no inicializado, devuelve 0.
 */
size_t arena_used(const Arena *a);

/*
 * Devuelve el número de bytes libres (capacity - offset).
 * Para un arena no inicializado, devuelve 0.
 */
size_t arena_remaining(const Arena *a);

/*
 * Marca de posición para "checkpoint". Devuelve el offset actual del
 * cursor. Se puede usar junto con arena_rewind para deshacer un
 * conjunto de asignaciones cuando ya no son necesarias.
 */
size_t arena_mark(const Arena *a);

/*
 * Rebobina el cursor al offset indicado. El offset debe haber sido
 * obtenido con arena_mark sobre el mismo arena y no puede ser mayor
 * que el offset actual. Rebobinar no libera nada al sistema operativo,
 * solo reutiliza la región a partir de esa marca.
 *
 * Esto permite un patrón de "scope": marcar antes de un bloque de
 * trabajo, reservar lo que haga falta, y rebobinar al salir para
 * recuperar la memoria intermedia, manteniendo solo los objetos que
 * se hayan copiado fuera del arena.
 *
 * Si el offset es inválido (mayor que el actual), la función no hace
 * nada.
 */
void arena_rewind(Arena *a, size_t mark);

#ifdef __cplusplus
}
#endif

#endif /* ISODENDRON_ARENA_H */
