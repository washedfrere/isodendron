/*
 * counts.h — Tabla hash int32_t -> uint32_t con open addressing
 *
 * Estructura pensada para el patrón central de ISODENDRON: contar cuántas
 * veces aparece cada valor de una feature dentro de un nodo. Se
 * reutiliza la misma tabla entre features y entre nodos llamando a
 * counts_reset, que limpia las entradas sin liberar memoria.
 *
 * Características:
 *
 *   - Clave int32_t, valor uint32_t (conteo).
 *   - Open addressing con linear probing.
 *   - La capacidad es siempre potencia de dos, de forma que el índice
 *     se calcula con un AND en lugar de un módulo.
 *   - Entrada "vacía" se representa con value == 0. Como en ISODENDRON solo
 *     incrementamos (nunca ponemos un conteo a cero explícitamente),
 *     esta convención es segura y evita un campo extra de ocupación.
 *   - Crece automáticamente al superar el 75% de carga, duplicando la
 *     capacidad y rehasheando.
 *   - No hay eliminación individual: solo reset completo. Eso elimina
 *     la necesidad de tombstones y simplifica la implementación.
 *
 * Gestión de memoria:
 *
 *   - Toda la memoria (la estructura Counts y el array de entradas)
 *     sale del arena que se pasa a counts_new.
 *   - No hay counts_free. La tabla vive hasta que el arena se destruye.
 *   - Al crecer, se reserva un nuevo array del arena y el antiguo queda
 *     huérfano dentro del arena. Esto es aceptable porque ISODENDRON usa
 *     tablas pequeñas y el crecimiento ocurre pocas veces.
 *
 * Uso típico en ISODENDRON:
 *
 *     Counts *c = counts_new(arena, initial_cap);
 *     for (int j = 0; j < f; j++) {
 *         counts_reset(c);
 *         for (i = 0; i < n; i++) {
 *             counts_inc(c, X[i * f + j]);
 *         }
 *         uint32_t unicos = counts_size(c);
 *         // ... evaluar la feature j ...
 *     }
 *
 * La tabla no es thread-safe. Cada hilo debe usar su propia instancia.
 */
#ifndef ISODENDRON_COUNTS_H
#define ISODENDRON_COUNTS_H

#include <stdint.h>
#include <stddef.h>

#include "arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Constantes                                                          */
/* ------------------------------------------------------------------ */

/*
 * Capacidad mínima. La capacidad efectiva siempre se redondea a la
 * siguiente potencia de dos >= este valor.
 */
#define COUNTS_MIN_CAPACITY  16u

/*
 * Umbral de crecimiento: se crece cuando size >= capacity * 3 / 4.
 * Mantener la carga por debajo de 0.75 es lo habitual en open
 * addressing para evitar aglomeraciones de linear probing.
 */
#define COUNTS_LOAD_NUM  3u
#define COUNTS_LOAD_DEN  4u

/* ------------------------------------------------------------------ */
/* Estructura                                                          */
/* ------------------------------------------------------------------ */

/*
 * Una entrada de la tabla.
 *   key     Clave. Solo significativa si value != 0.
 *   value   Conteo. 0 indica entrada vacía.
 */
typedef struct {
    int32_t  key;
    uint32_t value;
} CountsEntry;

/*
 * Tabla de conteos.
 *
 * Campos:
 *   entries    Array de entradas, tamaño `capacity`. Todas las entradas
 *              vacías tienen value == 0.
 *   capacity   Número de slots. Siempre potencia de dos.
 *   size       Número de entradas ocupadas (claves distintas).
 *   arena      Arena del que se sirve la memoria.
 *   failed     1 si alguna reserva (inicial o de crecimiento) falló.
 *              Una vez activada, las operaciones de escritura se
 *              convierten en no-op y las de lectura devuelven 0.
 */
typedef struct {
    CountsEntry *entries;
    uint32_t     capacity;
    uint32_t     size;
    Arena       *arena;
    int          failed;
} Counts;

/* ------------------------------------------------------------------ */
/* Ciclo de vida                                                       */
/* ------------------------------------------------------------------ */

/*
 * Crea una nueva tabla de conteos.
 *
 * Parámetros:
 *   a             Arena del que se sirve la memoria. Debe seguir vivo
 *                 mientras se use la tabla.
 *   initial_cap   Capacidad inicial deseada. Se redondea hacia arriba
 *                 a la siguiente potencia de dos, con un mínimo de
 *                 COUNTS_MIN_CAPACITY. Si es 0, se usa el mínimo.
 *
 * Devuelve:
 *   Puntero a la tabla, o NULL si el arena no puede servir la memoria.
 *
 * La tabla se devuelve vacía (size == 0). Todas las entradas tienen
 * value == 0.
 */
Counts *counts_new(Arena *a, uint32_t initial_cap);

/*
 * Limpia todas las entradas de la tabla sin liberar memoria.
 *
 * Tras esta llamada, size == 0 y todas las entradas están vacías. La
 * capacidad se conserva, de modo que reutilizar la tabla entre
 * features no requiere realojar nada.
 *
 * Si la tabla está en estado failed, esta función la deja vacía
 * pero conserva el flag (el llamante puede decidir reinicializarla
 * creando una nueva).
 */
void counts_reset(Counts *c);

/* ------------------------------------------------------------------ */
/* Inserción                                                           */
/* ------------------------------------------------------------------ */

/*
 * Incrementa el conteo asociado a `key` en 1.
 *
 * Si la clave no existe, la inserta con valor 1. Si ya existe, la
 * incrementa. Si el factor de carga supera el 75%, la tabla se duplica
 * automáticamente antes de insertar.
 *
 * Si la tabla está en estado failed, esta función no hace nada.
 */
void counts_inc(Counts *c, int32_t key);

/*
 * Suma `delta` al conteo asociado a `key`.
 *
 * Equivalente a llamar a counts_inc `delta` veces, pero en O(1). Útil
 * cuando se propaga un conteo desde el padre en lugar de contarlo desde
 * cero.
 *
 * Si delta == 0, no hace nada. Si la tabla está en estado failed, no
 * hace nada.
 */
void counts_add(Counts *c, int32_t key, uint32_t delta);

/* ------------------------------------------------------------------ */
/* Consulta                                                            */
/* ------------------------------------------------------------------ */

/*
 * Devuelve el conteo asociado a `key`, o 0 si la clave no existe.
 * Devuelve 0 si la tabla está en estado failed.
 */
uint32_t counts_get(const Counts *c, int32_t key);

/*
 * Devuelve 1 si `key` está presente, 0 en caso contrario.
 */
int counts_has(const Counts *c, int32_t key);

/*
 * Número de claves distintas almacenadas. Equivale a "valores únicos"
 * en el contexto de ISODENDRON.
 */
uint32_t counts_size(const Counts *c);

/*
 * Capacidad actual de la tabla (número de slots).
 */
uint32_t counts_capacity(const Counts *c);

/*
 * Devuelve 1 si la tabla ha sufrido algún fallo de reserva, 0 en caso
 * contrario. Una tabla NULL se considera fallida.
 */
int counts_failed(const Counts *c);

/* ------------------------------------------------------------------ */
/* Iteración                                                           */
/* ------------------------------------------------------------------ */

/*
 * Callback de iteración. Recibe la clave, el conteo y el puntero de
 * usuario. Si devuelve 0, la iteración se detiene.
 */
typedef int (*counts_iter_fn)(int32_t key, uint32_t value, void *user);

/*
 * Recorre todas las entradas ocupadas invocando `fn`. El orden de
 * recorrido es el orden interno del array de slots, que depende de la
 * función hash y de la secuencia de inserciones; no es estable entre
 * ejecuciones si cambia el orden de inserción.
 *
 * La iteración se detiene cuando el callback devuelve 0. Devuelve el
 * número de entradas visitadas.
 */
uint32_t counts_iterate(const Counts *c, counts_iter_fn fn, void *user);

/* ------------------------------------------------------------------ */
/* Búsqueda del máximo                                                 */
/* ------------------------------------------------------------------ */

/*
 * Devuelve la clave con el conteo más alto.
 *
 * En caso de empate, se devuelve la clave que aparezca primero en el
 * recorrido interno del array. Si se necesita un desempate estable
 * (por ejemplo, por orden de inserción), el llamante debe usar
 * counts_iterate y aplicar su propio criterio.
 *
 * Si la tabla está vacía o en estado failed, el comportamiento es
 * indefinido. El llamante debe comprobar counts_size > 0 antes de
 * llamar.
 */
int32_t counts_max_key(const Counts *c);

/*
 * Igual que counts_max_key pero devuelve también el conteo asociado.
 * Si `out_value` no es NULL, se escribe ahí el conteo máximo.
 */
int32_t counts_max_key_value(const Counts *c, uint32_t *out_value);

#ifdef __cplusplus
}
#endif

#endif /* ISODENDRON_COUNTS_H */
