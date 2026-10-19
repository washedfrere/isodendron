/*
 * blob.h — Serialización binaria del árbol ISODENDRON.
 *
 * El blob es la única estructura que cruza la frontera hacia el
 * llamante. Es un buffer de bytes little-endian con un header fijo de
 * 32 bytes seguido de N nodos de 32 bytes cada uno, en orden de
 * preorden (raíz, subárbol ==, subárbol !=).
 *
 * Layout exacto:
 *
 *   Header (32 bytes):
 *     0   uint32   magic             = 0x444F5349 ("ISOD" en LE)
 *     4   uint32   version           = 1
 *     8   uint32   n_features
 *     12  uint32   n_nodes
 *     16  uint32   max_deep
 *     20  uint32   p_min_num
 *     24  uint32   p_min_den
 *     28  uint32   reserved          = 0
 *
 *   Nodo (32 bytes). Los nodos internos usan:
 *     0   uint32   is_leaf           0 = interno, 1 = hoja
 *     4   int32    feature_idx       índice de feature
 *     8   int32    value             valor umbral
 *     12  int32    N                 observaciones en el nodo
 *     16  int32    K                 valores distintos del target
 *     20  int32    mode              valor del target más frecuente
 *     24  int32    n_mode            obs. con target == mode
 *     28  uint32   left_subtree_size nº de nodos del subárbol ==. Permite
 *                                    localizar el hijo != en preorden sin
 *                                    recorrer el subárbol izquierdo.
 *
 *   Las HOJAS no tienen feature ni umbral, así que reutilizan esos campos
 *   para la distribución del target. Los mismos 32 bytes significan:
 *     0   uint32   is_leaf          = 1
 *     4   uint32   payload_off      offset absoluto de su distribución
 *     8   uint32   payload_len      nº de entradas guardadas (<= K)
 *     12  int32    N                observaciones de la hoja
 *     16  int32    K                valores distintos del target
 *     20  int32    mode             target más frecuente
 *     24  int32    n_mode           obs. con target == mode
 *     28  uint32   (sin uso)        = 0
 *
 *   Así el nodo no crece y todos los offsets se mantienen.
 *
 * El payload va tras los nodos y contiene, por hoja y de forma
 * consecutiva, `payload_len` pares (target, count) de 8 bytes, ordenados
 * por conteo descendente y, a igualdad, por target ascendente. Las
 * entradas guardadas pueden sumar menos que N: el resto de la masa es lo
 * descartado por el tope, recuperable como N - suma de conteos.
 *
 * El uso de left_subtree_size elimina la necesidad de recorrer el
 * subárbol izquierdo para encontrar el hijo derecho durante la
 * predicción. El coste es 4 bytes por nodo, y el beneficio es que el
 * descenso del árbol es O(profundidad) estricto en lugar de
 * O(profundidad²) en el peor caso.
 *
 * Endianness:
 *   El blob se escribe y se lee siempre en little-endian, con
 *   independencia de la máquina. Las funciones de acceso hacen la
 *   conversión si la plataforma es big-endian. En la práctica, todas
 *   las plataformas que vamos a tocar son LE, así que la conversión
 *   es una identidad.
 *
 * Portabilidad:
 *   Los structs BlobHeader y BlobNode se definen con campos de ancho
 *   fijo (uint32_t / int32_t) y sin padding, de forma que su layout
 *   en memoria coincide con el layout del blob en plataformas LE con
 *   alineación natural a 4 bytes. Las funciones de escritura y lectura
 *   no dependen de esa coincidencia: escriben y leen campo a campo,
 *   así que funcionan también en plataformas con padding o BE.
 */
#ifndef ISODENDRON_BLOB_H
#define ISODENDRON_BLOB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Constantes                                                          */
/* ------------------------------------------------------------------ */

#define BLOB_MAGIC           0x444F5349u   /* "ISOD" en LE */
#define BLOB_VERSION         3u

#define BLOB_HEADER_SIZE     32u
#define BLOB_NODE_SIZE       32u

/* Tamaño de una entrada del payload: (target int32, count int32). */
#define BLOB_DIST_ENTRY_SIZE 8u

/* Offsets dentro del header. */
#define BLOB_OFF_MAGIC            0u
#define BLOB_OFF_VERSION          4u
#define BLOB_OFF_N_FEATURES       8u
#define BLOB_OFF_N_NODES          12u
#define BLOB_OFF_MAX_DEEP         16u
#define BLOB_OFF_P_MIN_NUM        20u
#define BLOB_OFF_P_MIN_DEN        24u
#define BLOB_OFF_PAYLOAD_BYTES    28u

/* Offsets dentro de un nodo. */
#define BLOB_NODE_OFF_IS_LEAF           0u
#define BLOB_NODE_OFF_FEATURE_IDX       4u
#define BLOB_NODE_OFF_VALUE             8u
#define BLOB_NODE_OFF_N                 12u
#define BLOB_NODE_OFF_K                 16u
#define BLOB_NODE_OFF_MODE              20u
#define BLOB_NODE_OFF_N_MODE            24u
#define BLOB_NODE_OFF_LEFT_SUBTREE      28u

/* ------------------------------------------------------------------ */
/* Estructuras                                                         */
/* ------------------------------------------------------------------ */

/*
 * Header del blob, en forma de struct para comodidad del llamante.
 * No se escribe ni se lee como bloque; se serializa campo a campo.
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_features;
    uint32_t n_nodes;
    uint32_t max_deep;
    uint32_t p_min_num;
    uint32_t p_min_den;
    uint32_t payload_bytes;
} BlobHeader;

/*
 * Nodo del blob. Misma consideración que el header: es una vista
 * cómoda, no una representación en disco.
 *
 * En hojas, feature_idx y value son 0, y left_subtree_size es 0.
 */
typedef struct {
    uint32_t is_leaf;
    int32_t  feature_idx;
    int32_t  value;
    int32_t  N;
    int32_t  K;
    int32_t  mode;
    int32_t  n_mode;
    uint32_t left_subtree_size;
} BlobNode;

/* ------------------------------------------------------------------ */
/* Cálculo de tamaño                                                   */
/* ------------------------------------------------------------------ */

/*
 * Devuelve el tamaño total en bytes de un blob con `n_nodes` nodos y
 * `payload_bytes` de distribuciones:
 *   BLOB_HEADER_SIZE + n_nodes * BLOB_NODE_SIZE + payload_bytes
 */
size_t blob_total_size(uint32_t n_nodes, uint32_t payload_bytes);

/* ------------------------------------------------------------------ */
/* Escritura                                                           */
/* ------------------------------------------------------------------ */

/*
 * Escribe el header en `buf`. `buf` debe tener al menos
 * BLOB_HEADER_SIZE bytes disponibles. No comprueba el tamaño: es
 * responsabilidad del llamante haber reservado blob_total_size().
 *
 * Los campos `magic` y `version` se escriben con los valores
 * canónicos (BLOB_MAGIC y BLOB_VERSION). El campo `reserved` se
 * escribe a 0.
 *
 * Los parámetros n_features, n_nodes, max_deep, p_min_num y p_min_den
 * se escriben tal cual. El llamante es responsable de que n_nodes
 * coincida con el número de nodos que escribirá después.
 */
void blob_write_header(uint8_t         *buf,
                       uint32_t         n_features,
                       uint32_t         n_nodes,
                       uint32_t         max_deep,
                       uint32_t         p_min_num,
                       uint32_t         p_min_den,
                       uint32_t         payload_bytes);

/*
 * Escribe y lee una entrada de la distribución del target. `off` es el
 * offset absoluto desde el comienzo del blob.
 */
void blob_write_dist(uint8_t *buf, uint32_t off, int32_t target,
                     int32_t count);
void blob_read_dist(const uint8_t *buf, uint32_t off, int32_t *target,
                    int32_t *count);

/*
 * Escribe el nodo `index` (0-based, en preorden) en `buf`, en la
 * posición BLOB_HEADER_SIZE + index * BLOB_NODE_SIZE.
 *
 * `buf` debe tener al menos blob_total_size(index + 1) bytes
 * disponibles. No comprueba el tamaño.
 */
void blob_write_node(uint8_t *buf, uint32_t index, const BlobNode *node);

/* ------------------------------------------------------------------ */
/* Lectura                                                             */
/* ------------------------------------------------------------------ */

/*
 * Códigos de retorno para las funciones de lectura.
 */
#define BLOB_OK             0
#define BLOB_ERR_NULL       1   /* buf o out son NULL */
#define BLOB_ERR_TOO_SHORT  2   /* len < tamaño mínimo esperado */
#define BLOB_ERR_MAGIC      3   /* magic no coincide */
#define BLOB_ERR_VERSION    4   /* versión desconocida */
#define BLOB_ERR_SIZE       5   /* n_nodes no cuadra con len */

/*
 * Lee y valida el header de `buf`.
 *
 * Comprueba:
 *   - buf y out no son NULL.
 *   - len >= BLOB_HEADER_SIZE.
 *   - magic coincide.
 *   - version es la esperada.
 *   - n_nodes * BLOB_NODE_SIZE + BLOB_HEADER_SIZE == len.
 *
 * Devuelve uno de los códigos BLOB_*.
 */
int blob_read_header(const uint8_t *buf,
                     size_t         len,
                     BlobHeader    *out);

/*
 * Lee el nodo `index` de `buf`.
 *
 * Comprueba que buf y out no son NULL y que
 * BLOB_HEADER_SIZE + (index + 1) * BLOB_NODE_SIZE <= len.
 *
 * No vuelve a validar el header: se asume que el llamante ya lo hizo
 * con blob_read_header. Esto permite que isodendron_predict lea el header
 * una vez y luego lea nodos en un bucle sin repetir validaciones.
 *
 * Devuelve uno de los códigos BLOB_*.
 */
int blob_read_node(const uint8_t *buf,
                   size_t         len,
                   uint32_t       index,
                   BlobNode      *out);

/*
 * Devuelve un puntero directo al nodo `index` dentro de `buf`, sin
 * copiarlo. El puntero es válido mientras `buf` lo sea.
 *
 * El nodo devuelto está en el layout del blob (little-endian), no en
 * el layout de BlobNode. Solo es seguro usarlo en plataformas LE y
 * con acceso a los campos a través de los offsets BLOB_NODE_OFF_*.
 *
 * Pensada para el bucle caliente de predicción en plataformas LE.
 * En plataformas BE, usar blob_read_node.
 *
 * Devuelve NULL si buf es NULL o si el índice está fuera de rango
 * (asumiendo len coherente con el header).
 */
const uint8_t *blob_node_ptr(const uint8_t *buf,
                             size_t         len,
                             uint32_t       index);

/* ------------------------------------------------------------------ */
/* Validación de header contra parámetros esperados                    */
/* ------------------------------------------------------------------ */

/*
 * Comprueba que el header es compatible con los parámetros de una
 * llamada de predicción: que n_features coincide con `f`.
 *
 * No comprueba magic, version ni tamaño; para eso está
 * blob_read_header. Esta función es un chequeo adicional.
 *
 * Devuelve BLOB_OK si son compatibles, BLOB_ERR_SIZE si no.
 */
int blob_check_compat(const BlobHeader *header, uint32_t f);

#ifdef __cplusplus
}
#endif

#endif /* ISODENDRON_BLOB_H */
