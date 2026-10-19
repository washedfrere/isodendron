/*
 * blob.c — Serialización y deserialización del formato binario ISODENDRON.
 *
 * Ver blob.h para el contrato y el layout exacto.
 *
 * Estrategia de endianness:
 *   Todas las funciones de lectura y escritura usan helpers que
 *   escriben/leen uint32_t en little-endian. En plataformas LE, el
 *   helper es un memcpy de 4 bytes. En plataformas BE, hace el byte
 *   swap. Los int32_t se manejan como uint32_t y se reinterpretan al
 *   vuelo (en C, la conversión de uint32_t a int32_t con valor > 2^31
 *   es implementación-definida, así que se hace con cast explícito).
 *
 *   Esta indirección cuesta prácticamente nada y garantiza que el
 *   blob es portable entre arquitecturas sin cambiar el formato.
 */
#include "blob.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Detección de endianness                                             */
/* ------------------------------------------------------------------ */

static int is_little_endian(void)
{
    const uint16_t x = 0x0001;
    return *(const uint8_t *)&x == 0x01;
}

/* ------------------------------------------------------------------ */
/* Helpers de lectura/escritura LE                                     */
/* ------------------------------------------------------------------ */

static void write_u32_le(uint8_t *p, uint32_t v)
{
    if (is_little_endian()) {
        memcpy(p, &v, 4);
        return;
    }
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static uint32_t read_u32_le(const uint8_t *p)
{
    if (is_little_endian()) {
        uint32_t v;
        memcpy(&v, p, 4);
        return v;
    }
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* Reinterpretación segura de uint32_t a int32_t sin UB para valores
 * fuera del rango de int32_t positivo. En C99/C11 con dos's complement
 * (que es lo que asumimos implícitamente en todo ISODENDRON), el cast
 * directo funciona, pero hacerlo explícito documenta la intención. */
static int32_t u32_to_i32(uint32_t v)
{
    return (int32_t)v;
}

static uint32_t i32_to_u32(int32_t v)
{
    return (uint32_t)v;
}

/* ------------------------------------------------------------------ */
/* Cálculo de tamaño                                                   */
/* ------------------------------------------------------------------ */

size_t blob_total_size(uint32_t n_nodes, uint32_t payload_bytes)
{
    return (size_t)BLOB_HEADER_SIZE
         + (size_t)n_nodes * (size_t)BLOB_NODE_SIZE
         + (size_t)payload_bytes;
}

/* ------------------------------------------------------------------ */
/* Escritura                                                           */
/* ------------------------------------------------------------------ */

void blob_write_header(uint8_t  *buf,
                       uint32_t  n_features,
                       uint32_t  n_nodes,
                       uint32_t  max_deep,
                       uint32_t  p_min_num,
                       uint32_t  p_min_den,
                       uint32_t  payload_bytes)
{
    if (buf == NULL) {
        return;
    }
    write_u32_le(buf + BLOB_OFF_MAGIC,       BLOB_MAGIC);
    write_u32_le(buf + BLOB_OFF_VERSION,     BLOB_VERSION);
    write_u32_le(buf + BLOB_OFF_N_FEATURES,  n_features);
    write_u32_le(buf + BLOB_OFF_N_NODES,     n_nodes);
    write_u32_le(buf + BLOB_OFF_MAX_DEEP,    max_deep);
    write_u32_le(buf + BLOB_OFF_P_MIN_NUM,   p_min_num);
    write_u32_le(buf + BLOB_OFF_P_MIN_DEN,   p_min_den);
    write_u32_le(buf + BLOB_OFF_PAYLOAD_BYTES, payload_bytes);
}

void blob_write_dist(uint8_t *buf, uint32_t off, int32_t target, int32_t count)
{
    if (buf == NULL) {
        return;
    }
    write_u32_le(buf + off,       i32_to_u32(target));
    write_u32_le(buf + off + 4u,  i32_to_u32(count));
}

void blob_read_dist(const uint8_t *buf, uint32_t off, int32_t *target,
                    int32_t *count)
{
    if (buf == NULL) {
        return;
    }
    if (target != NULL) {
        *target = u32_to_i32(read_u32_le(buf + off));
    }
    if (count != NULL) {
        *count = u32_to_i32(read_u32_le(buf + off + 4u));
    }
}

void blob_write_node(uint8_t *buf, uint32_t index, const BlobNode *node)
{
    if (buf == NULL || node == NULL) {
        return;
    }
    uint8_t *p = buf + BLOB_HEADER_SIZE + (size_t)index * BLOB_NODE_SIZE;

    write_u32_le(p + BLOB_NODE_OFF_IS_LEAF,       node->is_leaf);
    write_u32_le(p + BLOB_NODE_OFF_FEATURE_IDX,   i32_to_u32(node->feature_idx));
    write_u32_le(p + BLOB_NODE_OFF_VALUE,         i32_to_u32(node->value));
    write_u32_le(p + BLOB_NODE_OFF_N,             i32_to_u32(node->N));
    write_u32_le(p + BLOB_NODE_OFF_K,             i32_to_u32(node->K));
    write_u32_le(p + BLOB_NODE_OFF_MODE,         i32_to_u32(node->mode));
    write_u32_le(p + BLOB_NODE_OFF_N_MODE,       i32_to_u32(node->n_mode));
    write_u32_le(p + BLOB_NODE_OFF_LEFT_SUBTREE,  node->left_subtree_size);
}

/* ------------------------------------------------------------------ */
/* Lectura                                                             */
/* ------------------------------------------------------------------ */

int blob_read_header(const uint8_t *buf,
                     size_t         len,
                     BlobHeader    *out)
{
    if (buf == NULL || out == NULL) {
        return BLOB_ERR_NULL;
    }
    if (len < BLOB_HEADER_SIZE) {
        return BLOB_ERR_TOO_SHORT;
    }

    out->magic      = read_u32_le(buf + BLOB_OFF_MAGIC);
    out->version    = read_u32_le(buf + BLOB_OFF_VERSION);
    out->n_features = read_u32_le(buf + BLOB_OFF_N_FEATURES);
    out->n_nodes    = read_u32_le(buf + BLOB_OFF_N_NODES);
    out->max_deep   = read_u32_le(buf + BLOB_OFF_MAX_DEEP);
    out->p_min_num  = read_u32_le(buf + BLOB_OFF_P_MIN_NUM);
    out->p_min_den  = read_u32_le(buf + BLOB_OFF_P_MIN_DEN);
    out->payload_bytes = read_u32_le(buf + BLOB_OFF_PAYLOAD_BYTES);

    if (out->magic != BLOB_MAGIC) {
        return BLOB_ERR_MAGIC;
    }
    if (out->version != BLOB_VERSION) {
        return BLOB_ERR_VERSION;
    }

    /*
     * Validar que el tamaño del buffer cuadra exactamente con
     * n_nodes. Se hace con cuidado para no desbordar size_t si
     * n_nodes es enorme: primero comprobamos que n_nodes no excede
     * lo que len puede acomodar, y solo después calculamos el
     * tamaño esperado.
     */
    if (out->n_nodes > (len - BLOB_HEADER_SIZE) / BLOB_NODE_SIZE) {
        return BLOB_ERR_SIZE;
    }
    size_t expected = blob_total_size(out->n_nodes, out->payload_bytes);
    if (expected != len) {
        return BLOB_ERR_SIZE;
    }

    return BLOB_OK;
}

int blob_read_node(const uint8_t *buf,
                   size_t         len,
                   uint32_t       index,
                   BlobNode      *out)
{
    if (buf == NULL || out == NULL) {
        return BLOB_ERR_NULL;
    }

    /*
     * Comprobar que el nodo cabe en el buffer. Evitamos multiplicar
     * index * BLOB_NODE_SIZE sin comprobar antes, para prevenir
     * desbordamiento si index es muy grande.
     */
    if (index > (len - BLOB_HEADER_SIZE) / BLOB_NODE_SIZE) {
        return BLOB_ERR_SIZE;
    }
    size_t offset = BLOB_HEADER_SIZE + (size_t)index * BLOB_NODE_SIZE;
    if (offset + BLOB_NODE_SIZE > len) {
        return BLOB_ERR_SIZE;
    }

    const uint8_t *p = buf + offset;

    out->is_leaf          = read_u32_le(p + BLOB_NODE_OFF_IS_LEAF);
    out->feature_idx      = u32_to_i32(read_u32_le(p + BLOB_NODE_OFF_FEATURE_IDX));
    out->value            = u32_to_i32(read_u32_le(p + BLOB_NODE_OFF_VALUE));
    out->N                = u32_to_i32(read_u32_le(p + BLOB_NODE_OFF_N));
    out->K                = u32_to_i32(read_u32_le(p + BLOB_NODE_OFF_K));
    out->mode             = u32_to_i32(read_u32_le(p + BLOB_NODE_OFF_MODE));
    out->n_mode           = u32_to_i32(read_u32_le(p + BLOB_NODE_OFF_N_MODE));
    out->left_subtree_size = read_u32_le(p + BLOB_NODE_OFF_LEFT_SUBTREE);

    /*
     * En una hoja, los campos 4 y 8 no están libres: llevan el offset y
     * la longitud de su distribución, así que NO se normalizan. Solo
     * left_subtree_size se fuerza a 0, porque no tiene sentido en hoja.
     */
    if (out->is_leaf) {
        out->left_subtree_size = 0;
    }

    return BLOB_OK;
}

const uint8_t *blob_node_ptr(const uint8_t *buf,
                             size_t         len,
                             uint32_t       index)
{
    if (buf == NULL || len < BLOB_HEADER_SIZE) {
        return NULL;
    }
    if (index > (len - BLOB_HEADER_SIZE) / BLOB_NODE_SIZE) {
        return NULL;
    }
    size_t offset = BLOB_HEADER_SIZE + (size_t)index * BLOB_NODE_SIZE;
    if (offset + BLOB_NODE_SIZE > len) {
        return NULL;
    }
    return buf + offset;
}

/* ------------------------------------------------------------------ */
/* Validación de compatibilidad                                        */
/* ------------------------------------------------------------------ */

int blob_check_compat(const BlobHeader *header, uint32_t f)
{
    if (header == NULL) {
        return BLOB_ERR_NULL;
    }
    if (header->n_features != f) {
        return BLOB_ERR_SIZE;
    }
    return BLOB_OK;
}
