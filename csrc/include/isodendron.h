/*
 * isodendron.h — Interfaz pública de ISODENDRON
 *
 * ISODENDRON es un árbol de decisión binario, determinista, sobre features y
 * target discretos (int32_t). Los splits son de la forma:
 *
 *     feature == valor   vs   feature != valor
 *
 * La función de ganancia es puramente el balance de tamaños:
 *
 *     score(f, v) = min(n_igual, n_distinto)
 *
 * El target no participa en la elección del split; solo se usa para
 * resumir cada nodo con (N, K, mode, n_mode).
 *
 * Esta cabecera define la frontera entre la implementación C y los
 * clientes externos (Python vía ctypes, C puro, etc.). El contrato es:
 *
 *   - C solo lee arrays de int32_t y produce un blob de bytes.
 *   - El blob es la única estructura que cruza la frontera.
 *   - Toda memoria asignada por la biblioteca se libera con isodendron_free.
 */
#ifndef ISODENDRON_H
#define ISODENDRON_H

#include <stdint.h>
#include <stddef.h>

/*
 * Decoración de exportación.
 *
 *   En Windows:
 *     - Al compilar la DLL (ISODENDRON_BUILD_DLL definido) -> dllexport.
 *     - Al consumirla desde otro módulo               -> dllimport.
 *
 *   En Linux/macOS con GCC o Clang:
 *     - visibility("default"), explícito pero inocuo si no se usa
 *       -fvisibility=hidden.
 *
 *   En cualquier otro caso: vacío.
 */
#if defined(_WIN32)
#  if defined(ISODENDRON_BUILD_DLL)
#    define ISODENDRON_API __declspec(dllexport)
#  else
#    define ISODENDRON_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define ISODENDRON_API __attribute__((visibility("default")))
#else
#  define ISODENDRON_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Códigos de retorno                                                  */
/* ------------------------------------------------------------------ */

#define ISODENDRON_OK        0   /* éxito */
#define ISODENDRON_ERR_MEM   1   /* fallo de asignación de memoria */
#define ISODENDRON_ERR_ARG   2   /* argumento inválido */

/* ------------------------------------------------------------------ */
/* Formato del blob                                                    */
/* ------------------------------------------------------------------ */

/*
 * El blob es un buffer de bytes little-endian con la siguiente forma.
 * Todos los enteros son unsigned 32-bit little-endian excepto donde se
 * indique lo contrario.
 *
 * Header (32 bytes):
 *   offset  size  campo
 *   0       4     magic         = ISODENDRON_BLOB_MAGIC
 *   4       4     version       = ISODENDRON_BLOB_VERSION
 *   8       4     n_features    número de features del modelo
 *   12      4     n_nodes       número total de nodos (internos + hojas)
 *   16      4     max_deep      profundidad máxima usada en entrenamiento
 *   20      4     p_min_num     numerador de P_MIN
 *   24      4     p_min_den     denominador de P_MIN
 *   28      4     reserved      = 0
 *
 * Nodos (32 bytes cada uno, en orden de preorden: primero el nodo raíz,
 * luego recursivamente la rama ==, luego la rama !=):
 *   offset  size  campo
 *   0       4     is_leaf       0 = interno, 1 = hoja
 *   4       4     feature_idx   índice de la feature (solo si !is_leaf)
 *   8       4     value         valor umbral int32_t (solo si !is_leaf)
 *   12      4     N             observaciones en el nodo
 *   16      4     K             valores distintos del target
 *   20      4     mode          target más frecuente (int32_t); empate ->
 *                                intermedio menor de los empatados
 *   24      4     n_mode        nº de obs. con target == mode
 *   28      4     reserved      = 0
 *
 * En una hoja, feature_idx y value son 0.
 *
 * El blob está alineado a 4 bytes en todos sus campos, de forma que se
 * puede leer con struct.unpack_from sin copias ni realineamientos.
 */
#define ISODENDRON_BLOB_MAGIC     0x444F5349u   /* "ISOD" en LE */
#define ISODENDRON_BLOB_VERSION   3u

/* Número máximo de entradas de distribución que guarda cada hoja. */
#define ISODENDRON_MAX_TARGETS    10u

#define ISODENDRON_BLOB_HEADER_SIZE  32u
#define ISODENDRON_BLOB_NODE_SIZE    32u

/* ------------------------------------------------------------------ */
/* Versión                                                             */
/* ------------------------------------------------------------------ */

/*
 * Devuelve una cadena estática con la versión de la biblioteca, por
 * ejemplo "1.0.0". El llamante no debe liberarla.
 */
ISODENDRON_API const char *isodendron_version(void);

/* ------------------------------------------------------------------ */
/* Entrenamiento                                                       */
/* ------------------------------------------------------------------ */

/*
 * Entrena un árbol ISODENDRON y devuelve su serialización en un blob.
 *
 * Parámetros:
 *   X           Matriz de features en row-major, tamaño n * f.
 *               X[i * f + j] es la feature j de la muestra i.
 *   y           Vector de targets, tamaño n.
 *   n           Número de muestras (> 0).
 *   f           Número de features (> 0).
 *   max_deep    Profundidad máxima del árbol (>= 1). La raíz está a
 *               profundidad 0; una hoja se crea al alcanzar max_deep.
 *   p_min_num   Numerador de P_MIN.
 *   p_min_den   Denominador de P_MIN (> 0). P_MIN = p_min_num / p_min_den.
 *               Un nodo deja de dividirse si la probabilidad de su
 *               la moda, P = n_mode / N, es mayor o igual que P_MIN.
 *
 *               Además hay una parada por pureza SIEMPRE activa, al
 *               margen de P_MIN: si todos los targets del nodo son
 *               iguales (K == 1) el nodo se queda como hoja, porque
 *               dividirlo no cambiaría ninguna predicción. Consecuencia:
 *               P_MIN = 1/1 coincide con esa regla y ya no aporta nada
 *               distinto, y los árboles con P_MIN menor salen más
 *               pequeños sin perder exactitud.
 *               La condición aplica a todos los nodos, incluida la raíz.
 *
 * Salida:
 *   out_blob    Recibe un puntero a un buffer asignado por la biblioteca
 *               con el blob serializado. El llamante debe liberarlo con
 *               isodendron_free. No se modifica si la función devuelve error.
 *   out_len     Recibe el tamaño del blob en bytes.
 *
 * Devuelve:
 *   ISODENDRON_OK       éxito
 *   ISODENDRON_ERR_ARG  algún argumento es inválido
 *   ISODENDRON_ERR_MEM  fallo de asignación
 *
 * Notas:
 *   - La función no retiene ninguna referencia a X ni a y tras volver.
 *     El llamante puede liberarlos inmediatamente.
 *   - Es segura para llamadas concurrentes: cada invocación usa su
 *     propio arena interno.
 */
/*
 * Igual que isodendron_train_mt, pero con el tope de entradas de distribución
 * por hoja como parámetro. `max_targets = 0` desactiva el payload: las
 * hojas quedan con payload_len = 0 y el blob no lleva distribuciones, lo
 * que reproduce el comportamiento anterior al formato v3.
 */
ISODENDRON_API int isodendron_train_mt_cap(const int32_t *X,
                  const int32_t *y,
                  uint32_t       n,
                  uint32_t       f,
                  uint32_t       max_deep,
                  uint32_t       p_min_num,
                  uint32_t       p_min_den,
                  uint32_t       max_targets,
                  uint32_t       n_threads,
                  uint8_t      **out_blob,
                  size_t        *out_len);

ISODENDRON_API int isodendron_train(const int32_t *X,
                const int32_t *y,
                uint32_t       n,
                uint32_t       f,
                uint32_t       max_deep,
                uint32_t       p_min_num,
                uint32_t       p_min_den,
                uint8_t      **out_blob,
                size_t        *out_len);

/*
 * Igual que isodendron_train, pero reparte la construcción del árbol entre
 * `n_threads` hilos. n_threads == 0 o 1 equivale a isodendron_train.
 *
 * El reparto es transparente: las decisiones de split no dependen de
 * qué hilo construya cada subárbol, así que el blob resultante es
 * idéntico byte a byte al secuencial. La paralelización solo afecta al
 * tiempo, nunca al modelo.
 *
 * Coste de memoria: cada hilo reserva su propio arena, así que el pico
 * es aproximadamente n_threads veces el de la versión secuencial.
 */
ISODENDRON_API int isodendron_train_mt(const int32_t *X,
                   const int32_t *y,
                   uint32_t       n,
                   uint32_t       f,
                   uint32_t       max_deep,
                   uint32_t       p_min_num,
                   uint32_t       p_min_den,
                   uint32_t       n_threads,
                   uint8_t      **out_blob,
                   size_t        *out_len);

/*
 * ACUMULA un conjunto nuevo en las hojas de un blob, sin tocar la estructura del arbol: los nodos internos quedan intactos y solo cambian N, K, mode, n_mode y la distribucion de cada hoja. OJO: SUMA, no recalcula; antes de sumar aplica un decaimiento a todas las hojas, visitadas o no, restando (m - 1) a sus conteos con m el minimo de esa hoja, de modo que su minimo queda en 1, y bajando N en la misma cantidad para conservar el residuo. Una hoja que el conjunto nuevo no visita NO se vacia: conserva su evidencia decaida.
 *
 * Parámetros:
 *   blob_in      Blob de partida (versión actual).
 *   blob_len     Tamaño del blob de partida.
 *   X, y         Conjunto de calibración, row-major, n * f.
 *   n, f         Filas y features; f debe coincidir con el del blob.
 *   max_targets  Tope de entradas por hoja; 0 deja el payload vacío.
 *   out_blob     Blob nuevo, a liberar con isodendron_free.
 *   out_len      Tamaño del blob nuevo.
 *
 * Devuelve ISODENDRON_OK, ISODENDRON_ERR_ARG o ISODENDRON_ERR_MEM.
 */
typedef struct {
    uint32_t nodo;
    int32_t  N_previo;
    uint32_t payload_len;
    uint32_t resta;
    int32_t  N_final;
} IsodendronHojaAnomala;

typedef struct {
    uint64_t puntos;
    uint32_t hojas;
    uint32_t decayadas;
    uint32_t vaciadas;
    uint32_t resta_maxima;
    uint32_t anomalas;
    uint32_t capacidad;
    IsodendronHojaAnomala *detalle;
} IsodendronPartialDiag;

ISODENDRON_API int isodendron_partial_fit_diag(const uint8_t *blob_in, size_t blob_len,
                  const int32_t *X,
                  const int32_t *y,
                  uint32_t       n,
                  uint32_t       f,
                  uint32_t       max_targets,
                  uint8_t      **out_blob,
                  size_t        *out_len,
                  IsodendronPartialDiag *diag);

ISODENDRON_API int isodendron_partial_fit(const uint8_t *blob_in, size_t blob_len,
                  const int32_t *X,
                  const int32_t *y,
                  uint32_t       n,
                  uint32_t       f,
                  uint32_t       max_targets,
                  uint8_t      **out_blob,
                  size_t        *out_len);

/* ------------------------------------------------------------------ */
/* Predicción                                                          */
/* ------------------------------------------------------------------ */

/*
 * Predice para un lote de muestras usando un blob previamente entrenado.
 *
 * Para cada muestra, recorre el árbol según las condiciones de sus
 * nodos internos y devuelve la moda del nodo hoja alcanzado: el valor
 * del target más frecuente en esa región, que para un target categórico
 * es la clase más probable.
 *
 * Parámetros:
 *   blob        Buffer con el árbol serializado.
 *   blob_len    Tamaño del blob en bytes (>= ISODENDRON_BLOB_HEADER_SIZE).
 *   X           Matriz de features en row-major, tamaño n * f.
 *               El número de features debe coincidir con el del blob.
 *   n           Número de muestras (> 0).
 *   f           Número de features (> 0). Debe coincidir con el del blob.
 *
 * Salida:
 *   out_pred    Vector de tamaño n con las predicciones (la moda de cada
 *               hoja alcanzada).
 *
 * Devuelve:
 *   ISODENDRON_OK       éxito
 *   ISODENDRON_ERR_ARG  argumento inválido, blob malformado o incompatible
 *
 * Notas:
 *   - Es segura para llamadas concurrentes sobre el mismo blob: solo
 *     lee, nunca escribe.
 *   - No asigna memoria dinámica; todo el trabajo es sobre la pila.
 */
ISODENDRON_API int isodendron_predict(const uint8_t *blob,
                  size_t         blob_len,
                  const int32_t *X,
                  uint32_t       n,
                  uint32_t       f,
                  int32_t       *out_pred);

/*
 * Igual que isodendron_predict, pero devuelve además el índice del nodo hoja
 * en el que acaba cada muestra.
 *
 * Es lo que permite recuperar la distribución de la hoja (offset y
 * longitud en los campos 4 y 8 del nodo) y agregar probabilidades en
 * lugar de votos. El índice es el del nodo en preorden, el mismo que
 * usa el blob.
 *
 * `out_pred` y `out_leaf_idx` son obligatorios.
 */
ISODENDRON_API int isodendron_predict_leaf(const uint8_t *blob,
                  size_t         blob_len,
                  const int32_t *X,
                  uint32_t       n,
                  uint32_t       f,
                  int32_t       *out_pred,
                  uint32_t      *out_leaf_idx);

/*
 * Igual que isodendron_predict, pero devuelve además el resumen de la hoja
 * alcanzada por cada muestra:
 *
 *   out_n_mode[i]     observaciones de esa hoja cuyo target vale su moda
 *   out_leaf_n[i]     observaciones totales de esa hoja
 *
 * De ahí sale la pureza de la hoja, P = out_n_mode / out_leaf_n, que es
 * la exactitud del modelo dentro de esa región sobre el conjunto de
 * entrenamiento. Sirve para ponderar predicciones —por P o directamente
 * por out_n_mode, según convenga— y queda a criterio del llamante.
 *
 * Advertencia: es una medida de ENTRENAMIENTO, no de acierto esperado.
 * En árboles muy profundos casi todas las hojas tienen N = 1 y por
 * tanto P = 1, así que ahí no discrimina nada.
 *
 * Cualquiera de los dos arrays extra puede ser NULL; en ese caso no se
 * rellena y el resultado es idéntico a isodendron_predict.
 */
ISODENDRON_API int isodendron_predict_full(const uint8_t *blob,
                  size_t         blob_len,
                  const int32_t *X,
                  uint32_t       n,
                  uint32_t       f,
                  int32_t       *out_pred,
                  int32_t       *out_n_mode,
                  int32_t       *out_leaf_n);

/* ------------------------------------------------------------------ */
/* Liberación de memoria                                               */
/* ------------------------------------------------------------------ */

/*
 * Libera un buffer asignado por la biblioteca (por ejemplo, el blob
 * devuelto por isodendron_train). Llamar con NULL es seguro y no hace nada.
 *
 * No debe usarse para liberar memoria que no haya sido asignada por
 * esta biblioteca.
 */
ISODENDRON_API void isodendron_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* ISODENDRON_H */
