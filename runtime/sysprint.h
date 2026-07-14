/* sysprint.h -- structured kernel output routing.
 * Kernels emit class-tagged records into a buffer; the host drains it and
 * dispatches each record to the sink registered for its class. SYSPRINT, for GPUs. */

#ifndef BARRACUDA_SYSPRINT_H
#define BARRACUDA_SYSPRINT_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* ---- Limits ----
 * Fixed pools. Need more classes than this and you have other problems. */

#define BC_SP_MAX_CLASSES        128
#define BC_SP_MAX_CLASS_NAME     64
#define BC_SP_MAX_SINKS          16
#define BC_SP_DEFAULT_BUF_BYTES  (64 * 1024)

#define BC_SP_CLASS_NONE         0u

/* ---- Record ----
 * class_id, payload length, then payload bytes; the payload is opaque to the
 * dispatcher and meaningful only to the sink. */

typedef struct {
    uint32_t class_id;
    uint32_t payload_len;
    /* payload bytes follow, length payload_len, then 4-byte aligned. */
} bc_sp_record_t;

/* ---- Buffer ----
 * Linear append; full means silent drop. Layout is mirrored in sysprint_device.h,
 * and the guard below stops a double typedef. */

#ifndef BC_SP_BUF_T_DEFINED
#define BC_SP_BUF_T_DEFINED
typedef struct {
    uint8_t  *data;
    uint32_t  size;
    uint32_t  head;
    uint32_t  dropped;
} bc_sp_buf_t;
#endif

/* ---- Sink ----
 * Called once per matching record; the runtime passes the resolved class name. */

typedef void (*bc_sp_sink_t)(uint32_t class_id,
                             const char *class_name,
                             const void *payload,
                             uint32_t payload_len,
                             void *user);

/* ---- Public API ---- */

/* Initialise a buffer over caller-owned backing storage. */
void bc_sp_buf_init(bc_sp_buf_t *buf, void *bytes, uint32_t size);

/* Intern a class name to a stable id; same name returns the same id, or
 * BC_SP_CLASS_NONE if the table is full. */
uint32_t bc_sp_intern(const char *class_name);

/* Return the name an id was interned under, or NULL if unknown. */
const char *bc_sp_class_name(uint32_t class_id);

/* Emit a record; payload is copied in. Full buffer drops it and bumps dropped. */
void bc_sp_emit(bc_sp_buf_t *buf, uint32_t class_id,
                const void *payload, uint32_t payload_len);

/* Emit a printf-formatted record; formatted via a small stack buffer, truncated
 * if it overflows. */
void bc_sp_emitf(bc_sp_buf_t *buf, uint32_t class_id,
                 const char *fmt, ...);

/* Register a sink for class names matching pattern. Trailing-* is a prefix
 * wildcard ("STEP1.*"), "*" matches all. First match in registration order wins;
 * no multicast. */
int bc_sp_register_sink(const char *pattern,
                        bc_sp_sink_t sink,
                        void *user);

/* Drain the buffer, dispatching each record to its sink and leaving it empty.
 * Records matching no sink are discarded. */
void bc_sp_drain(bc_sp_buf_t *buf);

/* Reset the interning table and the sink registry. Mostly for
 * tests; nothing in production needs to forget a class. */
void bc_sp_reset_globals(void);

/* Two stock sinks for the common cases. */
void bc_sp_sink_stdout(uint32_t class_id, const char *class_name,
                       const void *payload, uint32_t len, void *user);
void bc_sp_sink_stderr(uint32_t class_id, const char *class_name,
                       const void *payload, uint32_t len, void *user);

/* FILE* sink; pass the FILE* as user. Writes one line per record: class name,
 * colon, payload as text. */
void bc_sp_sink_file(uint32_t class_id, const char *class_name,
                     const void *payload, uint32_t len, void *user);

#endif /* BARRACUDA_SYSPRINT_H */
