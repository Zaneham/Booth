/* sysprint.h -- structured kernel output routing.
 *
 * Kernels emit class-tagged records into a buffer. The host drains
 * the buffer and dispatches each record to a sink registered for
 * its class. This is the SYSPRINT idiom from IBM mainframes, which
 * solved this problem in 1965 and has been waiting patiently for
 * GPUs to notice ever since. */

#ifndef BARRACUDA_SYSPRINT_H
#define BARRACUDA_SYSPRINT_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* ---- Limits ----
 * Fixed pools, in the tradition of compute that knows what it's
 * doing. If you need more classes than this you have problems
 * the runtime cannot help you with. */

#define BC_SP_MAX_CLASSES        128
#define BC_SP_MAX_CLASS_NAME     64
#define BC_SP_MAX_SINKS          16
#define BC_SP_DEFAULT_BUF_BYTES  (64 * 1024)

#define BC_SP_CLASS_NONE         0u

/* ---- Record ----
 * Each record is class_id, payload length, payload bytes. The
 * payload is opaque to the dispatcher; the sink knows what to do
 * with it. We do not concern ourselves with the sink's business. */

typedef struct {
    uint32_t class_id;
    uint32_t payload_len;
    /* payload bytes follow, length payload_len, then 4-byte aligned. */
} bc_sp_record_t;

/* ---- Buffer ----
 * Linear append. When full, subsequent emits are silently dropped,
 * as is proper for telemetry that may or may not have a reader. */

typedef struct {
    uint8_t  *data;
    uint32_t  size;
    uint32_t  head;
    uint32_t  dropped;
} bc_sp_buf_t;

/* ---- Sink ----
 * Called once per record matching the sink's pattern. The runtime
 * supplies the class name in resolved form so the sink does not
 * have to look it up. */

typedef void (*bc_sp_sink_t)(uint32_t class_id,
                             const char *class_name,
                             const void *payload,
                             uint32_t payload_len,
                             void *user);

/* ---- Public API ---- */

/* Initialise a buffer with the given backing storage. The caller
 * owns the bytes; we just record where they are. */
void bc_sp_buf_init(bc_sp_buf_t *buf, void *bytes, uint32_t size);

/* Intern a class name and return its id. Repeated calls with the
 * same name return the same id. Returns BC_SP_CLASS_NONE if the
 * table is full, which is the kernel author's problem. */
uint32_t bc_sp_intern(const char *class_name);

/* Look up the name a class id was interned under. Returns NULL
 * for ids we have no record of. */
const char *bc_sp_class_name(uint32_t class_id);

/* Emit a record. Payload bytes are copied into the buffer. If
 * the buffer is full, the record is dropped and the buffer's
 * dropped counter is incremented. */
void bc_sp_emit(bc_sp_buf_t *buf, uint32_t class_id,
                const void *payload, uint32_t payload_len);

/* Convenience: emit a record whose payload is a printf-formatted
 * string. The string is formatted into a small stack buffer; if
 * your message exceeds it, it is truncated, in accordance with the
 * usual customs. */
void bc_sp_emitf(bc_sp_buf_t *buf, uint32_t class_id,
                 const char *fmt, ...);

/* Register a sink for records whose class name matches the
 * pattern. Patterns support trailing-* wildcards: "STEP1.*"
 * matches STEP1.TRACE and STEP1.ERROR but not STEP2.TRACE. The
 * pattern "*" matches everything, as it always has. Sinks are
 * checked in registration order; the first match wins. We do not
 * multicast. If you wanted multicast you should have said so. */
int bc_sp_register_sink(const char *pattern,
                        bc_sp_sink_t sink,
                        void *user);

/* Drain the buffer, dispatching each record to its sink. After
 * the drain the buffer is empty and ready for reuse. Records
 * whose class matches no sink are silently discarded, on the
 * grounds that nobody asked. */
void bc_sp_drain(bc_sp_buf_t *buf);

/* Reset the interning table and the sink registry. Mostly for
 * tests; nothing in production needs to forget a class. */
void bc_sp_reset_globals(void);

/* Two stock sinks for the common cases, supplied so the kernel
 * author does not have to write them. */
void bc_sp_sink_stdout(uint32_t class_id, const char *class_name,
                       const void *payload, uint32_t len, void *user);
void bc_sp_sink_stderr(uint32_t class_id, const char *class_name,
                       const void *payload, uint32_t len, void *user);

/* A FILE* sink. Pass the FILE* as the user argument when
 * registering. Records are written one per line, class name then
 * a colon then the payload interpreted as text. The format is
 * deliberately plain; if you wanted JSON you should have asked. */
void bc_sp_sink_file(uint32_t class_id, const char *class_name,
                     const void *payload, uint32_t len, void *user);

#endif /* BARRACUDA_SYSPRINT_H */
