/* sysprint.c -- structured kernel output routing.
 * Class table and sink registry are file-scope singletons, one set per process. */

#include "sysprint.h"

#include <string.h>
#include <stdarg.h>
#include <stdio.h>

/* ---- Class Interning ----
 * Linear search; the table is small and the lookup isn't hot. Sort it if that
 * ever changes. */

typedef struct {
    char name[BC_SP_MAX_CLASS_NAME];
    uint8_t len;
    uint8_t pad[3];
} bc_sp_class_t;

static bc_sp_class_t g_classes[BC_SP_MAX_CLASSES];
static uint32_t g_num_classes;

uint32_t bc_sp_intern(const char *class_name)
{
    if (!class_name) return BC_SP_CLASS_NONE;
    size_t n = strlen(class_name);
    if (n == 0 || n >= BC_SP_MAX_CLASS_NAME) return BC_SP_CLASS_NONE;

    for (uint32_t i = 1; i <= g_num_classes; i++) {
        const bc_sp_class_t *c = &g_classes[i - 1];
        if (c->len == (uint8_t)n && memcmp(c->name, class_name, n) == 0) {
            return i;
        }
    }
    if (g_num_classes >= BC_SP_MAX_CLASSES) return BC_SP_CLASS_NONE;
    bc_sp_class_t *c = &g_classes[g_num_classes++];
    memcpy(c->name, class_name, n);
    c->name[n] = '\0';
    c->len = (uint8_t)n;
    return g_num_classes;
}

const char *bc_sp_class_name(uint32_t class_id)
{
    if (class_id == BC_SP_CLASS_NONE) return NULL;
    if (class_id > g_num_classes) return NULL;
    return g_classes[class_id - 1].name;
}

/* ---- Sink Registry ----
 * Patterns stored in order; first match wins. */

typedef struct {
    char         pattern[BC_SP_MAX_CLASS_NAME];
    uint8_t      plen;
    uint8_t      is_prefix;   /* trailing-* wildcard */
    uint8_t      pad[2];
    bc_sp_sink_t sink;
    void        *user;
} bc_sp_sink_entry_t;

static bc_sp_sink_entry_t g_sinks[BC_SP_MAX_SINKS];
static uint32_t g_num_sinks;

int bc_sp_register_sink(const char *pattern, bc_sp_sink_t sink, void *user)
{
    if (!pattern || !sink) return -1;
    if (g_num_sinks >= BC_SP_MAX_SINKS) return -1;
    size_t n = strlen(pattern);
    if (n == 0 || n >= BC_SP_MAX_CLASS_NAME) return -1;

    bc_sp_sink_entry_t *e = &g_sinks[g_num_sinks];
    /* Trailing * marks a prefix match; everything else is exact. */
    if (pattern[n - 1] == '*') {
        e->is_prefix = 1;
        memcpy(e->pattern, pattern, n - 1);
        e->pattern[n - 1] = '\0';
        e->plen = (uint8_t)(n - 1);
    } else {
        e->is_prefix = 0;
        memcpy(e->pattern, pattern, n);
        e->pattern[n] = '\0';
        e->plen = (uint8_t)n;
    }
    e->sink = sink;
    e->user = user;
    g_num_sinks++;
    return 0;
}

static bc_sp_sink_entry_t *sp_match_sink(const char *class_name, uint8_t cname_len)
{
    for (uint32_t i = 0; i < g_num_sinks; i++) {
        bc_sp_sink_entry_t *e = &g_sinks[i];
        if (e->is_prefix) {
            if (e->plen == 0) return e;        /* bare "*" matches all */
            if (cname_len >= e->plen &&
                memcmp(class_name, e->pattern, e->plen) == 0) {
                return e;
            }
        } else {
            if (cname_len == e->plen &&
                memcmp(class_name, e->pattern, e->plen) == 0) {
                return e;
            }
        }
    }
    return NULL;
}

/* ---- Buffer ----
 * Linear append, records 4-byte aligned after an 8-byte header (class_id,
 * payload_len). No wrap: when full, emits drop and bump the dropped counter. */

void bc_sp_buf_init(bc_sp_buf_t *buf, void *bytes, uint32_t size)
{
    buf->data = (uint8_t *)bytes;
    buf->size = size;
    buf->head = 0;
    buf->dropped = 0;
}

static uint32_t sp_align4(uint32_t n) { return (n + 3u) & ~3u; }

void bc_sp_emit(bc_sp_buf_t *buf, uint32_t class_id,
                const void *payload, uint32_t payload_len)
{
    if (!buf || !buf->data || class_id == BC_SP_CLASS_NONE) {
        if (buf) buf->dropped++;
        return;
    }
    uint32_t record_bytes = (uint32_t)sizeof(bc_sp_record_t) +
                             sp_align4(payload_len);
    if (buf->head + record_bytes > buf->size) {
        buf->dropped++;
        return;
    }
    bc_sp_record_t hdr;
    hdr.class_id = class_id;
    hdr.payload_len = payload_len;
    memcpy(buf->data + buf->head, &hdr, sizeof(hdr));
    if (payload_len > 0 && payload) {
        memcpy(buf->data + buf->head + sizeof(hdr), payload, payload_len);
    }
    buf->head += record_bytes;
}

void bc_sp_emitf(bc_sp_buf_t *buf, uint32_t class_id,
                 const char *fmt, ...)
{
    /* 256-byte stack buffer; longer messages get truncated. */
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof(tmp) - 1) n = (int)sizeof(tmp) - 1;
    bc_sp_emit(buf, class_id, tmp, (uint32_t)n);
}

void bc_sp_drain(bc_sp_buf_t *buf)
{
    if (!buf || !buf->data) return;
    uint32_t pos = 0;
    while (pos + sizeof(bc_sp_record_t) <= buf->head) {
        bc_sp_record_t hdr;
        memcpy(&hdr, buf->data + pos, sizeof(hdr));
        uint32_t record_bytes = (uint32_t)sizeof(hdr) +
                                 sp_align4(hdr.payload_len);
        if (pos + record_bytes > buf->head) break;
        const uint8_t *payload = buf->data + pos + sizeof(hdr);

        const char *cname = bc_sp_class_name(hdr.class_id);
        if (cname) {
            uint8_t cnlen = (hdr.class_id <= g_num_classes)
                            ? g_classes[hdr.class_id - 1].len
                            : (uint8_t)strlen(cname);
            bc_sp_sink_entry_t *e = sp_match_sink(cname, cnlen);
            if (e) {
                e->sink(hdr.class_id, cname, payload,
                        hdr.payload_len, e->user);
            }
        }
        pos += record_bytes;
    }
    buf->head = 0;
}

/* ---- Stock Sinks ----
 * Deliberately plain reference sinks; write your own for JSON or structured
 * logging. */

void bc_sp_sink_stdout(uint32_t class_id, const char *class_name,
                       const void *payload, uint32_t len, void *user)
{
    (void)class_id; (void)user;
    fprintf(stdout, "%s: %.*s\n", class_name, (int)len,
            (const char *)payload);
}

void bc_sp_sink_stderr(uint32_t class_id, const char *class_name,
                       const void *payload, uint32_t len, void *user)
{
    (void)class_id; (void)user;
    fprintf(stderr, "%s: %.*s\n", class_name, (int)len,
            (const char *)payload);
}

void bc_sp_sink_file(uint32_t class_id, const char *class_name,
                     const void *payload, uint32_t len, void *user)
{
    (void)class_id;
    FILE *fp = (FILE *)user;
    if (!fp) return;
    fprintf(fp, "%s: %.*s\n", class_name, (int)len,
            (const char *)payload);
}

/* ---- Reset ----
 * Clears the class table and sink registry between test cases; production never
 * needs it. */

void bc_sp_reset_globals(void)
{
    memset(g_classes, 0, sizeof(g_classes));
    g_num_classes = 0;
    memset(g_sinks, 0, sizeof(g_sinks));
    g_num_sinks = 0;
}
