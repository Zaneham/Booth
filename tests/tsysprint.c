/* tsysprint.c -- tests for the SYSPRINT runtime. */

#include "tharns.h"
#include "sysprint.h"
#include <string.h>

static uint8_t buf_storage[8192];

/* ---- Capture Sink ----
 * A sink that copies the most recent record into a static buffer
 * so the test can inspect what was dispatched. */

static char  cap_class[64];
static char  cap_payload[256];
static uint32_t cap_len;
static int   cap_count;

static void cap_sink(uint32_t class_id, const char *class_name,
                     const void *payload, uint32_t len, void *user)
{
    (void)class_id; (void)user;
    cap_count++;
    snprintf(cap_class, sizeof(cap_class), "%s", class_name ? class_name : "?");
    uint32_t n = len < sizeof(cap_payload) - 1 ? len : sizeof(cap_payload) - 1;
    if (payload && n) memcpy(cap_payload, payload, n);
    cap_payload[n] = '\0';
    cap_len = len;
}

static void cap_reset(void)
{
    cap_class[0] = '\0';
    cap_payload[0] = '\0';
    cap_len = 0;
    cap_count = 0;
}

/* Per-class capture: separate buffers so multi-sink tests can
 * assert each sink got exactly what it expected. */

static int  err_count, trace_count;
static char err_last[256], trace_last[256];

static void err_sink(uint32_t cid, const char *cname,
                     const void *p, uint32_t n, void *u)
{
    (void)cid; (void)cname; (void)u;
    err_count++;
    uint32_t k = n < sizeof(err_last) - 1 ? n : sizeof(err_last) - 1;
    memcpy(err_last, p, k);
    err_last[k] = '\0';
}

static void trace_sink(uint32_t cid, const char *cname,
                       const void *p, uint32_t n, void *u)
{
    (void)cid; (void)cname; (void)u;
    trace_count++;
    uint32_t k = n < sizeof(trace_last) - 1 ? n : sizeof(trace_last) - 1;
    memcpy(trace_last, p, k);
    trace_last[k] = '\0';
}

/* ---- Tests ---- */

static void sp_intern_returns_same_id(void)
{
    bc_sp_reset_globals();
    uint32_t a = bc_sp_intern("STEP1.TRACE");
    uint32_t b = bc_sp_intern("STEP1.TRACE");
    uint32_t c = bc_sp_intern("STEP1.ERROR");
    CHNE(a, BC_SP_CLASS_NONE);
    CHEQ(a, b);
    CHNE(a, c);
    PASS();
}
TH_REG("sysprint", sp_intern_returns_same_id)

static void sp_class_name_roundtrip(void)
{
    bc_sp_reset_globals();
    uint32_t id = bc_sp_intern("STEP2.ERROR");
    const char *name = bc_sp_class_name(id);
    CHECK(name != NULL);
    CHSTR(name, "STEP2.ERROR");
    PASS();
}
TH_REG("sysprint", sp_class_name_roundtrip)

static void sp_intern_rejects_empty_or_long(void)
{
    bc_sp_reset_globals();
    CHEQ(bc_sp_intern(""), BC_SP_CLASS_NONE);
    char too_long[BC_SP_MAX_CLASS_NAME + 4];
    memset(too_long, 'A', sizeof(too_long));
    too_long[sizeof(too_long) - 1] = '\0';
    CHEQ(bc_sp_intern(too_long), BC_SP_CLASS_NONE);
    PASS();
}
TH_REG("sysprint", sp_intern_rejects_empty_or_long)

static void sp_emit_drain_roundtrip(void)
{
    bc_sp_reset_globals();
    cap_reset();
    bc_sp_buf_t buf;
    bc_sp_buf_init(&buf, buf_storage, sizeof(buf_storage));
    uint32_t cid = bc_sp_intern("STEP1.TRACE");
    bc_sp_register_sink("*", cap_sink, NULL);

    const char *msg = "hello from the kernel";
    bc_sp_emit(&buf, cid, msg, (uint32_t)strlen(msg));
    bc_sp_drain(&buf);

    CHEQ(cap_count, 1);
    CHSTR(cap_class, "STEP1.TRACE");
    CHSTR(cap_payload, "hello from the kernel");
    PASS();
}
TH_REG("sysprint", sp_emit_drain_roundtrip)

static void sp_emitf_formats(void)
{
    bc_sp_reset_globals();
    cap_reset();
    bc_sp_buf_t buf;
    bc_sp_buf_init(&buf, buf_storage, sizeof(buf_storage));
    uint32_t cid = bc_sp_intern("CALC");
    bc_sp_register_sink("*", cap_sink, NULL);

    bc_sp_emitf(&buf, cid, "tid=%d val=%.2f", 7, 3.14);
    bc_sp_drain(&buf);

    CHEQ(cap_count, 1);
    CHSTR(cap_payload, "tid=7 val=3.14");
    PASS();
}
TH_REG("sysprint", sp_emitf_formats)

static void sp_prefix_pattern_routes(void)
{
    /* The Sysprint People's Front and the People's Front of
     * Sysprint shall both match STEP1.*, which is to say neither
     * shall be slighted. */
    bc_sp_reset_globals();
    err_count = 0; trace_count = 0;
    bc_sp_buf_t buf;
    bc_sp_buf_init(&buf, buf_storage, sizeof(buf_storage));
    uint32_t trace_id = bc_sp_intern("STEP1.TRACE");
    uint32_t error_id = bc_sp_intern("STEP1.ERROR");
    uint32_t step2_id = bc_sp_intern("STEP2.TRACE");

    bc_sp_register_sink("STEP1.ERROR", err_sink, NULL);
    bc_sp_register_sink("STEP1.*", trace_sink, NULL);

    bc_sp_emit(&buf, trace_id, "t", 1);
    bc_sp_emit(&buf, error_id, "e", 1);
    bc_sp_emit(&buf, step2_id, "x", 1);
    bc_sp_drain(&buf);

    /* STEP1.ERROR matched the exact pattern first; STEP1.TRACE
     * fell through to the prefix; STEP2.TRACE matched nothing
     * and was silently discarded, as is its lot. */
    CHEQ(err_count, 1);
    CHEQ(trace_count, 1);
    CHSTR(err_last, "e");
    CHSTR(trace_last, "t");
    PASS();
}
TH_REG("sysprint", sp_prefix_pattern_routes)

static void sp_star_matches_everything(void)
{
    bc_sp_reset_globals();
    cap_reset();
    bc_sp_buf_t buf;
    bc_sp_buf_init(&buf, buf_storage, sizeof(buf_storage));
    uint32_t a = bc_sp_intern("WHATEVER");
    uint32_t b = bc_sp_intern("ANOTHER.THING");
    bc_sp_register_sink("*", cap_sink, NULL);

    bc_sp_emit(&buf, a, "1", 1);
    bc_sp_emit(&buf, b, "2", 1);
    bc_sp_drain(&buf);

    CHEQ(cap_count, 2);
    PASS();
}
TH_REG("sysprint", sp_star_matches_everything)

static void sp_overflow_drops(void)
{
    /* Telemetry that exceeds the buffer is silently lost. We do
     * not allocate more storage on the kernel's behalf. The
     * dropped counter is there for the curious. */
    bc_sp_reset_globals();
    uint8_t tiny[64];
    bc_sp_buf_t buf;
    bc_sp_buf_init(&buf, tiny, sizeof(tiny));
    uint32_t cid = bc_sp_intern("X");

    /* Each record is 8 (header) + aligned(payload). 32-byte payload
     * with 8-byte header is 40 bytes per record, so the second
     * record (80 bytes total) won't fit in 64. */
    char payload[32];
    memset(payload, 'a', sizeof(payload));
    bc_sp_emit(&buf, cid, payload, sizeof(payload));
    bc_sp_emit(&buf, cid, payload, sizeof(payload));

    CHEQ(buf.dropped, 1);
    PASS();
}
TH_REG("sysprint", sp_overflow_drops)

static void sp_unmatched_class_silently_dropped(void)
{
    bc_sp_reset_globals();
    err_count = 0; trace_count = 0;
    bc_sp_buf_t buf;
    bc_sp_buf_init(&buf, buf_storage, sizeof(buf_storage));
    uint32_t cid = bc_sp_intern("STEP3.UNHEARD");
    /* Sinks for STEP1 only, deliberately leaving STEP3 unrouted. */
    bc_sp_register_sink("STEP1.*", trace_sink, NULL);

    bc_sp_emit(&buf, cid, "hello?", 6);
    bc_sp_drain(&buf);

    CHEQ(trace_count, 0);
    PASS();
}
TH_REG("sysprint", sp_unmatched_class_silently_dropped)
