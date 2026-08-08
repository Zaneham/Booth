/* MIT licensed, not Apache like the rest of this tree.
 * Vendored from https://github.com/Zaneham/kauri
 */

/*
 * kauri.h: Memory safety for C99
 *
 * Single-header library (stb-style). In ONE .c file, before #include:
 *   #define KAURI_IMPL
 *   #include "kauri.h"
 *
 * Named for the kauri tree: grows slowly, lives millennia, doesn't fall over.
 * Unlike your heap allocator.
 *
 * Thread safety: none. One arena per thread, like one pen per astronaut.
 *               If you share arenas across threads you deserve what happens.
 *
 * (c) 2026. Zane's school of wonderful curiosities and maybe a compiler.
 */

#ifndef KAURI_H
#define KAURI_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- Config ---- */

#ifndef KAURI_DEBUG
#define KAURI_DEBUG 0
#endif

#ifndef KAURI_ABORT
#define KAURI_ABORT 0
#endif

/* ---- Error Codes + ka_res_t ----
 * Negative = bad day. Zero = everything's fine.
 * Much like altitude readings in aviation. */

#define KA_OK     0
#define KA_OOB   -1   /* out of bounds. the array has edges, respect them */
#define KA_OOM   -2   /* out of memory. the arena is not infinite, sorry  */
#define KA_OVFL  -3   /* overflow. numbers have limits, who knew           */
#define KA_INVAL -4   /* invalid argument. rubbish in, rubbish out         */

typedef struct {
    int         code;
    const char *msg;   /* static string, never allocated. we're not animals */
#if KAURI_DEBUG
    const char *file;
    int         line;
#endif
} ka_res_t;

/* Build a result. Debug mode tattoos the source location onto it,
 * like a black box recorder for your allocator. */
#if KAURI_DEBUG
#define KA_RES(c, m) ((ka_res_t){ (c), (m), __FILE__, __LINE__ })
#else
#define KA_RES(c, m) ((ka_res_t){ (c), (m) })
#endif

#define KA_RES_OK KA_RES(KA_OK, "ok")

/* Optional early return on error. Some people prefer to stare at the
 * error code and contemplate their life choices. */
#define KA_TRY(expr) do {          \
    ka_res_t _r = (expr);         \
    if (_r.code != KA_OK) return _r; \
} while (0)

/* ---- Utility Macros ----
 * The safety nets. Deploy them or enjoy the splat. */

/* Alignment: __alignof__ if the compiler is civilised, fallback for C99.
 * The fallback uses the struct-padding trick. Not pretty, but correct,
 * which is more than can be said for most C alignment code. */
#ifdef __GNUC__
#define KA_ALIGN(T) __alignof__(T)
#else
#define KA_ALIGN(T) offsetof(struct { char _c; T _t; }, _t)
#endif

/* Bounds check. In debug+abort mode this stops the program dead,
 * which is better than silently scribbling over adjacent memory
 * like a toddler with a permanent marker.
 * Returns 1 if out of bounds, 0 if fine. */
#if KAURI_DEBUG && KAURI_ABORT
#define KA_CHK(i, max) ka__chk((uint32_t)(i), (uint32_t)(max), __FILE__, __LINE__)
#elif KAURI_DEBUG
#define KA_CHK(i, max) ((uint32_t)(i) >= (uint32_t)(max)                       \
    ? (ka__oob(__FILE__, __LINE__, (uint32_t)(i), (uint32_t)(max)), 1) : 0)
#else
#define KA_CHK(i, max) ((uint32_t)(i) >= (uint32_t)(max))
#endif

/* Pool allocate: index 0 is the sentinel (the "nobody's home" value).
 * If the pool is full, returns 0. If not, hands you the next slot.
 * Evaluates cnt twice. Fine for simple lvalues, catastrophic for
 * expressions with side effects. Don't be clever. */
#define KA_PNEW(cnt, max) ((uint32_t)(cnt) >= (uint32_t)(max) ? 0u : (uint32_t)(cnt)++)

/* Guard counter for bounded loops. Declares a uint32_t that counts down,
 * preventing infinite loops like a responsible adult.
 * Usage: KA_GUARD(g, 1000); while (cond && g--) { ... } */
#define KA_GUARD(g, max) uint32_t g = (uint32_t)(max)

/* Overflow-safe multiply. Returns 0 if the product wraps past 32 bits,
 * which ka_alloc rejects. Turning "allocated 12 bytes for 4 billion
 * structs" into a polite refusal. */
static inline uint32_t
ka__smul(uint32_t a, uint32_t b)
{
    uint64_t r = (uint64_t)a * (uint64_t)b;
    return r > 0xFFFFFFFFu ? 0u : (uint32_t)r;
}

/* Typed arena alloc. Saves you from writing the cast and sizeof.
 * Returns NULL if the arena says no. */
#define KA_NEW(A, T)       ((T *)ka_alloc((A), (uint32_t)sizeof(T), (uint32_t)KA_ALIGN(T)))
#define KA_NEWN(A, T, n)   ((T *)ka_alloc((A), ka__smul((uint32_t)sizeof(T), (uint32_t)(n)), (uint32_t)KA_ALIGN(T)))

/* ---- Arena ----
 * A bump allocator. Goes forward, never backward (except on reset).
 * Like time, but useful.
 *
 * The first block is inline: no malloc for the common case where
 * you hand it a stack buffer. Chaining is opt-in because malloc in
 * a safety library is like a fire extinguisher that's also flammable. */

#define KA_CHAIN  0x01u  /* allow chaining additional blocks via malloc */

/* Internal flag: head block was heap-allocated (ka_init with buf=NULL) */
#define KA_F_HEAP 0x80000000u

#if KAURI_DEBUG
#define KA__DEAD 0xDE          /* poison byte. if you see this in production, repent */
#define KA__CVAL 0xDEADCA75u   /* canary sentinel. "dead cats", because coal mines are passé */
#define KA__CSZ  4             /* canary size in bytes */
#define KA__LMAX 256           /* max tracked allocs. if you need more, rethink your life */

typedef struct { uint8_t *ptr; uint32_t size; } ka__log_t;
#endif

typedef struct ka_blk_t {
    struct ka_blk_t *next;
    uint8_t         *base;  /* data region */
    uint32_t         cap;
    uint32_t         pos;
} ka_blk_t;

/* Snapshot for mark/rewind. Like a save point in a video game,
 * except the stakes are higher. */
typedef struct {
    ka_blk_t *blk;
    uint32_t  pos;
#if KAURI_DEBUG
    uint32_t  n_log;  /* saved canary log position */
#endif
} ka_mark_t;

typedef struct {
    ka_blk_t  head;     /* inline first block. no malloc needed */
    ka_blk_t *cur;      /* current active block */
    uint32_t  flags;
    uint32_t  max_blk;  /* chain limit, default 64 */
    uint32_t  n_blk;    /* blocks in chain (including head) */
#if KAURI_DEBUG
    uint32_t  n_alloc;  /* total allocations. for the curious */
    uint32_t  peak;     /* high-water n_alloc across resets */
    uint32_t  n_log;    /* entries in canary log */
    ka__log_t log[KA__LMAX]; /* the post-mortem ledger. 3KB well spent. */
#endif
} ka_arena_t;

/* Initialise arena. buf=user buffer (stack or static), cap=size in bytes.
 * If buf is NULL, allocates cap bytes from the heap (requires free later).
 * flags: KA_CHAIN to allow overflow into malloc'd blocks. */
void       ka_init (ka_arena_t *A, void *buf, uint32_t cap, uint32_t flags);

/* Bump-allocate size bytes with given alignment (must be power of 2).
 * Returns NULL if no room (and chaining is off or chain limit reached). */
void      *ka_alloc(ka_arena_t *A, uint32_t size, uint32_t align);

/* Reset arena: free chain blocks, rewind head to pos 0.
 * Does NOT free the head block itself. That's ka_free's job. */
void       ka_rst  (ka_arena_t *A);

/* Free everything. If the head was heap-backed, frees that too.
 * After this, the arena is a husk. Don't touch it. */
void       ka_free (ka_arena_t *A);

/* How many bytes are currently occupied / total capacity.
 * Walks the chain, bounded by max_blk. */
uint32_t   ka_used (const ka_arena_t *A);
uint32_t   ka_cap  (const ka_arena_t *A);

/* Save/restore point. Rewind frees any chain blocks allocated after
 * the mark. Does NOT zero the memory. Ghosts of old data remain.
 * (In debug mode it does, actually. 0xDE everywhere. Trust nothing.) */
ka_mark_t  ka_mark (ka_arena_t *A);
void       ka_rwind(ka_arena_t *A, ka_mark_t m);

/* Peak allocation count across resets. Debug only. Tells you how
 * big your arena actually needs to be, since guessing is for gamblers. */
uint32_t   ka_peak (ka_arena_t *A);

/* Allocate and copy. Like strdup but for arbitrary blobs.
 * Returns NULL on OOM. */
void      *ka_dup  (ka_arena_t *A, const void *src, uint32_t size, uint32_t align);

/* String duplicate into arena. Adds NUL terminator.
 * len=0 means "measure it for me" (strlen). If you pass len=0
 * on a non-NUL-terminated string, that's on you. Returns NULL on OOM. */
char      *ka_sdup (ka_arena_t *A, const char *s, uint32_t len);


/* ---- String Builder ----
 * A non-owning string buffer. You provide the backing memory
 * (stack, arena, whatever), it provides the safety.
 * Truncates gracefully. Always NUL-terminated, never overflows. */

typedef struct {
    char    *ptr;    /* NOT owned. caller manages lifetime */
    uint32_t len;
    uint32_t cap;
} ka_str_t;

/* Init string builder with user-supplied buffer.
 * cap must be >= 1 (need room for the NUL). */
int  ka_sinit(ka_str_t *S, char *buf, uint32_t cap);

/* Append slen bytes from src. Returns 0 on success, -1 if truncated. */
int  ka_scat (ka_str_t *S, const char *src, uint32_t slen);

/* Printf into the string. Returns 0 on success, -1 if truncated. */
#ifdef __GNUC__
__attribute__((format(printf, 2, 3)))
#endif
int  ka_sfmt (ka_str_t *S, const char *fmt, ...);

/* Append a single character. Returns 0 on success, -1 if truncated. */
int  ka_schr (ka_str_t *S, char c);

/* Clear the string to empty (len=0, ptr[0]='\0'). */
void ka_sclr (ka_str_t *S);

/* Compare two string builders lexicographically. Returns <0, 0, >0. */
int  ka_scmp (const ka_str_t *a, const ka_str_t *b);


/* ---- Debug Internals ---- */

void ka__oob(const char *file, int line, uint32_t idx, uint32_t max);

#if KAURI_DEBUG && KAURI_ABORT
static inline int ka__chk(uint32_t i, uint32_t max, const char *file, int line)
{
    if (i >= max) { ka__oob(file, line, i, max); }
    return i >= max;
}
#endif

#endif /* KAURI_H */


/* ================================================================
 *                      IMPLEMENTATION
 * ================================================================
 * Define KAURI_IMPL in exactly one .c file before including this.
 * Two definitions and the linker will express its displeasure. */

#ifdef KAURI_IMPL

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

/* ---- Debug ---- */

void
ka__oob(const char *file, int line, uint32_t idx, uint32_t max)
{
    /* The pilot's last words: "that index looked fine to me" */
    fprintf(stderr, "kauri: OOB %s:%d idx=%u max=%u\n",
            file, line, (unsigned)idx, (unsigned)max);
#if KAURI_ABORT
    abort();
#endif
}

/* ---- Arena ---- */

void
ka_init(ka_arena_t *A, void *buf, uint32_t cap, uint32_t flags)
{
    memset(A, 0, sizeof(*A));
    A->flags   = flags;
    A->max_blk = 64;
    A->n_blk   = 1;

    if (buf) {
        A->head.base = (uint8_t *)buf;
        A->head.cap  = cap;
    } else {
        /* Heap-backed: malloc the data region separately.
         * Because sometimes the stack just isn't big enough
         * for your ambitions. */
        uint8_t *p = (uint8_t *)malloc(cap);
        if (!p) {
            /* Well. This is embarrassing. A safety library that
             * can't allocate. At least we fail gracefully. */
            A->head.base = NULL;
            A->head.cap  = 0;
            return;
        }
        A->head.base = p;
        A->head.cap  = cap;
        A->flags    |= KA_F_HEAP;
    }
    A->head.pos  = 0;
    A->head.next = NULL;
    A->cur       = &A->head;
}

/* Align pos upward to `align` (must be power of 2). */
static uint32_t
ka__aup(uint32_t pos, uint32_t align)
{
    return (pos + align - 1u) & ~(align - 1u);
}

/* Try to allocate from a specific block. Returns NULL if no room. */
static void *
ka__blka(ka_blk_t *b, uint32_t size, uint32_t align)
{
    uint32_t apos, end;
    if (!b || !b->base) return NULL;
    apos = ka__aup(b->pos, align);
    end  = apos + size;
    if (end < apos) return NULL;  /* overflow. nice try */
    if (end > b->cap) return NULL;
    b->pos = end;
    return b->base + apos;
}

/* Grow the chain by one block. Returns the new block or NULL.
 * NOTE: assumes fresh blocks start at pos=0, so ka__blka won't
 * waste space re-aligning. If you ever add block headers, revisit. */
static ka_blk_t *
ka__grow(ka_arena_t *A, uint32_t need)
{
    ka_blk_t *nb;
    uint32_t  cap;

    if (A->n_blk >= A->max_blk) return NULL;

    /* New block is at least as big as what we need, or the head cap,
     * whichever is larger. No point adding a tiny block. */
    cap = A->head.cap;
    if (need > cap) cap = need;

    nb = (ka_blk_t *)malloc(sizeof(ka_blk_t) + cap);
    if (!nb) return NULL;

    nb->base = (uint8_t *)(nb + 1);
    nb->cap  = cap;
    nb->pos  = 0;
    nb->next = NULL;

    /* Append to chain after current block */
    A->cur->next = nb;
    A->cur       = nb;
    A->n_blk++;
    return nb;
}

#if KAURI_DEBUG
/* Log an allocation for canary tracking. The ledger has a fixed
 * size because unbounded logs are an oxymoron in safety code. */
static void
ka__alog(ka_arena_t *A, uint8_t *ptr, uint32_t size)
{
    uint32_t cval = KA__CVAL;
    memcpy(ptr + size, &cval, KA__CSZ);
    if (A->n_log < KA__LMAX) {
        A->log[A->n_log].ptr  = ptr;
        A->log[A->n_log].size = size;
        A->n_log++;
    } else if (A->n_log == KA__LMAX) {
        fprintf(stderr, "kauri: canary log full (%u). further allocs untracked\n",
                (unsigned)KA__LMAX);
        A->n_log++;  /* only warn once */
    }
}

/* Walk the canary log [from..to) and verify each sentinel.
 * If something scribbled past an allocation, we want to know
 * before it scribbles past something important. */
static void
ka__cchk(ka_arena_t *A, uint32_t from, uint32_t to)
{
    uint32_t i;
    for (i = from; i < to; i++) {
        uint32_t v;
        memcpy(&v, A->log[i].ptr + A->log[i].size, KA__CSZ);
        if (v != KA__CVAL) {
            fprintf(stderr, "kauri: CANARY CORRUPT ptr=%p size=%u\n",
                    (void *)A->log[i].ptr, (unsigned)A->log[i].size);
        }
    }
}
#endif

void *
ka_alloc(ka_arena_t *A, uint32_t size, uint32_t align)
{
    void *p;
    uint32_t rsz;
    if (!A || !size || !align) return NULL;

#if KAURI_DEBUG
    rsz = size + KA__CSZ;  /* room for the canary's perch */
#else
    rsz = size;
#endif

    /* Try current block first */
    p = ka__blka(A->cur, rsz, align);
    if (p) {
#if KAURI_DEBUG
        ka__alog(A, (uint8_t *)p, size);
        A->n_alloc++;
#endif
        return p;
    }

    /* No room. If chaining is allowed, grow. */
    if (A->flags & KA_CHAIN) {
        ka_blk_t *nb = ka__grow(A, ka__aup(rsz, align));
        if (nb) {
            p = ka__blka(nb, rsz, align);
#if KAURI_DEBUG
            if (p) {
                ka__alog(A, (uint8_t *)p, size);
                A->n_alloc++;
            }
#endif
            return p;
        }
    }

    return NULL;
}

void
ka_rst(ka_arena_t *A)
{
    ka_blk_t *b, *next;
    KA_GUARD(g, 64);

    if (!A) return;

#if KAURI_DEBUG
    /* Check the dead cats before clearing the ledger */
    ka__cchk(A, 0, A->n_log);
    if (A->n_alloc > A->peak) A->peak = A->n_alloc;

    /* Poison the head block. Stale data is the enemy */
    if (A->head.base && A->head.pos > 0)
        memset(A->head.base, KA__DEAD, A->head.pos);
#endif

    /* Free chain blocks (everything after head) */
    b = A->head.next;
    while (b && g--) {
        next = b->next;
#if KAURI_DEBUG
        if (b->base && b->pos > 0) memset(b->base, KA__DEAD, b->pos);
#endif
        free(b);
        b = next;
    }
    A->head.next = NULL;
    A->head.pos  = 0;
    A->cur       = &A->head;
    A->n_blk     = 1;
#if KAURI_DEBUG
    A->n_alloc   = 0;
    A->n_log     = 0;
#endif
}

void
ka_free(ka_arena_t *A)
{
    if (!A) return;
    ka_rst(A);

    if (A->flags & KA_F_HEAP) {
#if KAURI_DEBUG
        if (A->head.base && A->head.cap > 0)
            memset(A->head.base, KA__DEAD, A->head.cap);
#endif
        free(A->head.base);
    }
    memset(A, 0, sizeof(*A));
}

uint32_t
ka_used(const ka_arena_t *A)
{
    const ka_blk_t *b;
    uint32_t  total = 0;
    KA_GUARD(g, 65);

    if (!A) return 0;
    b = &A->head;
    while (b && g--) {
        total += b->pos;
        b = b->next;
    }
    return total;
}

uint32_t
ka_cap(const ka_arena_t *A)
{
    const ka_blk_t *b;
    uint32_t  total = 0;
    KA_GUARD(g, 65);

    if (!A) return 0;
    b = &A->head;
    while (b && g--) {
        total += b->cap;
        b = b->next;
    }
    return total;
}

uint32_t
ka_peak(ka_arena_t *A)
{
#if KAURI_DEBUG
    if (!A) return 0;
    return A->peak;
#else
    (void)A;
    return 0;
#endif
}

ka_mark_t
ka_mark(ka_arena_t *A)
{
    ka_mark_t m;
    m.blk = A->cur;
    m.pos = A->cur->pos;
#if KAURI_DEBUG
    m.n_log = A->n_log;
#endif
    return m;
}

void
ka_rwind(ka_arena_t *A, ka_mark_t m)
{
    ka_blk_t *b, *next;
    KA_GUARD(g, 64);

    if (!A) return;

#if KAURI_DEBUG
    /* Check canaries for allocations since the mark */
    ka__cchk(A, m.n_log, A->n_log);
    A->n_log = m.n_log;

    /* Poison the rewound region of the marked block */
    if (m.blk->base && m.blk->pos > m.pos)
        memset(m.blk->base + m.pos, KA__DEAD, m.blk->pos - m.pos);
#endif

    /* Free blocks after the marked block */
    b = m.blk->next;
    while (b && g--) {
        next = b->next;
        A->n_blk--;
#if KAURI_DEBUG
        if (b->base && b->pos > 0) memset(b->base, KA__DEAD, b->pos);
#endif
        free(b);
        b = next;
    }
    m.blk->next = NULL;
    m.blk->pos  = m.pos;
    A->cur       = m.blk;
}

void *
ka_dup(ka_arena_t *A, const void *src, uint32_t size, uint32_t align)
{
    void *p;
    if (!src || !size) return NULL;
    p = ka_alloc(A, size, align);
    if (p) memcpy(p, src, size);
    return p;
}

char *
ka_sdup(ka_arena_t *A, const char *s, uint32_t len)
{
    char *p;
    if (!s) return NULL;
    if (len == 0) len = (uint32_t)strlen(s);
    p = (char *)ka_alloc(A, len + 1, 1);
    if (!p) return NULL;
    memcpy(p, s, len);
    p[len] = '\0';
    return p;
}

/* ---- String Builder ---- */

int
ka_sinit(ka_str_t *S, char *buf, uint32_t cap)
{
    if (!S || !buf || cap < 1) return -1;
    S->ptr    = buf;
    S->cap    = cap;
    S->len    = 0;
    S->ptr[0] = '\0';
    return 0;
}

int
ka_scat(ka_str_t *S, const char *src, uint32_t slen)
{
    uint32_t avail, cpy;
    int      trunc = 0;

    if (!S || !src) return -1;
    avail = S->cap - S->len - 1;
    cpy   = slen;
    if (cpy > avail) {
        cpy   = avail;
        trunc = -1;
    }
    if (cpy > 0) {
        memcpy(S->ptr + S->len, src, cpy);
        S->len += cpy;
    }
    S->ptr[S->len] = '\0';
    return trunc;
}

int
ka_sfmt(ka_str_t *S, const char *fmt, ...)
{
    va_list ap;
    int     n;
    uint32_t avail;

    if (!S || !fmt) return -1;
    avail = S->cap - S->len;
    if (avail == 0) return -1;

    va_start(ap, fmt);
    n = vsnprintf(S->ptr + S->len, avail, fmt, ap);
    va_end(ap);

    if (n < 0) return -1;
    if ((uint32_t)n >= avail) {
        /* Truncated. vsnprintf already NUL-terminated at the boundary. */
        S->len = S->cap - 1;
        return -1;
    }
    S->len += (uint32_t)n;
    return 0;
}

int
ka_schr(ka_str_t *S, char c)
{
    if (!S) return -1;
    if (S->len + 1 >= S->cap) return -1;
    S->ptr[S->len++] = c;
    S->ptr[S->len]   = '\0';
    return 0;
}

void
ka_sclr(ka_str_t *S)
{
    if (!S || !S->ptr) return;
    S->len    = 0;
    S->ptr[0] = '\0';
}

int
ka_scmp(const ka_str_t *a, const ka_str_t *b)
{
    uint32_t mlen;
    int      r;

    if (!a || !b) return 0;
    mlen = a->len < b->len ? a->len : b->len;
    r    = memcmp(a->ptr, b->ptr, mlen);
    if (r != 0) return r;
    if (a->len < b->len) return -1;
    if (a->len > b->len) return  1;
    return 0;
}

#endif /* KAURI_IMPL */
