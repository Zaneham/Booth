#include "tdf.h"
#include <stdio.h>
#include <string.h>

/* ---- Chip geometry ---- */

/*
 * L1 sizes from tt-metal dev_mem_map.h, MEM_L1_SIZE and MEM_MAX_KERNEL_SIZE.
 * Wormhole's NCRISC executes from a 16 KiB IRAM and is the binding core, so its
 * text ceiling is that rather than MEM_MAX_KERNEL_SIZE. Blackhole has no IRAM
 * and gives every core the full budget.
 */
static const struct {
    const char *name;
    uint32_t    l1end;
    uint32_t    txtmax;
} k_chips[TD_CHIP_COUNT] = {
    { "wormhole",  1464u * 1024u,   16u * 1024u },
    { "blackhole", 1536u * 1024u, 1497u * 1024u },
};

static td_chip_t g_chip = TD_CHIP_BH;

static td_chip_t chipok(td_chip_t c)
{
    return c < TD_CHIP_COUNT ? c : TD_CHIP_BH;
}

uint32_t td_l1end(td_chip_t c)  { return k_chips[chipok(c)].l1end; }
uint32_t td_txtmax(td_chip_t c) { return k_chips[chipok(c)].txtmax; }
const char *td_cname(td_chip_t c) { return k_chips[chipok(c)].name; }

uint32_t td_shbase(td_chip_t c)
{
    return td_l1end(c) - TD_L1_SHARED_SIZE;
}

int td_pchip(const char *s, td_chip_t *out)
{
    if (!s || !out) return BC_ERR_TDF;
    for (uint32_t i = 0; i < TD_CHIP_COUNT; i++) {
        if (strcmp(s, k_chips[i].name) == 0) {
            *out = (td_chip_t)i;
            return BC_OK;
        }
    }
    return BC_ERR_TDF;
}

void td_setchip(td_chip_t c) { g_chip = chipok(c); }
td_chip_t td_chip(void)      { return g_chip; }

/*
 * Tile DataFlow builder, lookups, and dumper.
 *
 * Pure bookkeeping. The interesting work is in tdf_lower.c, which is
 * still being written; this file just gives the frontend a place to
 * hang the graph it builds out of a __global__. Limits are checked
 * against the arrays in tdf.h, on overflow the builders return
 * TD_BAD_ID and the caller is expected to bail.
 */

/* ---- Name tables ---- */

static const char *k_role[TD_RG_COUNT] = {
    "NONE", "RDR", "CMP", "WRT", "SOLO"
};

static const char *k_arc[TD_AR_COUNT] = {
    "RSV", "PUSH", "WAIT", "POP", "RD", "WR"
};

static const char *k_lay[TD_LAY_COUNT] = {
    "INTRL", "SHARD", "L1"
};

static const char *k_tgt[TD_TGT_COUNT] = {
    "AMD", "NVIDIA", "TENSIX"
};

static const char *k_core[5] = {
    "B", "T0", "T1", "T2", "NC"
};

const char *td_role_name(int role)
{
    if (role < 0 || role >= TD_RG_COUNT) return "?";
    return k_role[role];
}

const char *td_arc_kind_name(int kind)
{
    if (kind < 0 || kind >= TD_AR_COUNT) return "?";
    return k_arc[kind];
}

/* ---- Init ---- */

void td_init(td_mod_t *M, int target)
{
    memset(M, 0, sizeof(*M));
    M->target = (uint8_t)target;
}

/* ---- Frontend convenience ---- */

int td_build_solo_from_bir(td_mod_t *M, int target, bir_module_t *body)
{
    if (!body) {
        fprintf(stderr, "tdf: cannot wrap a NULL BIR body\n");
        return BC_ERR_TDF;
    }
    td_init(M, target);
    uint16_t r = td_mkrgn(M, TD_RG_SOLO);
    if (r == TD_BAD_ID) return BC_ERR_TDF;
    M->rgns[r].body = body;
    return BC_OK;
}

/* ---- Builders ----
 *
 * Each builder either returns a valid id in [0, max-1] or TD_BAD_ID
 * if the per-module limit is hit. Frontend callers should treat
 * TD_BAD_ID as a hard fail rather than carrying on with a wrong index,
 * because the module arenas are fixed and there is no growth path.
 */

uint16_t td_mkrgn(td_mod_t *M, int role)
{
    if (M->nrgn >= TD_MAX_RGNS) {
        fprintf(stderr, "tdf: region limit %d reached\n", TD_MAX_RGNS);
        return TD_BAD_ID;
    }
    uint16_t id = M->nrgn++;
    td_rgn_t *r = &M->rgns[id];
    memset(r, 0, sizeof(*r));
    r->id   = id;
    r->role = (uint8_t)role;
    return id;
}

uint16_t td_link(td_mod_t *M, uint16_t prod, uint16_t cons,
                 td_tag_t tag, uint16_t depth)
{
    if (M->ncha >= TD_MAX_CHANS) {
        fprintf(stderr, "tdf: channel limit %d reached\n", TD_MAX_CHANS);
        return TD_BAD_ID;
    }
    if (prod >= M->nrgn || cons >= M->nrgn) {
        fprintf(stderr, "tdf: link references unknown region\n");
        return TD_BAD_ID;
    }
    uint16_t id = M->ncha++;
    td_chan_t *c = &M->chans[id];
    c->id     = id;
    c->prod   = prod;
    c->cons   = cons;
    c->depth  = depth;
    c->tag    = tag;
    c->l1_off = 0;          /* filled by placement */
    return id;
}

uint16_t td_mkarc(td_mod_t *M, uint16_t rgn, int kind,
                  uint16_t chan, uint16_t cnt, uint32_t anchor)
{
    if (M->narc >= TD_MAX_ARCS) {
        fprintf(stderr, "tdf: arc limit %d reached\n", TD_MAX_ARCS);
        return TD_BAD_ID;
    }
    if (rgn >= M->nrgn) {
        fprintf(stderr, "tdf: arc on unknown region %u\n", rgn);
        return TD_BAD_ID;
    }
    /* CB arcs need a real channel; NoC arcs pass TD_BAD_ID for chan. */
    if (kind <= TD_AR_POP && chan >= M->ncha) {
        fprintf(stderr, "tdf: CB arc on unknown channel %u\n", chan);
        return TD_BAD_ID;
    }
    uint16_t id = M->narc++;
    td_arc_t *a = &M->arcs[id];
    a->kind     = (uint8_t)kind;
    a->noc_id   = 0;
    a->rgn      = rgn;
    a->chan     = chan;
    a->cnt      = cnt;
    a->bir_inst = anchor;
    a->length   = 0;
    return id;
}

/* ---- Lookups ---- */

td_rgn_t *td_rgn(td_mod_t *M, uint16_t id)
{
    if (id >= M->nrgn) return NULL;
    return &M->rgns[id];
}

td_chan_t *td_chan(td_mod_t *M, uint16_t id)
{
    if (id >= M->ncha) return NULL;
    return &M->chans[id];
}

td_arc_t *td_arc(td_mod_t *M, uint16_t id)
{
    if (id >= M->narc) return NULL;
    return &M->arcs[id];
}

/* ---- Dump ----
 *
 * Human-readable, one line per item, no JSON. Plain text holds up
 * better in a terminal at three in the morning than anything with
 * brackets in it.
 */

static const char *dtype_short(uint8_t dt)
{
    switch (dt) {
    case BIR_TYPE_INT:    return "i";
    case BIR_TYPE_FLOAT:  return "f";
    case BIR_TYPE_BFLOAT: return "bf";
    case BIR_TYPE_VOID:   return "v";
    default:              return "?";
    }
}

static void dump_tag(const td_tag_t *t, FILE *fp)
{
    fprintf(fp, "%ux%u %s%s",
            t->rows, t->cols, dtype_short(t->dtype),
            (t->layout < TD_LAY_COUNT) ? k_lay[t->layout] : "?");
}

void td_dump(const td_mod_t *M, FILE *fp)
{
    const char *tgt = (M->target < TD_TGT_COUNT) ? k_tgt[M->target] : "?";
    fprintf(fp, "TDF module (target=%s)\n", tgt);
    fprintf(fp, "  regions:  %u\n", M->nrgn);
    fprintf(fp, "  channels: %u\n", M->ncha);
    fprintf(fp, "  arcs:     %u\n", M->narc);

    int is_tt = (M->target == TD_TGT_TENSIX);
    for (uint16_t i = 0; i < M->nrgn; i++) {
        const td_rgn_t *r = &M->rgns[i];
        /* core, x, y, twa only mean something on Tensix. On AMD and NVIDIA the
         * region stands in for the whole kernel, so those fields are zero and
         * would only confuse the reader of the dump. */
        if (is_tt) {
            const char *core = (r->core < 5) ? k_core[r->core] : "-";
            fprintf(fp, "\n  region %u: %s core=%s twa=%u x=%u y=%u\n",
                    r->id, td_role_name(r->role), core,
                    r->twa_sz, r->x, r->y);
        } else {
            fprintf(fp, "\n  region %u: %s\n",
                    r->id, td_role_name(r->role));
        }
        int have_arg = 0;
        for (int k = 0; k < TD_NARG; k++) {
            if (r->args[k] != 0) { have_arg = 1; break; }
        }
        if (have_arg) {
            fprintf(fp, "    args:");
            for (int k = 0; k < TD_NARG; k++) {
                if (r->args[k] != 0)
                    fprintf(fp, " [%d]=0x%x", k, r->args[k]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "    body: %s\n", r->body ? "<bir attached>" : "<none>");
    }

    if (M->ncha > 0) fprintf(fp, "\n");
    for (uint16_t i = 0; i < M->ncha; i++) {
        const td_chan_t *c = &M->chans[i];
        fprintf(fp, "  chan %u: rgn%u -> rgn%u  tile ",
                c->id, c->prod, c->cons);
        dump_tag(&c->tag, fp);
        fprintf(fp, " depth=%u l1=0x%x\n", c->depth, c->l1_off);
    }

    if (M->narc > 0) fprintf(fp, "\n");
    for (uint16_t i = 0; i < M->narc; i++) {
        const td_arc_t *a = &M->arcs[i];
        fprintf(fp, "  arc %u: rgn%u %s", i, a->rgn, td_arc_kind_name(a->kind));
        if (a->chan != TD_BAD_ID)
            fprintf(fp, " chan%u", a->chan);
        if (a->kind == TD_AR_RD || a->kind == TD_AR_WR) {
            fprintf(fp, " noc%u len=%u", a->noc_id, a->length);
        }
        fprintf(fp, " cnt=%u @bir[%u]\n", a->cnt, a->bir_inst);
    }
}
