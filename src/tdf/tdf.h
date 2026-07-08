#ifndef BARRACUDA_TDF_H
#define BARRACUDA_TDF_H

#include "barracuda.h"
#include "bir.h"
#include "rv_buf.h"     /* rv_buf_t, for the CB-arc emitter */

/*
 * Tile DataFlow IR.
 *
 * Sits above BIR. A CUDA __global__ lowers to a td_mod_t, a graph of regions
 * communicating via channels. On AMD and NVIDIA the graph collapses to one
 * region and forwards straight to BIR isel; on Tensix it fans out, one region
 * per baby core, channels become L1 circular buffers, and arcs become inlined
 * NoC + CB ops. The vocabulary is mainframe on purpose: regions are CICS tasks
 * with their own working area, channels are the TPF circular block lists the
 * regions hand tiles through, and arcs are the ENQ/DEQ pairs the hardware
 * enforces with semaphores. None of this is novel; the novelty is the chip, not
 * the bookkeeping pattern that ran airline reservations on a 360 in 1968. No
 * malloc, the module is sized at sema time and lives in one arena.
 */

/* ---- Limits ---- */

#define TD_MAX_RGNS     64      /* regions per module                   */
#define TD_MAX_CHANS    128     /* channels per module                  */
#define TD_MAX_ARCS     1024    /* arcs per module                      */
#define TD_NARG         16      /* runtime args, TPF CBRW count is 16   */
#define TD_BAD_ID       0xFFFFu /* returned by builders on overflow     */

/* TDF-layer error code, parallel to BC_ERR_TRITON in triton.h. */
#define BC_ERR_TDF      -11

/* ---- Wormhole L1 placement constants ----
 *
 * Source citations sit next to each value because the gap between spec and
 * reverse-engineered-from-tt-metal is wide and load-bearing, and future-us needs
 * to know which number is gospel and which is a sensible guess. Wormhole L1 is
 * 1,464 KiB usable from 0x00000000 to 0x0016FFFF, organised as 16 banks of
 * 91.5 KiB (WormholeB0/TensixTile/L1.md). The 32 KiB code reservation is OUR
 * call, since the docs fix no maximum kernel size and only say baby cores fetch
 * instructions from L1; it is conservative, matches typical Metalium kernels, and
 * we can adjust once real RV32IM emission lets us measure. NoC reads/writes to L1
 * require C16 congruence (16-byte alignment) or tile-data buffers start off a
 * boundary and get silently corrupted (WormholeB0/NoC/Alignment.md). The 8-byte
 * FIFO header is a software convention for an L1-resident read/write pair when we
 * run out of the eight hardware semaphores (indices 0..7); the hardware path
 * lives in 0xFFE80000 MMIO and costs no L1, so the L1 FIFO path serves only
 * channels nine and onward (WormholeB0/.../SyncUnit.md).
 */
#define TD_L1_BASE        0x00000000u
#define TD_L1_END         0x00170000u    /* one past last usable byte */
#define TD_L1_CODE_RSV    0x00008000u    /* 32 KiB for code + stack  */
/*
 * Runtime args slab sits between code and CBs. The kernel reads its CUDA
 * coordinate intrinsics (threadIdx etc.) and its scalar parameters from here,
 * populated by the host launcher before dispatch, with the layout in
 * tensix/rt_args.h. We reserve 0x100 (256 B) for the 112-byte struct plus growth
 * room, and the base keeps a 12-bit-zero low half so the isel can materialise it
 * with a single LUI (0x00008000 = 0x8 << 12).
 */
#define TD_L1_RTARG_BASE  TD_L1_CODE_RSV
#define TD_L1_RTARG_SIZE  0x00000100u
#define TD_L1_CB_BASE     (TD_L1_RTARG_BASE + TD_L1_RTARG_SIZE)
#define TD_L1_ALIGN       16u            /* C16 NoC alignment        */
#define TD_FIFO_BYTES     8u             /* head + tail, L1 fallback */
#define TD_HW_SEMS        8u             /* hardware semaphore count */

/* ---- Wormhole NoC constants ----
 *
 * From WormholeB0/NoC/MemoryMap.md (64-bit unicast address layout) and
 * ./Alignment.md (8192-byte max transfer). A unicast address packs the 36-bit
 * byte-granular local memory address in bits [35:0], the 6-bit destination X in
 * [41:36], the 6-bit destination Y in [47:42], and leaves [63:48] reserved and
 * ignored by the hardware. Two physical NoCs run in opposite directions: NoC 0
 * favours reads because it flows top-left to bottom-right, matching the
 * Tensix-to-DRAM direction for column-16 banks, and NoC 1 favours writes for the
 * same reason in reverse (RoutingPaths.md plus the DRAMTile/README.md congestion
 * table). Static VC allocation matters for large writes but is below this layer's
 * concern; the RV32IM emitter sets it when materialising the NoC instruction.
 */
#define TD_NOC0           0u
#define TD_NOC1           1u
#define TD_NOC_MAX_XFER   8192u
#define TD_NOC_LOCAL_BITS 36u
#define TD_NOC_LOCAL_MASK 0x0000000FFFFFFFFFull
#define TD_NOC_COORD_MASK 0x3Fu

/* ---- Region role ---- */

/*
 * The three classical Metalium kernel roles plus a slot for the bare
 * single-region case used by AMD and NVIDIA, and the cleanup region
 * that the EXITC sweep is bound to. More can be added later when the
 * fission scheme grows past three programs.
 */
typedef enum {
    TD_RG_NONE,
    TD_RG_RDR,      /* reader,   DM0 baby core, NoC -> L1 */
    TD_RG_CMP,      /* compute,  T0/T1/T2 cores, FPU/SFPU */
    TD_RG_WRT,      /* writer,   DM1 baby core, L1 -> NoC */
    TD_RG_SOLO,     /* single region for non-Tensix backends */
    TD_RG_COUNT
} td_role_t;

/* ---- Arc kind ---- */

/*
 * Six kinds. Four for the producer/consumer CB protocol, two for the
 * NoC boundary. RSV waits until the producer has free slots, PUSH
 * commits tiles into the channel, WAIT blocks the consumer until tiles
 * are ready, POP releases slots back to the producer. RD and WR cross
 * the L1/DRAM boundary; both are async by default and the backend
 * inserts the matching barrier when the next dependent arc fires.
 */
typedef enum {
    TD_AR_RSV,      /* cb_reserve_back  */
    TD_AR_PUSH,     /* cb_push_back     */
    TD_AR_WAIT,     /* cb_wait_front    */
    TD_AR_POP,      /* cb_pop_front     */
    TD_AR_RD,       /* noc_async_read   */
    TD_AR_WR,       /* noc_async_write  */
    TD_AR_COUNT
} td_arc_kind_t;

/* ---- Channel tag ---- */

/*
 * A channel carries tiles of one shape and one dtype. The tag packs
 * both, Burroughs-descriptor style, so a misaligned producer/consumer
 * trips an assertion at module-build time rather than a silent
 * misformat at runtime. Tile shape is rows x cols. Dtype matches
 * bir_type_t numerics. Layout is the placement strategy: interleaved
 * across DRAM banks, sharded by some axis, or pinned to L1.
 */
typedef enum {
    TD_LAY_INTRL,   /* interleaved across DRAM banks    */
    TD_LAY_SHARD,   /* sharded to specific cores        */
    TD_LAY_L1,      /* pinned to a single L1            */
    TD_LAY_COUNT
} td_layout_t;

typedef struct {
    uint16_t rows;
    uint16_t cols;
    uint8_t  dtype;     /* one of BIR_TYPE_* numerics       */
    uint8_t  layout;    /* td_layout_t                      */
    uint16_t _pad;
} td_tag_t;     /* 8 bytes */

/* ---- Region ---- */

/*
 * A region is one transactional program. It has a role, a binding to
 * a baby core that placement may fill in later, a TWA giving how much
 * L1 it wants for its own working storage, and a runtime args block
 * for parameters the host hands in at dispatch. The body is a BIR
 * module containing exactly one function which is the region's entry.
 *
 * The 16-slot args array is sized to match TPF's ECB which has 16
 * CBRWs at data levels D0 through DF. That number turns up in
 * Metalium too, in the runtime-args API, and is not a coincidence.
 */
typedef struct {
    uint16_t     id;
    uint8_t      role;          /* td_role_t                        */
    uint8_t      core;          /* 0=B, 1=T0, 2=T1, 3=T2, 4=NC      */
    uint16_t     x, y;          /* NoC coord, filled in by placer   */
    uint32_t     twa_sz;        /* L1 working area, bytes           */
    uint32_t     args[TD_NARG]; /* runtime args, TPF CBRW-shaped    */
    bir_module_t *body;         /* region's BIR, one func inside    */
} td_rgn_t;

/* ---- Channel ---- */

/*
 * A channel is a circular buffer between two regions. Depth is the
 * number of tiles that can sit in the ring at once. The l1_off field
 * is filled by placement once the producer core is fixed.
 */
typedef struct {
    uint16_t id;
    uint16_t prod;          /* producer region id               */
    uint16_t cons;          /* consumer region id               */
    uint16_t depth;         /* CB slot count                    */
    td_tag_t tag;           /* tile shape + dtype + layout      */
    uint32_t l1_off;        /* L1 base offset, placed later     */
} td_chan_t;

/* ---- Arc ---- */

/*
 * An arc is a single CB or NoC op that the region issues at some
 * point in its body. The bir_inst field points at the BIR instruction
 * the arc was lowered from, so the codegen can drop the synchronisation
 * call into the right slot in the region's instruction stream.
 */
typedef struct {
    uint8_t  kind;          /* td_arc_kind_t                    */
    uint8_t  noc_id;        /* 0 or 1, NoC choice for RD/WR;    */
                            /* ignored for CB arcs              */
    uint16_t rgn;           /* region id this arc belongs to    */
    uint16_t chan;          /* channel id                       */
    uint16_t cnt;           /* tile credit count                */
    uint32_t bir_inst;      /* BIR instruction index, anchor    */
    uint32_t length;        /* bytes per NoC fire; 0 for CB arcs*/
} td_arc_t;

/* ---- Module ---- */

/*
 * The whole graph for one source-level kernel. Fixed-size arrays
 * because the limits hold and the sizes are small. Anything that
 * needs more than 64 regions is doing something the IR was not
 * designed for and should fail loudly rather than quietly grow.
 */
typedef struct {
    td_rgn_t  rgns[TD_MAX_RGNS];      uint16_t nrgn;
    td_chan_t chans[TD_MAX_CHANS];    uint16_t ncha;
    td_arc_t  arcs[TD_MAX_ARCS];      uint16_t narc;
    uint8_t   target;                 /* see td_target_t          */
    uint8_t   _pad[3];
} td_mod_t;

typedef enum {
    TD_TGT_AMD,
    TD_TGT_NVIDIA,
    TD_TGT_TENSIX,
    TD_TGT_COUNT
} td_target_t;

/* ---- Builder API ---- */

/*
 * The frontend calls these to assemble a module. They return ids,
 * not pointers, because the storage is the module arena and pointer
 * stability across allocations is not promised. Lookups go through
 * td_rgn(), td_chan(), td_arc().
 */

void      td_init(td_mod_t *M, int target);

uint16_t  td_mkrgn(td_mod_t *M, int role);
uint16_t  td_link(td_mod_t *M, uint16_t prod, uint16_t cons,
                  td_tag_t tag, uint16_t depth);
uint16_t  td_mkarc(td_mod_t *M, uint16_t rgn, int kind,
                   uint16_t chan, uint16_t cnt, uint32_t anchor);

td_rgn_t  *td_rgn(td_mod_t *M, uint16_t id);
td_chan_t *td_chan(td_mod_t *M, uint16_t id);
td_arc_t  *td_arc(td_mod_t *M, uint16_t id);

/* ---- Lowering API ---- */

/*
 * Target-aware. For AMD and NVIDIA, td_lower collapses the module to
 * its solo region and hands the body straight to the existing isel.
 * For Tensix, td_lower fans the module out into N RV32IM-bound BIR
 * modules, materialises L1 layouts for channels, and walks arcs into
 * the appropriate region bodies. The output is a vector of BIR
 * modules plus a manifest that the host loader uses to wire up CBs.
 */

/*
 * Lowering output. Holds one BIR module pointer per region that
 * came out the far side of td_lower. On the AMD/NVIDIA path nmods
 * is one and the pointer aliases the SOLO region's body. On the
 * Tensix path nmods will equal the number of baby cores the kernel
 * was fissioned across, with each pointer owning its own module
 * out of an arena td_lower allocates. The manifest fields below
 * accrete as the Tensix backend grows.
 */
typedef struct {
    bir_module_t *mods[TD_MAX_RGNS];
    uint8_t       owns[TD_MAX_RGNS];    /* 1 = td_lower owns, 0 = alias */
    uint16_t      nmods;
} td_lout_t;

int       td_lower(td_mod_t *M, td_lout_t *out);

/*
 * Convenience: wrap an existing BIR module in a one-region SOLO
 * TDF module. This is what the C99 and Triton frontends will call
 * once they have a BIR but before they hand anything to a backend;
 * for AMD and NVIDIA the resulting module lowers straight back to
 * the same BIR, and for Tensix the SOLO region is the first stop
 * on the eventual fission path.
 */
int       td_build_solo_from_bir(td_mod_t *M, int target,
                                 bir_module_t *body);

/*
 * Tensix fission. Takes a SOLO module whose body is a CUDA __global__ BIR and
 * rewrites the TDF graph in place into the canonical RDR/CMP/WRT three-region
 * shape: loaded pointer parameters become channels from the reader into compute,
 * stored ones become channels from compute out to the writer. The BIR body stays
 * attached to the compute region for now, and the eventual BIR-level fission that
 * splits loads off the front and stores off the back into their own bodies lands
 * in a follow-up sitting. Returns BC_ERR_TDF if the input is not a Tensix SOLO
 * module with a body, has no recognisable kernel function, or contains constructs
 * that defeat the analysis (currently device function calls).
 */
int       td_fission_tensix(td_mod_t *M);

/*
 * L1 placement. Walks every channel, packs each into the CB region of L1 from
 * TD_L1_CB_BASE and sets the channel's l1_off field, first-fit in declaration
 * order; a future pass can sort by size or core once placement has more than one
 * core to think about. Returns BC_ERR_TDF if the channels do not fit after the
 * code reservation, a ceiling of roughly 1,432 KiB (about 350 fp32 32x32 tiles in
 * flight), so the error case mostly guards against runaway depth values rather
 * than a real workload limit. Also exposes a helper for the byte size of one tile
 * given a channel tag, which the L1 emitter and the NoC orchestrator both want.
 */
uint32_t  td_tile_bytes(td_tag_t tag);
int       td_place_l1(td_mod_t *M);

/*
 * NoC orchestration. Walks every RD and WR arc, fills in noc_id (which of the two
 * physical NoCs to use) and length (bytes per fire), and refuses any transfer that
 * busts the TD_NOC_MAX_XFER 8 KiB single-request limit from Alignment.md; bigger
 * transfers must be split by a future pass, and until then we bail rather than
 * silently truncate. The encoder is exposed too so the eventual RV32IM emitter can
 * reach for it directly rather than reinventing the bit-packing.
 */
uint64_t  td_noc_addr(uint8_t x, uint8_t y, uint64_t local);
int       td_noc_orchestrate(td_mod_t *M);
int       td_emit_cb_arc(td_mod_t *M, const td_arc_t *a, rv_buf_t *code);
int       td_emit_dma_loop(rv_buf_t *code, int is_write,
                           uint32_t dram_arg_slot, uint32_t ntiles_arg_slot,
                           uint32_t l1_buf, uint32_t depth,
                           uint32_t dram_mid, uint32_t l1_mid,
                           uint32_t tile_bytes,
                           uint32_t recv_addr, uint32_t free_addr);

/* ---- Inspection ---- */

void      td_dump(const td_mod_t *M, FILE *fp);
const char *td_role_name(int role);
const char *td_arc_kind_name(int kind);

#endif /* BARRACUDA_TDF_H */
