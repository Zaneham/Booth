#ifndef BARRACUDA_BIR_STRUCT_H
#define BARRACUDA_BIR_STRUCT_H

#include "bir.h"

/*
 * BIR control-flow structuriser.
 *
 * Down in the IR a function is a pile of basic blocks held together by
 * branches, which is a perfectly honest way to live if the thing reading
 * it is a chip that thinks in jumps. Metal is not that thing. Metal Shading
 * Language wants its control flow tucked in and structured, ifs inside ifs
 * and loops with proper walls, and it reacts to a bare goto the way a
 * Wellington landlord reacts to a request for insulation: technically it
 * heard the words, but no.
 *
 * So this pass is a translator between two ways of seeing the same river.
 * It reads the block soup and works out what shape it secretly was all
 * along, then writes that shape down as a structure tree, the kind of
 * nested if / loop / break / continue a person would have typed if a person
 * had been involved. Crucially it touches absolutely nothing. The blocks,
 * the branches, the instructions, all of it stays exactly where it was,
 * undisturbed. The machine-code backends (AMD, NVIDIA, Tensix, CPU, the
 * RISC-V mob) speak fluent jump and wouldn't thank anyone for structure
 * cluttering up the place, so they never call this and never notice it
 * exists. Only the backends that emit actual source and have to keep a
 * fussy frontend happy (Metal today, SPIR-V or WGSL or plain C down the
 * track) walk the tree, and they get to do it without ever inventing a
 * goto and slinking off feeling cheap.
 *
 * The method is relooper-shaped recovery on a reducible graph, following
 * the recipe in Ramsey, "Beyond Relooper: recursive translation of
 * unstructured control flow to structured control flow" (ICFP 2022), which
 * is the good modern write-up and worth the read before poking this. Work
 * out the dominators and a reverse-postorder numbering, and the rest falls
 * out: a loop header is anything a back edge points at, a merge is any
 * block two or more forward edges crash into, and a tidy recursive walk
 * rebuilds the nesting from there. The input arriving here was lowered from
 * structured CUDA, HIP, or Triton and then run through mem2reg, so it is
 * reducible nearly always, and this single path covers the whole real
 * workload with output clean enough to read.
 *
 * ROADMAP, or the bits left on the to-do list pinned to the fridge with a
 * Buzzy Bee magnet: an irreducible region (the only road to one from those
 * frontends is a hand-rolled goto, so it is rare as a dry Auckland summer)
 * makes the pass down tools and set ok = 0 rather than confidently emit
 * rubbish, and the caller is trusted to make a fuss. The fix, when a real
 * irreducible kernel finally turns up and forces the issue, is the
 * universal while(true){ switch(state) } dispatcher, sitting in the chilly
 * bin until then. BIR_SWITCH gets the same treatment and bails the same way.
 */

/* ---- Limits ---- */

#define BST_MAX_BLOCKS  4096        /* keeps step with M2R_MAX_BLOCKS */
#define BST_MAX_PREDS   64          /* a merge or loop-follow can gather a crowd */
#define BST_MAX_NODES   (1 << 14)   /* the structure-tree node pool */
#define BST_MAX_KIDS    (1 << 14)   /* SEQ child-list pool */
#define BST_MAX_COPIES  (1 << 13)   /* phi-destruction copy pool */
#define BST_MAX_LOOPS   256         /* how deep the loop-nesting is allowed to burrow */
#define BST_MAX_RDEPTH  1024        /* recursion cap on if-nesting, so the C stack stays on the table */

#define BST_NONE        0xFFFFFFFFu

/* ---- Structure-Tree Node Kinds ----
 * The whole vocabulary of a structured language, no more and no less,
 * which is a short list once the gotos have been shown the door. */

typedef enum {
    BST_SEQ,            /* a run of children in order: kids[kid_start .. +nkids) */
    BST_BLOCK,          /* leaf: the honest straight-line work of `block`, no phi, no terminator */
    BST_IF,             /* the fork on `block`'s terminator: then_kid, and else_kid or NONE */
    BST_LOOP,           /* a forever-loop round `block` the header, body hanging off then_kid */
    BST_BREAK,          /* out of the innermost loop, mind the step */
    BST_CONTINUE,       /* back round to the innermost loop header for another lap */
    BST_RET,            /* return; the value, if any, is read off `block`'s terminator */
    BST_UNREACHABLE,    /* the bit of code at the end of the No Exit road */
    BST_COPY            /* phi destruction: lay down copies[copy_start .. +copy_count) */
} bst_kind_t;

/* One phi-settling move along a CFG edge, of the form dest = src. dest is
 * the absolute instruction index of the phi (the SSA name it answers to);
 * src is whatever value the predecessor turned up holding, be it a
 * constant in disguise or another instruction's result. Structured code
 * has no phi nodes and no patience for them, so each one gets paid off in
 * plain assignments on the edges that feed it. */
typedef struct {
    uint32_t dest;
    uint32_t src;
} bst_copy_t;

typedef struct {
    uint16_t kind;          /* bst_kind_t */
    uint16_t nkids;         /* SEQ: how many children */
    uint32_t block;         /* BLOCK/RET: the source block. IF/LOOP: the cond/header block */
    uint32_t cond;          /* IF: the condition value (a BIR val ref) */
    uint32_t then_kid;      /* IF: the then arm. LOOP: the body */
    uint32_t else_kid;      /* IF: the else arm, or BST_NONE if it went without one */
    uint32_t kid_start;     /* SEQ: where the children start in kids[] */
    uint32_t copy_start;    /* COPY: where the copies start in copies[] */
    uint32_t copy_count;    /* COPY: how many copies to lay down */
} bst_node_t;

/* ---- Output ----
 * The caller owns this, usually parks it on the heap, and hands the same
 * one back for the next function (the pass wipes the pools clean each
 * time). All the scaffolding the build leans on (dominators, the RPO
 * numbering, the loop tables) lives in a file-static working struct over
 * in bir_struct.c, same arrangement mem2reg already runs, because pouring
 * that much scratch through the caller's struct would be rude. */

typedef struct bst_tree_t {
    bst_node_t  nodes[BST_MAX_NODES];
    uint32_t    num_nodes;

    uint32_t    kids[BST_MAX_KIDS];
    uint32_t    num_kids;

    bst_copy_t  copies[BST_MAX_COPIES];
    uint32_t    num_copies;

    uint32_t    root;           /* the root node, or BST_NONE for an empty function */
    uint32_t    func;           /* which function this tree is the shape of */

    int         ok;             /* 1 = came out structured, 0 = downed tools */
    const char *bail_reason;    /* a note in plain words for when ok == 0 */
} bst_tree_t;

/* ---- API ----
 * Take function `func_index` of module M and work out its shape into `out`.
 * A clean run returns BC_OK with out->ok == 1. A CFG that will not sit
 * still also returns BC_OK, but with out->ok == 0 and out->bail_reason
 * filled in, leaving it to the backend to decide how loudly to grumble. A
 * hard BC_ERR_* is kept in reserve for pool overflow and the other faults
 * that mean something has gone properly pear-shaped. */
int bir_structurize(const bir_module_t *M, uint32_t func_index, bst_tree_t *out);

#endif /* BARRACUDA_BIR_STRUCT_H */
