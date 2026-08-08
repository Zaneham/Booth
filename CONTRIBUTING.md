# Contributing to Booth

You're more than welcome to submit a PR. I'm happy to look at it and give it a fair shake.

If you're reading this and going "Ah heck I don't think I can do that, I'll get it wrong", that's perfectly fine. Submit the PR anyway and I am always happy to guide and assist. I am always learning myself.

## The Style

Booth is written in a defensive C99 style. I spent too much time staring at NASA code in my Halmat project and it stuck. The rules exist because they eliminate entire categories of bugs by construction rather than by testing.

**No dynamic allocation in hot paths.** Pre-allocated, fixed-size buffers. If a pool overflows, return a sentinel, never corrupt a counter. If you can't have unbounded allocation you can't have memory leaks.

**No recursion.** Every function call is a known-depth call. Stack usage is predictable. If you can't recurse you can't blow the stack.

**All loops must be bounded.** If you're iterating to a fixpoint, there's a guard counter. No infinite loops, no "this should always converge." If every loop is bounded you can't hang.

**Bounds-check array accesses** from external or untrusted indices. Trust internal bookkeeping, verify everything else.

**Stack-allocated where possible.** Deterministic behaviour, deterministic cleanup.

**Strict error checking.** Check return values. Handle the failure path.

**No floats where integers will do.** If you're comparing ratios, cross-multiply. Floating point is for GPU shader maths, not compiler internals.

### Kauri

Three of those rules used to be things I just tried to remember. Now there are macros for them. `barracuda.h` pulls in [Kauri](https://github.com/Zaneham/kauri), so they're already available anywhere in the tree and you don't need to include anything.

```c
/* Bounded loop. g counts down, so a cone that won't converge stops
 * instead of hanging. Pick a bound you can justify. */
KA_GUARD(g, 64);
while (changed && g--) {
    changed = fold_once(M);
}

/* Untrusted index. Returns 1 when out of bounds, so it reads as
 * "if that's rubbish, refuse". Internal bookkeeping doesn't need it. */
if (KA_CHK(idx, M->num_insts)) return BC_ERR_VERIFY;

/* Pool allocation. Index 0 is the sentinel, so 0 means the pool is
 * full. Pass the sentinel upwards rather than winding the counter back. */
uint32_t ni = KA_PNEW(M->num_insts, BIR_MAX_INSTS);
if (ni == 0) return 0;
```

Most of the existing code was written before these existed, so there's plenty that could use them. If you're reading a file and spot a loop that could take a guard, or an index coming in from outside that isn't checked, a PR is very welcome. If you're not sure whether a particular one wants it, raise an issue and we can work it out, that's a more interesting conversation than it sounds.

Two things worth knowing. `KA_PNEW` evaluates its counter twice, so give it a plain lvalue and nothing with side effects. And Kauri has an arena allocator and a result type that Booth doesn't use, since allocation happens once per phase and errors come back as `BC_ERR_*`, so there's no need to reach for `KA_TRY`.

### Naming

Function and variable names are short, 4-7 characters. Think of it like reading a motorway sign at 100km/h, you want "SH1 NORTH" not "STATE_HIGHWAY_ONE_NORTHBOUND_DIRECTION". When you're reading a thousand lines of instruction selector at 2am you want `ra_gc` not `regalloc_graphcolor`. Look at the newer code for the pattern: `isel_emit`, `mk_hash`, `enc_vop3`, `xt_meta`, `dce_copy`.

Some older code still uses longer names from when I wanted things readable for reviewers on Reddit. That's changed now. If you're touching a file and spot a verbose name, feel free to shorten it.

### Comments

Comments explain the *why*, not the *what*. Any C programmer can read the code and understand what it does. The comments are there to explain intent, context, and the reasoning behind non-obvious decisions.

Section headers look like this:
```c
/* ---- Section Name ---- */
```

Humour is welcome and encouraged. You're welcome to add your own personality and wit to anything you write. Arrogance is not welcome. Self-deprecating dry wit is the house style but it's not mandatory.

When refactoring or moving code, please keep existing comments with it. They took thought to write.

## Language / Langue / Sprache / Idioma / 言語

**Issues and PRs can be written in any language.** The only rule: provide an English translation alongside your native text. Translation tools are amazing now. Please feel free to use them.

This is a deliberate choice. The best ideas in computing weren't all written in English. I learned technical Russian to build a Setun-70 ternary emulator. The BESM-6 manual isn't in English. Neither is the original Z3 documentation. If I can learn to read Cyrillic for a trinary computer, the least I can do is welcome a PR in Portuguese.

**How it works in practice:**

For issues and PR descriptions, write in whatever language you think in. Then add an English translation below it. The translation doesn't have to be perfect. Google Translate, DeepL, LLM's, whatever you've got. We'll work it out together.

If you use an LLM (Chatgpt, Deepseek, Mistral, A russian LLM run in a basement somewhere) please provide the prompt you used with the output!

For code comments, same deal. Multilingual comments are welcome and encouraged. Humour especially. if you've got a good joke in French or a dry observation in Japanese, put it in. Just add the English so everyone can laugh.

```c
/* Ceci n'est pas une pipe(line).
 * (This is not a pipe(line).) */
```

For identifiers, the HLASM naming convention handles this naturally. When your function names are 4-7 characters of Latin text — `ra_gc`, `mk_hash`, `enc_vop3` — there's nothing culturally English about them. They're just short labels. No umlauts or accents though, sorry (especially you Germans), ASCII only in identifiers. The compiler's lexer would have opinions.

I've had messages from developers around the world, all of whom speak English to varying degrees. Some of them see things I don't. Developer convenience is never a factor in my dependency decisions, and that principle extends to people.

## On LLM's

Speaking of translations and using LLM's, let us address the elephant in the room. An LLM like Chatgpt or any other model is a tool and tools are perfectly acceptable to use depending on how you use them.

I'll start off by saying that I have used LLM's. They're very nifty and perfectly fine. When AMD's documentation says one thing but the output and behaviour say another, an LLM can look up an obscure bug report from a forum post made in 2019 when someone else hit the same bug. It can also look into multiple languages, not just English. I've used LLM's like Ollama to quickly jot down some documentation, like how I would dictate to a voice recorder whenever I've hit a bug or an edge case somewhere and wanted details for when I get back to it. I've had LLM's write me tests to throw at my compilers, and if I'm tired and writing my 42nd array I might just let intellisense handle the rest.

All of these are perfectly acceptable uses of LLM's.

When I was a kid learning Lua on Roblox, I would actually copy and paste scripts from forums when I genuinely got stuck. It is a fantastic way to 1) learn and 2) fix a problem if you struggle with it. Intellisense, Stack Overflow and all of these things are tools. The Mainframe community passed around assembler macros and borrowed off each other's work on literal magnetic tapes. This isn't new. LLM's are just another tool in a long line of tool development that has happened over the years.


**What's acceptable**

- Code review - I use LLM's to have a "second pair of eyes" when I'm writing code, it's pretty nifty at spotting unbounded memory violations I occasionally write or uncommented code that another person might need to read. There are limits to LLM's. If you ask Chatgpt about how to make a compiler it will probably recommend you to use Rust and LLVM which this project purposefully does not use. Code review is fine, architecture is not.
- Research and search - Finding edge cases, documentation, summarisation are all fine
- Test generation - Generate edge case galore and throw it at the compiler, If you do use an LLM just make sure to say so in the issues.
- Documentation - Writing up said bugs when you run into them or for your personal notes

**What's not acceptable**

- **Generating code you don't understand** - When I was writing Callout, my Call and Dispatch engine (it's what Emergency services use when dispatching a firetruck because you burnt toast and now the alarm is going off), I hit a wall. I know systems but had no idea on how to properly add a button or a UI element. I found myself relying on my Ollama model too much and eventually couldn't understand what I was making. Booth requires bit level precision as it emits machine code. If you want to submit a PR but don't understand a section of the codebase or don't understand everything, that is fine, that's being human. You are more than welcome to submit a PR, even an incomplete one, and we can discuss tradeoffs and implementations. We are all learning. Learning is what makes us, us.
- **Wholly synthetic undeclared code** is something we'll have to send back or rework together. If you've used an LLM, just say so — declared LLM-assisted code that you understand and can defend is absolutely fine. The copyright picture is genuinely unsettled though, so occasionally I might ask you to rewrite a section from scratch. Here's why there's caution:
  - **Licence contamination** — LLM training data can include proprietary or incompatibly-licensed code. If it leaks into a PR, it poisons the Apache 2.0 licence for the whole project.
  - **Copyright** — the wonderful folks in that small outfit known as "The United States Federal Government" have ruled that a human has to substantially author or alter code for it to be copyrightable. Unaltered LLM output may not be copyrightable at all, which means it can't actually be licensed under Apache 2.0. Now I'm not in the US, I'm in New Zealand, and our laws are actually more reasonable, but US lawyers aren't exactly well known for their geography knowledge.
  - **Quality** — this is a compiler that emits GPU machine code. One wrong bit is silent data corruption. You need to understand what you're shipping or know when you don't know.

- **Architecture** - As above please don't make architectural decisions using a chatbot. Even then if you're making a big change in the code anyway feel free to contact me, I'm always happy to chat and open to new ideas.


## Where to Help

Check `Issues` for current priorities. In general the most impactful areas are:

**Language features** that real CUDA code needs, things the parser or sema doesn't handle yet. If you've got a .cu file that doesn't compile, that's a useful bug report even if you don't have a fix.

**Backend work.** New architecture targets are always interesting. The compiler is designed for this, BIR is backend-agnostic and each target is self-contained. If you're a deep tech startup and need CUDA support for your hardware, reach out.

**Test cases.** Real CUDA kernels that break things are genuinely valuable. The weirder the better.

**Translating error messages.** The compiler now supports multilingual error output via `--lang <file>`. There's an English reference at `lang/en.txt` and a te reo Maori translation at `lang/mi.txt`. If you speak another language, translating error messages is a fantastic way to contribute — no compiler knowledge required, just copy `lang/en.txt`, translate the text after each `=`, and keep the `%s`/`%d` placeholders in place. Every error has a language-neutral ID (like `E020`) so developers can search for help regardless of what language their errors are in.

**Optimisation passes.** DCE, constant folding, and instruction scheduling already exist. If you want to add something like loop unrolling or better spill heuristics, open an issue first so we can chat about the approach.

If you're not sure whether something is worth a PR, open an issue. I also love to, as we say here in New Zealand, spin a few yarns. A quick conversation up front saves everyone time.

## Backends

Booth generates code for the following targets. This is codegen support; the silicon it's actually been run and validated on is listed in [docs/hardware.md](docs/hardware.md).

- **AMD CDNA 2** (gfx90a, MI250)
- **AMD CDNA 3** (gfx942, MI300X)
- **AMD RDNA 2** (gfx1030 family)
- **AMD RDNA 3** (gfx1100 family, gfx1150 family)
- **AMD RDNA 4** (gfx1200 family)
- **NVIDIA PTX** (sm_60 and above)
- **Tenstorrent Wormhole**: native RV32IM for baby cores via `--rv-elf`, plus Metalium C++ via `--tensix`
- **Apple Metal** (stub, hardware validation pending)
- **Intel Arc / Xe SPIR-V** (stub)

The frontend lowers to BIR (Booth IR) in SSA form. Each backend is a self-contained target that consumes BIR. Adding a new architecture means writing a new instruction selector and emitter; the rest of the pipeline is shared.

Tenstorrent additionally sits above BIR through TDF (Tile DataFlow), a small IR that models L1 placement, circular buffers, and NoC arcs as first-class concepts. Adding a new Tenstorrent-shaped target (other tile-based accelerators, RVV-style processors) can reuse TDF and skip writing all of the orchestration from scratch.

## Building & Testing

```bash
# Build
make

# Run the test suite (currently 274 tests across the frontends,
# IR, backends, runtime, and SYSPRINT)
make test

# Run the emulator test suite (RDNA3, requires tinygrad mockgpu in WSL)
python tests/emu/run_emu.py
```

Verify your changes don't introduce encoding regressions:
```bash
llvm-objdump -d --mcpu=gfx1100 output.hsaco
# Zero decode failures = good
```

A note on the Makefile: header dependencies aren't auto-tracked. If you change a `.h` file that other modules include (especially the larger shared structs in `triton.h`, `bir.h`, `tensix.h`), run `make clean && make` rather than incremental rebuild. Stale `.o` files compiled against the old struct layout crash at runtime in interesting ways.

## Branching & Pull Requests

Ongoing work happens on feature branches off `master`. The currently active long-running branch is `triton-apple-mega` (Triton frontend, shape inference, SYSPRINT and adjacent work). Pull requests merge into `master` via **squash merge** so the main branch history stays readable: one squashed commit per landed feature, rather than the full back-and-forth of the development branch.

For external contributors: fork the repo, make a branch off `master`, send a PR. Multiple small PRs are easier to land than one large PR. If you're working on something speculative, mark the PR as a draft and we can discuss the approach before it's review-ready.

Commits before PR merge can have any history shape you like: squash, rebase, merge commits, all fine. The squash at merge time flattens it all down. Write the commit message you want to see in `master`'s log.

## Changelog

Any user-visible change should come with a `CHANGELOG.md` entry, added under
`## Unreleased` in the section that fits (Frontend, Tensix, Backends, CI and
tests, and so on). Follow the existing shape:

```
- #123: what changed, in a sentence that makes sense to someone reading the
  release notes rather than the diff
  (Your Name, 2026-07-28)
```

Lead with the PR number where there is one, keep numbered entries in ascending
order, and credit yourself with the date the work landed. The rest of the file
is organised by date, so entries carry one too. Credit reviewers and bug
reporters as well where they shaped the change.

At release time these entries get gathered into the prose summary that heads
each dated section, so write them as if someone else will read them cold.

Small changes such as typo fixes, comment tidying or `.gitignore` additions
don't need an entry. Apply the **`no-changelog-needed`** label to the PR and CI
will skip the check.

## License

Booth is Apache 2.0. By submitting a PR, you agree your contribution is licensed under the same terms and you represent that you have the right to do so — meaning the code is your own work, or derived from compatibly-licensed sources, and not copied from proprietary material.

---

You've read this long, here's your prize, the island across the harbour from my house!

![Rangitoto](docs/harbour.png)
