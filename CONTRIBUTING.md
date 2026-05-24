# Contributing to BarraCUDA

You're more than welcome to submit a PR. I'm happy to look at it and give it a fair shake.

If you're reading this and going "Ah heck I don't think I can do that, I'll get it wrong", that's perfectly fine. Submit the PR anyway and I am always happy to guide and assist. I am always learning myself.

## The Style

BarraCUDA is written in a defensive C99 style. I spent too much time staring at NASA code in my Halmat project and it stuck. The rules exist because they eliminate entire categories of bugs by construction rather than by testing.

**No dynamic allocation in hot paths.** Pre-allocated, fixed-size buffers. If a pool overflows, return a sentinel, never corrupt a counter. If you can't have unbounded allocation you can't have memory leaks.

**No recursion.** Every function call is a known-depth call. Stack usage is predictable. If you can't recurse you can't blow the stack.

**All loops must be bounded.** If you're iterating to a fixpoint, there's a guard counter. No infinite loops, no "this should always converge." If every loop is bounded you can't hang.

**Bounds-check array accesses** from external or untrusted indices. Trust internal bookkeeping, verify everything else.

**Stack-allocated where possible.** Deterministic behaviour, deterministic cleanup.

**Strict error checking.** Check return values. Handle the failure path.

**No floats where integers will do.** If you're comparing ratios, cross-multiply. Floating point is for GPU shader maths, not compiler internals.

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

- **Generating code you don't understand** - When I was writing Callout, my Call and Dispatch engine (it's what Emergency services use when dispatching a firetruck because you burnt toast and now the alarm is going off), I hit a wall. I know systems but had no idea on how to properly add a button or a UI element. I found myself relying on my Ollama model too much and eventually couldn't understand what I was making. BarraCUDA requires bit level precision as it emits machine code. If you want to submit a PR but don't understand a section of the codebase or don't understand everything, that is fine, that's being human. You are more than welcome to submit a PR, even an incomplete one, and we can discuss tradeoffs and implementations. We are all learning. Learning is what makes us, us.
- **Wholly synthetic undeclared code** is something we'll have to send back or rework together. If you've used an LLM, just say so — declared LLM-assisted code that you understand and can defend is absolutely fine. The copyright picture is genuinely unsettled though, so occasionally I might ask you to rewrite a section from scratch. Here's why there's caution:
  - **Licence contamination** — LLM training data can include proprietary or incompatibly-licensed code. If it leaks into a PR, it poisons the Apache 2.0 licence for the whole project.
  - **Copyright** — the wonderful folks in that small outfit known as "The United States Federal Government" have ruled that a human has to substantially author or alter code for it to be copyrightable. Unaltered LLM output may not be copyrightable at all, which means it can't actually be licensed under Apache 2.0. Now I'm not in the US, I'm in New Zealand, and our laws are actually more reasonable, but US lawyers aren't exactly well known for their geography knowledge.
  - **Quality** — this is a compiler that emits GPU machine code. One wrong bit is silent data corruption. You need to understand what you're shipping.

  The short version: declare your tools, understand your code, and if I ask you to rewrite something it's not personal — it's just the reality of shipping code in a world where the law hasn't caught up yet. Treat LLM output like a snippet from Stack Overflow — make sure you can explain it. And if you're struggling, that's okay. Mark a PR as a draft, raise a discussion, and I'll help when I can.
- **Architecture** - As above please don't make architectural decisions using a chatbot. Even then if you're making a big change in the code anyway feel free to contact me, I'm always happy to chat and open to new ideas.

## On the Mighty Emdash
You'll notice emdashes everywhere — in comments, in commit messages, in this document. I've been drawing hyphens a bit too long since primary school. I have my old books from when I was seven, and there they are — emdashes. Or hyphens. Or maybe I just didn't know what I was writing. 7-year-old me didn't leave a comment.

The point is: use them. They're better than parentheses for asides, better than semicolons for joining related thoughts, and they look good in monospaced fonts. If your PR has emdashes, you'll fit right in.

The hate for emdashes is superficial and weird. I understand its because of LLM's. But I'm not going to let some Robot dictate how I share my own thoughts in my own language. 

## Indigenous and Endangered Languages 

The first non-English language in BarraCUDA's error messages was te reo Māori. Not because it was strategic, but because I live here, these are my neighbours, and this is one of the three official languages of Aotearoa New Zealand. You can run `barracuda --lang lang/mi.txt` right now and get your errors in te reo. Kia ora, GPU.

There are roughly 7,000 languages spoken on Earth. About 40% of them are endangered. When those languages disappear from technology — when every error message, every man page, every compiler diagnostic is English-only — it sends a quiet message: this isn't for you. That matters more than most developers realise. If your tools don't speak your language, you're not just reading code, you're translating core infrastructure. The research on cognitive load is absolutely clear: that translation costs time, costs accuracy, and costs people who would have been brilliant engineers. When you're reading one of my glorious Abend dumps because you decided it would be a great idea to chuck your MEGAGPT10 onto your RDNA 2 stick you should at least be able to read the consequences of your bad decisions in your own tongue. 

Indigenous and endangered languages are especially welcome here. Te reo Māori, Welsh, Hawaiian, Navajo, Scots Gaelic, Samoan, any of the hundreds of languages that technology has quietly decided don't matter — if you want to see your language in a compiler diagnostic, this is that project. The translation format is dead simple and needs zero compiler knowledge, which also makes it a neat entry point into compiler development too! See the Where to Help section below for how.

## Where to Help

Check `Issues` for current priorities. In general the most impactful areas are:

**Language features** that real CUDA code needs, things the parser or sema doesn't handle yet. If you've got a .cu file that doesn't compile, that's a useful bug report even if you don't have a fix.

**Backend work.** New architecture targets are always interesting. The compiler is designed for this, BIR is backend-agnostic and each target is self-contained. If you're a deep tech startup and need CUDA support for your hardware, reach out.

**Test cases.** Real CUDA kernels that break things are genuinely valuable. The weirder the better.

**Translating error messages.** The compiler now supports multilingual error output via `--lang <file>`. There's an English reference at `lang/en.txt` and a te reo Maori translation at `lang/mi.txt`. If you speak another language, translating error messages is a fantastic way to contribute — no compiler knowledge required, just copy `lang/en.txt`, translate the text after each `=`, and keep the `%s`/`%d` placeholders in place. Every error has a language-neutral ID (like `E020`) so developers can search for help regardless of what language their errors are in.

**Optimisation passes.** DCE, constant folding, and instruction scheduling already exist. If you want to add something like loop unrolling or better spill heuristics, open an issue first so we can chat about the approach.

If you're not sure whether something is worth a PR, open an issue. I also love to, as we say here in New Zealand, spin a few yarns. A quick conversation up front saves everyone time.

## Backends

BarraCUDA currently supports:

- **AMD CDNA 2** (gfx90a, MI250)
- **AMD CDNA 3** (gfx942, MI300X)
- **AMD RDNA 2** (gfx1030 family)
- **AMD RDNA 3** (gfx1100 family, gfx1150 family)
- **AMD RDNA 4** (gfx1200 family)
- **NVIDIA PTX** (sm_60 and above)
- **Tenstorrent Wormhole**: native RV32IM for baby cores via `--rv-elf`, plus Metalium C++ via `--tensix`
- **Apple Metal** (stub, hardware validation pending)
- **Intel Arc / Xe SPIR-V** (stub)

The frontend lowers to BIR (BarraCUDA IR) in SSA form. Each backend is a self-contained target that consumes BIR. Adding a new architecture means writing a new instruction selector and emitter; the rest of the pipeline is shared.

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

## License

BarraCUDA is Apache 2.0. By submitting a PR, you agree your contribution is licensed under the same terms and you represent that you have the right to do so — meaning the code is your own work, or derived from compatibly-licensed sources, and not copied from proprietary material.

---

You've read this long, here's your prize, the island across the harbour from my house!

![Rangitoto](docs/harbour.png)
