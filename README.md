<div align="center">
  <img src="Booth_logo.png" alt="Booth logo" width="400">
  <h1>Booth</h1>
</div>

**Booth** is an open-source CUDA, HIP and Triton compiler targeting multiple GPU architectures, either natively by emitting machine code or as close as we can possibly get. Now with distinctly less fish.

It is named to honour Kathleen Booth: creator of the first assembly language, co-builder of the computer it first ran on, an early researcher into neural nets, a champion of women in computing, the daughter of a tax clerk, English by birth and Canadian by choice, and a mother. I believe it is fitting to name this after an incredible woman whose work this is built on. She built the machines her assembler ran on, which is exactly the bootstrapped "yeah, why not ay?" attitude this whole compiler is going for.

A running log of what's changed is in [CHANGELOG.md](CHANGELOG.md).

**Update:** ggml-cuda, the CUDA half of llama.cpp, now lowers. All 67 of its
files reach Booth's IR, and 47 of its kernels go through the new native SASS
backend, which writes a cubin the card loads directly. Until now NVIDIA meant
PTX and the driver's JIT. Try it with `kath --nvidia-cubin`. And I never want
to have to read a machine dump ever again, lmao omg it's been rough.

## What It Does

Takes CUDA C, HIP, or Triton source (the same files you'd hand to `nvcc`, `ROCm`, or Triton's JIT) and turns them into AMD RDNA 2/3/4 binaries, NVIDIA PTX, Tenstorrent Metalium C++ or native RV32IM, or just plain x86-64 you can run on a laptop with no GPU in it.

That last one still surprises me a bit. You can write a Triton kernel, matmul and all, and run it on a machine that's never seen a GPU, from scratch, no LLVM, straight to native. I haven't come across anyone else doing Triton like this, but I'd happily be proven wrong, so give me a yell if you've seen it somewhere.

Fortran `do concurrent` kernels go down the same path through
[LFortran](https://lfortran.org/), checked against SLATEC values in CI. See
[Using Fortran](docs/usage.md#fortran).

OCaml kernels go down it too, written as plain functions and type-checked by
`ocamlc` before Booth reads the `.cmt`. See [Using OCaml](docs/usage.md#ocaml).

It also borrows a pile of operational discipline from the mainframe world: real crash dumps when a kernel faults, structured output routed by class, parameter snapshots on entry. See [docs/mainframe.md](docs/mainframe.md) if that sounds like your kind of thing.

## Getting it

If you're lazy like me and just want to run and go, then you can install it [here](https://github.com/Zaneham/Booth/releases/latest). It comes with no dependencies and you don't have to run `make` to use it. There's a build for Linux, macOS and Windows.

```bash
tar xzf booth-*-linux-x86_64.tar.gz
cd booth-*-linux-x86_64
./kath --version
```

## Build

If you'd rather build it, or you're on something I don't ship a binary for:

```bash
make
```

That's the whole thing. You need a C99 compiler (gcc, clang, whatever you've got) and nothing else.

Two frontends read what another compiler produced, so you only need that
compiler if you want that frontend. Neither is needed to build Booth:

- Fortran kernels want [LFortran](https://lfortran.org/), which emits the CUDA
  source Booth compiles the rest of the way.
- OCaml kernels want [OCaml](https://ocaml.org/) 5.x and dune, which type-check
  the kernel and leave the `.cmt` Booth reads.


```bash
# compile a CUDA kernel to an AMD GPU binary
./kath --amdgpu-bin kernel.cu -o kernel.hsaco
```

The binary is `kath` (after Kathleen but if she picked Australia or New Zealand instead of Canada), not `booth`: there's already a `booth` in the Linux HA stack, so you'll likely end up with both on your PATH. The full command reference, every backend and flag, lives in [docs/usage.md](docs/usage.md).

## Documentation

- **[Usage](docs/usage.md)** — every backend, every flag, and the runtime launcher
- **[CMake](docs/cmake.md)** — installing Booth and compiling kernels from a CMake project
- **[Feature status](docs/features.md)** — what compiles today, and what doesn't yet
- **[Mainframe curios](docs/mainframe.md)** — ABEND dumps, SNAP, SYSPRINT, TDF
- **[Validated hardware](docs/hardware.md)** — the silicon it's been tested on, and the test suite
- **[Roadmap](docs/roadmap.md)** — where it's headed
- **[Contributing](CONTRIBUTING.md)** — style, naming, and where to help (PRs in any language welcome)

## Supporting this compiler

If you're considering supporting this compiler please feel free to get in touch.

 However if this compiler has been particularly helpful for you then please consider the [Kate Edger Foundation](https://kateedgerfoundation.org.nz/) which funds women in education across Auckland and Northland, named for Kate Edger, the first woman in New Zealand to earn a university degree (Latin and mathematics, 1877). If Booth's namesake means anything to you, they're well worth a look.

## License

Apache 2.0. Do whatever you want. If this compiler somehow ends up in production, I'd love to hear about it, mostly so I can update my LinkedIn with something more interesting than wrote a CUDA compiler for fun.

## Contact

Found a bug? Want to discuss the finer points of AMDGPU instruction encoding? Need someone to commiserate with about the state of GPU computing?

**zanehambly@gmail.com**

Open an issue if there's anything you want to discuss. Or don't. I'm not your mum.

Based in New Zealand, where it's already tomorrow and the GPUs are just as confused as everywhere else.

## Acknowledgements

- **Fernando Magno Quintão Pereira** and the **Compilers Lab at UFMG** (Universidade Federal de Minas Gerais). Fernando reached out after seeing the project, pointed me to the divergence analysis papers, and offered guidance. The SSA register allocator exists because of that conversation.
- **[Jorge Galvez](https://github.com/JorgeG94)** for sending me his `do concurrent` Fortran benchmarks and letting me run tests on them. Three real frontend bugs turned up in an afternoon, which is exactly what you want somebody else's code to do.
- **Jon Stevens** from Hot Aisle who has very generously supported this compiler by providing access to AMD CDNA GPUs. You can find more about Hot Aisle [here](https://hotaisle.xyz/).
- **The academic community**: Cooper, Harvey & Kennedy for dominators; Braun & Hack for SSA spilling; Sampaio, Souza, Collange & Pereira for divergence analysis. I'm just a hobbyist who reads papers and writes C. The actual hard work was done by the researchers.
- **Steven Muchnick** for *Advanced Compiler Design and Implementation*. If this compiler does anything right, that book is why.
- **Low Level** for the Zero to Hero C course and the YouTube channel. That's where I learnt C.
- **Abe Kornelis** for being an amazing teacher. His work on the [z390 Portable Mainframe Assembler](https://github.com/z390development/z390) project is well worth your time.
- To the people who've sent messages of kindness and critique, thank you from a forever student and a happy hobbyist.
- **Lola**, my sister, for the logo :-)
- My Granny, Grandad, Nana and Baka. Love you x

*He aha te mea nui o te ao. He tāngata, he tāngata, he tāngata.*

What is the most important thing in the world? It is people, it is people, it is people.
