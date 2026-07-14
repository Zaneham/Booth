#!/usr/bin/env python3
"""Booth RDNA3/RDNA4 emulator harness. Powered by tinygrad's mockgpu."""
import struct, sys, ctypes, os

TGDIR = os.environ.get('TINYGRAD_PATH', '')
if TGDIR:
    sys.path.insert(0, TGDIR)

from test.mockgpu.amd.emu import run_asm

LIBC  = ctypes.CDLL("libc.so.6")
LIBC.mmap.restype  = ctypes.c_void_p
LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]

# ELF e_flags → tinygrad arch string
ARCHM = { 0x41: "rdna3", 0x48: "rdna4" }

def lo_mem(NBYTE):
    """mmap in low address space (MAP_32BIT)."""
    MADDR = LIBC.mmap(0, NBYTE, 0x3, 0x22 | 0x40, -1, 0)
    if MADDR == ctypes.c_void_p(-1).value:
        raise RuntimeError("mmap failed — check ulimit or address space")
    return MADDR

def asm_dw(*DWORD):
    """Pack dwords into low-mem code buffer."""
    NBYTE = len(DWORD) * 4
    MADDR = lo_mem(NBYTE)
    CBUF  = (ctypes.c_char * NBYTE).from_address(MADDR)
    for i, W in enumerate(DWORD):
        struct.pack_into('<I', CBUF, i * 4, W)
    return MADDR, NBYTE

# ---- Smoke Tests (IPL Checks) ----

def ipl1(ARCHV="rdna3"):
    """IPL1: s_endpgm only. Verify basic dispatch/terminate."""
    CADDR, CSIZ = asm_dw(0xBFB00000)
    RETCD = run_asm(CADDR, CSIZ, 1, 1, 1, 32, 1, 1, 0, 0x08, 0, ARCHV, [0, 0])
    PASSD = "PASS" if RETCD == 0 else "FAIL"
    print(f"  IPL1 s_endpgm:         {PASSD} (rc={RETCD})")
    return RETCD == 0

def ipl2(ARCHV="rdna3"):
    """IPL2: store 42.0f via VOP1 + global_store (SADDR=null)."""
    MEMSZ = 4096
    MBASE = lo_mem(MEMSZ)
    OUTBF = (ctypes.c_float * 1).from_address(MBASE)
    OUTBF[0] = 0.0

    if ARCHV == "rdna4":
        # GFX12: 96-bit global_store_b32 (OP at [21:14] per AMD ISA XML)
        # v_mov_b32 v1, s0 / v2, s1 / v3, 42.0f
        # global_store_b32 v[1:2], v3, off  (3 dwords: enc|op|saddr, vsrc, vaddr)
        GSTD0 = (0xEE << 24) | (0x1A << 14) | 0x7C  # enc=GLOBAL OP=STORE_B32 SADDR=null
        GSTD1 = (3 << 23)                             # VSRC=v3
        GSTD2 = 1                                     # VADDR=v1
        CADDR, CSIZ = asm_dw(
            0x7E020200,             # v_mov_b32 v1, s0
            0x7E040201,             # v_mov_b32 v2, s1
            0x7E0602FF, 0x42280000, # v_mov_b32 v3, 42.0f (literal)
            GSTD0, GSTD1, GSTD2,   # global_store_b32 v[1:2], v3, off
            0xBFB00000,             # s_endpgm
        )
    else:
        # GFX11: 64-bit global_store_dword
        CADDR, CSIZ = asm_dw(
            0x7E020200,             # v_mov_b32 v1, s0
            0x7E040201,             # v_mov_b32 v2, s1
            0x7E0602FF, 0x42280000, # v_mov_b32 v3, 42.0f (literal)
            0xDC6A0000, 0x007C0301, # global_store_dword SADDR=null VADDR=v1 DATA=v3
            0xBFB00000,             # s_endpgm
        )

    USRDT = [MBASE & 0xFFFFFFFF, (MBASE >> 32) & 0xFFFFFFFF]
    RETCD = run_asm(CADDR, CSIZ, 1, 1, 1, 1, 1, 1, 0, 0x08, 0, ARCHV, USRDT)
    GVAL  = OUTBF[0]
    PASSD = "PASS" if (RETCD == 0 and abs(GVAL - 42.0) < 1e-5) else "FAIL"
    print(f"  IPL2 vop1+gstore:      {PASSD} (rc={RETCD}, got={GVAL})")
    return RETCD == 0 and abs(GVAL - 42.0) < 1e-5

def ipl3(ARCHV="rdna3"):
    """IPL3: s_load_dwordx2 + global_store. Full SMEM pipeline."""
    MEMSZ = 4096
    MBASE = lo_mem(MEMSZ)

    # layout: [0:7]=ptr to outbuf, [8:11]=padding, [256:259]=output float
    KARGP = MBASE
    OUTAD = MBASE + 256
    OUTBF = (ctypes.c_float * 1).from_address(OUTAD)
    OUTBF[0] = 0.0
    struct.pack_into('<Q', (ctypes.c_uint8 * 8).from_address(KARGP), 0, OUTAD)

    if ARCHV == "rdna4":
        # GFX12 SMEM: OP at [18:13] (6-bit), 24-bit IOFFSET
        SMLD0 = (0x3D << 26) | (0x01 << 13) | (4 << 6) | 0
        SMLD1 = (0x7C << 25) | 0

        # GFX12 global_store_b32 v1, v2, s[4:5] — 96-bit (OP at [21:14])
        GSTD0 = (0xEE << 24) | (0x1A << 14) | 4
        GSTD1 = (2 << 23)      # VSRC=v2
        GSTD2 = 1              # VADDR=v1

        CADDR, CSIZ = asm_dw(
            SMLD0, SMLD1,           # s_load_dwordx2 s[4:5], s[0:1], 0
            0xBFC70000,             # s_wait_kmcnt 0x0
            0x7E020280,             # v_mov_b32 v1, 0
            0x7E0402FF, 0x42280000, # v_mov_b32 v2, 42.0f
            GSTD0, GSTD1, GSTD2,   # global_store_b32 v1, v2, s[4:5]
            0xBFB00000,             # s_endpgm
        )
    else:
        # GFX11 SMEM: OP at [25:18] (8-bit), 21-bit offset
        SMLD0 = (0x3D << 26) | (0x01 << 18) | (4 << 6) | 0
        SMLD1 = (0x7C << 25) | 0

        # GFX11 global_store_dword v1, v2, s[4:5] — 64-bit
        GSTD0 = (0x37 << 26) | (0x1A << 18) | (2 << 16)
        GSTD1 = (0 << 24) | (4 << 16) | (2 << 8) | 1

        CADDR, CSIZ = asm_dw(
            SMLD0, SMLD1,           # s_load_dwordx2 s[4:5], s[0:1], 0
            0xBF89FC07,             # s_waitcnt lgkmcnt(0)
            0x7E020280,             # v_mov_b32 v1, 0
            0x7E0402FF, 0x42280000, # v_mov_b32 v2, 42.0f
            GSTD0, GSTD1,          # global_store_dword v1, v2, s[4:5]
            0xBFB00000,             # s_endpgm
        )

    USRDT = [KARGP & 0xFFFFFFFF, (KARGP >> 32) & 0xFFFFFFFF]
    RETCD = run_asm(CADDR, CSIZ, 1, 1, 1, 1, 1, 1, 0, 0x08, 0, ARCHV, USRDT)
    GVAL  = OUTBF[0]
    PASSD = "PASS" if (RETCD == 0 and abs(GVAL - 42.0) < 1e-5) else "FAIL"
    print(f"  IPL3 smem+gstore:      {PASSD} (rc={RETCD}, got={GVAL})")
    return RETCD == 0 and abs(GVAL - 42.0) < 1e-5

# ---- ELF Parser ----

def prs_elf(FDATA):
    """Extract .rodata (KD) + .text (code) from ELF64 LE.
    Returns KD bytes + code bytes concatenated so prs_kd still
    reads offset 0 as the kernel descriptor."""
    if FDATA[:4] != b'\x7fELF':
        raise ValueError("not ELF")
    SHOFF = struct.unpack_from('<Q', FDATA, 40)[0]
    SHENT = struct.unpack_from('<H', FDATA, 58)[0]
    SHNUM = struct.unpack_from('<H', FDATA, 60)[0]
    STIDX = struct.unpack_from('<H', FDATA, 62)[0]
    SSOFF = struct.unpack_from('<Q', FDATA, SHOFF + STIDX * SHENT + 24)[0]
    SSSIZ = struct.unpack_from('<Q', FDATA, SHOFF + STIDX * SHENT + 32)[0]
    SSTAB = FDATA[SSOFF : SSOFF + SSSIZ]
    SECTS = {}
    for i in range(SHNUM):
        SHBAS = SHOFF + i * SHENT
        NMOFF = struct.unpack_from('<I', FDATA, SHBAS)[0]
        SNAME = SSTAB[NMOFF : SSTAB.index(b'\0', NMOFF)].decode()
        SOFF  = struct.unpack_from('<Q', FDATA, SHBAS + 24)[0]
        SSIZ  = struct.unpack_from('<Q', FDATA, SHBAS + 32)[0]
        SECTS[SNAME] = FDATA[SOFF : SOFF + SSIZ]
    # New layout: KD in .rodata, code in .text
    if '.rodata' in SECTS and '.text' in SECTS:
        KDATA = SECTS['.rodata']
        TDATA = SECTS['.text']
        # Pad KD to 256 bytes (entry_byte_offset) then append code
        PAD   = b'\x00' * (256 - len(KDATA))
        return KDATA + PAD + TDATA
    # Legacy layout: KD + code both in .text
    if '.text' in SECTS:
        return SECTS['.text']
    raise ValueError("no .text or .rodata")

def prs_arch(FDATA):
    """Extract arch from ELF e_flags. Returns 'rdna3' or 'rdna4'."""
    EFLGS = struct.unpack_from('<I', FDATA, 48)[0]
    return ARCHM.get(EFLGS, "rdna3")

def prs_kd(TXDAT):
    """Parse 64-byte kernel descriptor."""
    LDSSZ = struct.unpack_from('<I', TXDAT, 0)[0]
    SCRSZ = struct.unpack_from('<I', TXDAT, 4)[0]
    KASIZ = struct.unpack_from('<I', TXDAT, 8)[0]
    RSRC1 = struct.unpack_from('<I', TXDAT, 48)[0]
    RSRC2 = struct.unpack_from('<I', TXDAT, 52)[0]
    KPROP = struct.unpack_from('<H', TXDAT, 56)[0]
    return RSRC1, RSRC2, KASIZ, LDSSZ, SCRSZ, KPROP

# ---- vectorAdd Execution ----

def run_vadd(TXDAT, RSRC2, SCRSZ, ARCHV="rdna3", KPROP=0x03):
    """Execute vectorAdd and verify every element."""
    KCODE = TXDAT[256:]
    KCSIZ = len(KCODE)
    CADDR = lo_mem(KCSIZ)
    ctypes.memmove(CADDR, KCODE, KCSIZ)

    NELMS = 256
    BKSIZ = 64
    NGRPS = NELMS // BKSIZ
    FSIZE = NELMS * 4

    MEMSZ = FSIZE * 3 + 64 + 64
    MBASE = lo_mem(MEMSZ)

    AADDR = MBASE
    BADDR = MBASE + FSIZE
    CADR2 = MBASE + FSIZE * 2
    KAOFF = MBASE + FSIZE * 3
    DPOFF = KAOFF + 64

    BUFA  = (ctypes.c_float * NELMS).from_address(AADDR)
    BUFB  = (ctypes.c_float * NELMS).from_address(BADDR)
    BUFC  = (ctypes.c_float * NELMS).from_address(CADR2)
    KARGS = (ctypes.c_uint8 * 64).from_address(KAOFF)
    DSPKT = (ctypes.c_uint8 * 64).from_address(DPOFF)

    for i in range(NELMS):
        BUFA[i] = float(i)
        BUFB[i] = float(i * 2)
        BUFC[i] = 0.0

    print(f"  bufa=0x{AADDR:016x} bufb=0x{BADDR:016x} bufc=0x{CADR2:016x}")

    struct.pack_into('<Q', KARGS, 0,  AADDR)
    struct.pack_into('<Q', KARGS, 8,  BADDR)
    struct.pack_into('<Q', KARGS, 16, CADR2)
    struct.pack_into('<I', KARGS, 24, NELMS)

    # Hidden kernargs at offset 32: block_count (3×u32) + group_size (3×u16)
    # Matches isel.c hk_base layout for BIR_BLOCK_DIM / BIR_GRID_DIM
    struct.pack_into('<I', KARGS, 32, NGRPS)   # block_count_x
    struct.pack_into('<I', KARGS, 36, 1)        # block_count_y
    struct.pack_into('<I', KARGS, 40, 1)        # block_count_z
    struct.pack_into('<H', KARGS, 44, BKSIZ)    # group_size_x
    struct.pack_into('<H', KARGS, 46, 1)        # group_size_y
    struct.pack_into('<H', KARGS, 48, 1)        # group_size_z

    struct.pack_into('<H', DSPKT, 4,  BKSIZ)
    struct.pack_into('<H', DSPKT, 6,  1)
    struct.pack_into('<H', DSPKT, 8,  1)
    struct.pack_into('<I', DSPKT, 12, NELMS)

    print(f"  karg=0x{KAOFF:016x} disp=0x{DPOFF:016x} code=0x{CADDR:016x}")

    # Build user SGPRs from kernel_code_properties bits:
    #   bit 0 = ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER (4 SGPRs)
    #   bit 1 = ENABLE_SGPR_DISPATCH_PTR (2 SGPRs)
    #   bit 2 = ENABLE_SGPR_QUEUE_PTR (2 SGPRs)
    #   bit 3 = ENABLE_SGPR_KERNARG_SEGMENT_PTR (2 SGPRs)
    USRDT = []
    if KPROP & (1 << 1):  # dispatch_ptr
        USRDT += [DPOFF & 0xFFFFFFFF, (DPOFF >> 32) & 0xFFFFFFFF]
    if KPROP & (1 << 3):  # kernarg_ptr
        USRDT += [KAOFF & 0xFFFFFFFF, (KAOFF >> 32) & 0xFFFFFFFF]

    RETCD = run_asm(CADDR, KCSIZ, NGRPS, 1, 1, BKSIZ, 1, 1,
                    KAOFF, RSRC2, SCRSZ, ARCHV, USRDT)
    if RETCD != 0:
        return RETCD, 0, NELMS

    DIAG = [0, 1, 2, 63, 64, 128, 192, 255]
    for i in DIAG:
        EXPCT = float(i) + float(i * 2)
        GVAL  = BUFC[i]
        TAG   = "OK" if abs(GVAL - EXPCT) < 1e-5 else "WRONG"
        print(f"    c[{i:3d}]: expected {EXPCT:8.1f}, got {GVAL:8.1f} [{TAG}]")

    NFAIL = 0
    for i in range(NELMS):
        EXPCT = float(i) + float(i * 2)
        GVAL  = BUFC[i]
        if abs(GVAL - EXPCT) > 1e-5:
            NFAIL += 1

    return 0, NFAIL, NELMS

def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <file.hsaco>", file=sys.stderr)
        return 1

    with open(sys.argv[1], 'rb') as f:
        FDATA = f.read()

    ARCHV = prs_arch(FDATA)

    print(f"=== IPL checks ({ARCHV}) ===")
    TPAS1 = ipl1(ARCHV)
    TPAS2 = ipl2(ARCHV)
    TPAS3 = ipl3(ARCHV)

    if not TPAS1:
        print("ABORT: emulator cannot even s_endpgm. Check tinygrad install.")
        return 1

    print(f"\n=== vectorAdd ({ARCHV}) ===")
    TXDAT = prs_elf(FDATA)
    RSRC1, RSRC2, KASIZ, LDSSZ, SCRSZ, KPROP = prs_kd(TXDAT)
    KCSIZ = len(TXDAT) - 256

    print(f"  rsrc1=0x{RSRC1:08x} rsrc2=0x{RSRC2:08x} "
          f"kernarg={KASIZ} lds={LDSSZ} scratch={SCRSZ} code={KCSIZ}B")

    RETCD, NFAIL, NELMS = run_vadd(TXDAT, RSRC2, SCRSZ, ARCHV, KPROP)

    if RETCD != 0:
        print(f"FAIL: emulator rc={RETCD}")
        return 1
    if NFAIL > 0:
        print(f"FAIL: {NFAIL}/{NELMS} elements wrong")
        return 1

    print(f"PASS: vectorAdd {NELMS} elements verified ({ARCHV})")
    return 0

if __name__ == '__main__':
    sys.exit(main())
