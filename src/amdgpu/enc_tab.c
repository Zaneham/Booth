#include "amdgpu.h"

/*
 * AMDGCN opcode encoding tables.
 * Maps amd_op_t -> (format, hw_opcode, mnemonic) for each target generation.
 * GFX11 (gfx1100, RDNA 3) and GFX10 (gfx1030, RDNA 2) opcodes extracted
 * by assembling with llvm-mc and reading the bones.
 * Two manuals, twice the fun.
 */

/* ---- GFX11 Encoding Table (gfx1100, RDNA 3) ---- */

const amd_enc_entry_t amd_enc_table[AMD_OP_COUNT] = {
    /* SOP2 */
    [AMD_S_ADD_U32]         = { AMD_FMT_SOP2, 0x00, "s_add_u32"         },
    [AMD_S_ADD_I32]         = { AMD_FMT_SOP2, 0x02, "s_add_i32"         },
    [AMD_S_SUB_U32]         = { AMD_FMT_SOP2, 0x01, "s_sub_u32"         },
    [AMD_S_MUL_I32]         = { AMD_FMT_SOP2, 0x2C, "s_mul_i32"         },
    [AMD_S_AND_B32]         = { AMD_FMT_SOP2, 0x16, "s_and_b32"         },
    [AMD_S_OR_B32]          = { AMD_FMT_SOP2, 0x18, "s_or_b32"          },
    [AMD_S_XOR_B32]         = { AMD_FMT_SOP2, 0x1A, "s_xor_b32"         },
    [AMD_S_LSHL_B32]        = { AMD_FMT_SOP2, 0x08, "s_lshl_b32"        },
    [AMD_S_LSHR_B32]        = { AMD_FMT_SOP2, 0x0A, "s_lshr_b32"        },
    [AMD_S_ASHR_I32]        = { AMD_FMT_SOP2, 0x0C, "s_ashr_i32"        },
    [AMD_S_ANDN2_B32]       = { AMD_FMT_SOP2, 0x22, "s_andn2_b32"       },
    [AMD_S_ORN2_B32]        = { AMD_FMT_SOP2, 0x24, "s_orn2_b32"        },
    [AMD_S_BFE_I32]         = { AMD_FMT_SOP2, 0x27, "s_bfe_i32"         },
    [AMD_S_CSELECT_B32]     = { AMD_FMT_SOP2, 0x30, "s_cselect_b32"     },

    /* SOP1 */
    [AMD_S_MOV_B32]         = { AMD_FMT_SOP1, 0x00, "s_mov_b32"         },
    [AMD_S_NOT_B32]         = { AMD_FMT_SOP1, 0x1E, "s_not_b32"         },
    [AMD_S_AND_SAVEEXEC_B32]= { AMD_FMT_SOP1, 0x20, "s_and_saveexec_b32"},
    [AMD_S_SETPC_B64]       = { AMD_FMT_SOP1, 0x48, "s_setpc_b64"       },
    [AMD_S_SWAPPC_B64]      = { AMD_FMT_SOP1, 0x49, "s_swappc_b64"      },
    [AMD_S_GETPC_B64]       = { AMD_FMT_SOP1, 0x47, "s_getpc_b64"       },

    /* SOPC */
    [AMD_S_CMP_EQ_U32]      = { AMD_FMT_SOPC, 0x06, "s_cmp_eq_u32"      },
    [AMD_S_CMP_NE_U32]      = { AMD_FMT_SOPC, 0x07, "s_cmp_lg_u32"      },
    [AMD_S_CMP_LT_U32]      = { AMD_FMT_SOPC, 0x0A, "s_cmp_lt_u32"      },
    [AMD_S_CMP_LE_U32]      = { AMD_FMT_SOPC, 0x0B, "s_cmp_le_u32"      },
    [AMD_S_CMP_GT_U32]      = { AMD_FMT_SOPC, 0x08, "s_cmp_gt_u32"      },
    [AMD_S_CMP_GE_U32]      = { AMD_FMT_SOPC, 0x09, "s_cmp_ge_u32"      },
    [AMD_S_CMP_LT_I32]      = { AMD_FMT_SOPC, 0x04, "s_cmp_lt_i32"      },
    [AMD_S_CMP_LE_I32]      = { AMD_FMT_SOPC, 0x05, "s_cmp_le_i32"      },
    [AMD_S_CMP_GT_I32]      = { AMD_FMT_SOPC, 0x02, "s_cmp_gt_i32"      },
    [AMD_S_CMP_GE_I32]      = { AMD_FMT_SOPC, 0x03, "s_cmp_ge_i32"      },
    [AMD_S_CMP_EQ_I32]      = { AMD_FMT_SOPC, 0x00, "s_cmp_eq_i32"      },
    [AMD_S_CMP_NE_I32]      = { AMD_FMT_SOPC, 0x01, "s_cmp_lg_i32"      },

    /* SOPP */
    [AMD_S_BRANCH]           = { AMD_FMT_SOPP, 0x20, "s_branch"           },
    [AMD_S_CBRANCH_SCC0]     = { AMD_FMT_SOPP, 0x21, "s_cbranch_scc0"     },
    [AMD_S_CBRANCH_SCC1]     = { AMD_FMT_SOPP, 0x22, "s_cbranch_scc1"     },
    [AMD_S_CBRANCH_EXECZ]    = { AMD_FMT_SOPP, 0x25, "s_cbranch_execz"    },
    [AMD_S_CBRANCH_EXECNZ]   = { AMD_FMT_SOPP, 0x26, "s_cbranch_execnz"   },
    [AMD_S_ENDPGM]           = { AMD_FMT_SOPP, 0x30, "s_endpgm"           },
    [AMD_S_BARRIER]          = { AMD_FMT_SOPP, 0x3D, "s_barrier"          },
    [AMD_S_TRAP]             = { AMD_FMT_SOPP, 0x10, "s_trap"             },
    [AMD_S_WAITCNT]          = { AMD_FMT_SOPP, 0x09, "s_waitcnt"          },
    [AMD_S_NOP]              = { AMD_FMT_SOPP, 0x00, "s_nop"              },
    [AMD_S_WAIT_LOADCNT]     = { AMD_FMT_SOPP, 0x40, "s_wait_loadcnt"    },
    [AMD_S_WAIT_STORECNT]    = { AMD_FMT_SOPP, 0x41, "s_wait_storecnt"   },
    [AMD_S_WAIT_DSCNT]       = { AMD_FMT_SOPP, 0x46, "s_wait_dscnt"     },
    [AMD_S_WAIT_KMCNT]       = { AMD_FMT_SOPP, 0x47, "s_wait_kmcnt"     },

    /* SMEM — GFX11: prefix=0x3D, OP at [25:18] */
    [AMD_S_LOAD_DWORD]       = { AMD_FMT_SMEM, 0x00, "s_load_dword"       },
    [AMD_S_LOAD_DWORDX2]     = { AMD_FMT_SMEM, 0x01, "s_load_dwordx2"     },
    [AMD_S_LOAD_DWORDX4]     = { AMD_FMT_SMEM, 0x02, "s_load_dwordx4"     },

    /* VOP2 */
    [AMD_V_ADD_U32]          = { AMD_FMT_VOP2, 0x25, "v_add_u32"          },
    [AMD_V_SUB_U32]          = { AMD_FMT_VOP2, 0x26, "v_sub_u32"          },
    [AMD_V_MUL_LO_U32]      = { AMD_FMT_VOP3, 0x32C,"v_mul_lo_u32"       },
    [AMD_V_MUL_HI_U32]      = { AMD_FMT_VOP3, 0x32D,"v_mul_hi_u32"       },
    [AMD_V_AND_B32]          = { AMD_FMT_VOP2, 0x1B, "v_and_b32"          },
    [AMD_V_OR_B32]           = { AMD_FMT_VOP2, 0x1C, "v_or_b32"           },
    [AMD_V_XOR_B32]          = { AMD_FMT_VOP2, 0x1D, "v_xor_b32"          },
    [AMD_V_LSHLREV_B32]      = { AMD_FMT_VOP2, 0x18, "v_lshlrev_b32"      },
    [AMD_V_LSHRREV_B32]      = { AMD_FMT_VOP2, 0x19, "v_lshrrev_b32"      },
    [AMD_V_ASHRREV_I32]      = { AMD_FMT_VOP2, 0x1A, "v_ashrrev_i32"      },
    [AMD_V_ADD_F32]          = { AMD_FMT_VOP2, 0x03, "v_add_f32"          },
    [AMD_V_SUB_F32]          = { AMD_FMT_VOP2, 0x04, "v_sub_f32"          },
    [AMD_V_MUL_F32]          = { AMD_FMT_VOP2, 0x08, "v_mul_f32"          },
    [AMD_V_CNDMASK_B32]      = { AMD_FMT_VOP2, 0x01, "v_cndmask_b32"      },
    [AMD_V_MIN_F32]          = { AMD_FMT_VOP2, 0x0F, "v_min_f32"          },
    [AMD_V_MAX_F32]          = { AMD_FMT_VOP2, 0x10, "v_max_f32"          },
    [AMD_V_MIN_U32]          = { AMD_FMT_VOP2, 0x13, "v_min_u32"          },

    /* VOP1 */
    [AMD_V_MOV_B32]          = { AMD_FMT_VOP1, 0x01, "v_mov_b32"          },
    [AMD_V_CVT_F32_I32]      = { AMD_FMT_VOP1, 0x05, "v_cvt_f32_i32"      },
    [AMD_V_CVT_F32_U32]      = { AMD_FMT_VOP1, 0x06, "v_cvt_f32_u32"      },
    [AMD_V_CVT_I32_F32]      = { AMD_FMT_VOP1, 0x08, "v_cvt_i32_f32"      },
    [AMD_V_CVT_U32_F32]      = { AMD_FMT_VOP1, 0x07, "v_cvt_u32_f32"      },
    [AMD_V_CVT_F32_F16]      = { AMD_FMT_VOP1, 0x0B, "v_cvt_f32_f16"      },
    [AMD_V_CVT_F16_F32]      = { AMD_FMT_VOP1, 0x0A, "v_cvt_f16_f32"      },
    [AMD_V_CVT_F64_F32]      = { AMD_FMT_VOP1, 0x10, "v_cvt_f64_f32"      },
    [AMD_V_CVT_F32_F64]      = { AMD_FMT_VOP1, 0x0F, "v_cvt_f32_f64"      },
    [AMD_V_CVT_F32_BF16]     = { AMD_FMT_VOP1, 0x72, "v_cvt_f32_bf16"    },
    [AMD_V_RCP_F32]          = { AMD_FMT_VOP1, 0x2A, "v_rcp_f32"          },
    [AMD_V_SQRT_F32]         = { AMD_FMT_VOP1, 0x33, "v_sqrt_f32"         },
    [AMD_V_RSQ_F32]          = { AMD_FMT_VOP1, 0x2E, "v_rsq_f32"          },
    [AMD_V_EXP_F32]          = { AMD_FMT_VOP1, 0x25, "v_exp_f32"          },
    [AMD_V_LOG_F32]          = { AMD_FMT_VOP1, 0x27, "v_log_f32"          },
    [AMD_V_SIN_F32]          = { AMD_FMT_VOP1, 0x35, "v_sin_f32"          },
    [AMD_V_COS_F32]          = { AMD_FMT_VOP1, 0x36, "v_cos_f32"          },
    [AMD_V_FLOOR_F32]        = { AMD_FMT_VOP1, 0x24, "v_floor_f32"        },
    [AMD_V_CEIL_F32]         = { AMD_FMT_VOP1, 0x22, "v_ceil_f32"         },
    [AMD_V_TRUNC_F32]        = { AMD_FMT_VOP1, 0x21, "v_trunc_f32"        },
    [AMD_V_RNDNE_F32]        = { AMD_FMT_VOP1, 0x23, "v_rndne_f32"        },
    [AMD_V_FRACT_F32]        = { AMD_FMT_VOP1, 0x20, "v_fract_f32"        },
    [AMD_V_NOT_B32]          = { AMD_FMT_VOP1, 0x37, "v_not_b32"          },
    [AMD_V_READFIRSTLANE_B32]= { AMD_FMT_VOP1, 0x02, "v_readfirstlane_b32"},
    [AMD_V_BFREV_B32]        = { AMD_FMT_VOP1, 0x38, "v_bfrev_b32"        },
    [AMD_V_FFBH_U32]         = { AMD_FMT_VOP1, 0x39, "v_clz_i32_u32"      },
    [AMD_V_FFBL_B32]         = { AMD_FMT_VOP1, 0x3A, "v_ctz_i32_b32"      },

    /* VOP3 — GFX11: prefix=0x35, 10-bit opcodes */
    [AMD_V_BCNT_U32_B32]     = { AMD_FMT_VOP3, 0x31E,"v_bcnt_u32_b32"     },
    [AMD_V_MAD_U32_U24]      = { AMD_FMT_VOP3, 0x20B,"v_mad_u32_u24"      },
    [AMD_V_BFE_I32]          = { AMD_FMT_VOP3, 0x211,"v_bfe_i32"          },
    [AMD_V_BFE_U32]          = { AMD_FMT_VOP3, 0x210,"v_bfe_u32"          },
    [AMD_V_LSHL_ADD_U32]     = { AMD_FMT_VOP3, 0x246,"v_lshl_add_u32"     },
    [AMD_V_ADD3_U32]         = { AMD_FMT_VOP3, 0x255,"v_add3_u32"         },

    /* VOPC — GFX11: OP at [24:17], 8-bit field */
    [AMD_V_CMP_EQ_U32]       = { AMD_FMT_VOPC, 0x4A, "v_cmp_eq_u32"       },
    [AMD_V_CMP_NE_U32]       = { AMD_FMT_VOPC, 0x4D, "v_cmp_ne_u32"       },
    [AMD_V_CMP_LT_U32]       = { AMD_FMT_VOPC, 0x49, "v_cmp_lt_u32"       },
    [AMD_V_CMP_LE_U32]       = { AMD_FMT_VOPC, 0x4B, "v_cmp_le_u32"       },
    [AMD_V_CMP_GT_U32]       = { AMD_FMT_VOPC, 0x4C, "v_cmp_gt_u32"       },
    [AMD_V_CMP_GE_U32]       = { AMD_FMT_VOPC, 0x4E, "v_cmp_ge_u32"       },
    [AMD_V_CMP_LT_I32]       = { AMD_FMT_VOPC, 0x41, "v_cmp_lt_i32"       },
    [AMD_V_CMP_LE_I32]       = { AMD_FMT_VOPC, 0x43, "v_cmp_le_i32"       },
    [AMD_V_CMP_GT_I32]       = { AMD_FMT_VOPC, 0x44, "v_cmp_gt_i32"       },
    [AMD_V_CMP_GE_I32]       = { AMD_FMT_VOPC, 0x46, "v_cmp_ge_i32"       },
    [AMD_V_CMP_EQ_I32]       = { AMD_FMT_VOPC, 0x42, "v_cmp_eq_i32"       },
    [AMD_V_CMP_NE_I32]       = { AMD_FMT_VOPC, 0x45, "v_cmp_ne_i32"       },
    [AMD_V_CMP_EQ_F32]       = { AMD_FMT_VOPC, 0x12, "v_cmp_eq_f32"       },
    [AMD_V_CMP_NE_F32]       = { AMD_FMT_VOPC, 0x1D, "v_cmp_neq_f32"      },
    [AMD_V_CMP_LT_F32]       = { AMD_FMT_VOPC, 0x11, "v_cmp_lt_f32"       },
    [AMD_V_CMP_LE_F32]       = { AMD_FMT_VOPC, 0x13, "v_cmp_le_f32"       },
    [AMD_V_CMP_GT_F32]       = { AMD_FMT_VOPC, 0x14, "v_cmp_gt_f32"       },
    [AMD_V_CMP_GE_F32]       = { AMD_FMT_VOPC, 0x16, "v_cmp_ge_f32"       },
    [AMD_V_CMP_O_F32]        = { AMD_FMT_VOPC, 0x17, "v_cmp_o_f32"        },
    [AMD_V_CMP_U_F32]        = { AMD_FMT_VOPC, 0x18, "v_cmp_u_f32"        },
    [AMD_V_CMP_NLT_F32]      = { AMD_FMT_VOPC, 0x1E, "v_cmp_nlt_f32"      },
    [AMD_V_CMP_NLE_F32]      = { AMD_FMT_VOPC, 0x1C, "v_cmp_nle_f32"      },
    [AMD_V_CMP_NGT_F32]      = { AMD_FMT_VOPC, 0x1B, "v_cmp_ngt_f32"      },
    [AMD_V_CMP_NGE_F32]      = { AMD_FMT_VOPC, 0x19, "v_cmp_nge_f32"      },
    [AMD_V_CMP_NEQ_F32]      = { AMD_FMT_VOPC, 0x1D, "v_cmp_neq_f32"      },

    /* DS */
    [AMD_DS_READ_B32]         = { AMD_FMT_DS, 0x36, "ds_load_b32"         },
    [AMD_DS_WRITE_B32]        = { AMD_FMT_DS, 0x0D, "ds_store_b32"        },
    [AMD_DS_ADD_RTN_U32]      = { AMD_FMT_DS, 0x20, "ds_add_rtn_u32"      },
    [AMD_DS_SUB_RTN_U32]      = { AMD_FMT_DS, 0x21, "ds_sub_rtn_u32"      },
    [AMD_DS_AND_RTN_B32]      = { AMD_FMT_DS, 0x29, "ds_and_rtn_b32"      },
    [AMD_DS_OR_RTN_B32]       = { AMD_FMT_DS, 0x2A, "ds_or_rtn_b32"       },
    [AMD_DS_XOR_RTN_B32]      = { AMD_FMT_DS, 0x2B, "ds_xor_rtn_b32"      },
    [AMD_DS_MIN_RTN_I32]      = { AMD_FMT_DS, 0x25, "ds_min_rtn_i32"      },
    [AMD_DS_MAX_RTN_I32]      = { AMD_FMT_DS, 0x26, "ds_max_rtn_i32"      },
    [AMD_DS_SWIZZLE_B32]      = { AMD_FMT_DS, 0x35, "ds_swizzle_b32"      },
    [AMD_DS_BPERMUTE_B32]     = { AMD_FMT_DS, 0xB3, "ds_bpermute_b32"     },

    /* FLAT_GBL — GFX11: OP at [25:18], seg at [17:16] (global=2) */
    [AMD_GLOBAL_LOAD_DWORD]      = { AMD_FMT_FLAT_GBL, 0x14, "global_load_dword"      },
    [AMD_GLOBAL_STORE_DWORD]     = { AMD_FMT_FLAT_GBL, 0x1A, "global_store_dword"     },
    [AMD_GLOBAL_LOAD_DWORDX2]    = { AMD_FMT_FLAT_GBL, 0x15, "global_load_dwordx2"    },
    [AMD_GLOBAL_STORE_DWORDX2]   = { AMD_FMT_FLAT_GBL, 0x1B, "global_store_dwordx2"   },
    [AMD_GLOBAL_ATOMIC_ADD]      = { AMD_FMT_FLAT_GBL, 0x35, "global_atomic_add"      },
    [AMD_GLOBAL_ATOMIC_SUB]      = { AMD_FMT_FLAT_GBL, 0x36, "global_atomic_sub"      },
    [AMD_GLOBAL_ATOMIC_AND]      = { AMD_FMT_FLAT_GBL, 0x3C, "global_atomic_and"      },
    [AMD_GLOBAL_ATOMIC_OR]       = { AMD_FMT_FLAT_GBL, 0x3D, "global_atomic_or"       },
    [AMD_GLOBAL_ATOMIC_XOR]      = { AMD_FMT_FLAT_GBL, 0x3E, "global_atomic_xor"      },
    [AMD_GLOBAL_ATOMIC_SMIN]     = { AMD_FMT_FLAT_GBL, 0x38, "global_atomic_smin"     },
    [AMD_GLOBAL_ATOMIC_SMAX]     = { AMD_FMT_FLAT_GBL, 0x3A, "global_atomic_smax"     },
    [AMD_GLOBAL_ATOMIC_SWAP]     = { AMD_FMT_FLAT_GBL, 0x33, "global_atomic_swap"     },
    [AMD_GLOBAL_ATOMIC_CMPSWAP]  = { AMD_FMT_FLAT_GBL, 0x34, "global_atomic_cmpswap"  },

    /* FLAT_SCR — GFX11: seg=1 (scratch) */
    [AMD_SCRATCH_LOAD_DWORD]     = { AMD_FMT_FLAT_SCR, 0x14, "scratch_load_dword"     },
    [AMD_SCRATCH_STORE_DWORD]    = { AMD_FMT_FLAT_SCR, 0x1A, "scratch_store_dword"    },

    /* FLAT — not used on RDNA, CDNA only */
    [AMD_FLAT_LOAD_DWORD]        = { AMD_FMT_FLAT, 0x00, NULL },
    [AMD_FLAT_STORE_DWORD]       = { AMD_FMT_FLAT, 0x00, NULL },

    /* Pseudo */
    [AMD_PSEUDO_PHI]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_PHI"  },
    [AMD_PSEUDO_COPY]            = { AMD_FMT_PSEUDO, 0, "PSEUDO_COPY" },
    [AMD_PSEUDO_DEF]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_DEF"  },
};

/* ---- GFX10 Encoding Table (gfx1030, RDNA 2) ---- */

const amd_enc_entry_t amd_enc_table_gfx10[AMD_OP_COUNT] = {
    /* SOP2 — extensively renumbered from GFX11 */
    [AMD_S_ADD_U32]         = { AMD_FMT_SOP2, 0x00, "s_add_u32"         },
    [AMD_S_ADD_I32]         = { AMD_FMT_SOP2, 0x02, "s_add_i32"         },
    [AMD_S_SUB_U32]         = { AMD_FMT_SOP2, 0x01, "s_sub_u32"         },
    [AMD_S_MUL_I32]         = { AMD_FMT_SOP2, 0x24, "s_mul_i32"         },
    [AMD_S_AND_B32]         = { AMD_FMT_SOP2, 0x0E, "s_and_b32"         },
    [AMD_S_OR_B32]          = { AMD_FMT_SOP2, 0x0F, "s_or_b32"          },
    [AMD_S_XOR_B32]         = { AMD_FMT_SOP2, 0x10, "s_xor_b32"         },
    [AMD_S_LSHL_B32]        = { AMD_FMT_SOP2, 0x08, "s_lshl_b32"        },
    [AMD_S_LSHR_B32]        = { AMD_FMT_SOP2, 0x09, "s_lshr_b32"        },
    [AMD_S_ASHR_I32]        = { AMD_FMT_SOP2, 0x0A, "s_ashr_i32"        },
    [AMD_S_ANDN2_B32]       = { AMD_FMT_SOP2, 0x12, "s_andn2_b32"       },
    [AMD_S_ORN2_B32]        = { AMD_FMT_SOP2, 0x13, "s_orn2_b32"        },
    [AMD_S_BFE_I32]         = { AMD_FMT_SOP2, 0x28, "s_bfe_i32"         },
    [AMD_S_CSELECT_B32]     = { AMD_FMT_SOP2, 0x0A, "s_cselect_b32"     },

    /* SOP1 — all renumbered */
    [AMD_S_MOV_B32]         = { AMD_FMT_SOP1, 0x03, "s_mov_b32"         },
    [AMD_S_NOT_B32]         = { AMD_FMT_SOP1, 0x07, "s_not_b32"         },
    [AMD_S_AND_SAVEEXEC_B32]= { AMD_FMT_SOP1, 0x21, "s_and_saveexec_b32"},
    [AMD_S_SETPC_B64]       = { AMD_FMT_SOP1, 0x1D, "s_setpc_b64"       },
    [AMD_S_SWAPPC_B64]      = { AMD_FMT_SOP1, 0x1E, "s_swappc_b64"      },
    [AMD_S_GETPC_B64]       = { AMD_FMT_SOP1, 0x1C, "s_getpc_b64"       },

    /* SOPC — GFX9: NE is "lg" (less-or-greater) */
    [AMD_S_CMP_EQ_U32]      = { AMD_FMT_SOPC, 0x06, "s_cmp_eq_u32"      },
    [AMD_S_CMP_NE_U32]      = { AMD_FMT_SOPC, 0x07, "s_cmp_lg_u32"      },
    [AMD_S_CMP_LT_U32]      = { AMD_FMT_SOPC, 0x0A, "s_cmp_lt_u32"      },
    [AMD_S_CMP_LE_U32]      = { AMD_FMT_SOPC, 0x0B, "s_cmp_le_u32"      },
    [AMD_S_CMP_GT_U32]      = { AMD_FMT_SOPC, 0x08, "s_cmp_gt_u32"      },
    [AMD_S_CMP_GE_U32]      = { AMD_FMT_SOPC, 0x09, "s_cmp_ge_u32"      },
    [AMD_S_CMP_LT_I32]      = { AMD_FMT_SOPC, 0x04, "s_cmp_lt_i32"      },
    [AMD_S_CMP_LE_I32]      = { AMD_FMT_SOPC, 0x05, "s_cmp_le_i32"      },
    [AMD_S_CMP_GT_I32]      = { AMD_FMT_SOPC, 0x02, "s_cmp_gt_i32"      },
    [AMD_S_CMP_GE_I32]      = { AMD_FMT_SOPC, 0x03, "s_cmp_ge_i32"      },
    [AMD_S_CMP_EQ_I32]      = { AMD_FMT_SOPC, 0x00, "s_cmp_eq_i32"      },
    [AMD_S_CMP_NE_I32]      = { AMD_FMT_SOPC, 0x01, "s_cmp_lg_i32"      },

    /* SOPP — completely renumbered */
    [AMD_S_BRANCH]           = { AMD_FMT_SOPP, 0x02, "s_branch"           },
    [AMD_S_CBRANCH_SCC0]     = { AMD_FMT_SOPP, 0x04, "s_cbranch_scc0"     },
    [AMD_S_CBRANCH_SCC1]     = { AMD_FMT_SOPP, 0x05, "s_cbranch_scc1"     },
    [AMD_S_CBRANCH_EXECZ]    = { AMD_FMT_SOPP, 0x08, "s_cbranch_execz"    },
    [AMD_S_CBRANCH_EXECNZ]   = { AMD_FMT_SOPP, 0x09, "s_cbranch_execnz"   },
    [AMD_S_ENDPGM]           = { AMD_FMT_SOPP, 0x01, "s_endpgm"           },
    [AMD_S_BARRIER]          = { AMD_FMT_SOPP, 0x0A, "s_barrier"          },
    [AMD_S_TRAP]             = { AMD_FMT_SOPP, 0x12, "s_trap"             },
    [AMD_S_WAITCNT]          = { AMD_FMT_SOPP, 0x0C, "s_waitcnt"          },
    [AMD_S_NOP]              = { AMD_FMT_SOPP, 0x00, "s_nop"              },
    [AMD_S_WAIT_LOADCNT]     = { AMD_FMT_SOPP, 0x00, NULL                 },
    [AMD_S_WAIT_STORECNT]    = { AMD_FMT_SOPP, 0x00, NULL                 },
    [AMD_S_WAIT_DSCNT]       = { AMD_FMT_SOPP, 0x00, NULL                 },
    [AMD_S_WAIT_KMCNT]       = { AMD_FMT_SOPP, 0x00, NULL                 },

    /* SMEM — opcodes same, encoding structure differs (handled in encode.c) */
    [AMD_S_LOAD_DWORD]       = { AMD_FMT_SMEM, 0x00, "s_load_dword"       },
    [AMD_S_LOAD_DWORDX2]     = { AMD_FMT_SMEM, 0x01, "s_load_dwordx2"     },
    [AMD_S_LOAD_DWORDX4]     = { AMD_FMT_SMEM, 0x02, "s_load_dwordx4"     },

    /* VOP2 — float ops stable, int/logic shifted */
    [AMD_V_ADD_U32]          = { AMD_FMT_VOP2, 0x25, "v_add_u32"          },
    [AMD_V_SUB_U32]          = { AMD_FMT_VOP2, 0x26, "v_sub_u32"          },
    [AMD_V_MUL_LO_U32]      = { AMD_FMT_VOP3, 0x169,"v_mul_lo_u32"       },
    [AMD_V_MUL_HI_U32]      = { AMD_FMT_VOP3, 0x16A,"v_mul_hi_u32"       },
    [AMD_V_AND_B32]          = { AMD_FMT_VOP2, 0x15, "v_and_b32"          },
    [AMD_V_OR_B32]           = { AMD_FMT_VOP2, 0x16, "v_or_b32"           },
    [AMD_V_XOR_B32]          = { AMD_FMT_VOP2, 0x17, "v_xor_b32"          },
    [AMD_V_LSHLREV_B32]      = { AMD_FMT_VOP2, 0x12, "v_lshlrev_b32"      },
    [AMD_V_LSHRREV_B32]      = { AMD_FMT_VOP2, 0x13, "v_lshrrev_b32"      },
    [AMD_V_ASHRREV_I32]      = { AMD_FMT_VOP2, 0x14, "v_ashrrev_i32"      },
    [AMD_V_ADD_F32]          = { AMD_FMT_VOP2, 0x03, "v_add_f32"          },
    [AMD_V_SUB_F32]          = { AMD_FMT_VOP2, 0x04, "v_sub_f32"          },
    [AMD_V_MUL_F32]          = { AMD_FMT_VOP2, 0x08, "v_mul_f32"          },
    [AMD_V_CNDMASK_B32]      = { AMD_FMT_VOP2, 0x00, "v_cndmask_b32"      },
    [AMD_V_MIN_F32]          = { AMD_FMT_VOP2, 0x0F, "v_min_f32"          },
    [AMD_V_MAX_F32]          = { AMD_FMT_VOP2, 0x10, "v_max_f32"          },
    [AMD_V_MIN_U32]          = { AMD_FMT_VOP2, 0x13, "v_min_u32"          },

    /* VOP1 — largely stable between GFX10 and GFX11 */
    [AMD_V_MOV_B32]          = { AMD_FMT_VOP1, 0x01, "v_mov_b32"          },
    [AMD_V_CVT_F32_I32]      = { AMD_FMT_VOP1, 0x05, "v_cvt_f32_i32"      },
    [AMD_V_CVT_F32_U32]      = { AMD_FMT_VOP1, 0x06, "v_cvt_f32_u32"      },
    [AMD_V_CVT_I32_F32]      = { AMD_FMT_VOP1, 0x08, "v_cvt_i32_f32"      },
    [AMD_V_CVT_U32_F32]      = { AMD_FMT_VOP1, 0x07, "v_cvt_u32_f32"      },
    [AMD_V_CVT_F32_F16]      = { AMD_FMT_VOP1, 0x0B, "v_cvt_f32_f16"      },
    [AMD_V_CVT_F16_F32]      = { AMD_FMT_VOP1, 0x0A, "v_cvt_f16_f32"      },
    [AMD_V_CVT_F64_F32]      = { AMD_FMT_VOP1, 0x10, "v_cvt_f64_f32"      },
    [AMD_V_CVT_F32_F64]      = { AMD_FMT_VOP1, 0x0F, "v_cvt_f32_f64"      },
    [AMD_V_RCP_F32]          = { AMD_FMT_VOP1, 0x2A, "v_rcp_f32"          },
    [AMD_V_SQRT_F32]         = { AMD_FMT_VOP1, 0x33, "v_sqrt_f32"         },
    [AMD_V_RSQ_F32]          = { AMD_FMT_VOP1, 0x2E, "v_rsq_f32"          },
    [AMD_V_EXP_F32]          = { AMD_FMT_VOP1, 0x25, "v_exp_f32"          },
    [AMD_V_LOG_F32]          = { AMD_FMT_VOP1, 0x27, "v_log_f32"          },
    [AMD_V_SIN_F32]          = { AMD_FMT_VOP1, 0x35, "v_sin_f32"          },
    [AMD_V_COS_F32]          = { AMD_FMT_VOP1, 0x36, "v_cos_f32"          },
    [AMD_V_FLOOR_F32]        = { AMD_FMT_VOP1, 0x24, "v_floor_f32"        },
    [AMD_V_CEIL_F32]         = { AMD_FMT_VOP1, 0x22, "v_ceil_f32"         },
    [AMD_V_TRUNC_F32]        = { AMD_FMT_VOP1, 0x21, "v_trunc_f32"        },
    [AMD_V_RNDNE_F32]        = { AMD_FMT_VOP1, 0x23, "v_rndne_f32"        },
    [AMD_V_FRACT_F32]        = { AMD_FMT_VOP1, 0x20, "v_fract_f32"        },
    [AMD_V_NOT_B32]          = { AMD_FMT_VOP1, 0x37, "v_not_b32"          },
    [AMD_V_READFIRSTLANE_B32]= { AMD_FMT_VOP1, 0x02, "v_readfirstlane_b32"},
    [AMD_V_BFREV_B32]        = { AMD_FMT_VOP1, 0x38, "v_bfrev_b32"        },
    [AMD_V_FFBH_U32]         = { AMD_FMT_VOP1, 0x39, "v_ffbh_u32"         },
    [AMD_V_FFBL_B32]         = { AMD_FMT_VOP1, 0x3A, "v_ffbl_b32"         },

    /* VOP3 — GFX10: prefix=0x34, 10-bit opcodes all renumbered */
    [AMD_V_BCNT_U32_B32]     = { AMD_FMT_VOP3, 0x364,"v_bcnt_u32_b32"     },
    [AMD_V_MAD_U32_U24]      = { AMD_FMT_VOP3, 0x1C3,"v_mad_u32_u24"      },
    [AMD_V_BFE_I32]          = { AMD_FMT_VOP3, 0x1C5,"v_bfe_i32"          },
    [AMD_V_BFE_U32]          = { AMD_FMT_VOP3, 0x1C4,"v_bfe_u32"          },
    [AMD_V_LSHL_ADD_U32]     = { AMD_FMT_VOP3, 0x1FC,"v_lshl_add_u32"     },
    [AMD_V_ADD3_U32]         = { AMD_FMT_VOP3, 0x1FF,"v_add3_u32"         },

    /* VOPC — GCN heritage: float=0x00+, int-signed=0x80+, int-unsigned=0xC0+ */
    [AMD_V_CMP_EQ_U32]       = { AMD_FMT_VOPC, 0xC2, "v_cmp_eq_u32"       },
    [AMD_V_CMP_NE_U32]       = { AMD_FMT_VOPC, 0xC5, "v_cmp_lg_u32"       },
    [AMD_V_CMP_LT_U32]       = { AMD_FMT_VOPC, 0xC1, "v_cmp_lt_u32"       },
    [AMD_V_CMP_LE_U32]       = { AMD_FMT_VOPC, 0xC3, "v_cmp_le_u32"       },
    [AMD_V_CMP_GT_U32]       = { AMD_FMT_VOPC, 0xC4, "v_cmp_gt_u32"       },
    [AMD_V_CMP_GE_U32]       = { AMD_FMT_VOPC, 0xC6, "v_cmp_ge_u32"       },
    [AMD_V_CMP_LT_I32]       = { AMD_FMT_VOPC, 0x81, "v_cmp_lt_i32"       },
    [AMD_V_CMP_LE_I32]       = { AMD_FMT_VOPC, 0x83, "v_cmp_le_i32"       },
    [AMD_V_CMP_GT_I32]       = { AMD_FMT_VOPC, 0x84, "v_cmp_gt_i32"       },
    [AMD_V_CMP_GE_I32]       = { AMD_FMT_VOPC, 0x86, "v_cmp_ge_i32"       },
    [AMD_V_CMP_EQ_I32]       = { AMD_FMT_VOPC, 0x82, "v_cmp_eq_i32"       },
    [AMD_V_CMP_NE_I32]       = { AMD_FMT_VOPC, 0x85, "v_cmp_lg_i32"       },
    [AMD_V_CMP_EQ_F32]       = { AMD_FMT_VOPC, 0x02, "v_cmp_eq_f32"       },
    [AMD_V_CMP_NE_F32]       = { AMD_FMT_VOPC, 0x0D, "v_cmp_neq_f32"      },
    [AMD_V_CMP_LT_F32]       = { AMD_FMT_VOPC, 0x01, "v_cmp_lt_f32"       },
    [AMD_V_CMP_LE_F32]       = { AMD_FMT_VOPC, 0x03, "v_cmp_le_f32"       },
    [AMD_V_CMP_GT_F32]       = { AMD_FMT_VOPC, 0x04, "v_cmp_gt_f32"       },
    [AMD_V_CMP_GE_F32]       = { AMD_FMT_VOPC, 0x06, "v_cmp_ge_f32"       },
    [AMD_V_CMP_O_F32]        = { AMD_FMT_VOPC, 0x07, "v_cmp_o_f32"        },
    [AMD_V_CMP_U_F32]        = { AMD_FMT_VOPC, 0x08, "v_cmp_u_f32"        },
    [AMD_V_CMP_NLT_F32]      = { AMD_FMT_VOPC, 0x0E, "v_cmp_nlt_f32"      },
    [AMD_V_CMP_NLE_F32]      = { AMD_FMT_VOPC, 0x0C, "v_cmp_nle_f32"      },
    [AMD_V_CMP_NGT_F32]      = { AMD_FMT_VOPC, 0x0B, "v_cmp_ngt_f32"      },
    [AMD_V_CMP_NGE_F32]      = { AMD_FMT_VOPC, 0x09, "v_cmp_nge_f32"      },
    [AMD_V_CMP_NEQ_F32]      = { AMD_FMT_VOPC, 0x0D, "v_cmp_neq_f32"      },

    /* DS — opcodes stable, mnemonics: read/write (not load/store) */
    [AMD_DS_READ_B32]         = { AMD_FMT_DS, 0x36, "ds_read_b32"         },
    [AMD_DS_WRITE_B32]        = { AMD_FMT_DS, 0x0D, "ds_write_b32"        },
    [AMD_DS_ADD_RTN_U32]      = { AMD_FMT_DS, 0x20, "ds_add_rtn_u32"      },
    [AMD_DS_SUB_RTN_U32]      = { AMD_FMT_DS, 0x21, "ds_sub_rtn_u32"      },
    [AMD_DS_AND_RTN_B32]      = { AMD_FMT_DS, 0x29, "ds_and_rtn_b32"      },
    [AMD_DS_OR_RTN_B32]       = { AMD_FMT_DS, 0x2A, "ds_or_rtn_b32"       },
    [AMD_DS_XOR_RTN_B32]      = { AMD_FMT_DS, 0x2B, "ds_xor_rtn_b32"      },
    [AMD_DS_MIN_RTN_I32]      = { AMD_FMT_DS, 0x25, "ds_min_rtn_i32"      },
    [AMD_DS_MAX_RTN_I32]      = { AMD_FMT_DS, 0x26, "ds_max_rtn_i32"      },
    [AMD_DS_SWIZZLE_B32]      = { AMD_FMT_DS, 0x35, "ds_swizzle_b32"      },
    [AMD_DS_BPERMUTE_B32]     = { AMD_FMT_DS, 0xB3, "ds_bpermute_b32"     },

    /* FLAT_GBL — load opcodes renumbered (0x0C/0x0D), stores same as GFX11 */
    [AMD_GLOBAL_LOAD_DWORD]      = { AMD_FMT_FLAT_GBL, 0x0C, "global_load_dword"      },
    [AMD_GLOBAL_STORE_DWORD]     = { AMD_FMT_FLAT_GBL, 0x1C, "global_store_dword"     },
    [AMD_GLOBAL_LOAD_DWORDX2]    = { AMD_FMT_FLAT_GBL, 0x0D, "global_load_dwordx2"    },
    [AMD_GLOBAL_STORE_DWORDX2]   = { AMD_FMT_FLAT_GBL, 0x1D, "global_store_dwordx2"   },
    [AMD_GLOBAL_ATOMIC_ADD]      = { AMD_FMT_FLAT_GBL, 0x32, "global_atomic_add"      },
    [AMD_GLOBAL_ATOMIC_SUB]      = { AMD_FMT_FLAT_GBL, 0x33, "global_atomic_sub"      },
    [AMD_GLOBAL_ATOMIC_AND]      = { AMD_FMT_FLAT_GBL, 0x39, "global_atomic_and"      },
    [AMD_GLOBAL_ATOMIC_OR]       = { AMD_FMT_FLAT_GBL, 0x3A, "global_atomic_or"       },
    [AMD_GLOBAL_ATOMIC_XOR]      = { AMD_FMT_FLAT_GBL, 0x3B, "global_atomic_xor"      },
    [AMD_GLOBAL_ATOMIC_SMIN]     = { AMD_FMT_FLAT_GBL, 0x35, "global_atomic_smin"     },
    [AMD_GLOBAL_ATOMIC_SMAX]     = { AMD_FMT_FLAT_GBL, 0x37, "global_atomic_smax"     },
    [AMD_GLOBAL_ATOMIC_SWAP]     = { AMD_FMT_FLAT_GBL, 0x30, "global_atomic_swap"     },
    [AMD_GLOBAL_ATOMIC_CMPSWAP]  = { AMD_FMT_FLAT_GBL, 0x31, "global_atomic_cmpswap"  },

    /* FLAT_SCR — load renumbered (0x0C), store same */
    [AMD_SCRATCH_LOAD_DWORD]     = { AMD_FMT_FLAT_SCR, 0x0C, "scratch_load_dword"     },
    [AMD_SCRATCH_STORE_DWORD]    = { AMD_FMT_FLAT_SCR, 0x1C, "scratch_store_dword"    },

    /* FLAT — not used on RDNA, CDNA only */
    [AMD_FLAT_LOAD_DWORD]        = { AMD_FMT_FLAT, 0x00, NULL },
    [AMD_FLAT_STORE_DWORD]       = { AMD_FMT_FLAT, 0x00, NULL },

    /* Pseudo */
    [AMD_PSEUDO_PHI]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_PHI"  },
    [AMD_PSEUDO_COPY]            = { AMD_FMT_PSEUDO, 0, "PSEUDO_COPY" },
    [AMD_PSEUDO_DEF]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_DEF"  },
};

/* ---- GFX9 Encoding Table (gfx90a, CDNA 2, Wave64) ---- */
/*
 * Opcodes verified by assembling with llvm-mc --mcpu=gfx90a and
 * reading the entrails. The _b64 variants exist here because Wave64
 * EXEC is 64 bits wide and we need to manipulate it without shame.
 */

const amd_enc_entry_t amd_enc_table_gfx9[AMD_OP_COUNT] = {
    /* SOP2 — GFX9 interleaves _b32/_b64 as consecutive opcodes */
    [AMD_S_ADD_U32]         = { AMD_FMT_SOP2, 0x00, "s_add_u32"         },
    [AMD_S_ADD_I32]         = { AMD_FMT_SOP2, 0x02, "s_add_i32"         },
    [AMD_S_SUB_U32]         = { AMD_FMT_SOP2, 0x01, "s_sub_u32"         },
    [AMD_S_MUL_I32]         = { AMD_FMT_SOP2, 0x24, "s_mul_i32"         },
    [AMD_S_AND_B32]         = { AMD_FMT_SOP2, 0x0C, "s_and_b32"         },
    [AMD_S_OR_B32]          = { AMD_FMT_SOP2, 0x0E, "s_or_b32"          },
    [AMD_S_XOR_B32]         = { AMD_FMT_SOP2, 0x10, "s_xor_b32"         },
    [AMD_S_LSHL_B32]        = { AMD_FMT_SOP2, 0x1C, "s_lshl_b32"        },
    [AMD_S_LSHR_B32]        = { AMD_FMT_SOP2, 0x1E, "s_lshr_b32"        },
    [AMD_S_ASHR_I32]        = { AMD_FMT_SOP2, 0x20, "s_ashr_i32"        },
    [AMD_S_ANDN2_B32]       = { AMD_FMT_SOP2, 0x12, "s_andn2_b32"       },
    [AMD_S_ORN2_B32]        = { AMD_FMT_SOP2, 0x14, "s_orn2_b32"        },
    [AMD_S_BFE_I32]         = { AMD_FMT_SOP2, 0x26, "s_bfe_i32"         },
    [AMD_S_CSELECT_B32]     = { AMD_FMT_SOP2, 0x0A, "s_cselect_b32"     },
    /* _b64 SOP2 (Wave64 EXEC manipulation) */
    [AMD_S_AND_B64]         = { AMD_FMT_SOP2, 0x0D, "s_and_b64"         },
    [AMD_S_OR_B64]          = { AMD_FMT_SOP2, 0x0F, "s_or_b64"          },
    [AMD_S_XOR_B64]         = { AMD_FMT_SOP2, 0x11, "s_xor_b64"         },
    [AMD_S_ANDN2_B64]       = { AMD_FMT_SOP2, 0x13, "s_andn2_b64"       },
    [AMD_S_ORN2_B64]        = { AMD_FMT_SOP2, 0x15, "s_orn2_b64"        },

    /* SOP1 — s_mov_b32 back at 0x00 where it belongs */
    [AMD_S_MOV_B32]         = { AMD_FMT_SOP1, 0x00, "s_mov_b32"         },
    [AMD_S_NOT_B32]         = { AMD_FMT_SOP1, 0x04, "s_not_b32"         },
    [AMD_S_AND_SAVEEXEC_B32]= { AMD_FMT_SOP1, 0x00, NULL                }, /* Wave32 only */
    [AMD_S_MOV_B64]         = { AMD_FMT_SOP1, 0x01, "s_mov_b64"         },
    [AMD_S_NOT_B64]         = { AMD_FMT_SOP1, 0x05, "s_not_b64"         },
    [AMD_S_AND_SAVEEXEC_B64]= { AMD_FMT_SOP1, 0x20, "s_and_saveexec_b64"},
    [AMD_S_SETPC_B64]       = { AMD_FMT_SOP1, 0x1D, "s_setpc_b64"       },
    [AMD_S_SWAPPC_B64]      = { AMD_FMT_SOP1, 0x1E, "s_swappc_b64"      },
    [AMD_S_GETPC_B64]       = { AMD_FMT_SOP1, 0x1C, "s_getpc_b64"       },

    /* SOPC — GFX10: NE is "lg" (less-or-greater) */
    [AMD_S_CMP_EQ_U32]      = { AMD_FMT_SOPC, 0x06, "s_cmp_eq_u32"      },
    [AMD_S_CMP_NE_U32]      = { AMD_FMT_SOPC, 0x07, "s_cmp_lg_u32"      },
    [AMD_S_CMP_LT_U32]      = { AMD_FMT_SOPC, 0x0A, "s_cmp_lt_u32"      },
    [AMD_S_CMP_LE_U32]      = { AMD_FMT_SOPC, 0x0B, "s_cmp_le_u32"      },
    [AMD_S_CMP_GT_U32]      = { AMD_FMT_SOPC, 0x08, "s_cmp_gt_u32"      },
    [AMD_S_CMP_GE_U32]      = { AMD_FMT_SOPC, 0x09, "s_cmp_ge_u32"      },
    [AMD_S_CMP_LT_I32]      = { AMD_FMT_SOPC, 0x04, "s_cmp_lt_i32"      },
    [AMD_S_CMP_LE_I32]      = { AMD_FMT_SOPC, 0x05, "s_cmp_le_i32"      },
    [AMD_S_CMP_GT_I32]      = { AMD_FMT_SOPC, 0x02, "s_cmp_gt_i32"      },
    [AMD_S_CMP_GE_I32]      = { AMD_FMT_SOPC, 0x03, "s_cmp_ge_i32"      },
    [AMD_S_CMP_EQ_I32]      = { AMD_FMT_SOPC, 0x00, "s_cmp_eq_i32"      },
    [AMD_S_CMP_NE_I32]      = { AMD_FMT_SOPC, 0x01, "s_cmp_lg_i32"      },

    /* SOPP — same as GFX10. Stable from GFX9 through the heat death */
    [AMD_S_BRANCH]           = { AMD_FMT_SOPP, 0x02, "s_branch"           },
    [AMD_S_CBRANCH_SCC0]     = { AMD_FMT_SOPP, 0x04, "s_cbranch_scc0"     },
    [AMD_S_CBRANCH_SCC1]     = { AMD_FMT_SOPP, 0x05, "s_cbranch_scc1"     },
    [AMD_S_CBRANCH_EXECZ]    = { AMD_FMT_SOPP, 0x08, "s_cbranch_execz"    },
    [AMD_S_CBRANCH_EXECNZ]   = { AMD_FMT_SOPP, 0x09, "s_cbranch_execnz"   },
    [AMD_S_ENDPGM]           = { AMD_FMT_SOPP, 0x01, "s_endpgm"           },
    [AMD_S_BARRIER]          = { AMD_FMT_SOPP, 0x0A, "s_barrier"          },
    [AMD_S_TRAP]             = { AMD_FMT_SOPP, 0x12, "s_trap"             },
    [AMD_S_WAITCNT]          = { AMD_FMT_SOPP, 0x0C, "s_waitcnt"          },
    [AMD_S_NOP]              = { AMD_FMT_SOPP, 0x00, "s_nop"              },
    [AMD_S_WAIT_LOADCNT]     = { AMD_FMT_SOPP, 0x00, NULL                 }, /* GFX12 only */
    [AMD_S_WAIT_STORECNT]    = { AMD_FMT_SOPP, 0x00, NULL                 },
    [AMD_S_WAIT_DSCNT]       = { AMD_FMT_SOPP, 0x00, NULL                 },
    [AMD_S_WAIT_KMCNT]       = { AMD_FMT_SOPP, 0x00, NULL                 },

    /* SMEM — same opcodes, different prefix (0x30 not 0x3D, handled in encode.c) */
    [AMD_S_LOAD_DWORD]       = { AMD_FMT_SMEM, 0x00, "s_load_dword"       },
    [AMD_S_LOAD_DWORDX2]     = { AMD_FMT_SMEM, 0x01, "s_load_dwordx2"     },
    [AMD_S_LOAD_DWORDX4]     = { AMD_FMT_SMEM, 0x02, "s_load_dwordx4"     },

    /* VOP2 — GFX9 added v_add_u32 (0x34) / v_sub_u32 (0x35) without carry.
       0x19/0x1A are the carry variants (v_add_co_u32) that clobber VCC. */
    [AMD_V_ADD_U32]          = { AMD_FMT_VOP2, 0x34, "v_add_u32"          },
    [AMD_V_SUB_U32]          = { AMD_FMT_VOP2, 0x35, "v_sub_u32"          },
    [AMD_V_MUL_LO_U32]      = { AMD_FMT_VOP3, 0x285,"v_mul_lo_u32"       },
    [AMD_V_MUL_HI_U32]      = { AMD_FMT_VOP3, 0x286,"v_mul_hi_u32"       },
    [AMD_V_AND_B32]          = { AMD_FMT_VOP2, 0x13, "v_and_b32"          },
    [AMD_V_OR_B32]           = { AMD_FMT_VOP2, 0x14, "v_or_b32"           },
    [AMD_V_XOR_B32]          = { AMD_FMT_VOP2, 0x15, "v_xor_b32"          },
    [AMD_V_LSHLREV_B32]      = { AMD_FMT_VOP2, 0x12, "v_lshlrev_b32"      },
    [AMD_V_LSHRREV_B32]      = { AMD_FMT_VOP2, 0x10, "v_lshrrev_b32"      },
    [AMD_V_ASHRREV_I32]      = { AMD_FMT_VOP2, 0x11, "v_ashrrev_i32"      },
    [AMD_V_ADD_F32]          = { AMD_FMT_VOP2, 0x01, "v_add_f32"          },
    [AMD_V_SUB_F32]          = { AMD_FMT_VOP2, 0x02, "v_sub_f32"          },
    [AMD_V_MUL_F32]          = { AMD_FMT_VOP2, 0x05, "v_mul_f32"          },
    [AMD_V_CNDMASK_B32]      = { AMD_FMT_VOP2, 0x00, "v_cndmask_b32"      },
    [AMD_V_MIN_F32]          = { AMD_FMT_VOP2, 0x0A, "v_min_f32"          },
    [AMD_V_MAX_F32]          = { AMD_FMT_VOP2, 0x0B, "v_max_f32"          },
    [AMD_V_MIN_U32]          = { AMD_FMT_VOP2, 0x0E, "v_min_u32"          },

    /* VOP1 — conversions stable, transcendentals renumbered */
    [AMD_V_MOV_B32]          = { AMD_FMT_VOP1, 0x01, "v_mov_b32"          },
    [AMD_V_CVT_F32_I32]      = { AMD_FMT_VOP1, 0x05, "v_cvt_f32_i32"      },
    [AMD_V_CVT_F32_U32]      = { AMD_FMT_VOP1, 0x06, "v_cvt_f32_u32"      },
    [AMD_V_CVT_I32_F32]      = { AMD_FMT_VOP1, 0x08, "v_cvt_i32_f32"      },
    [AMD_V_CVT_U32_F32]      = { AMD_FMT_VOP1, 0x07, "v_cvt_u32_f32"      },
    [AMD_V_CVT_F32_F16]      = { AMD_FMT_VOP1, 0x0B, "v_cvt_f32_f16"      },
    [AMD_V_CVT_F16_F32]      = { AMD_FMT_VOP1, 0x0A, "v_cvt_f16_f32"      },
    [AMD_V_CVT_F64_F32]      = { AMD_FMT_VOP1, 0x10, "v_cvt_f64_f32"      },
    [AMD_V_CVT_F32_F64]      = { AMD_FMT_VOP1, 0x0F, "v_cvt_f32_f64"      },
    [AMD_V_RCP_F32]          = { AMD_FMT_VOP1, 0x22, "v_rcp_f32"          },
    [AMD_V_SQRT_F32]         = { AMD_FMT_VOP1, 0x27, "v_sqrt_f32"         },
    [AMD_V_RSQ_F32]          = { AMD_FMT_VOP1, 0x24, "v_rsq_f32"          },
    [AMD_V_EXP_F32]          = { AMD_FMT_VOP1, 0x20, "v_exp_f32"          },
    [AMD_V_LOG_F32]          = { AMD_FMT_VOP1, 0x21, "v_log_f32"          },
    [AMD_V_SIN_F32]          = { AMD_FMT_VOP1, 0x29, "v_sin_f32"          },
    [AMD_V_COS_F32]          = { AMD_FMT_VOP1, 0x2A, "v_cos_f32"          },
    [AMD_V_FLOOR_F32]        = { AMD_FMT_VOP1, 0x1F, "v_floor_f32"        },
    [AMD_V_CEIL_F32]         = { AMD_FMT_VOP1, 0x1D, "v_ceil_f32"         },
    [AMD_V_TRUNC_F32]        = { AMD_FMT_VOP1, 0x1C, "v_trunc_f32"        },
    [AMD_V_RNDNE_F32]        = { AMD_FMT_VOP1, 0x1E, "v_rndne_f32"        },
    [AMD_V_FRACT_F32]        = { AMD_FMT_VOP1, 0x1B, "v_fract_f32"        },
    [AMD_V_NOT_B32]          = { AMD_FMT_VOP1, 0x2B, "v_not_b32"          },
    [AMD_V_READFIRSTLANE_B32]= { AMD_FMT_VOP1, 0x02, "v_readfirstlane_b32"},
    [AMD_V_BFREV_B32]        = { AMD_FMT_VOP1, 0x2C, "v_bfrev_b32"        },
    [AMD_V_FFBH_U32]         = { AMD_FMT_VOP1, 0x2D, "v_ffbh_u32"         },
    [AMD_V_FFBL_B32]         = { AMD_FMT_VOP1, 0x2E, "v_ffbl_b32"         },

    /* VOP3 — prefix 0x34 (not 0x35), VDST at [15:8] (handled in encode.c) */
    [AMD_V_BCNT_U32_B32]     = { AMD_FMT_VOP3, 0x28B,"v_bcnt_u32_b32"     },
    [AMD_V_MAD_U32_U24]      = { AMD_FMT_VOP3, 0x1C3,"v_mad_u32_u24"      },
    [AMD_V_BFE_I32]          = { AMD_FMT_VOP3, 0x1C9,"v_bfe_i32"          },
    [AMD_V_BFE_U32]          = { AMD_FMT_VOP3, 0x1C8,"v_bfe_u32"          },
    [AMD_V_LSHL_ADD_U32]     = { AMD_FMT_VOP3, 0x1FD,"v_lshl_add_u32"     },
    [AMD_V_ADD3_U32]         = { AMD_FMT_VOP3, 0x1FF,"v_add3_u32"         },

    /* VOPC — GCN heritage: opcodes doubled to make room for v_cmpx_ */
    [AMD_V_CMP_EQ_U32]       = { AMD_FMT_VOPC, 0xCA, "v_cmp_eq_u32"       },
    [AMD_V_CMP_NE_U32]       = { AMD_FMT_VOPC, 0xCD, "v_cmp_ne_u32"       },
    [AMD_V_CMP_LT_U32]       = { AMD_FMT_VOPC, 0xC9, "v_cmp_lt_u32"       },
    [AMD_V_CMP_LE_U32]       = { AMD_FMT_VOPC, 0xCB, "v_cmp_le_u32"       },
    [AMD_V_CMP_GT_U32]       = { AMD_FMT_VOPC, 0xCC, "v_cmp_gt_u32"       },
    [AMD_V_CMP_GE_U32]       = { AMD_FMT_VOPC, 0xCE, "v_cmp_ge_u32"       },
    [AMD_V_CMP_LT_I32]       = { AMD_FMT_VOPC, 0xC1, "v_cmp_lt_i32"       },
    [AMD_V_CMP_LE_I32]       = { AMD_FMT_VOPC, 0xC3, "v_cmp_le_i32"       },
    [AMD_V_CMP_GT_I32]       = { AMD_FMT_VOPC, 0xC4, "v_cmp_gt_i32"       },
    [AMD_V_CMP_GE_I32]       = { AMD_FMT_VOPC, 0xC6, "v_cmp_ge_i32"       },
    [AMD_V_CMP_EQ_I32]       = { AMD_FMT_VOPC, 0xC2, "v_cmp_eq_i32"       },
    [AMD_V_CMP_NE_I32]       = { AMD_FMT_VOPC, 0xC5, "v_cmp_ne_i32"       },
    [AMD_V_CMP_EQ_F32]       = { AMD_FMT_VOPC, 0x42, "v_cmp_eq_f32"       },
    [AMD_V_CMP_NE_F32]       = { AMD_FMT_VOPC, 0x4D, "v_cmp_neq_f32"      },
    [AMD_V_CMP_LT_F32]       = { AMD_FMT_VOPC, 0x41, "v_cmp_lt_f32"       },
    [AMD_V_CMP_LE_F32]       = { AMD_FMT_VOPC, 0x43, "v_cmp_le_f32"       },
    [AMD_V_CMP_GT_F32]       = { AMD_FMT_VOPC, 0x44, "v_cmp_gt_f32"       },
    [AMD_V_CMP_GE_F32]       = { AMD_FMT_VOPC, 0x46, "v_cmp_ge_f32"       },
    [AMD_V_CMP_O_F32]        = { AMD_FMT_VOPC, 0x47, "v_cmp_o_f32"        },
    [AMD_V_CMP_U_F32]        = { AMD_FMT_VOPC, 0x48, "v_cmp_u_f32"        },
    [AMD_V_CMP_NLT_F32]      = { AMD_FMT_VOPC, 0x4E, "v_cmp_nlt_f32"      },
    [AMD_V_CMP_NLE_F32]      = { AMD_FMT_VOPC, 0x4C, "v_cmp_nle_f32"      },
    [AMD_V_CMP_NGT_F32]      = { AMD_FMT_VOPC, 0x4B, "v_cmp_ngt_f32"      },
    [AMD_V_CMP_NGE_F32]      = { AMD_FMT_VOPC, 0x49, "v_cmp_nge_f32"      },
    [AMD_V_CMP_NEQ_F32]      = { AMD_FMT_VOPC, 0x4D, "v_cmp_neq_f32"      },

    /* DS — read/write (not load/store). swizzle/bpermute renumbered */
    [AMD_DS_READ_B32]         = { AMD_FMT_DS, 0x36, "ds_read_b32"         },
    [AMD_DS_WRITE_B32]        = { AMD_FMT_DS, 0x0D, "ds_write_b32"        },
    [AMD_DS_ADD_RTN_U32]      = { AMD_FMT_DS, 0x20, "ds_add_rtn_u32"      },
    [AMD_DS_SUB_RTN_U32]      = { AMD_FMT_DS, 0x21, "ds_sub_rtn_u32"      },
    [AMD_DS_AND_RTN_B32]      = { AMD_FMT_DS, 0x29, "ds_and_rtn_b32"      },
    [AMD_DS_OR_RTN_B32]       = { AMD_FMT_DS, 0x2A, "ds_or_rtn_b32"       },
    [AMD_DS_XOR_RTN_B32]      = { AMD_FMT_DS, 0x2B, "ds_xor_rtn_b32"      },
    [AMD_DS_MIN_RTN_I32]      = { AMD_FMT_DS, 0x25, "ds_min_rtn_i32"      },
    [AMD_DS_MAX_RTN_I32]      = { AMD_FMT_DS, 0x26, "ds_max_rtn_i32"      },
    [AMD_DS_SWIZZLE_B32]      = { AMD_FMT_DS, 0x3D, "ds_swizzle_b32"      },
    [AMD_DS_BPERMUTE_B32]     = { AMD_FMT_DS, 0x3F, "ds_bpermute_b32"     },

    /* FLAT_GBL — loads/stores renumbered, atomics in 0x40+ range */
    [AMD_GLOBAL_LOAD_DWORD]      = { AMD_FMT_FLAT_GBL, 0x14, "global_load_dword"      },
    [AMD_GLOBAL_STORE_DWORD]     = { AMD_FMT_FLAT_GBL, 0x1C, "global_store_dword"     },
    [AMD_GLOBAL_LOAD_DWORDX2]    = { AMD_FMT_FLAT_GBL, 0x15, "global_load_dwordx2"    },
    [AMD_GLOBAL_STORE_DWORDX2]   = { AMD_FMT_FLAT_GBL, 0x1D, "global_store_dwordx2"   },
    [AMD_GLOBAL_ATOMIC_ADD]      = { AMD_FMT_FLAT_GBL, 0x42, "global_atomic_add"      },
    [AMD_GLOBAL_ATOMIC_SUB]      = { AMD_FMT_FLAT_GBL, 0x43, "global_atomic_sub"      },
    [AMD_GLOBAL_ATOMIC_AND]      = { AMD_FMT_FLAT_GBL, 0x48, "global_atomic_and"      },
    [AMD_GLOBAL_ATOMIC_OR]       = { AMD_FMT_FLAT_GBL, 0x49, "global_atomic_or"       },
    [AMD_GLOBAL_ATOMIC_XOR]      = { AMD_FMT_FLAT_GBL, 0x4A, "global_atomic_xor"      },
    [AMD_GLOBAL_ATOMIC_SMIN]     = { AMD_FMT_FLAT_GBL, 0x44, "global_atomic_smin"     },
    [AMD_GLOBAL_ATOMIC_SMAX]     = { AMD_FMT_FLAT_GBL, 0x46, "global_atomic_smax"     },
    [AMD_GLOBAL_ATOMIC_SWAP]     = { AMD_FMT_FLAT_GBL, 0x40, "global_atomic_swap"     },
    [AMD_GLOBAL_ATOMIC_CMPSWAP]  = { AMD_FMT_FLAT_GBL, 0x41, "global_atomic_cmpswap"  },

    /* FLAT_SCR — same opcodes as FLAT_GBL, differentiated by SEG field */
    [AMD_SCRATCH_LOAD_DWORD]     = { AMD_FMT_FLAT_SCR, 0x14, "scratch_load_dword"     },
    [AMD_SCRATCH_STORE_DWORD]    = { AMD_FMT_FLAT_SCR, 0x1C, "scratch_store_dword"    },

    /* FLAT — true flat (SEG=0). CDNA uses private aperture for scratch. */
    [AMD_FLAT_LOAD_DWORD]        = { AMD_FMT_FLAT, 0x14, "flat_load_dword"        },
    [AMD_FLAT_STORE_DWORD]       = { AMD_FMT_FLAT, 0x1C, "flat_store_dword"       },

    /* VOP3P-MAI — MFMA matrix ops. Opcodes from llvm-mc, format from ISA PDF.
       These only exist on CDNA, which is why they're here and not upstairs
       with the RDNA tables. RDNA users can have matrix envy in silence. */
    [AMD_V_MFMA_F32_4X4X4_F16]      = { AMD_FMT_VOP3P_MAI, 0x4A, "v_mfma_f32_4x4x4f16"      },
    [AMD_V_MFMA_F32_16X16X16_F16]   = { AMD_FMT_VOP3P_MAI, 0x4D, "v_mfma_f32_16x16x16f16"    },
    [AMD_V_MFMA_F32_32X32X8_F16]    = { AMD_FMT_VOP3P_MAI, 0x4C, "v_mfma_f32_32x32x8f16"     },
    [AMD_V_MFMA_F32_4X4X4_BF16_1K]  = { AMD_FMT_VOP3P_MAI, 0x5F, "v_mfma_f32_4x4x4bf16_1k"  },
    [AMD_V_MFMA_F32_16X16X16_BF16_1K]={ AMD_FMT_VOP3P_MAI, 0x61, "v_mfma_f32_16x16x16bf16_1k"},
    [AMD_V_MFMA_F32_32X32X8_BF16_1K]= { AMD_FMT_VOP3P_MAI, 0x60, "v_mfma_f32_32x32x8bf16_1k" },
    [AMD_V_MFMA_F32_4X4X1_F32]      = { AMD_FMT_VOP3P_MAI, 0x42, "v_mfma_f32_4x4x1f32"      },
    [AMD_V_MFMA_F32_16X16X4_F32]    = { AMD_FMT_VOP3P_MAI, 0x45, "v_mfma_f32_16x16x4f32"     },
    [AMD_V_MFMA_F32_32X32X2_F32]    = { AMD_FMT_VOP3P_MAI, 0x44, "v_mfma_f32_32x32x2f32"     },
    [AMD_V_MFMA_I32_4X4X4_I8]       = { AMD_FMT_VOP3P_MAI, 0x52, "v_mfma_i32_4x4x4i8"       },
    [AMD_V_MFMA_I32_16X16X16_I8]    = { AMD_FMT_VOP3P_MAI, 0x55, "v_mfma_i32_16x16x16i8"     },
    [AMD_V_MFMA_I32_32X32X8_I8]     = { AMD_FMT_VOP3P_MAI, 0x54, "v_mfma_i32_32x32x8i8"      },
    /* FP8/BF8 — gfx942 CDNA3. Tastes like mixed-precision sushi. */
    [AMD_V_MFMA_F32_16X16X32_FP8_FP8] = { AMD_FMT_VOP3P_MAI, 0x73, "v_mfma_f32_16x16x32_fp8_fp8" },
    [AMD_V_MFMA_F32_16X16X32_FP8_BF8] = { AMD_FMT_VOP3P_MAI, 0x72, "v_mfma_f32_16x16x32_fp8_bf8" },
    [AMD_V_MFMA_F32_16X16X32_BF8_FP8] = { AMD_FMT_VOP3P_MAI, 0x71, "v_mfma_f32_16x16x32_bf8_fp8" },
    [AMD_V_MFMA_F32_16X16X32_BF8_BF8] = { AMD_FMT_VOP3P_MAI, 0x70, "v_mfma_f32_16x16x32_bf8_bf8" },
    [AMD_V_MFMA_F32_32X32X16_FP8_FP8] = { AMD_FMT_VOP3P_MAI, 0x77, "v_mfma_f32_32x32x16_fp8_fp8" },
    [AMD_V_MFMA_F32_32X32X16_FP8_BF8] = { AMD_FMT_VOP3P_MAI, 0x76, "v_mfma_f32_32x32x16_fp8_bf8" },
    [AMD_V_MFMA_F32_32X32X16_BF8_FP8] = { AMD_FMT_VOP3P_MAI, 0x75, "v_mfma_f32_32x32x16_bf8_fp8" },
    [AMD_V_MFMA_F32_32X32X16_BF8_BF8] = { AMD_FMT_VOP3P_MAI, 0x74, "v_mfma_f32_32x32x16_bf8_bf8" },
    [AMD_V_MFMA_I32_16X16X32_I8]      = { AMD_FMT_VOP3P_MAI, 0x57, "v_mfma_i32_16x16x32_i8" },
    [AMD_V_MFMA_I32_32X32X16_I8]      = { AMD_FMT_VOP3P_MAI, 0x56, "v_mfma_i32_32x32x16_i8" },
    /* F64 — for when your matrix really needs 52 bits of mantissa */
    [AMD_V_MFMA_F64_4X4X4_F64]        = { AMD_FMT_VOP3P_MAI, 0x6F, "v_mfma_f64_4x4x4f64"         },
    [AMD_V_MFMA_F64_16X16X4_F64]      = { AMD_FMT_VOP3P_MAI, 0x6E, "v_mfma_f64_16x16x4f64"       },

    /* Pseudo */
    [AMD_PSEUDO_PHI]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_PHI"  },
    [AMD_PSEUDO_COPY]            = { AMD_FMT_PSEUDO, 0, "PSEUDO_COPY" },
    [AMD_PSEUDO_DEF]             = { AMD_FMT_PSEUDO, 0, "PSEUDO_DEF"  },
};

static const amd_mfma_t mfma_shapes[] = {
    { AMD_V_MFMA_F32_4X4X4_F16,           2, 2,  4, 0x4A, 0x4A },
    { AMD_V_MFMA_F32_16X16X16_F16,        2, 2,  4, 0x4D, 0x4D },
    { AMD_V_MFMA_F32_32X32X8_F16,         2, 2, 16, 0x4C, 0x4C },
    { AMD_V_MFMA_F32_4X4X4_BF16_1K,       2, 2,  4, 0x65, 0x5F },
    { AMD_V_MFMA_F32_16X16X16_BF16_1K,    2, 2,  4, 0x67, 0x61 },
    { AMD_V_MFMA_F32_32X32X8_BF16_1K,     2, 2, 16, 0x66, 0x60 },
    { AMD_V_MFMA_F32_4X4X1_F32,           1, 1,  4, 0x42, 0x42 },
    { AMD_V_MFMA_F32_16X16X4_F32,         1, 1,  4, 0x45, 0x45 },
    { AMD_V_MFMA_F32_32X32X2_F32,         1, 1, 16, 0x44, 0x44 },
    { AMD_V_MFMA_I32_4X4X4_I8,            1, 1,  4, 0x52, 0x52 },
    { AMD_V_MFMA_I32_16X16X16_I8,         1, 1,  4, 0x55, MFMA_NONE },
    { AMD_V_MFMA_I32_32X32X8_I8,          1, 1, 16, 0x54, MFMA_NONE },
    { AMD_V_MFMA_I32_16X16X32_I8,         2, 2,  4, MFMA_NONE, 0x57 },
    { AMD_V_MFMA_I32_32X32X16_I8,         2, 2, 16, MFMA_NONE, 0x56 },
    { AMD_V_MFMA_F32_16X16X32_FP8_FP8,    2, 2,  4, MFMA_NONE, 0x73 },
    { AMD_V_MFMA_F32_16X16X32_FP8_BF8,    2, 2,  4, MFMA_NONE, 0x72 },
    { AMD_V_MFMA_F32_16X16X32_BF8_FP8,    2, 2,  4, MFMA_NONE, 0x71 },
    { AMD_V_MFMA_F32_16X16X32_BF8_BF8,    2, 2,  4, MFMA_NONE, 0x70 },
    { AMD_V_MFMA_F32_32X32X16_FP8_FP8,    2, 2, 16, MFMA_NONE, 0x77 },
    { AMD_V_MFMA_F32_32X32X16_FP8_BF8,    2, 2, 16, MFMA_NONE, 0x76 },
    { AMD_V_MFMA_F32_32X32X16_BF8_FP8,    2, 2, 16, MFMA_NONE, 0x75 },
    { AMD_V_MFMA_F32_32X32X16_BF8_BF8,    2, 2, 16, MFMA_NONE, 0x74 },
    { AMD_V_MFMA_F64_4X4X4_F64,           2, 2,  2, 0x6F, 0x6F },
    { AMD_V_MFMA_F64_16X16X4_F64,         2, 2,  8, 0x6E, 0x6E },
};

const amd_mfma_t *amd_mfsh(uint16_t op)
{
    for (unsigned i = 0; i < sizeof mfma_shapes / sizeof mfma_shapes[0]; i++)
        if (mfma_shapes[i].op == op) return &mfma_shapes[i];
    return NULL;
}

uint8_t amd_mfop(const amd_mfma_t *sh, int target)
{
    if (!sh) return MFMA_NONE;
    if (target == AMD_TARGET_GFX90A) return sh->op90a;
    if (target == AMD_TARGET_GFX942) return sh->op942;
    return MFMA_NONE;
}
