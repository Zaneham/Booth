#ifndef BARRACUDA_AMDGPU_ENCODE_H
#define BARRACUDA_AMDGPU_ENCODE_H

#include "amdgpu.h"

/* Binary-encode a single machine function into A->code[].
   Called by emit.c during ELF generation. */
void encode_function(amd_module_t *A, uint32_t mf_idx);

/* Return the encoding table for the current target */
const amd_enc_entry_t *get_enc_table(const amd_module_t *A);

/* Resolve one opcode against the target table, honouring GFX12 overrides */
const amd_enc_entry_t *amd_enc_ent(const amd_module_t *A, uint16_t op);

#endif /* BARRACUDA_AMDGPU_ENCODE_H */
