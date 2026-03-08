# Practical Guide: gfx12 (RDNA4) WMMA Output Lane Mapping

## TL;DR

The `v_wmma_f32_16x16x16_f16` instruction on RDNA4 (gfx12) uses a **column-distributed fragment layout**: lanes index columns, accumulator registers index rows. This can be expressed as:

```
VGPR[lane][j] = matrix[(lane / 16) * 8 + j][lane % 16]
```

Or equivalently, from the matrix perspective:

```
matrix[row][col] -> VGPR[(row / 8) * 16 + col][row % 8]
```

No AMD guide explains this in practical terms for HIP kernel developers. If you're writing WMMA kernels for RDNA4 and getting transposed results within each 16x16 tile — you're assuming lanes index rows. They don't. This guide gives you the correct store pattern and explains why.

## Background

We were building a fused MXFP4 WMMA GEMM kernel for AMD Radeon AI PRO R9700 (gfx1201). The kernel compiled and ran without errors, but every 16x16 output tile was **transposed**: `C_gpu[i,j] == C_ref[j,i]` within each WMMA tile.

The root cause: we assumed `lane % 16 = row`, but the hardware does `lane % 16 = column`.

## The Mapping

### Lane-to-element mapping for WMMA on gfx12 (Wave32)

```
Lane i (0..31) holds:
  Column:  i % 16
  Rows:    (i / 16) * 8  ..  (i / 16) * 8 + 7
  acc[j] = element at row (i/16)*8 + j, column i%16
```

Concretely:
- **Lanes 0-15** (SubGroup 0): columns 0-15, rows 0-7
- **Lanes 16-31** (SubGroup 1): columns 0-15, rows 8-15

Full lane table:

| Lane | Column | Rows | acc[0]..acc[7] |
|------|--------|------|----------------|
| 0 | 0 | 0-7 | D[0,0] D[1,0] ... D[7,0] |
| 1 | 1 | 0-7 | D[0,1] D[1,1] ... D[7,1] |
| ... | ... | ... | ... |
| 15 | 15 | 0-7 | D[0,15] D[1,15] ... D[7,15] |
| 16 | 0 | 8-15 | D[8,0] D[9,0] ... D[15,0] |
| ... | ... | ... | ... |
| 31 | 15 | 8-15 | D[8,15] D[9,15] ... D[15,15] |

### This is a column-distributed fragment layout

Each lane "owns" one column and 8 consecutive rows. The fast-varying dimension across lanes is the **column index** (N), not the row index (M).

This is **not** "column-major" in the CUDA/CUTLASS memory layout sense. It is a **column-distributed fragment** — lanes distribute across columns, vector elements distribute across rows.

### Source: CK diagram

AMD's Composable Kernels library (`wmma_gemm.hpp`) documents this layout as an ASCII diagram:

```
WAVE32 — 16x16 output tile
    col 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
   ---------------------------------------------------
   |RC0|                                              |  SubGroup 0
   |RC1|  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T|  (lanes 0-15)
   |RC2|  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1|
   |RC3|  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5|  8 rows (RC0-RC7)
   |RC4|                                              |
   |RC5|                                              |
   |RC6|                                              |
   |RC7|                                              |
   ---------------------------------------------------
   |   |                                              |  SubGroup 1
   |   |  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T|  (lanes 16-31)
   |   |  1  1  1  1  2  2  2  2  2  2  2  2  2  2  3|
   |   |  6  7  8  9  0  1  2  3  4  5  6  7  8  9  0|  8 rows (RC0-RC7)
   |   |                                          3  |
   |   |                                          1  |
   |   |                                              |
   |   |                                              |
   ---------------------------------------------------

Threads run along N (columns). RC registers run along M (rows).
```

The diagram shows threads (T01-T15, T16-T31) spanning the column axis horizontally, while RC0-RC7 span the row axis vertically. CK's code comment confirms: *"num_thread_per_subgroups always along N direction"*.

### Algebraic derivation

From the diagram we can derive the mapping formula:

```
lane = (row / 8) * 16 + col        // matrix → VGPR index
j    = row % 8                     // matrix → register index

col      = lane % 16               // VGPR → matrix (kernel developer form)
row_base = (lane / 16) * 8
row      = row_base + j
```

We verified this algebraic mapping against our empirical results — they match exactly.

### The common mistake

Many developers assume the opposite (lane = row, acc = column):

```cpp
// WRONG for gfx12 WMMA:
int row = lane % 16;
int col_base = (lane / 16) * 8;
// acc[j] at position (row, col_base + j)  <-- WRONG, produces transposed tiles
```

The correct mapping:

```cpp
// CORRECT for gfx12 WMMA:
int col = lane % 16;
int row_base = (lane / 16) * 8;
// acc[j] at position (row_base + j, col)  <-- CORRECT
```

## Intrinsic Signatures

```cpp
typedef __attribute__((ext_vector_type(8))) _Float16 half8_t;
typedef __attribute__((ext_vector_type(8))) float    float8_t;

// FP16 inputs, FP32 accumulator
float8_t __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
    half8_t a,      // 8 FP16 values per lane (A fragment)
    half8_t b,      // 8 FP16 values per lane (B fragment)
    float8_t acc    // 8 FP32 values per lane (accumulator)
);

// INT4 inputs, INT32 accumulator
typedef int v8i __attribute__((ext_vector_type(8)));
v8i __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
    int neg_a, int a,    // 8 INT4 values packed in 1 int
    int neg_b, int b,    // 8 INT4 values packed in 1 int
    v8i acc, int clamp   // 8 INT32 accumulator values
);
```

## Correct Store Pattern

When writing WMMA results to row-major global memory:

```cpp
// After WMMA accumulation:
int lane = threadIdx.x % 32;
int out_col  = lane % 16;          // lane selects column (not row!)
int out_row0 = (lane / 16) * 8;    // lane group selects row block

#pragma unroll
for (int j = 0; j < 8; j++) {
    int r = tile_row_base + out_row0 + j;
    int c = tile_col_base + out_col;
    if (r < M && c < N)
        C[r * N + c] = (fp16_t)acc[j];  // acc[j] is at row out_row0+j, col out_col
}
```

## Applies to All WMMA Data Types

The CK library uses the same output fragment layout for all WMMA variants. We verified this empirically for FP16 and INT4:

**FP16 WMMA** (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`):
- Verified with identity matrices, small known matrices, and large matrices up to 17408x5120
- CPU FP32 reference vs GPU WMMA output, element-by-element, max relative error < 0.08%

**INT4 WMMA** (`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`):
- Verified with asymmetric matrices: `A[i][j] = (i + j*3) % 7`, `B[i][j] = (i*2 + j) % 5`
- 117/120 elements of C_ref are asymmetric (C[i][j] != C[j][i]), eliminating ambiguity
- Column-distributed store: total diff = 0 (correct); row-distributed store: total diff = 4720 (wrong)

**Lesson learned**: identity-times-constant tests cannot distinguish between row and column distribution because the output is symmetric. Always test with asymmetric matrices.

## Relationship to CDNA MFMA Instructions

The WMMA lane mapping on RDNA4 follows the **same principle** as MFMA (Matrix Fused Multiply-Add) on CDNA (MI200/MI300). From CK's `xdlops_gemm.hpp`:

**MFMA `f32_16x16x16f16`** (CDNA, Wave64):
```
num_threads_per_blk = 16, group_size = 4
blk_td = laneId % 16   ->  N offset (column)
blk_id = laneId / 16   ->  blk_id * 4 = M offset (row base)
// 4 subgroups of 16 threads, 4 rows each
```

**WMMA `f32_16x16x16f16`** (RDNA4, Wave32):
```
num_thread_per_subgroups = 16
lane % 16               ->  column (N direction)
(lane / 16) * 8          ->  row base (M direction)
// 2 subgroups of 16 threads, 8 rows each
```

Both follow the same structure: `lane % 16` selects the column, the lane group selects the row block.

The only difference is subgroup count vs rows-per-subgroup:
- MFMA Wave64: 4 subgroups x 4 rows = 16 rows, 64 lanes
- WMMA Wave32: 2 subgroups x 8 rows = 16 rows, 32 lanes

This suggests WMMA on RDNA4 shares the same underlying matrix core design as MFMA on CDNA, adapted for Wave32.

## Application: Fused MXFP4 WMMA GEMM Kernel

Understanding this mapping was a prerequisite for building what is, to our knowledge, the first fused MXFP4 WMMA GEMM kernel on consumer RDNA4 hardware:

- **Architecture**: load MXFP4 weights (4-bit E2M1 + E8M0 block scales) -> dequant via LDS-based LUT -> FP16 WMMA 16x16x16 with FP32 accumulator
- **Key design choices**: TILE_K=32 matches E8M0 block size (zero scale interpolation), LDS LUT for FP4 decode (1 read vs 6-8 ALU ops), `ldexpf` for E8M0 (1 ALU op), 128-bit vectorized A loads
- **Tiling**: 2x2 register tiling, 64x64 block, 4 warps, ~9KB shared memory
- **Performance**: 40.8 TFLOPS peak (53% of FP16 WMMA theoretical)
- **Speedup**: 3.8x faster than separate dequant + hipBLAS GEMM for batch size <= 32 (eliminates global memory round-trip for dequantized weights)
- **Correctness**: 100% across all tested matrix sizes (up to Qwen3-14B layer dimensions: 17408x5120)

## Hardware & Software

- GPU: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB)
- ROCm: 7.1.0
- Compiler: hipcc (clang-19)
- PyTorch: 2.10.0+rocm7.1

**Note**: Verified on ROCm 7.1.0 with clang-19. The lane mapping is defined by hardware, so it should be stable across ROCm versions. When in doubt, verify with a simple asymmetric matrix test.

## References

- [AMD RDNA4 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf) — WMMA instruction encoding
- [GPUOpen: Using Matrix Cores on RDNA4](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) — intrinsic signatures and fragment descriptions
- [AMD Composable Kernels](https://github.com/ROCm/composable_kernel) — `wmma_gemm.hpp` (lines 31-80) contains the authoritative lane mapping diagram; `xdlops_gemm.hpp` contains the MFMA equivalent
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) — tool for verifying lane mappings across data types

## License

This document is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt it with attribution.
