# CUDA GEMM Kernels

This project contains CUDA implementations of matrix multiplication (GEMM).

- `naive_gemm.cu`: a straightforward GEMM kernel
- `opt_gemm.cu`: an optimized version using tiling and shared memory
- and so on..

## 1. Problem Definition

- Input:
  - Matrix \(A\): size \(m \times k\)
  - Matrix \(B\): size \(k \times n\)
- Output:
  - Matrix \(C\): size \(m \times n\), where \(C = A \times B\)

All matrices are assumed to be row‑major `float` arrays.

## 2. Kernel Description

- **`naive_gemm`**
  - Each thread computes a single element `C[row, col]`.
  - The kernel directly reads `A[row, :]` and `B[:, col]` from global memory and performs the multiply–accumulate loop.
  - The implementation is simple but does not optimize memory reuse or access patterns.

- **`opt_gemm`**
  - Uses `TILE_SIZE × TILE_SIZE` tiles for computation.
  - Uses `__shared__` memory to cache tiles of `A` and `B` and reuse them within a block.
  - Organizes global memory accesses to be coalesced, improving bandwidth utilization.
