#include <cuda.h>

#define TILE_SIZE 16

// m x k, k x n -> m x n
__global__ void opt_gemm(float *C, const float *A, const float *B, const int m,
                         const int k, const int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    // opt: memory coalescing, shared memory reuse
    if (row < m && (t * TILE_SIZE + threadIdx.x) < k) {
      A_tile[threadIdx.y][threadIdx.x] =
          A[row * k + (t * TILE_SIZE + threadIdx.x)];
    } else {
      A_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (col < n && (t * TILE_SIZE + threadIdx.y) < k) {
      B_tile[threadIdx.y][threadIdx.x] =
          B[(t * TILE_SIZE + threadIdx.y) * n + col];
    } else {
      B_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
    }
    __syncthreads()
  }

  if (row < m && col < n) {
    C[row * n + col] = sum;
  }
}