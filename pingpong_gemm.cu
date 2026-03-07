#include <cuda.h>

#define TILE_SIZE 16

// m x k, k x n -> m x n
__global__ void opt_gemm(float *C, const float *A, const float *B, const int m,
                         const int k, const int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float A_tile[2][TILE_SIZE][TILE_SIZE];
  __shared__ float B_tile[2][TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

  // prologue
  A_tile[0][threadIdx.y][threadIdx.x] = A[row * k + threadIdx.x];
  B_tile[0][threadIdx.y][threadIdx.x] = B[threadIdx.y * n + col];
  __syncthreads();

  for (int t = 0; t < num_tiles; t++) {
    int comp_idx = t % 2;         // cur buf
    int load_idx = (t + 1) % 2;   // next buf
    int next_idx = t + 1;         // next tile

    float a_next = 0.0f;
    float b_next = 0.0f;

    // async load the next tiles of A and B
    if (next_idx < num_tiles) {
      float a_next = A[row * k + (next_idx * TILE_SIZE + threadIdx.x)];
      float b_next = B[(next_idx * TILE_SIZE + threadIdx.y) * n + col];
    }

    for (int i = 0; i < TILE_SIZE; i++) {
      sum += A_tile[comp_idx][threadIdx.y][i] * B_tile[comp_idx][i][threadIdx.x];
    }
    __syncthreads();

    if (next_idx < num_tiles) {
      A_tile[load_idx][threadIdx.y][threadIdx.x] = a_next;
      B_tile[load_idx][threadIdx.y][threadIdx.x] = b_next;
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * n + col] = sum;
  }
}
