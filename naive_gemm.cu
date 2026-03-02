#include <cuda.h>

// m x k, k x n -> m x n
__global__ void naive_gemm(float *C, const float *A, const float *B, 
                           const int m, const int k, const int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
      sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}