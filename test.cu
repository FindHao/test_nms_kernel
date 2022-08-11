
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <chrono>
#include <bits/stdc++.h>

#define ceil_div(x, y) (x + y - 1) / y

typedef unsigned long long ull;
using namespace std;

const int threadsPerBlock = 64;
const int dets_num = 1024*64;
const int col_blocks = ceil_div(dets_num, threadsPerBlock);

vector<ull> h_remv_cpu(col_blocks);

ull *h_mask_cpu;
ull *d_mask_cpu;
ull *h_mask_gpu;
ull *d_mask_gpu;
int64_t *h_keep_out_cpu;
int64_t *d_keep_out_cpu;
int64_t *d_keep_out_gpu;
int64_t *h_keep_out_gpu;
ull *h_remv_gpu;
ull *d_remv_gpu;
__device__ __managed__ int num_to_keep;
void generate_input()
{
  h_mask_cpu = new ull[dets_num * col_blocks];
  for (int i = 0; i < dets_num * col_blocks; i++)
  {
    h_mask_cpu[i] = rand();
  }
  h_keep_out_cpu = new int64_t[dets_num];
  h_keep_out_gpu = new int64_t[dets_num];
  h_remv_gpu = new ull[col_blocks];
  checkCudaErrors(cudaMalloc((void **)&d_remv_gpu, sizeof(ull) * col_blocks));
  // checkCudaErrors(cudaMemset(d_remv_gpu, 0, sizeof(ull) * col_blocks));
  checkCudaErrors(cudaMalloc((void **)&d_mask_gpu, sizeof(ull) * dets_num * col_blocks));
  checkCudaErrors(cudaMalloc((void **)&d_mask_cpu, sizeof(ull) * dets_num * col_blocks));
  checkCudaErrors(cudaMemcpy(d_mask_cpu, h_mask_cpu, sizeof(ull) * dets_num * col_blocks, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_mask_gpu, h_mask_cpu, sizeof(ull) * dets_num * col_blocks, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_keep_out_gpu, sizeof(int64_t) * dets_num));
  checkCudaErrors(cudaMalloc((void **)&d_keep_out_cpu, sizeof(int64_t) * dets_num));
}

__global__ void gpu_func(ull* d_remv_gpu, ull* d_mask_gpu, int64_t* d_keep_out_gpu)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  __shared__ ull s_remv_gpu[col_blocks];
  for (int i = 0; i < col_blocks; i++)
  {
    s_remv_gpu[i] = 0;
  }

  if (idx < col_blocks)
  {
    for (int i = 0; i < dets_num; i++)
    {
      int nblock = i / threadsPerBlock;
      int inblock = i % threadsPerBlock;
      // if (!(d_remv_gpu[nblock] & (1ull << inblock)))
      if (!(s_remv_gpu[nblock] & (1ull << inblock)))
      {
        if (tid == 0)
          d_keep_out_gpu[num_to_keep++] = i;
        if (tid < col_blocks )
          // d_remv_gpu[tid] |= d_mask_gpu[i * col_blocks + tid];
          s_remv_gpu[tid] |= d_mask_gpu[i * col_blocks + tid];
      }
      __syncthreads();
    }
  }
}

void cpu_func()
{
  checkCudaErrors(cudaMemcpy(h_mask_cpu, d_mask_gpu, sizeof(ull) * dets_num * col_blocks, cudaMemcpyDeviceToHost));
  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++)
  {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(h_remv_cpu[nblock] & (1ULL << inblock)))
    {
      h_keep_out_cpu[num_to_keep++] = i;
      unsigned long long *p = h_mask_cpu + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++)
      {
        h_remv_cpu[j] |= p[j];
      }
    }
  }
  checkCudaErrors(cudaMemcpy(d_keep_out_cpu, h_keep_out_cpu, sizeof(int64_t) * dets_num, cudaMemcpyHostToDevice));
}

int main()
{
  generate_input();
  auto start = chrono::high_resolution_clock::now();
  cpu_func();
  auto end = chrono::high_resolution_clock::now();
  // Calculating total time taken by the program.
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  time_taken *= 1e-6;
  cout << "Time taken by program is : " << fixed 
        << time_taken << setprecision(9);
  cout << " ms" << endl;

  start = chrono::high_resolution_clock::now();
  int block_size = col_blocks;
  dim3 threads(block_size);
  dim3 grids(1);
  gpu_func<<<grids, threads >>>(d_remv_gpu, d_mask_gpu, d_keep_out_gpu);
  cudaDeviceSynchronize();

  end = chrono::high_resolution_clock::now();
  // Calculating total time taken by the program.
  time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  time_taken *= 1e-6;
  cout << "Time taken by program is : " << fixed 
        << time_taken << setprecision(9);
  cout << " ms" << endl;

  checkCudaErrors(cudaMemcpy(h_keep_out_gpu, d_keep_out_gpu, sizeof(int64_t) * dets_num, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_remv_gpu, d_remv_gpu, sizeof(ull) * col_blocks, cudaMemcpyDeviceToHost));
  printf("%d\n", num_to_keep);
  // for (int i = 0; i < col_blocks; i++)
  // {
  //   if (h_remv_gpu[i] != h_remv_cpu[i])
  //   {
  //     printf("%d %llu %llu\n", i, h_remv_gpu[i], h_remv_cpu[i]);
  //   }
  //   // printf("%llu\n", h_remv_gpu[i]);
  // }
  for (int i = 0; i < num_to_keep; i++)
  {
    if (h_keep_out_cpu[i] != h_keep_out_gpu[i])
    {
      printf("%ld %ld\n", h_keep_out_cpu[i], h_keep_out_gpu[i]);
    }
  }
  cudaFree(d_mask_gpu);
  cudaFree(d_mask_cpu);
  cudaFree(d_keep_out_gpu);
  cudaFree(d_keep_out_cpu);
  cudaFree(d_remv_gpu);
  free(h_mask_cpu);
  delete[] h_keep_out_cpu;
  delete[] h_keep_out_gpu;
  delete[] h_remv_gpu;
  return 0;
}
