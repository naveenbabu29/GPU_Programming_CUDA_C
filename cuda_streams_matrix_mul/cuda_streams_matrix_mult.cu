#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <math.h>

#define TILE_SIZE 16

// CUDA kernel for row-major matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float temp = 0.0f;

    for (int sub = 0; sub < (N + TILE_SIZE - 1) / TILE_SIZE; ++sub) {
        int a_col = sub * TILE_SIZE + threadIdx.x;
        if (row < N && a_col < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = sub * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            temp += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = temp;
    }
}

// CPU matrix multiplication (row-major)
void cpuMatrixMul(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
        }
    }
}

// Transpose matrix: src (row-major) -> dst (column-major)
void transposeMatrix(const float *src, float *dst, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dst[j * N + i] = src[i * N + j]; 
        }
    }
}

bool checkCorrectness(const float *ref, const float *res, int N, float tol) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(ref[i] - res[i]) > tol) {
            printf("Mismatch at index %d: ref=%f, res=%f\n", i, ref[i], res[i]);
            return false;
        }
    }
    return true;
}

void initializeArray(float *arr, int N) {
    for (int i = 0; i < N * N; ++i) {
        arr[i] = (float)(rand() % 100);
    }
}

void runExperiment(int N) {
    size_t size = N * N * sizeof(float);

    // Host allocations
    float *h_A, *h_B, *h_C_cublas, *h_C_kernel, *h_C_cpu;
    float *h_A_col, *h_B_col;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C_cublas, size);
    cudaMallocHost(&h_C_kernel, size);
    cudaMallocHost(&h_C_cpu, size);
    cudaMallocHost(&h_A_col, size);
    cudaMallocHost(&h_B_col, size);

    initializeArray(h_A, N);
    initializeArray(h_B, N);

    // Prepare column-major copies for cuBLAS
    transposeMatrix(h_A, h_A_col, N);
    transposeMatrix(h_B, h_B_col, N);

    // Device allocations
    float *d_A_cublas, *d_B_cublas, *d_C_cublas;
    float *d_A_kernel, *d_B_kernel, *d_C_kernel;
    cudaMalloc(&d_A_cublas, size);
    cudaMalloc(&d_B_cublas, size);
    cudaMalloc(&d_C_cublas, size);
    cudaMalloc(&d_A_kernel, size);
    cudaMalloc(&d_B_kernel, size);
    cudaMalloc(&d_C_kernel, size);

    // Copy host to device
    cudaMemcpy(d_A_cublas, h_A_col, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_cublas, h_B_col, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_kernel, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_kernel, h_B, size, cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // Timing WITHOUT CUDA streams (sequential on default stream)
    cudaEvent_t start_no_stream, stop_no_stream;
    cudaEventCreate(&start_no_stream);
    cudaEventCreate(&stop_no_stream);
    cudaEventRecord(start_no_stream, 0);

    // cuBLAS multiplication
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A_cublas, N,
                d_B_cublas, N,
                &beta,
                d_C_cublas, N);
    
    cudaDeviceSynchronize();

    // Kernel launch
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulKernel<<<grid, block>>>(d_A_kernel, d_B_kernel, d_C_kernel, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop_no_stream, 0);
    cudaEventSynchronize(stop_no_stream);

    float time_no_streams = 0;
    cudaEventElapsedTime(&time_no_streams, start_no_stream, stop_no_stream);

    cudaEventDestroy(start_no_stream);
    cudaEventDestroy(stop_no_stream);

    // Timing WITH CUDA streams (concurrent execution)
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start_streams, stop_streams;
    cudaEventCreate(&start_streams);
    cudaEventCreate(&stop_streams);

    cudaEventRecord(start_streams, stream1);

    // cuBLAS multiplication on stream1
    cublasSetStream(handle, stream1);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A_cublas, N,
                d_B_cublas, N,
                &beta,
                d_C_cublas, N);

    // Kernel launch on stream2
    matrixMulKernel<<<grid, block, 0, stream2>>>(d_A_kernel, d_B_kernel, d_C_kernel, N);

    // Record stop event on default stream (to measure total elapsed time)
    cudaEventRecord(stop_streams, 0);

    // Wait for all streams to finish
    cudaEventSynchronize(stop_streams);

    float time_with_streams = 0;
    cudaEventElapsedTime(&time_with_streams, start_streams, stop_streams);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start_streams);
    cudaEventDestroy(stop_streams);

    // Reset cuBLAS stream to default
    cublasSetStream(handle, 0);

    // Copy results back to host
    cudaMemcpy(h_C_cublas, d_C_cublas, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_kernel, d_C_kernel, size, cudaMemcpyDeviceToHost);

    // Transpose cuBLAS result from column-major to row-major for comparison
    float *h_C_cublas_row = (float*)malloc(size);
    transposeMatrix(h_C_cublas, h_C_cublas_row, N);

    // CPU reference multiplication
    cpuMatrixMul(h_A, h_B, h_C_cpu, N);

    // Verify correctness
    bool cublas_correct = checkCorrectness(h_C_cpu, h_C_cublas_row, N, 1e-2f);
    bool kernel_correct = checkCorrectness(h_C_cpu, h_C_kernel, N, 1e-2f);

    printf("Matrix size: %d x %d\n", N, N);
    printf("cuBLAS result is %s compared to CPU reference.\n", cublas_correct ? "correct" : "incorrect");
    printf("Custom kernel result is %s compared to CPU reference.\n", kernel_correct ? "correct" : "incorrect");
    printf("Execution time WITHOUT CUDA streams (sequential): %.3f ms\n", time_no_streams);
    printf("Execution time WITH CUDA streams (concurrent):    %.3f ms\n", time_with_streams);

    if (time_with_streams < time_no_streams) {
        printf("CUDA streams improved performance by %.2f%%\n",
               100.0f * (time_no_streams - time_with_streams) / time_no_streams);
    } else {
        printf("CUDA streams did NOT improve performance in this case.\n");
    }

    printf("-----------------------------------------------------\n");

    // Cleanup
    free(h_C_cublas_row);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_cublas);
    cudaFreeHost(h_C_kernel);
    cudaFreeHost(h_C_cpu);
    cudaFreeHost(h_A_col);
    cudaFreeHost(h_B_col);
    cudaFree(d_A_cublas);
    cudaFree(d_B_cublas);
    cudaFree(d_C_cublas);
    cudaFree(d_A_kernel);
    cudaFree(d_B_kernel);
    cudaFree(d_C_kernel);
    cublasDestroy(handle);
}

int main() {
    srand(time(NULL));

    int sizes[] = {100, 500, 1000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; ++i) {
        runExperiment(sizes[i]);
    }

    return 0;
}
