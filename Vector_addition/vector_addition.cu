#include <stdio.h>
#include <cuda_runtime.h>

cudaDeviceProp deviceProp;

#define START_N 1  // Start with 1 element
#define MAX_N deviceProp.maxGridSize[0] // Maximum allowed size from gridDim.x (2147483647) in grace2 Quadro RTX 600 GPU

#define START_P 1 //Start with 1 threads
#define MAX_THREADS_PER_BLOCK deviceProp.maxThreadsPerBlock

// CUDA error checking
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(result);
    }
    return result;
}

// Initialize arrays with random values
void initializeArray(float *arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
    }
}

// CUDA Kernel for Vector Addition (1 thread per block)
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Check results
inline void verifyResult(float *C, float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("ERROR at index %d: Expected %f, Found %f\n", i, A[i] + B[i], C[i]);
            exit(1);
        }
    }
}

int main() {
    
    //Device properties
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    checkCuda(err);
    
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        checkCuda(cudaGetDeviceProperties(&deviceProp, 0));
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("\nDevice %d: %s\n", device, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f MB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024));
        printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }

    int N = START_N, P = START_P;
    int threads;
    while (N <= deviceProp.maxGridSize[0]) {
        printf("\nPerforming vector addition for N = %d and P = %d\n", N, P);

        // Allocate host memory
        float *h_A = (float*)malloc(N * sizeof(float));
        float *h_B = (float*)malloc(N * sizeof(float));
        float *h_C = (float*)malloc(N * sizeof(float));

        // Initialize arrays
        initializeArray(h_A, N);
        initializeArray(h_B, N);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        checkCuda(cudaMalloc((void**)&d_A, N * sizeof(float)));
        checkCuda(cudaMalloc((void**)&d_B, N * sizeof(float)));
        checkCuda(cudaMalloc((void**)&d_C, N * sizeof(float)));

        // Copy data to device
        checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

        // Kernel launch configuration
        threads = (P > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : P;
        int blocks = (N + threads - 1) / threads; // Ensures full coverage

        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        //Launching Kernel
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
        checkCuda(cudaGetLastError());  // Check for kernel launch errors
        checkCuda(cudaDeviceSynchronize());  // Ensure execution completes

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Copy result back to host
        checkCuda(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify the result
        verifyResult(h_C, h_A, h_B, N);
        printf("Vector addition successful for N = %d and P = %d\n", N, P);

        // Compute execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time for N = %d: %.3f ms\n", N, milliseconds);

        // Free memory
        free(h_A);
        free(h_B);
        free(h_C);
        checkCuda(cudaFree(d_A));
        checkCuda(cudaFree(d_B));
        checkCuda(cudaFree(d_C));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (N > deviceProp.maxGridSize[0] / 2) 
        {
            break;
        }
        N *= 2; //Incrementing N in the power of two
    }
    
    printf("\n*************************************************************************************\n");
    N = deviceProp.maxGridSize[0] / 2;
    P = 2;
    while (N <= deviceProp.maxGridSize[0] && P <= MAX_THREADS_PER_BLOCK) {
        printf("\nPerforming vector addition for N = %d and P = %d\n", N, P);

        // Allocate host memory
        float *h_A = (float*)malloc(N * sizeof(float));
        float *h_B = (float*)malloc(N * sizeof(float));
        float *h_C = (float*)malloc(N * sizeof(float));

        // Initialize arrays
        initializeArray(h_A, N);
        initializeArray(h_B, N);

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        checkCuda(cudaMalloc((void**)&d_A, N * sizeof(float)));
        checkCuda(cudaMalloc((void**)&d_B, N * sizeof(float)));
        checkCuda(cudaMalloc((void**)&d_C, N * sizeof(float)));

        // Copy data to device
        checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

        // Kernel launch configuration
        //int blocks = N;
        //threads = P;
        
        threads = (P > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : P;
        int blocks = (N + threads - 1) / threads; // Ensures full coverage

        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        //Launching Kernel
        vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
        
        checkCuda(cudaGetLastError());  // Check for kernel launch errors
        checkCuda(cudaDeviceSynchronize());  // Ensure execution completes

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Copy result back to host
        checkCuda(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify the result
        verifyResult(h_C, h_A, h_B, N);
        printf("Vector addition successful for N = %d and P = %d\n", N, P);

        // Compute execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time for N = %d: %.3f ms\n", N, milliseconds);

        // Free memory
        free(h_A);
        free(h_B);
        free(h_C);
        checkCuda(cudaFree(d_A));
        checkCuda(cudaFree(d_B));
        checkCuda(cudaFree(d_C));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (P > MAX_THREADS_PER_BLOCK) 
        {
            break;
        }
        P *= 2;
    }

    return 0;
}
