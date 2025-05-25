#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <threads_per_block>\n", argv[0]);
        return 1;
    }
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double totalStartTime = MPI_Wtime();

    // Check GPU memory
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Dynamically calculate the maximum vector size that fits in GPU memory
    size_t requiredMemoryPerArray = sizeof(float) * 3; // A, B, and C arrays on the device (each array)
    size_t maxElements = freeMem / requiredMemoryPerArray;  // Maximum elements we can allocate

    // Adjust the size to allocate a smaller number of elements if necessary
    if (maxElements > 100000000) { // Limit the maximum size for safety
        maxElements = 100000000;
    }

    // Ensure that the vector size is a multiple of the number of processes
    int MAX_VECTOR_SIZE = (maxElements / size) * size;

    if (MAX_VECTOR_SIZE <= 0) {
        printf("Rank %d - Not enough memory to allocate even a small vector!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Partition vector
    for (int VECTOR_SIZE = 1000; VECTOR_SIZE <= MAX_VECTOR_SIZE; VECTOR_SIZE = VECTOR_SIZE*10) {
        printf("VECTOR_SIZE = %d:\n", VECTOR_SIZE);
        int elementsPerProcess = VECTOR_SIZE / size;
        int remainder = VECTOR_SIZE % size;
        int startIdx = rank * elementsPerProcess + (rank < remainder ? rank : remainder);
        int numElements = elementsPerProcess + (rank < remainder ? 1 : 0);
    
        // Allocate host memory
        float *h_A = (float*)malloc(numElements * sizeof(float));
        float *h_B = (float*)malloc(numElements * sizeof(float));
        float *h_C = (float*)malloc(numElements * sizeof(float));
    
        // Initialize A and B
        for (int i = 0; i < numElements; i++) {
            h_A[i] = startIdx + i;
            h_B[i] = (startIdx + i) * 2;
        }
    
        // Device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, numElements * sizeof(float));
        cudaMalloc((void**)&d_B, numElements * sizeof(float));
        cudaMalloc((void**)&d_C, numElements * sizeof(float));
    
        cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);
    
        // CUDA configuration
        int threadsPerBlock = atoi(argv[1]);
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
        // Kernel timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    
        // Launch the kernel
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    
        // Synchronize the device to ensure the kernel has completed
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float kernelTime = 0;
        cudaEventElapsedTime(&kernelTime, start, stop);
        printf("Rank %d - Kernel execution time: %.3f ms\n", rank, kernelTime);
        printf("Rank %d - Threads per block: %d\n", rank, threadsPerBlock);
    
        // Copy result
        cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    
        // Gather result to root
        float *globalC = NULL;
        float *globalA = NULL;
        float *globalB = NULL;
        int *recvCounts = NULL;
        int *displs = NULL;
    
        if (rank == 0) {
            globalC = (float*)malloc(VECTOR_SIZE * sizeof(float));
            globalA = (float*)malloc(VECTOR_SIZE * sizeof(float));
            globalB = (float*)malloc(VECTOR_SIZE * sizeof(float));
            recvCounts = (int*)malloc(size * sizeof(int));
            displs = (int*)malloc(size * sizeof(int));
    
            for (int i = 0; i < size; i++) {
                recvCounts[i] = elementsPerProcess + (i < remainder ? 1 : 0);
                displs[i] = i * elementsPerProcess + (i < remainder ? i : remainder);
            }
        }
    
        MPI_Gatherv(h_C, numElements, MPI_FLOAT,
                    globalC, recvCounts, displs, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    
        MPI_Gatherv(h_A, numElements, MPI_FLOAT,
                    globalA, recvCounts, displs, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    
        MPI_Gatherv(h_B, numElements, MPI_FLOAT,
                    globalB, recvCounts, displs, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    
        double totalEndTime = MPI_Wtime();
    
        // Write result & verify
        if (rank == 0) {
            int correct = 1;
            for (int i = 0; i < VECTOR_SIZE; i++) {
                float expected = globalA[i] + globalB[i];
                if (abs(globalC[i] - expected) > 1e-5) {
                    correct = 0;
                    printf("Mismatch at index %d: expected %.1f, got %.1f\n", i, expected, globalC[i]);
                    break;
                }
            }
    
            if (correct) {
                printf("Result is correct!\n");
            }
            else {
                printf("Result is incorrect!\n");
            }
    
            printf("Total Execution Time (including MPI): %.3f seconds\n\n", totalEndTime - totalStartTime);
        }
    
        // Cleanup
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C);
        free(h_A); 
        free(h_B); 
        free(h_C);
        if (rank == 0) {
            free(globalC); free(globalA); free(globalB);
            free(recvCounts); free(displs);
        }
    }

    MPI_Finalize();
    return 0;
}