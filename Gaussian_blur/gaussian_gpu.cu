#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Include STB image library for image loading and writing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Define constants for Gaussian blur
#define KERNEL_SIZE 15
#define SIGMA 3.0f

// Declare constant memory for the Gaussian kernel
__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Kernel for basic global memory Gaussian blur
__global__ void gaussianBlurGlobal(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (x >= width || y >= height) return;
   
    int radius = KERNEL_SIZE / 2;
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        // Apply the Gaussian kernel
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                sum += input[(py * width + px) * channels + c] * d_kernel[(ky + radius) * KERNEL_SIZE + (kx + radius)];
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

// Kernel for optimized shared memory Gaussian blur
__global__ void gaussianBlurShared(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Declare shared memory tile
    __shared__ unsigned char tile[16 + KERNEL_SIZE - 1][16 + KERNEL_SIZE - 1][3];
   
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
   
    for (int c = 0; c < channels; c++) {
        // Load data into shared memory
        if (x < width && y < height)
            tile[ty][tx][c] = input[(y * width + x) * channels + c];
        __syncthreads();

        if (x >= width || y >= height) return;
       
        float sum = 0.0f;
        int radius = KERNEL_SIZE / 2;
        // Apply the Gaussian kernel using shared memory
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int sx = tx + kx;
                int sy = ty + ky;
                sx = min(max(sx, 0), blockDim.x - 1);
                sy = min(max(sy, 0), blockDim.y - 1);
                sum += tile[sy][sx][c] * d_kernel[(ky + radius) * KERNEL_SIZE + (kx + radius)];
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

// Function to generate Gaussian kernel
void generateGaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center, dy = y - center;
            kernel[y * size + x] = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            sum += kernel[y * size + x];
        }
    }
    // Normalize the kernel
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc != 6) {
        printf("Usage: %s <input_image> <output_image_global> <output_image_shared> <block_size_x> <block_size_y>\n", argv[0]);
        return 1;
    }
    
    // Load input image
    int width, height, channels;
    unsigned char* input = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!input) {
        printf("Failed to load image\n");
        return 1;
    }
    
    // Allocate device memory and copy input data
    unsigned char *d_input, *d_output_global, *d_output_shared;
    size_t img_size = width * height * channels;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output_global, img_size);
    cudaMalloc(&d_output_shared, img_size);
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);
    
    // Generate and copy Gaussian kernel to constant memory
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    generateGaussianKernel(kernel, KERNEL_SIZE, SIGMA);
    cudaMemcpyToSymbol(d_kernel, kernel, sizeof(kernel));
    
    // Set up grid and block dimensions
    int block_size_x = atoi(argv[4]);
    int block_size_y = atoi(argv[5]);
    dim3 blockSize(block_size_x, block_size_y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Run and time global memory kernel
    cudaEventRecord(start);
    gaussianBlurGlobal<<<gridSize, blockSize>>>(d_input, d_output_global, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeGlobal;
    cudaEventElapsedTime(&timeGlobal, start, stop);
    printf("Global kernel: %.3f ms\n", timeGlobal);
    
    // Run and time shared memory kernel
    size_t sharedMemSize = (block_size_x + 2 * (KERNEL_SIZE / 2)) * (block_size_y + 2 * (KERNEL_SIZE / 2)) * channels;
    cudaEventRecord(start);
    gaussianBlurShared<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output_shared, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeShared;
    cudaEventElapsedTime(&timeShared, start, stop);
    printf("Shared kernel: %.3f ms\n", timeShared);
    
    // Copy results back to host and save output images
    unsigned char* output_global = (unsigned char*)malloc(img_size);
    unsigned char* output_shared = (unsigned char*)malloc(img_size);
    cudaMemcpy(output_global, d_output_global, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_shared, d_output_shared, img_size, cudaMemcpyDeviceToHost);
    
    stbi_write_png(argv[2], width, height, channels, output_global, width * channels);
    stbi_write_png(argv[3], width, height, channels, output_shared, width * channels);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output_global);
    cudaFree(d_output_shared);
    free(output_global);
    free(output_shared);
    stbi_image_free(input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
