#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void generateGaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            kernel[y * size + x] = exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            sum += kernel[y * size + x];
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

void gaussianBlur(unsigned char* input, unsigned char* output, 
                  int width, int height, int channels,
                  float* kernel, int kernelSize) {
    int radius = kernelSize / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        
                        px = (px < 0) ? 0 : ((px >= width) ? width - 1 : px);
                        py = (py < 0) ? 0 : ((py >= height) ? height - 1 : py);
                        
                        int pixel_idx = (py * width + px) * channels + c;
                        float kernel_val = kernel[(ky + radius) * kernelSize + (kx + radius)];
                        sum += input[pixel_idx] * kernel_val;
                    }
                }
                
                output[(y * width + x) * channels + c] = (unsigned char)sum;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* input = stbi_load(argv[1], &width, &height, &channels, 0);
    
    if (!input) {
        printf("Failed to load image: %s\n", argv[1]);
        return 1;
    }

    unsigned char* output = (unsigned char*)malloc(width * height * channels);
    
    // Start timing
    clock_t start_time = clock();

    // Generate and apply Gaussian kernel
    const int kernelSize = 15;
    const float sigma = 3.0f;
    float* kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    generateGaussianKernel(kernel, kernelSize, sigma);
    
    gaussianBlur(input, output, width, height, channels, kernel, kernelSize);

    // End timing
    clock_t end_time = clock();
    
    // Calculate CPU time used
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
    
    // Save result
    stbi_write_png(argv[2], width, height, channels, output, width * channels);
    
    // Cleanup
    stbi_image_free(input);
    free(output);
    free(kernel);
    
    return 0;
}
