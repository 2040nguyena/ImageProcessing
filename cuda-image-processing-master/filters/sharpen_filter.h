#ifndef SHARPEN_FILTER_H
#define SHARPEN_FILTER_H

#include "../image.h"
#include "convolve.h"

int sharpen_mask_3[] = { 0, -1,  0,
                        -1,  6, -1,
                         0, -1,  0};

int sharpen_mask_dimension_3 = 3;


stbi_uc** sharpenBatchStreams(stbi_uc** input_images, int input_size, int width, int height, int channels, int GPUrank);


stbi_uc** sharpenBatchStreams(stbi_uc** input_images, int input_size, int width, int height, int channels, int GPUrank) {
    Memory memory = Global;
    stbi_uc** output_images = convolveBatch(input_images, input_size, width, height, channels, sharpen_mask_3, sharpen_mask_dimension_3, memory, true, GPUrank);

    return output_images;
}

#endif