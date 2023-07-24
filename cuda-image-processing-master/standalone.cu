#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>

#include "image.h"

#include "filters/blur_filter.h"
#include "filters/sharpen_filter.h"
#include "filters/vertical_flip_filter.h"
#include "filters/horizontal_flip_filter.h"
#include "filters/grayscale_filter.h"
#include "filters/grayscale_weighted_filter.h"
#include "filters/edge_detection_filter.h"

//#include "filters/convolve.h"


//nvcc sharpen.cu `pkg-config opencv --cflags --libs`
//
//./a.out path_to_image_input path_to_image_output filter_arg mode_arg
//./a.out path_to_input_dir path_to_output_dir filter_arg mode_arg
//
// Single Example
//./a.out images/lena_rgb.png output/sharpen.png sharpen single
//
// Batch Example
//./a.out images/batch/ output/batch/ sharpen batch

const char* BLUR_FILTER = "blur";
const char* SHARPEN_FILTER = "sharpen";
const char* VERTICAL_FLIP_FILTER = "vflip";
const char* HORIZONTAL_FLIP_FILTER = "hflip";
const char* GRAYSCALE_FILTER = "gray";
const char* GRAYSCALE_WEIGHTED_FILTER = "grayweight";
const char* EDGE_DETECTION_FILTER = "edge";

const char* SINGLE_MODE = "single";
const char* BATCH_MODE = "batch";


int main(int argc, const char* argv[]) {
    if (argc != 5) {
        printf("Incorrect number of arguments.\n");
        return 1;
    }

    const char* path_to_input_image = argv[1];
    const char* path_to_output_image = argv[2];
    const char* filter = argv[3];
    const char* mode = argv[4];

    if (strcmp(mode, SINGLE_MODE) == 0) {
        printf("Applying filter %s to image %s.\n", filter, path_to_input_image);

        int width, height, channels;
        stbi_uc* image = loadImage(path_to_input_image, &width, &height, &channels);
        //stbi_uc** images[0] = loadImage(path_to_input_image, &width, &height, &channels);
        if (image == NULL) {
            printf("Could not load image %s.\n", path_to_input_image);
            return 1;
        }

        stbi_uc* filtered_image;
        //unsigned char** filtered_images;
        if (strcmp(filter, SHARPEN_FILTER) == 0) {
            //filtered_image = sharpen(image, width, height, channels);

            const int image_size = width * height * sizeof(stbi_uc*);

/*
            //allocate mem on device
            unsigned char* d_input_image;
            cudaMalloc(&d_input_image, image_size);
            cudaMemcpy(d_input_image, images[0], image_size, cudaMemcpyHostToDevice);

            unsigned char* d_output_image;
            cudaMalloc(&d_output_image, image_size);

            //set up grid and block sizes
            const dim3 blockSize(32, 32, 1);
            const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                                (height + blockSize.y - 1) / blockSize.y,
                                1);
*/
            filtered_image = sharpen_new(image, width, height, channels);
/*
            cudaMemcpy(filtered_images[0], d_output_image, image_size, cudaMemcpyDeviceToHost);

            cudaFree(d_input_image);
            cudaFree(d_output_image);
*/
        } else {
            printf("Invalid filter %s.\n", filter);
        }

        writeImage(path_to_output_image, filtered_image, width, height, channels);
        printf("image written and filter applied\n");
        //imageFree(image);
        //imageFree(filtered_image);


    } else if (strcmp(mode, BATCH_MODE) == 0) {

/*
        printf("in batch mode\n");
        int MAX_IMAGES = 100;
        int image_count = 0;
        //stbi_uc** images = (stbi_uc**) malloc(MAX_IMAGES * sizeof(stbi_uc*));
        unsigned char** images = (unsigned char**) malloc(MAX_IMAGES * sizeof(unsigned char*));

        // arrays for diff images
        int* width_arr; //=  (int*) malloc(image_count * sizeof(int));
        int* height_arr; //= (int*) malloc(image_count * sizeof(int));
        int* channels_arr; //= (int*) malloc(image_count * sizeof(int));

        DIR* input_directory;
        struct dirent* entry;
        input_directory = opendir(path_to_input_image);

        if (input_directory == NULL) {
            printf("Could not find directory %s.\n", path_to_input_image);
            return 1;            
        }

        while ((entry = readdir(input_directory)) != NULL && image_count < MAX_IMAGES) {
            if (entry->d_type == DT_REG) {
                char input_file[1024];
                sprintf(input_file, "%s%s", path_to_input_image, entry->d_name);
                images[image_count] = loadImage(input_file, &width_arr[image_count], &height_arr[image_count], &channels_arr[image_count]);
                
                image_count++;
            }
        }
        closedir(input_directory);
        printf("Read %d images\n", image_count);
        clock_t begin = clock();

        //stbi_uc** filtered_images;
        undsigned char** filtered_images;

        if (strcmp(filter, SHARPEN_FILTER) == 0) {
            printf("doing sharpen\n");
            //filtered_images = sharpenBatchStreams(images, image_count, width, height, channels);
            
            for(int i = 0; i < image_count; i++) {
                const int image_size = width_arr[i] * height_arr[i] * sizeof(unsigned char);

                //allocate mem on gpu
                unsigned char* d_input_image;
                cudaMalloc(&d_input_image, image_size);
                cudaMemcpy(d_input_image, images[i], image_size, cudaMemcpyHostToDevice);

                unsigned char* d_output_image;
                cudaMalloc(&d_output_image, image_size);

                //set up grid and block sizes
                const dim3 blockSize(32, 32, 1);
                const dim3 gridSize((width_arr[i] + blockSize.x -1) / blockSize.x,
                                    (height_arr[i] + blockSize.y -1) / blockSize.y,
                                    1);
                
                sharpen_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, width_arr[i], height_arr[i]);

                cudaMemcpy(filtered_images[i], d_output_image, image_size, cudaMemcpyDeviceToHost);

                cudaFree(d_input_image);
                cudaFree(d_output_image);
            }

        } else {
            printf("Invalid filter %s.\n", filter);
        }

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("streams time %f\n", time_spent);

        for (int i = 0; i < image_count; i++) {
            char output_file[100];

            sprintf(output_file, "%s%d.png", path_to_output_image, i);
            // printf("writing to %s\n", output_file);
            //writeImage(output_file, filtered_images[i], width, height, channels);
            writeImage(output_file, filtered_images[i], width_arr[i], height_arr[i], channels_arr[i]);
        } 
        printf("reached end of batch mode\n");
*/
    }
    
    return 0;
}
