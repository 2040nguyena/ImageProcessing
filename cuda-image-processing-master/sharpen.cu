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

/*

int sharpen_images(stbi_uc** images, int image_count, int width, int height, int channels, char* path_to_output_image) {
    printf("inside sharpen function\n");
    
    clock_t begin = clock();

    printf("doing sharpen\n");
    stbi_uc** filtered_images = sharpenBatchStreams(images, image_count, width, height, channels);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("streams time %f\n", time_spent);

    for (int i = 0; i < image_count; i++) {
        char output_file[100];

        sprintf(output_file, "%s%d.png", path_to_output_image, i);
        // printf("writing to %s\n", output_file);
        writeImage(output_file, filtered_images[i], width, height, channels);
    } 
    printf("reached end of sharpen function\n");
}

*/

int sharpen_images(char * path_to_input_image, char * path_to_output_image) {
        printf("in batch mode\n");
        int MAX_IMAGES = 1000;
        int image_count = 0;
        stbi_uc** images = (stbi_uc**) malloc(MAX_IMAGES * sizeof(stbi_uc*));

        // assumed to be the same for all images
        int width; 
        int height; 
        int channels;

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
                images[image_count] = loadImage(input_file, &width, &height, &channels);
                image_count++;
            }
        }
        closedir(input_directory);
        printf("Read %d images\n", image_count);
        clock_t begin = clock();

        
        stbi_uc** filtered_images = sharpenBatchStreams(images, image_count, width, height, channels);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("streams time %f\n", time_spent);

        for (int i = 0; i < image_count; i++) {
            char output_file[100];

            sprintf(output_file, "%s%d.png", path_to_output_image, i);
            // printf("writing to %s\n", output_file);
            writeImage(output_file, filtered_images[i], width, height, channels);
        } 
        printf("reached end of batch mode\n");
    return 0;
}

/*
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
        if (image == NULL) {
            printf("Could not load image %s.\n", path_to_input_image);
            return 1;
        }

        stbi_uc* filtered_image;
        if (strcmp(filter, BLUR_FILTER) == 0) {
            filtered_image = blur(image, width, height, channels);
        } else if (strcmp(filter, SHARPEN_FILTER) == 0) {
            filtered_image = sharpen(image, width, height, channels);
        } else if (strcmp(filter, VERTICAL_FLIP_FILTER) == 0) {
            filtered_image = verticalFlip(image, width, height, channels);
        } else if (strcmp(filter, HORIZONTAL_FLIP_FILTER) == 0) {
            filtered_image = horizontalFlip(image, width, height, channels);
        } else if (strcmp(filter, GRAYSCALE_FILTER) == 0) {
            filtered_image = gray(image, width, height, channels);
        } else if (strcmp(filter, GRAYSCALE_WEIGHTED_FILTER) == 0) {
            filtered_image = grayWeight(image, width, height, channels);
        } else if (strcmp(filter, EDGE_DETECTION_FILTER) == 0) {
            filtered_image = edgeDetection(image, width, height, channels);
            // filtered_image = edgeDetectionSharedMemory(image, width, height, channels);
            // filtered_image = edgeDetectionTextureMemory(image, width, height, channels);
            // filtered_image = edgeDetectionConstantMemory(image, width, height, channels);
        }  else {
            printf("Invalid filter %s.\n", filter);
        }

        writeImage(path_to_output_image, filtered_image, width, height, channels);
        imageFree(image);
        imageFree(filtered_image);
    } else if (strcmp(mode, BATCH_MODE) == 0) {
        printf("in batch mode\n");
        int MAX_IMAGES = 1000;
        int image_count = 0;
        stbi_uc** images = (stbi_uc**) malloc(MAX_IMAGES * sizeof(stbi_uc*));

        // assumed to be the same for all images
        int width; 
        int height; 
        int channels;

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
                images[image_count] = loadImage(input_file, &width, &height, &channels);
                image_count++;
            }
        }
        closedir(input_directory);
        printf("Read %d images\n", image_count);
        clock_t begin = clock();

        stbi_uc** filtered_images;

        if (strcmp(filter, BLUR_FILTER) == 0) {
            printf("doing blur but rly edge detection rn\n");
            //filtered_images = blur(images, width, height, channels);
            filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        } else if (strcmp(filter, SHARPEN_FILTER) == 0) {
            printf("doing sharpen\n");
            filtered_images = sharpenBatchStreams(images, image_count, width, height, channels);
        } else if (strcmp(filter, VERTICAL_FLIP_FILTER) == 0) {
            printf("doing vertical flip but rly edge detection rn\n");
            //filtered_images = verticalFlip(images, width, height, channels);
            filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        } else if (strcmp(filter, HORIZONTAL_FLIP_FILTER) == 0) {
            printf("doing horizontal flip but rly edge detection rn\n");
            //filtered_images = horizontalFlip(images, width, height, channels);
            filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        } else if (strcmp(filter, GRAYSCALE_FILTER) == 0) {
            printf("doing grayscale but rly edge detection rn\n");
            //filtered_images = gray(images, width, height, channels);
            filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        } else if (strcmp(filter, GRAYSCALE_WEIGHTED_FILTER) == 0) {
            printf("doing gray weight but rly edge detection rn\n");
            //filtered_images = grayWeight(images, width, height, channels);
            filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        } else if (strcmp(filter, EDGE_DETECTION_FILTER) == 0) {
            printf("doing edge detection\n");
            filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        }  else {
            printf("Invalid filter %s.\n", filter);
        }

        //stbi_uc** filtered_images = edgeDetectionBatchStreams(images, image_count, width, height, channels);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("streams time %f\n", time_spent);

        for (int i = 0; i < image_count; i++) {
            char output_file[100];

            sprintf(output_file, "%s%d.png", path_to_output_image, i);
            // printf("writing to %s\n", output_file);
            writeImage(output_file, filtered_images[i], width, height, channels);
        } 
        printf("reached end of batch mode\n");
    }
    
    return 0;
}
*/