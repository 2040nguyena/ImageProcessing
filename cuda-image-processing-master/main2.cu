#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "image.h"
#include "filters/sharpen_filter.h"

//nvcc sharpen.cu `pkg-config opencv --cflags --libs`
//
//./a.out path_to_image_input path_to_image_output filter_arg mode_arg
//./a.out path_to_input_dir path_to_output_dir filter_arg mode_arg
//

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int sharpen_images_main(char * path_to_output_image, int max, int GPUrank) {
        char * path_to_input_image = "./images_out/";
        printf("in batch mode\n");
        int MAX_IMAGES = max;
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

        //cudaSetDevice(GPUrank);
        //printf("device is set to gpu %d\n", GPUrank);

        stbi_uc** filtered_images = sharpenBatchStreams(images, image_count, width, height, channels, GPUrank);

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("streams time %f\n", time_spent);

        cudaDeviceSynchronize();

        for (int i = 0; i < image_count; i++) {
            char output_file[100];

            sprintf(output_file, "%sgpu%dpic%d.png", path_to_output_image, GPUrank, i);
            // printf("writing to %s\n", output_file);
            writeImage(output_file, filtered_images[i], width, height, channels);
        } 
        printf("reached end of batch mode\n");
    return 0;
}
