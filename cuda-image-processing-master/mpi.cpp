#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

// -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/11.2/lib64

// -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/include

//mpicxx -c mpi.cpp -o mpi.o -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/include
//nvcc -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/include -w -m64 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_70,code=sm_70 -c -w main2.cu
//mpicxx mpi.o main2.o -lcudart -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/11.2/lib64
//Execute: mpirun -np 3 ./a.out

int sharpen_images_main(char * path_to_output_image, int max, int GPUrank);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char *argv[])
{

 int rank, size;

 MPI_Init (&argc, &argv); /* starts MPI */

 MPI_Comm_rank (MPI_COMM_WORLD, &rank); /* get current process id */

 MPI_Comm_size (MPI_COMM_WORLD, &size); /* get number of processes */

 //int deviceCount;
 //cudaGetDeviceCount(&deviceCount);
 //printf("deviceCount: %d, rank: %d\n", deviceCount, rank);

 gpuErrchk(cudaSetDevice(rank));

 printf("device set to %d\n", rank);

 char* outputDir = "./output/batch/";
 int max = 1;

//  if(rank > 0) {
//     sharpen_images_main(outputDir, max, rank);
//  }

 cudaSetDevice(rank);
 sharpen_images_main(outputDir, max, rank);

 MPI_Finalize();

 return 0;
}