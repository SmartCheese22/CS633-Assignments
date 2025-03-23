//===============================================================================
// Program : src.c
// Purpose : Find local and global extremas in a 3D grid
//
// Compile : mpicc -o src.x src.c -O3
//===============================================================================
/*===============================LIBRARY INCLUDE FILES===========================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>

int PX, PY, PZ, NX, NY, NZ, NC;

// DATA: Convert linear index to 3D coordinates
void linearToCoord(int linearIndex, int *x, int *y, int *z) {
    *z = linearIndex / (NX * NY);
    int remainder = linearIndex % (NX * NY);
    *y = remainder / NX;
    *x = remainder % NX;
}

// DATA: Convert 3D coordinates to linear index
int coordToLinear(int x, int y, int z) {
    return x + y * NX + z * NX * NY;
}

// DATA -> RANK: Determine which process owns a specific coordinate
int coordToRank(int x, int y, int z) {
    int process_x = x / (NX / PX);
    int process_y = y / (NY / PY);
    int process_z = z / (NZ / PZ);
    return process_x + process_y * PX + process_z * PX * PY;
}

int coordToRankU(int x, int y, int z) {
    // Calculate which process owns x
    int div_x = NX / PX;
    int rem_x = NX % PX;
    int process_x;
    if (x < (div_x + 1) * rem_x) {
        // In the region of processes with div_x + 1 elements
        process_x = x / (div_x + 1);
    } else {
        // In the region of processes with div_x elements
        process_x = rem_x + (x - (div_x + 1) * rem_x) / div_x;
    }
    
    // Similar calculations for y and z dimensions
    int div_y = NY / PY;
    int rem_y = NY % PY;
    int process_y;
    if (y < (div_y + 1) * rem_y) {
        process_y = y / (div_y + 1);
    } else {
        process_y = rem_y + (y - (div_y + 1) * rem_y) / div_y;
    }
    
    int div_z = NZ / PZ;
    int rem_z = NZ % PZ;
    int process_z;
    if (z < (div_z + 1) * rem_z) {
        process_z = z / (div_z + 1);
    } else {
        process_z = rem_z + (z - (div_z + 1) * rem_z) / div_z;
    }
    
    return process_x + process_y * PX + process_z * PX * PY;
}

// PROCESS: Convert process rank to its position in the process grid
void rankToProcessCoord(int rank, int *process_x, int *process_y, int *process_z) {
    *process_x = rank % PX;
    *process_y = (rank / PX) % PY;
    *process_z = rank / (PX * PY);
}

// PROCESS: Convert linear index to rank
int idxToRank(int i) {
    int x, y, z;
    linearToCoord(i, &x, &y, &z);
    return coordToRank(x, y, z);
}

// Function to compute 3D indices in local array with ghost cells
int localIndex(int x, int y, int z, int t, int lnx, int lny, int lnz) {
    return (x + 1) + (y + 1) * lnx + (z + 1) * lnx * lny + t * lnx * lny * lnz;
}

int main(int argc, char* argv[]){
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    /* argument checking and setting */
    if(argc < 10){
        if(!rank)
            fprintf(stderr, "Usage: %s dataset PX PY PZ NX NY NZ NC outputfile\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char* datafile = argv[1];
    PX = atoi(argv[2]);
    PY = atoi(argv[3]);
    PZ = atoi(argv[4]);
    NX = atoi(argv[5]);
    NY = atoi(argv[6]);
    NZ = atoi(argv[7]);
    NC = atoi(argv[8]);
    char* outputfile = argv[9];

    if(PX * PY * PZ != size){
        if(!rank)
            fprintf(stderr, "Error: PX*PY*PZ must equal the total number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int proc_x = rank % PX;
    int proc_y = (rank / PX) % PY;
    int proc_z = rank / (PX * PY);

    int local_nx = NX / PX;
    int local_ny = NY / PY;
    int local_nz = NZ / PZ;

    int local_data_size = local_nx * local_ny * local_nz * NC; /* floats */
    
    int lnx = local_nx + 2;
    int lny = local_ny + 2;
    int lnz = local_nz + 2;

    int local_data_array_size = lnx * lny * lnz * NC;
    float* local_data = (float *)malloc(local_data_array_size * sizeof(float));

    if(local_data == NULL){
        fprintf(stderr, "Rank %d: Local data allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* assuming divisible grid */
    float* global_data = NULL;
    if(!rank){
        //read the file and scatter the appropriate data to the processes
        global_data = (float *)malloc(NX * NY * NZ * NC * sizeof(float));
        if(global_data == NULL){
            fprintf(stderr, "Rank 0: Global data allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        FILE *fp = fopen(datafile, "r");
        if(fp == NULL){
            fprintf(stderr, "Rank 0: Cannot open input file %s\n", datafile);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        float* base_ptr[size];
        for(int i = 0; i < size; i++)
            base_ptr[i] = global_data + i * local_data_size;

        for(int i = 0; i < NX*NY*NZ; i++){
            int rowRank = idxToRank(i);
            for(int t = 0; t < NC; t++){
                float* offset = base_ptr[rowRank] + t*local_nx*local_ny*local_nz;
                if(fscanf(fp, "%f", offset) != 1){
                    fprintf(stderr, "Rank 0: Error reading data from file.\n");
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
            base_ptr[rowRank]++;
        }
        
        fclose(fp);
    }

    //scatter the data to the processes
    MPI_Scatter(global_data, local_data_size, MPI_FLOAT, local_data, local_data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if(!rank)
        free(global_data);
    
    

    free(local_data);
    MPI_Finalize();
    return 0;
}
