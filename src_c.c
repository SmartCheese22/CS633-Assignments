//===============================================================================
// Program : src_c.c
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

enum{RIGHT = 0, LEFT, UP, DOWN, FRONT, BACK};

// DATA: Convert linear index to 3D coordinates
void linearToCoord(int linearIndex, int *x, int *y, int *z) {
    *z = linearIndex / (NX * NY);
    int remainder = linearIndex % (NX * NY);
    *y = remainder / NX;
    *x = remainder % NX;
}

// DATA -> RANK: Determine which process owns a specific coordinate
int coordToRank(int x, int y, int z) {
    int process_x = x / (NX / PX);
    int process_y = y / (NY / PY);
    int process_z = z / (NZ / PZ);
    // Apply bounds checking
    if (process_x >= PX) process_x = PX - 1;
    if (process_y >= PY) process_y = PY - 1;
    if (process_z >= PZ) process_z = PZ - 1;
    return process_x + process_y * PX + process_z * PX * PY;
}

// PROCESS: Convert linear index to rank
int idxToRank(int i) {
    int x, y, z;
    linearToCoord(i, &x, &y, &z);
    return coordToRank(x, y, z);
}

// Function to compute 3D indices in local array with ghost cells
int localIndex(int x, int y, int z, int t, int lnx, int lny, int lnz) {
    return x + (y) * lnx + (z) * lnx * lny + t * lnx * lny * lnz;
}

// Function to handle wrapping for process coordinates
int coordToRankProcess(int px, int py, int pz) {
    // Handle wrapping at boundaries to maintain periodic boundary conditions
    if (px < 0) px = PX - 1;
    if (py < 0) py = PY - 1;
    if (pz < 0) pz = PZ - 1;
    if (px >= PX) px = 0;
    if (py >= PY) py = 0;
    if (pz >= PZ) pz = 0;
    
    return px + py * PX + pz * PX * PY;
}

// Function to determine if a point is a local minimum
// Returns 1 if it's a local minimum, 0 otherwise
int isLocalMinimum(float* local_data, int x, int y, int z, int t, int lnx, int lny, int lnz) {
    // Check all 6 neighbors
    float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
    
    // Check right neighbor
    if (value > local_data[localIndex(x+1, y, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check left neighbor
    if (value > local_data[localIndex(x-1, y, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check up neighbor
    if (value > local_data[localIndex(x, y+1, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check down neighbor
    if (value > local_data[localIndex(x, y-1, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check front neighbor
    if (value > local_data[localIndex(x, y, z+1, t, lnx, lny, lnz)])
        return 0;
    
    // Check back neighbor
    if (value > local_data[localIndex(x, y, z-1, t, lnx, lny, lnz)])
        return 0;
    
    return 1; // It's a local minimum
}

// Similarly for local maximum
int isLocalMaximum(float* local_data, int x, int y, int z, int t, int lnx, int lny, int lnz) {
    float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
    
    // Check right neighbor
    if (value < local_data[localIndex(x+1, y, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check left neighbor
    if (value < local_data[localIndex(x-1, y, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check up neighbor
    if (value < local_data[localIndex(x, y+1, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check down neighbor
    if (value < local_data[localIndex(x, y-1, z, t, lnx, lny, lnz)])
        return 0;
    
    // Check front neighbor
    if (value < local_data[localIndex(x, y, z+1, t, lnx, lny, lnz)])
        return 0;
    
    // Check back neighbor
    if (value < local_data[localIndex(x, y, z-1, t, lnx, lny, lnz)])
        return 0;
    
    return 1; // It's a local maximum
}

// Find local minima and maxima for a specific time step
void findLocalExtrema(float *local_data, int t, int lnx, int lny, int lnz,
                      int local_nx, int local_ny, int local_nz,
                      int *min_count, int *max_count, float *global_min, float *global_max)
{
    *min_count = 0;
    *max_count = 0;
    *global_min = FLT_MAX;
    *global_max = -FLT_MAX;

    for (int z = 1; z <= local_nz; z++){
        for (int y = 1; y <= local_ny; y++){
            for (int x = 1; x <= local_nx; x++){
                float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
                
                // Update global extrema
                if (value < *global_min)
                    *global_min = value;
                if (value > *global_max)
                    *global_max = value;

                // Check for local extrema
                if (isLocalMinimum(local_data, x, y, z, t, lnx, lny, lnz)){
                    (*min_count)++;
                }

                if (isLocalMaximum(local_data, x, y, z, t, lnx, lny, lnz)){
                    (*max_count)++;
                }
            }
        }
    }
}

int main(int argc, char* argv[]){
    int size, rank;
    double start_time, read_time, main_time, total_time;
    start_time = MPI_Wtime();
    
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

    // Adjust for non-divisible dimensions if necessary
    if (proc_x == PX - 1) local_nx = NX - (PX - 1) * local_nx;
    if (proc_y == PY - 1) local_ny = NY - (PY - 1) * local_ny;
    if (proc_z == PZ - 1) local_nz = NZ - (PZ - 1) * local_nz;

    // Account for local dimensions with ghost cells
    int lnx = local_nx + 2;  // +2 for ghost cells on both sides
    int lny = local_ny + 2;
    int lnz = local_nz + 2;

    // Calculate local array sizes
    int local_data_size = local_nx * local_ny * local_nz * NC;
    int local_data_array_size = lnx * lny * lnz * NC;   // with ghost cells

    // Allocate memory for local data (with ghost cells)
    float* local_data = (float *)malloc(local_data_array_size * sizeof(float));
    if(local_data == NULL){
        fprintf(stderr, "Rank %d: Local data allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Initialize the entire local_data array to zero (including ghost cells)
    memset(local_data, 0, local_data_array_size * sizeof(float));

    // Calculate global coordinates of this process's subdomain
    int global_start_x = proc_x * (NX / PX);
    int global_start_y = proc_y * (NY / PY);
    int global_start_z = proc_z * (NZ / PZ);

    float* temp_buffer = NULL;
    if(rank == 0) {
        // Read the entire dataset only on rank 0
        FILE *fp = fopen(datafile, "r");
        if(fp == NULL){
            fprintf(stderr, "Rank 0: Cannot open input file %s\n", datafile);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Allocate buffer for the entire dataset
        temp_buffer = (float *)malloc(NX * NY * NZ * NC * sizeof(float));
        if(temp_buffer == NULL){
            fprintf(stderr, "Rank 0: Buffer allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Read the data from file
        for(int i = 0; i < NX * NY * NZ; i++){
            for(int t = 0; t < NC; t++){
                if(fscanf(fp, "%f", &temp_buffer[i * NC + t]) != 1){
                    fprintf(stderr, "Rank 0: Error reading data from file.\n");
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
        }
        fclose(fp);
    }

    // Allocate send and receive buffers
    float* send_buffer = NULL;
    float* recv_buffer = (float *)malloc(local_data_size * sizeof(float));
    if(recv_buffer == NULL){
        fprintf(stderr, "Rank %d: Receive buffer allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if(rank == 0){
        send_buffer = (float *)malloc(local_data_size * sizeof(float));
        if(send_buffer == NULL){
            fprintf(stderr, "Rank 0: Send buffer allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    read_time = MPI_Wtime() - start_time;
    double main_start_time = MPI_Wtime();

    // Distribute data from rank 0 to all processes
    for(int p = 0; p < size; p++){
        int p_x = p % PX;
        int p_y = (p / PX) % PY;
        int p_z = p / (PX * PY);
        
        int p_local_nx = NX / PX;
        int p_local_ny = NY / PY;
        int p_local_nz = NZ / PZ;
        
        // Adjust for non-divisible dimensions
        if (p_x == PX - 1) p_local_nx = NX - (PX - 1) * p_local_nx;
        if (p_y == PY - 1) p_local_ny = NY - (PY - 1) * p_local_ny;
        if (p_z == PZ - 1) p_local_nz = NZ - (PZ - 1) * p_local_nz;
        
        int p_start_x = p_x * (NX / PX);
        int p_start_y = p_y * (NY / PY);
        int p_start_z = p_z * (NZ / PZ);
        
        if(rank == 0){
            // Pack the data for the current process
            int idx = 0;
            for(int t = 0; t < NC; t++){
                for(int z = 0; z < p_local_nz; z++){
                    for(int y = 0; y < p_local_ny; y++){
                        for(int x = 0; x < p_local_nx; x++){
                            int global_idx = ((p_start_z + z) * NY * NX + 
                                             (p_start_y + y) * NX + 
                                             (p_start_x + x)) * NC + t;
                            send_buffer[idx++] = temp_buffer[global_idx];
                        }
                    }
                }
            }
        }
        
        // Send/receive data
        if(p == 0){
            // Rank 0 just copies its own data
            if(rank == 0){
                memcpy(recv_buffer, send_buffer, local_data_size * sizeof(float));
            }
        } else {
            // Send from rank 0 to process p
            if(rank == 0){
                MPI_Send(send_buffer, local_data_size, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
            }
            else if(rank == p){
                MPI_Recv(recv_buffer, local_data_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    if(rank == 0 && temp_buffer != NULL){
        free(temp_buffer);
        free(send_buffer);
    }

    // Copy received data into local_data array (interior cells only, ghost cells remain zero)
    int idx = 0;
    for(int t = 0; t < NC; t++){
        for(int z = 0; z < local_nz; z++){
            for(int y = 0; y < local_ny; y++){
                for(int x = 0; x < local_nx; x++){
                    local_data[localIndex(x + 1, y + 1, z + 1, t, lnx, lny, lnz)] = recv_buffer[idx++];
                }
            }
        }
    }
    free(recv_buffer);

    // Prepare MPI derived datatypes for ghost cell exchange
    MPI_Datatype x_face, y_face, z_face;
    
    // X-face: collection of cells in the YZ plane
    MPI_Type_vector(local_nz, local_ny, lnx * lny, MPI_FLOAT, &x_face);
    MPI_Type_commit(&x_face);
    
    // Y-face: collection of cells in the XZ plane
    MPI_Type_vector(local_nz, local_nx, lnx * lny, MPI_FLOAT, &y_face);
    MPI_Type_commit(&y_face);
    
    // Z-face: collection of cells in the XY plane
    MPI_Type_vector(local_ny, local_nx, lnx, MPI_FLOAT, &z_face);
    MPI_Type_commit(&z_face);

    // Allocate arrays for collecting extrema information
    int *local_min_counts = (int *)malloc(NC * sizeof(int));
    int *local_max_counts = (int *)malloc(NC * sizeof(int));
    float *local_global_mins = (float *)malloc(NC * sizeof(float));
    float *local_global_maxs = (float *)malloc(NC * sizeof(float));
    
    // Calculate neighbor ranks with proper wrapping
    int neighbours[6];
    neighbours[RIGHT] = coordToRankProcess(proc_x + 1, proc_y, proc_z);  // right
    neighbours[LEFT] = coordToRankProcess(proc_x - 1, proc_y, proc_z);   // left
    neighbours[UP] = coordToRankProcess(proc_x, proc_y + 1, proc_z);     // up
    neighbours[DOWN] = coordToRankProcess(proc_x, proc_y - 1, proc_z);   // down
    neighbours[FRONT] = coordToRankProcess(proc_x, proc_y, proc_z + 1);  // front
    neighbours[BACK] = coordToRankProcess(proc_x, proc_y, proc_z - 1);   // back

    // Main computation loop for each time step
    for(int t = 0; t < NC; t++){
        MPI_Request requests[12];
        MPI_Status statuses[12];
        int req_idx = 0;

        // Exchange ghost cells in X direction
        // Send right face (x=local_nx), receive left ghost (x=0)
        MPI_Isend(&local_data[localIndex(local_nx, 1, 1, t, lnx, lny, lnz)], 1, x_face, 
                  neighbours[RIGHT], 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(0, 1, 1, t, lnx, lny, lnz)], 1, x_face, 
                  neighbours[LEFT], 0, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send left face (x=1), receive right ghost (x=local_nx+1)
        MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, x_face, 
                  neighbours[LEFT], 1, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(local_nx+1, 1, 1, t, lnx, lny, lnz)], 1, x_face, 
                  neighbours[RIGHT], 1, MPI_COMM_WORLD, &requests[req_idx++]);

        // Exchange ghost cells in Y direction
        // Send top face (y=local_ny), receive bottom ghost (y=0)
        MPI_Isend(&local_data[localIndex(1, local_ny, 1, t, lnx, lny, lnz)], 1, y_face, 
                  neighbours[UP], 2, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, 0, 1, t, lnx, lny, lnz)], 1, y_face, 
                  neighbours[DOWN], 2, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send bottom face (y=1), receive top ghost (y=local_ny+1)
        MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, y_face, 
                  neighbours[DOWN], 3, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, local_ny+1, 1, t, lnx, lny, lnz)], 1, y_face, 
                  neighbours[UP], 3, MPI_COMM_WORLD, &requests[req_idx++]);

        // Exchange ghost cells in Z direction
        // Send front face (z=local_nz), receive back ghost (z=0)
        MPI_Isend(&local_data[localIndex(1, 1, local_nz, t, lnx, lny, lnz)], 1, z_face, 
                  neighbours[FRONT], 4, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, 1, 0, t, lnx, lny, lnz)], 1, z_face, 
                  neighbours[BACK], 4, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send back face (z=1), receive front ghost (z=local_nz+1)
        MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, z_face, 
                  neighbours[BACK], 5, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, 1, local_nz+1, t, lnx, lny, lnz)], 1, z_face, 
                  neighbours[FRONT], 5, MPI_COMM_WORLD, &requests[req_idx++]);

        // Wait for all communication to complete
        MPI_Waitall(req_idx, requests, statuses);
        
        // Find local extrema
        findLocalExtrema(local_data, t, lnx, lny, lnz, local_nx, local_ny, local_nz, 
                         &local_min_counts[t], &local_max_counts[t], 
                         &local_global_mins[t], &local_global_maxs[t]);
    }

    main_time = MPI_Wtime() - main_start_time;

    // Allocate arrays for gathering global results
    int *global_min_counts = NULL;
    int *global_max_counts = NULL;
    float *global_mins = NULL;
    float *global_maxs = NULL;

    if (rank == 0) {
        global_min_counts = (int*)malloc(NC * sizeof(int));
        global_max_counts = (int*)malloc(NC * sizeof(int));
        global_mins = (float*)malloc(NC * sizeof(float));
        global_maxs = (float*)malloc(NC * sizeof(float));
    }

    // Reduce min/max counts and global extrema values
    MPI_Reduce(local_min_counts, global_min_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_counts, global_max_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_global_mins, global_mins, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_global_maxs, global_maxs, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    total_time = MPI_Wtime() - start_time;

    // Rank 0 writes output to file
    if(rank == 0){
        FILE* fp = fopen(outputfile, "w");
        if(!fp){
            fprintf(stderr, "Rank 0: Cannot open output file %s\n", outputfile);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Line 1: counts of local minima and maxima
        for(int t = 0; t < NC; t++){
            fprintf(fp, "(%d, %d)", global_min_counts[t], global_max_counts[t]);
            if(t < NC - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, "\n");

        // Line 2: global minimum and maximum values
        for(int t = 0; t < NC; t++){
            fprintf(fp, "(%f, %f)", global_mins[t], global_maxs[t]);
            if(t < NC - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, "\n");

        // Line 3: timing information
        fprintf(fp, "%f, %f, %f\n", read_time, main_time, total_time);
        fclose(fp);
        
        // Free memory
        free(global_min_counts);
        free(global_max_counts);
        free(global_mins);
        free(global_maxs);
    }

    // Free memory
    free(local_data);
    free(local_min_counts);
    free(local_max_counts);
    free(local_global_mins);
    free(local_global_maxs);

    // Free MPI datatypes
    MPI_Type_free(&x_face);
    MPI_Type_free(&y_face);
    MPI_Type_free(&z_face);
    
    MPI_Finalize();
    return 0;
}