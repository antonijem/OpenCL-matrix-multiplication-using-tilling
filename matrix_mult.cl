#define TILE_SIZE 64
#define NW 16

__kernel void matrix_mult(__global float* A, __global float* B, __global float* C, const int m, const int n, const int k, const int n_works) {

    //Identifiers
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TILE_SIZE * get_group_id(0) + row;
    const int globalCol = TILE_SIZE * get_group_id(1) + col;

    //Local memory
    __local float sharedA[TILE_SIZE][TILE_SIZE];
    __local float sharedB[TILE_SIZE][TILE_SIZE];

    //Acumulation registers
    float sum[NW];
    for (int i = 0; i < n_works; i++) {
        sum[i] = 0.0f;
    }

    const int numTiles = n / TILE_SIZE;
    for (int i = 0; i < numTiles; i++) {
        for (int j = 0; j < NW; j++) {
            const int tiledRow = TILE_SIZE * i + row;
            const int tiledCol = TILE_SIZE * i + col;
            sharedA[row][col + j * TILE_SIZE / n_works] = A[globalRow * k + (tiledCol + j * TILE_SIZE / NW)];
            sharedB[row][col + j * TILE_SIZE / n_works] = B[tiledRow * n + (globalCol + j * TILE_SIZE / NW)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            for (int q = 0; q < NW; q++) {
                sum[q] += sharedA[row][k] * sharedB[k][col + q * TILE_SIZE / NW];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    
        for (int q = 0; q < NW; q++) {
            C[globalRow * n + (globalCol + q * TILE_SIZE / NW)] = sum[q];
        }
    }

}