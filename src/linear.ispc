export uniform float dot_product(uniform float a[],
                        uniform float b[],
                        uniform int n) {
    varying float partial = 0.0f;
    foreach (i = 0 ... n) {
        partial += a[i] * b[i];
    }
    uniform float res = reduce_add(partial);
    return res;
}



export void matmul(uniform float A[],
                   uniform float B[],
                   uniform float C[],
                   uniform int M,
                   uniform int N,
                   uniform int K) {

    foreach (row = 0 ... M, col = 0 ... N) {
        float sum = 0.0f;
        for (uniform int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


#if TARGET_WIDTH == 2
#define TILE_WIDTH 4
#elif TARGET_WIDTH == 4
#define TILE_WIDTH 8
#elif TARGET_WIDTH == 8
#define TILE_WIDTH 16
#elif TARGET_WIDTH == 16
#define TILE_WIDTH 32
#elif TARGET_WIDTH == 32
#define TILE_WIDTH 64
#elif TARGET_WIDTH == 64
#define TILE_WIDTH 128
#else
#error "Unsupported target width"
#endif
#define TILE_HEIGHT 8

export void matmul2(uniform float A[],
                   uniform float B[],
                   uniform float C[],
                   uniform int M,
                   uniform int N,
                   uniform int K) {
    if (M < TILE_HEIGHT || N < TILE_WIDTH) {
        matmul(A, B, C, M, N, K);
        return;
    }
    uniform float sumTile[TILE_HEIGHT][TILE_WIDTH];
    uniform float avals[TILE_HEIGHT];

    for (uniform unsigned int m = 0; m < M; m += TILE_HEIGHT) {
        for (uniform unsigned int n0 = 0; n0 < N; n0 += TILE_WIDTH) {
            foreach (ni = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    sumTile[i][ni] = 0.0f;
                }
            }
            for (uniform unsigned int k = 0; k < K; k++) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++)
                    avals[i] = A[(m + i) * K + k];

                
                foreach (nt = 0 ... TILE_WIDTH) {
                        
                    varying float B1 = B[k * N + n0 + nt];
                    for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {    
                        sumTile[i][nt] += avals[i] * B1;
                    }
                }
            }
            foreach (ni = 0 ... TILE_WIDTH) {
                for (uniform unsigned int i = 0; i < TILE_HEIGHT; i++) {
                    C[(m + i) * N + n0 + ni] = sumTile[i][ni];
                }
            }
        }
    }
}

export void add_vector(uniform float output[],
                       uniform float b[],
                       uniform int n) {
    foreach (i = 0 ... n) {
        output[i] += b[i];
    }
}

export void multiply_scalar(uniform float v[],
                             uniform float scalar,
                             uniform int n) {
    foreach (i = 0 ... n) {
        v[i] *= scalar;
    }
}
