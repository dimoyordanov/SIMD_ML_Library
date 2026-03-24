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
