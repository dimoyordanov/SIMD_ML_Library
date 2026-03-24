// Gradient kernels for backpropagation.
//
// Convention: every backward function receives the upstream gradient
// (grad_output) and writes the downstream gradient (grad_input).
// Activation backwards also receive the pre-computed forward output so
// the expensive transcendental is not repeated.

// ------------------------------------------------------------------ activations

// ReLU: grad flows through where input > 0, is blocked otherwise.
export void relu_backward(uniform float grad_output[],
                          uniform float input[],
                          uniform float grad_input[],
                          uniform int n) {
    foreach (i = 0 ... n) {
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
    }
}

// Sigmoid: d/dx sigma(x) = sigma(x) * (1 - sigma(x)).
// Takes pre-computed sigmoid_output to avoid recomputing exp.
export void sigmoid_backward(uniform float grad_output[],
                              uniform float sigmoid_output[],
                              uniform float grad_input[],
                              uniform int n) {
    foreach (i = 0 ... n) {
        grad_input[i] = grad_output[i]
                      * sigmoid_output[i]
                      * (1.0f - sigmoid_output[i]);
    }
}

// Tanh: d/dx tanh(x) = 1 - tanh(x)^2.
// Takes pre-computed tanh_output to avoid recomputing.
export void tanh_backward(uniform float grad_output[],
                           uniform float tanh_output[],
                           uniform float grad_input[],
                           uniform int n) {
    foreach (i = 0 ... n) {
        grad_input[i] = grad_output[i]
                      * (1.0f - tanh_output[i] * tanh_output[i]);
    }
}

// Softmax: full Jacobian contracted with upstream gradient.
//   grad_input[i] = softmax[i] * (grad_output[i] - dot(grad_output, softmax))
// Takes pre-computed softmax_output.
export void softmax_backward(uniform float grad_output[],
                              uniform float softmax_output[],
                              uniform float grad_input[],
                              uniform int n) {
    varying float vdot = 0.0f;
    foreach (i = 0 ... n) {
        vdot += grad_output[i] * softmax_output[i];
    }
    uniform float dot = reduce_add(vdot);

    foreach (i = 0 ... n) {
        grad_input[i] = softmax_output[i] * (grad_output[i] - dot);
    }
}

// ------------------------------------------------------------------ loss

// MSE: d/d(pred_i) mean((pred - target)^2) = 2*(pred_i - target_i) / n.
export void mse_backward(uniform float predictions[],
                          uniform float targets[],
                          uniform float grad[],
                          uniform int n) {
    foreach (i = 0 ... n) {
        grad[i] = 2.0f * (predictions[i] - targets[i]) / (float)n;
    }
}

// ------------------------------------------------------------------ linear layer

// dA = dC * B^T   where C = A*B,  A: M×K,  B: K×N,  C: M×N.
//   dA[r*K+c] = sum_n  dC[r*N+n] * B[c*N+n]
export void matmul_backward_A(uniform float dC[],
                               uniform float B[],
                               uniform float dA[],
                               uniform int M,
                               uniform int N,
                               uniform int K) {
    foreach (r = 0 ... M, c = 0 ... K) {
        float sum = 0.0f;
        for (uniform int n = 0; n < N; n++) {
            sum += dC[r * N + n] * B[c * N + n];
        }
        dA[r * K + c] = sum;
    }
}

// dB = A^T * dC   where C = A*B,  A: M×K,  B: K×N,  C: M×N.
//   dB[r*N+c] = sum_m  A[m*K+r] * dC[m*N+c]
export void matmul_backward_B(uniform float A[],
                               uniform float dC[],
                               uniform float dB[],
                               uniform int M,
                               uniform int N,
                               uniform int K) {
    foreach (r = 0 ... K, c = 0 ... N) {
        float sum = 0.0f;
        for (uniform int m = 0; m < M; m++) {
            sum += A[m * K + r] * dC[m * N + c];
        }
        dB[r * N + c] = sum;
    }
}
