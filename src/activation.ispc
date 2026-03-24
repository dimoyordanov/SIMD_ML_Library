export void relu(uniform float input[],
                 uniform float output[],
                 uniform int count) {
    foreach (i = 0 ... count) {
        output[i] = max(input[i], 0.0f);
    }
}

export void sigmoid(uniform float input[],
                    uniform float output[],
                    uniform int count) {
    foreach (i = 0 ... count) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

double tanh(double x) {
    double e_pos = exp(x);
    double e_neg = exp(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

export void tanh_activation(uniform float input[],
                             uniform float output[],
                             uniform int count) {
    foreach (i = 0 ... count) {
        output[i] = tanh(input[i]);
    }
}

export void softmax(uniform float input[],
                    uniform float output[],
                    uniform int n) {


    varying float vsum = 0.0f;
    foreach (i = 0 ... n) {
        output[i] = exp(input[i]);
        vsum += output[i];
    }
    uniform float total = reduce_add(vsum);

    foreach (i = 0 ... n) {
        output[i] /= total;
    }
}
