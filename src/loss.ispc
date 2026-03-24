export void mse_loss(uniform float predictions[],
                     uniform float targets[],
                     uniform int n,
                     uniform float result[]) {
    varying float partial = 0.0f;
    foreach (i = 0 ... n) {
        varying float diff = predictions[i] - targets[i];
        partial += diff * diff;
    }
    result[0] = reduce_add(partial) / (float)n;
}
