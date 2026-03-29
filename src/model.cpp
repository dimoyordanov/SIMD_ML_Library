#include "model.h"

#include "activation.h"
#include "backprop.h"
#include "linear.h"
#include "loss.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace ml {

// ------------------------------------------------------------------ ActivationFn factories
// Each activation encodes both its forward kernel and the caching policy
// (which value backward() needs) inside apply_and_cache. Captureless
// lambdas decay to function pointers, so no heap allocation occurs.

ActivationFn activation::relu() {
    ActivationFn fn;
    fn.apply_and_cache = [](float* pre, float* cache, float* out, int n) {
        std::copy(pre, pre + n, cache);  // relu backward needs pre-activation
        ispc::relu(pre, out, n);
    };
    fn.backward = [](float* go, float* saved, float* gi, int n) {
        ispc::relu_backward(go, saved, gi, n);
    };
    return fn;
}

ActivationFn activation::sigmoid() {
    ActivationFn fn;
    fn.apply_and_cache = [](float* pre, float* cache, float* out, int n) {
        ispc::sigmoid(pre, out, n);
        std::copy(out, out + n, cache);  // sigmoid backward needs post-activation
    };
    fn.backward = [](float* go, float* saved, float* gi, int n) {
        ispc::sigmoid_backward(go, saved, gi, n);
    };
    return fn;
}

ActivationFn activation::tanh() {
    ActivationFn fn;
    fn.apply_and_cache = [](float* pre, float* cache, float* out, int n) {
        ispc::tanh_activation(pre, out, n);
        std::copy(out, out + n, cache);  // tanh backward needs post-activation
    };
    fn.backward = [](float* go, float* saved, float* gi, int n) {
        ispc::tanh_backward(go, saved, gi, n);
    };
    return fn;
}

ActivationFn activation::softmax() {
    ActivationFn fn;
    fn.apply_and_cache = [](float* pre, float* cache, float* out, int n) {
        ispc::softmax(pre, out, n);
        std::copy(out, out + n, cache);  // softmax backward needs post-activation
    };
    fn.backward = [](float* go, float* saved, float* gi, int n) {
        ispc::softmax_backward(go, saved, gi, n);
    };
    return fn;
}

// ------------------------------------------------------------------ DenseLayer

DenseLayer::DenseLayer(int in_features, int out_features, ActivationFn act,
                       unsigned int seed)
    : in_features_{in_features}
    , out_features_{out_features}
    , act_{act}
    , weights_(static_cast<std::size_t>(out_features * in_features))
    , bias_(static_cast<std::size_t>(out_features), 0.0f)
    , act_cache_(static_cast<std::size_t>(out_features))
    , dweights_(static_cast<std::size_t>(out_features * in_features), 0.0f)
    , dbias_(static_cast<std::size_t>(out_features), 0.0f)
{
    std::mt19937 rng{seed};
    const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
    std::uniform_real_distribution<float> dist{-limit, limit};
    for (float& w : weights_) w = dist(rng);
}

std::vector<float> DenseLayer::forward(const std::vector<float>& input) {
    input_cache_ = input;

    // pre_act = W * input  (matmul: M=out_features_, N=1, K=in_features_)
    std::vector<float> pre_act(static_cast<std::size_t>(out_features_));
    ispc::matmul(weights_.data(),
                 const_cast<float*>(input.data()),
                 pre_act.data(),
                 out_features_, 1, in_features_);

    ispc::add_vector(pre_act.data(), bias_.data(), out_features_);

    // Apply activation, populating act_cache_ with whatever backward needs
    std::vector<float> output(static_cast<std::size_t>(out_features_));
    act_.apply_and_cache(pre_act.data(), act_cache_.data(), output.data(), out_features_);

    return output;
}

std::vector<float> DenseLayer::backward(const std::vector<float>& grad_out) {
    // Activation backward: d_pre_act from upstream gradient and cached value
    std::vector<float> d_pre_act(static_cast<std::size_t>(out_features_));
    act_.backward(const_cast<float*>(grad_out.data()),
                  act_cache_.data(),
                  d_pre_act.data(),
                  out_features_);

    // dW = d_pre_act ⊗ input^T  (matmul_backward_A: M=out, N=1, K=in)
    ispc::matmul_backward_A(d_pre_act.data(),
                             const_cast<float*>(input_cache_.data()),
                             dweights_.data(),
                             out_features_, 1, in_features_);

    // dbias = d_pre_act
    dbias_ = d_pre_act;

    // d_input = W^T * d_pre_act  (matmul_backward_B: M=out, N=1, K=in)
    std::vector<float> d_input(static_cast<std::size_t>(in_features_));
    ispc::matmul_backward_B(weights_.data(),
                             d_pre_act.data(),
                             d_input.data(),
                             out_features_, 1, in_features_);

    return d_input;
}

void DenseLayer::update_weights(float lr) {
    // weights -= lr * dweights: scale dweights by -lr then add into weights.
    // dweights are recomputed every backward pass so in-place scaling is safe.
    ispc::multiply_scalar(dweights_.data(), -lr, static_cast<int>(dweights_.size()));
    ispc::add_vector(weights_.data(), dweights_.data(), static_cast<int>(weights_.size()));

    ispc::multiply_scalar(dbias_.data(), -lr, static_cast<int>(dbias_.size()));
    ispc::add_vector(bias_.data(), dbias_.data(), static_cast<int>(bias_.size()));
}

// ------------------------------------------------------------------ Model

Model& Model::add(std::unique_ptr<Layer> l) {
    if (!layers_.empty()) {
        const int expected = layers_.back()->out_size();
        const int actual   = l->in_size();
        if (expected != actual)
            throw std::invalid_argument(
                "Layer size mismatch: previous layer outputs " +
                std::to_string(expected) + " but new layer expects " +
                std::to_string(actual));
    }
    layers_.push_back(std::move(l));
    return *this;
}

// ------------------------------------------------------------------ factories

std::unique_ptr<DenseLayer> make_dense(int in_features, int out_features,
                                       ActivationFn act, unsigned int seed) {
    return std::make_unique<DenseLayer>(in_features, out_features, act, seed);
}

// ------------------------------------------------------------------ inference

std::vector<float> forward(Model& m, const std::vector<float>& input) {
    if (m.layers().empty())
        throw std::invalid_argument("Model has no layers");
    if (static_cast<int>(input.size()) != m.layers().front()->in_size())
        throw std::invalid_argument("Input size mismatch");

    std::vector<float> current = input;
    for (auto& layer : m.layers()) current = layer->forward(current);
    return current;
}

// ------------------------------------------------------------------ training

float train_step(Model& m, const std::vector<float>& input,
                 const std::vector<float>& target, float lr) {
    if (m.layers().empty())
        throw std::invalid_argument("Model has no layers");
    if (static_cast<int>(target.size()) != m.layers().back()->out_size())
        throw std::invalid_argument("Target size mismatch");

    const std::vector<float> output = forward(m, input);

    float loss{};
    ispc::mse_loss(const_cast<float*>(output.data()),
                   const_cast<float*>(target.data()),
                   static_cast<int>(output.size()), &loss);

    std::vector<float> grad(output.size());
    ispc::mse_backward(const_cast<float*>(output.data()),
                       const_cast<float*>(target.data()),
                       grad.data(), static_cast<int>(output.size()));

    for (int i = static_cast<int>(m.layers().size()) - 1; i >= 0; --i)
        grad = m.layers()[static_cast<std::size_t>(i)]->backward(grad);

    for (auto& layer : m.layers()) layer->update_weights(lr);

    return loss;
}

}  // namespace ml
