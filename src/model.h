#pragma once

#include <memory>
#include <vector>

namespace ml {

// ------------------------------------------------------------------ ActivationFn
// Strategy pattern: each activation bundles its forward/backward ISPC kernels
// and a self-managed cache that stores whichever value the backward kernel
// needs: pre-activation for ReLU, post-activation for Sigmoid/Tanh/Softmax.

struct ActivationFn {
    // Applies the activation pre_act -> output AND writes to cache whatever the
    // backward kernel will need as its 'saved' argument.
    // ReLU writes pre_act into cache; Sigmoid/Tanh/Softmax write their output.
    void (*apply_and_cache)(float* pre_act, float* cache, float* output, int n);
    void (*backward)(float* grad_out, float* saved, float* grad_in, int n);
};

namespace activation {
    ActivationFn relu();
    ActivationFn sigmoid();
    ActivationFn tanh();
    ActivationFn softmax();
}

// ------------------------------------------------------------------ Layer
// Abstract base for all layer types. Stored polymorphically via unique_ptr.

class Layer {
public:
    virtual ~Layer() = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual std::vector<float> backward(const std::vector<float>& grad_out) = 0;
    virtual void update_weights(float lr) = 0;

    virtual int in_size() const noexcept = 0;
    virtual int out_size() const noexcept = 0;
    virtual const std::vector<float>& weights() const = 0;
    virtual const std::vector<float>& bias() const = 0;
    virtual void set_weights(std::vector<float>&& w) = 0;
    virtual void set_bias(std::vector<float>&& b) = 0;

protected:
    Layer() = default;
};

// ------------------------------------------------------------------ DenseLayer
// Fully-connected layer: output = activation(W * input + b).
// The ActivationFn strategy owns its forward-pass cache for use in backward.

class DenseLayer : public Layer {
public:
    DenseLayer(int in_features, int out_features, ActivationFn act,
               unsigned int seed = 42);

    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& grad_out) override;
    void update_weights(float lr) override;

    int in_size() const noexcept override { return in_features_; }
    int out_size() const noexcept override { return out_features_; }

    const std::vector<float>& weights() const override { return weights_; }
    const std::vector<float>& bias()    const override { return bias_; }
    void set_weights(std::vector<float>&& w) override { weights_ = std::move(w); }
    void set_bias(std::vector<float>&& b)    override { bias_    = std::move(b); }

private:
    int in_features_;
    int out_features_;
    ActivationFn act_;
    std::vector<float> weights_;     // row-major [out_features_ × in_features_]
    std::vector<float> bias_;        // [out_features_]
    std::vector<float> input_cache_; // saved input x for dW = d_pre_act ⊗ x^T
    std::vector<float> act_cache_;   // saved value needed by activation backward
    std::vector<float> dweights_;
    std::vector<float> dbias_;
};

// ------------------------------------------------------------------ Model
// Fluent builder: Model m; m.add(make_dense(...)).add(make_dense(...));

class Model {
public:
    Model& add(std::unique_ptr<Layer> l);

    const std::vector<std::unique_ptr<Layer>>& layers() const noexcept { return layers_; }
    std::vector<std::unique_ptr<Layer>>&       layers()       noexcept { return layers_; }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

// ------------------------------------------------------------------ factories

std::unique_ptr<DenseLayer> make_dense(int in_features, int out_features,
                                       ActivationFn act, unsigned int seed = 42);

// ------------------------------------------------------------------ inference

std::vector<float> forward(Model& m, const std::vector<float>& input);

// ------------------------------------------------------------------ training
// Runs forward + backward + SGD on one sample. Returns MSE loss.

float train_step(Model& m, const std::vector<float>& input,
                 const std::vector<float>& target, float lr);

}  // namespace ml
