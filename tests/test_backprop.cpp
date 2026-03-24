#include <gtest/gtest.h>
#include "backprop.h"

#include <cmath>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;

// ------------------------------------------------------------------ relu_backward
TEST(ReluBackwardTest, PositiveInputPassesGrad) {
    const std::vector<float> grad_out{1.0f, 2.0f, 3.0f};
    const std::vector<float> input{0.1f, 1.0f, 5.0f};
    std::vector<float> grad_in(3);

    ispc::relu_backward(const_cast<float*>(grad_out.data()),
                        const_cast<float*>(input.data()),
                        grad_in.data(), 3);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(grad_in[i], grad_out[i], kTol) << "at index " << i;
}

TEST(ReluBackwardTest, NegativeInputBlocksGrad) {
    const std::vector<float> grad_out{1.0f, 2.0f, 3.0f};
    const std::vector<float> input{-0.1f, -1.0f, -5.0f};
    std::vector<float> grad_in(3);

    ispc::relu_backward(const_cast<float*>(grad_out.data()),
                        const_cast<float*>(input.data()),
                        grad_in.data(), 3);

    for (int i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(grad_in[i], 0.0f) << "at index " << i;
}

TEST(ReluBackwardTest, ZeroInputBlocksGrad) {
    const std::vector<float> grad_out{5.0f};
    const std::vector<float> input{0.0f};
    std::vector<float> grad_in(1);

    ispc::relu_backward(const_cast<float*>(grad_out.data()),
                        const_cast<float*>(input.data()),
                        grad_in.data(), 1);

    EXPECT_FLOAT_EQ(grad_in[0], 0.0f);
}

TEST(SigmoidBackwardTest, ZeroOutputGivesQuarterGrad) {
    const std::vector<float> grad_out{1.0f};
    const std::vector<float> sig_out{0.5f};
    std::vector<float> grad_in(1);

    ispc::sigmoid_backward(const_cast<float*>(grad_out.data()),
                           const_cast<float*>(sig_out.data()),
                           grad_in.data(), 1);

    EXPECT_NEAR(grad_in[0], 0.25f, kTol);
}

TEST(SigmoidBackwardTest, SaturatedOutputGivesNearZeroGrad) {
    const std::vector<float> grad_out{1.0f};
    const std::vector<float> sig_out{0.9999f};
    std::vector<float> grad_in(1);

    ispc::sigmoid_backward(const_cast<float*>(grad_out.data()),
                           const_cast<float*>(sig_out.data()),
                           grad_in.data(), 1);

    EXPECT_NEAR(grad_in[0], 0.0f, 1e-3f);
}

TEST(TanhBackwardTest, ZeroTanhOutputGivesFullGrad) {
    const std::vector<float> grad_out{2.0f};
    const std::vector<float> tanh_out{0.0f};
    std::vector<float> grad_in(1);

    ispc::tanh_backward(const_cast<float*>(grad_out.data()),
                        const_cast<float*>(tanh_out.data()),
                        grad_in.data(), 1);

    EXPECT_NEAR(grad_in[0], 2.0f, kTol);
}

TEST(TanhBackwardTest, KnownDerivativeAtHalfPi) {
    const float tanh_val = std::tanh(1.0f);
    const float expected = 1.0f - tanh_val * tanh_val;

    const std::vector<float> grad_out{1.0f};
    const std::vector<float> tanh_out{tanh_val};
    std::vector<float> grad_in(1);

    ispc::tanh_backward(const_cast<float*>(grad_out.data()),
                        const_cast<float*>(tanh_out.data()),
                        grad_in.data(), 1);

    EXPECT_NEAR(grad_in[0], expected, kTol);
}

TEST(SoftmaxBackwardTest, UniformSoftmaxWithUniformGradGivesZero) {
    constexpr int n = 4;
    const float p = 1.0f / n;
    std::vector<float> grad_out(n, 1.0f);
    std::vector<float> softmax_out(n, p);
    std::vector<float> grad_in(n);

    ispc::softmax_backward(grad_out.data(), softmax_out.data(),
                           grad_in.data(), n);

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(grad_in[i], 0.0f, kTol) << "at index " << i;
}

TEST(SoftmaxBackwardTest, GradInputSumsToZero) {
    std::vector<float> grad_out{0.1f, 0.4f, 0.3f, 0.2f};
    std::vector<float> softmax_out{0.1f, 0.5f, 0.3f, 0.1f};
    std::vector<float> grad_in(4);

    ispc::softmax_backward(grad_out.data(), softmax_out.data(),
                           grad_in.data(), 4);

    float total = 0.0f;
    for (float g : grad_in) total += g;
    EXPECT_NEAR(total, 0.0f, kTol);
}

TEST(MseBackwardTest, KnownGradient) {
    const std::vector<float> pred{3.0f};
    const std::vector<float> target{1.0f};
    std::vector<float> grad(1);

    ispc::mse_backward(const_cast<float*>(pred.data()),
                       const_cast<float*>(target.data()),
                       grad.data(), 1);

    EXPECT_NEAR(grad[0], 4.0f, kTol);
}

TEST(MseBackwardTest, PerfectPredictionGivesZeroGrad) {
    const std::vector<float> pred{1.0f, 2.0f, 3.0f};
    const std::vector<float> target{1.0f, 2.0f, 3.0f};
    std::vector<float> grad(3);

    ispc::mse_backward(const_cast<float*>(pred.data()),
                       const_cast<float*>(target.data()),
                       grad.data(), 3);

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(grad[i], 0.0f, kTol) << "at index " << i;
}

TEST(MatmulBackwardATest, OneByOneCase) {
    std::vector<float> dC{3.0f};
    std::vector<float> B{2.0f};
    std::vector<float> dA(1);

    ispc::matmul_backward_A(dC.data(), B.data(), dA.data(), 1, 1, 1);

    EXPECT_NEAR(dA[0], 6.0f, kTol);
}

TEST(MatmulBackwardATest, GradientMatchesNumerical) {
    std::vector<float> dC{1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> B{5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> dA(4);
    const std::vector<float> expected{5.0f, 7.0f, 6.0f, 8.0f};

    ispc::matmul_backward_A(dC.data(), B.data(), dA.data(), 2, 2, 2);

    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(dA[i], expected[i], kTol) << "at index " << i;
}

TEST(MatmulBackwardBTest, OneByOneCase) {
    std::vector<float> A{3.0f};
    std::vector<float> dC{4.0f};
    std::vector<float> dB(1);

    ispc::matmul_backward_B(A.data(), dC.data(), dB.data(), 1, 1, 1);

    EXPECT_NEAR(dB[0], 12.0f, kTol);
}

TEST(MatmulBackwardBTest, GradientMatchesNumerical) {
    std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> dC{1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> dB(4);
    const std::vector<float> expected{1.0f, 3.0f, 2.0f, 4.0f};

    ispc::matmul_backward_B(A.data(), dC.data(), dB.data(), 2, 2, 2);

    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(dB[i], expected[i], kTol) << "at index " << i;
}

} // namespace
