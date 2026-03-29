#include <gtest/gtest.h>
#include "model.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;

// ------------------------------------------------------------------ DenseLayer construction

TEST(DenseLayerTest, CorrectDimensions) {
    const ml::DenseLayer layer{4, 8, ml::activation::relu()};
    EXPECT_EQ(layer.in_size(), 4);
    EXPECT_EQ(layer.out_size(), 8);
    EXPECT_EQ(layer.weights().size(), 32u);
    EXPECT_EQ(layer.bias().size(), 8u);
}

TEST(DenseLayerTest, WeightsAreNonZero) {
    const ml::DenseLayer layer{4, 8, ml::activation::relu()};
    const bool all_zero = std::all_of(layer.weights().begin(), layer.weights().end(),
                                      [](float w) { return w == 0.0f; });
    EXPECT_FALSE(all_zero);
}

TEST(DenseLayerTest, BiasesInitializedToZero) {
    const ml::DenseLayer layer{4, 8, ml::activation::relu()};
    for (float b : layer.bias()) EXPECT_FLOAT_EQ(b, 0.0f);
}

TEST(DenseLayerTest, WeightsWithinXavierRange) {
    constexpr int in = 4, out = 8;
    const ml::DenseLayer layer{in, out, ml::activation::sigmoid()};
    const float limit = std::sqrt(6.0f / static_cast<float>(in + out));
    for (float w : layer.weights()) {
        EXPECT_GE(w, -limit);
        EXPECT_LE(w,  limit);
    }
}

TEST(DenseLayerTest, SameSeedProducesSameWeights) {
    const ml::DenseLayer l1{4, 4, ml::activation::relu(), 99};
    const ml::DenseLayer l2{4, 4, ml::activation::relu(), 99};
    EXPECT_EQ(l1.weights(), l2.weights());
}

TEST(DenseLayerTest, DifferentSeedProducesDifferentWeights) {
    const ml::DenseLayer l1{4, 4, ml::activation::relu(), 1};
    const ml::DenseLayer l2{4, 4, ml::activation::relu(), 2};
    EXPECT_NE(l1.weights(), l2.weights());
}

// ------------------------------------------------------------------ forward

TEST(ForwardTest, OutputHasCorrectSize) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::relu()))
     .add(ml::make_dense(4, 1, ml::activation::sigmoid()));
    EXPECT_EQ(ml::forward(m, {0.5f, -0.3f}).size(), 1u);
}

TEST(ForwardTest, SigmoidOutputInOpenUnitInterval) {
    ml::Model m;
    m.add(ml::make_dense(2, 1, ml::activation::sigmoid()));
    const auto out = ml::forward(m, {1.0f, -1.0f});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_GT(out[0], 0.0f);
    EXPECT_LT(out[0], 1.0f);
}

TEST(ForwardTest, ReLUOutputNonNegative) {
    ml::Model m;
    m.add(ml::make_dense(3, 5, ml::activation::relu()));
    for (float v : ml::forward(m, {-1.0f, 0.5f, -2.0f})) EXPECT_GE(v, 0.0f);
}

TEST(ForwardTest, TanhOutputInOpenInterval) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::tanh()));
    for (float v : ml::forward(m, {3.0f, -3.0f})) {
        EXPECT_GT(v, -1.0f);
        EXPECT_LT(v,  1.0f);
    }
}

TEST(ForwardTest, SoftmaxOutputSumsToOne) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::softmax()));
    const auto out = ml::forward(m, {1.0f, -1.0f});
    EXPECT_NEAR(std::accumulate(out.begin(), out.end(), 0.0f), 1.0f, kTol);
}

TEST(ForwardTest, InputSizeMismatchThrows) {
    ml::Model m;
    m.add(ml::make_dense(3, 1, ml::activation::relu()));
    EXPECT_THROW(ml::forward(m, {1.0f, 2.0f}), std::invalid_argument);
}

// ------------------------------------------------------------------ Model::add compatibility

TEST(ModelAddTest, IncompatibleLayerSizesThrow) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::relu()));
    EXPECT_THROW(m.add(ml::make_dense(3, 1, ml::activation::sigmoid())),
                 std::invalid_argument);
}

TEST(ModelAddTest, CompatibleLayerSizesDoNotThrow) {
    ml::Model m;
    EXPECT_NO_THROW(
        m.add(ml::make_dense(2, 4, ml::activation::relu()))
         .add(ml::make_dense(4, 1, ml::activation::sigmoid()))
    );
}

// ------------------------------------------------------------------ train_step

TEST(TrainStepTest, ReturnsFiniteNonNegativeLoss) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::sigmoid()))
     .add(ml::make_dense(4, 1, ml::activation::sigmoid()));
    const float loss = ml::train_step(m, {0.0f, 1.0f}, {1.0f}, 0.1f);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GE(loss, 0.0f);
}

TEST(TrainStepTest, TargetSizeMismatchThrows) {
    ml::Model m;
    m.add(ml::make_dense(2, 1, ml::activation::sigmoid()));
    EXPECT_THROW(ml::train_step(m, {1.0f, 2.0f}, {0.0f, 1.0f}, 0.1f),
                 std::invalid_argument);
}

TEST(TrainStepTest, ZeroLearningRateDoesNotChangeWeights) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::sigmoid(), 42))
     .add(ml::make_dense(4, 1, ml::activation::sigmoid(), 43));
    const std::vector<float> before = m.layers()[0]->weights();
    ml::train_step(m, {0.5f, -0.5f}, {1.0f}, 0.0f);
    EXPECT_EQ(m.layers()[0]->weights(), before);
}

TEST(TrainStepTest, WeightsChangeAfterNonZeroLR) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::sigmoid(), 42))
     .add(ml::make_dense(4, 1, ml::activation::sigmoid(), 43));
    const std::vector<float> before = m.layers()[0]->weights();
    ml::train_step(m, {0.5f, -0.5f}, {1.0f}, 0.1f);
    EXPECT_NE(m.layers()[0]->weights(), before);
}

TEST(TrainStepTest, LossDecreasesOnXOR) {
    ml::Model m;
    m.add(ml::make_dense(2, 8, ml::activation::sigmoid(), 42))
     .add(ml::make_dense(8, 1, ml::activation::sigmoid(), 43));

    const std::vector<std::pair<std::vector<float>, std::vector<float>>> xor_data{
        {{0.0f, 0.0f}, {0.0f}},
        {{0.0f, 1.0f}, {1.0f}},
        {{1.0f, 0.0f}, {1.0f}},
        {{1.0f, 1.0f}, {0.0f}},
    };

    constexpr float lr = 0.5f;

    float first_loss = 0.0f;
    for (const auto& [x, y] : xor_data) first_loss += ml::train_step(m, x, y, lr);

    for (int epoch = 1; epoch < 5000; ++epoch)
        for (const auto& [x, y] : xor_data) ml::train_step(m, x, y, lr);

    float final_loss = 0.0f;
    for (const auto& [x, y] : xor_data) final_loss += ml::train_step(m, x, y, lr);

    EXPECT_LT(final_loss, first_loss);
}

}  // namespace
