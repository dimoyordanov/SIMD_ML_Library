#include <gtest/gtest.h>
#include "activation.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;

// ------------------------------------------------------------------ relu
TEST(ReluTest, NegativeInputsClampedToZero) {
    const std::vector<float> input{-3.0f, -1.0f, -0.001f};
    std::vector<float> output(input.size());

    ispc::relu(const_cast<float*>(input.data()), output.data(),
               static_cast<int>(input.size()));

    for (std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "at index " << i;
    }
}

TEST(ReluTest, ZeroPassesThrough) {
    const std::vector<float> input{0.0f};
    std::vector<float> output(1);

    ispc::relu(const_cast<float*>(input.data()), output.data(), 1);

    EXPECT_FLOAT_EQ(output[0], 0.0f);
}

TEST(ReluTest, PositiveInputsPassThrough) {
    const std::vector<float> input{0.001f, 1.0f, 42.0f};
    std::vector<float> output(input.size());

    ispc::relu(const_cast<float*>(input.data()), output.data(),
               static_cast<int>(input.size()));

    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], input[i], kTol) << "at index " << i;
    }
}

TEST(SigmoidTest, OutputAlwaysInOpenUnitInterval) {
    const std::vector<float> input{-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    std::vector<float> output(input.size());

    ispc::sigmoid(const_cast<float*>(input.data()), output.data(),
                  static_cast<int>(input.size()));

    for (std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_GT(output[i], 0.0f) << "at index " << i;
        EXPECT_LT(output[i], 1.0f) << "at index " << i;
    }
}

TEST(SigmoidTest, ZeroInputGivesHalf) {
    const std::vector<float> input{0.0f};
    std::vector<float> output(1);

    ispc::sigmoid(const_cast<float*>(input.data()), output.data(), 1);

    EXPECT_NEAR(output[0], 0.5f, kTol);
}

TEST(SigmoidTest, LargePositiveApproachesOne) {
    const std::vector<float> input{100.0f};
    std::vector<float> output(1);

    ispc::sigmoid(const_cast<float*>(input.data()), output.data(), 1);

    EXPECT_NEAR(output[0], 1.0f, kTol);
}

TEST(SigmoidTest, LargeNegativeApproachesZero) {
    const std::vector<float> input{-100.0f};
    std::vector<float> output(1);

    ispc::sigmoid(const_cast<float*>(input.data()), output.data(), 1);

    EXPECT_NEAR(output[0], 0.0f, kTol);
}

// ------------------------------------------------------------------ tanh
TEST(TanhActivationTest, ZeroInputGivesZero) {
    const std::vector<float> input{0.0f};
    std::vector<float> output(1);

    ispc::tanh_activation(const_cast<float*>(input.data()), output.data(), 1);

    EXPECT_NEAR(output[0], 0.0f, kTol);
}

TEST(TanhActivationTest, OutputAlwaysInOpenIntervalMinusOneToOne) {
    const std::vector<float> input{-8.0f, -1.0f, 0.0f, 1.0f, 8.0f};
    std::vector<float> output(input.size());

    ispc::tanh_activation(const_cast<float*>(input.data()), output.data(),
                          static_cast<int>(input.size()));

    for (std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_GT(output[i], -1.0f) << "at index " << i;
        EXPECT_LT(output[i],  1.0f) << "at index " << i;
    }
}

TEST(TanhActivationTest, MatchesStdTanh) {
    const std::vector<float> input{-2.0f, -0.5f, 0.5f, 2.0f};
    std::vector<float> output(input.size());

    ispc::tanh_activation(const_cast<float*>(input.data()), output.data(),
                          static_cast<int>(input.size()));

    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], std::tanh(input[i]), kTol) << "at index " << i;
    }
}

TEST(SoftmaxTest, OutputsSumToOne) {
    std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(input.size());

    ispc::softmax(const_cast<float*>(input.data()), output.data(),
                  static_cast<int>(input.size()));

    const float total = std::accumulate(output.begin(), output.end(), 0.0f);
    EXPECT_NEAR(total, 1.0f, kTol);
}

TEST(SoftmaxTest, AllOutputsInOpenUnitInterval) {
    std::vector<float> input{-1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(input.size());

    ispc::softmax(const_cast<float*>(input.data()), output.data(),
                  static_cast<int>(input.size()));

    for (std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_GT(output[i], 0.0f) << "at index " << i;
        EXPECT_LT(output[i], 1.0f) << "at index " << i;
    }
}

TEST(SoftmaxTest, UniformInputGivesEqualProbabilities) {
    const int n = 4;
    std::vector<float> input(n, 0.0f);
    std::vector<float> output(n);

    ispc::softmax(const_cast<float*>(input.data()), output.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(output[i], 1.0f / n, kTol) << "at index " << i;
    }
}

TEST(SoftmaxTest, LargestInputGetHighestProbability) {
    std::vector<float> input{1.0f, 5.0f, 2.0f, 3.0f};
    std::vector<float> output(input.size());

    ispc::softmax(const_cast<float*>(input.data()), output.data(),
                  static_cast<int>(input.size()));

    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i != 1) EXPECT_LT(output[i], output[1]) << "at index " << i;
    }
}


TEST(SoftmaxTest, KnownTwoClassValues) {
    std::vector<float> input{0.0f, 1.0f};
    std::vector<float> output(2);

    ispc::softmax(const_cast<float*>(input.data()), output.data(), 2);

    const float e = std::exp(1.0f);
    EXPECT_NEAR(output[0], 1.0f / (1.0f + e), kTol);
    EXPECT_NEAR(output[1], e / (1.0f + e),     kTol);
}


} // namespace
