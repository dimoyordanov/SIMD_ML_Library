#include <gtest/gtest.h>
#include "loss.h"

#include <vector>

namespace {

constexpr float kTol = 1e-5f;

// ------------------------------------------------------------------ mse_loss
TEST(MseLossTest, PerfectPredictionsGiveZeroLoss) {
    const std::vector<float> predictions{1.0f, 2.0f, 3.0f};
    const std::vector<float> targets{1.0f, 2.0f, 3.0f};
    float result{};

    ispc::mse_loss(const_cast<float*>(predictions.data()),
                   const_cast<float*>(targets.data()),
                   static_cast<int>(predictions.size()),
                   &result);

    EXPECT_NEAR(result, 0.0f, kTol);
}

TEST(MseLossTest, ConstantOffsetByOneGivesLossOfOne) {
    const std::vector<float> predictions{2.0f, 3.0f, 4.0f};
    const std::vector<float> targets{1.0f, 2.0f, 3.0f};
    float result{};

    ispc::mse_loss(const_cast<float*>(predictions.data()),
                   const_cast<float*>(targets.data()),
                   static_cast<int>(predictions.size()),
                   &result);

    EXPECT_NEAR(result, 1.0f, kTol);
}

TEST(MseLossTest, KnownTwoElementCase) {
    const std::vector<float> predictions{0.0f, 2.0f};
    const std::vector<float> targets{0.0f, 0.0f};
    float result{};

    ispc::mse_loss(const_cast<float*>(predictions.data()),
                   const_cast<float*>(targets.data()),
                   static_cast<int>(predictions.size()),
                   &result);

    EXPECT_NEAR(result, 2.0f, kTol);
}

TEST(MseLossTest, SingleElementCase) {
    const std::vector<float> predictions{5.0f};
    const std::vector<float> targets{3.0f};
    float result{};

    ispc::mse_loss(const_cast<float*>(predictions.data()),
                   const_cast<float*>(targets.data()),
                   1,
                   &result);

    EXPECT_NEAR(result, 4.0f, kTol);
}

} // namespace
