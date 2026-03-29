#include <gtest/gtest.h>
#include "model.h"
#include "weight_io.h"

#include <string>
#include <vector>

namespace {

std::string tmp(const std::string& name) {
    return "/tmp/mltest_" + name + ".bin";
}

// ------------------------------------------------------------------ round-trip

TEST(WeightIOTest, RoundTripPreservesWeightsAndBiases) {
    ml::Model m;
    m.add(ml::make_dense(3, 5, ml::activation::relu(),    1))
     .add(ml::make_dense(5, 2, ml::activation::sigmoid(), 2));

    ml::save_weights(m, tmp("round_trip"));

    ml::Model m2;
    m2.add(ml::make_dense(3, 5, ml::activation::relu(),    99))
      .add(ml::make_dense(5, 2, ml::activation::sigmoid(), 99));
    ml::load_weights(m2, tmp("round_trip"));

    for (std::size_t l = 0; l < m.layers().size(); ++l) {
        EXPECT_EQ(m.layers()[l]->weights(), m2.layers()[l]->weights())
            << "Weights differ at layer " << l;
        EXPECT_EQ(m.layers()[l]->bias(), m2.layers()[l]->bias())
            << "Biases differ at layer " << l;
    }
}

TEST(WeightIOTest, LoadAfterTrainingPreservesForwardOutput) {
    ml::Model m;
    m.add(ml::make_dense(2, 4, ml::activation::sigmoid(), 7))
     .add(ml::make_dense(4, 1, ml::activation::sigmoid(), 8));

    for (int i = 0; i < 50; ++i) ml::train_step(m, {0.0f, 1.0f}, {1.0f}, 0.1f);

    ml::save_weights(m, tmp("after_training"));

    ml::Model m2;
    m2.add(ml::make_dense(2, 4, ml::activation::sigmoid(), 99))
      .add(ml::make_dense(4, 1, ml::activation::sigmoid(), 99));
    ml::load_weights(m2, tmp("after_training"));

    const auto out1 = ml::forward(m,  {0.0f, 1.0f});
    const auto out2 = ml::forward(m2, {0.0f, 1.0f});

    ASSERT_EQ(out1.size(), out2.size());
    EXPECT_FLOAT_EQ(out1[0], out2[0]);
}

// ------------------------------------------------------------------ error cases

TEST(WeightIOTest, LoadIntoWrongLayerCountThrows) {
    ml::Model m;
    m.add(ml::make_dense(3, 5, ml::activation::relu()));
    ml::save_weights(m, tmp("wrong_count"));

    ml::Model m2;
    m2.add(ml::make_dense(3, 5, ml::activation::relu()))
      .add(ml::make_dense(5, 1, ml::activation::sigmoid()));
    EXPECT_THROW(ml::load_weights(m2, tmp("wrong_count")), std::runtime_error);
}

TEST(WeightIOTest, LoadIntoWrongShapeThrows) {
    ml::Model m;
    m.add(ml::make_dense(3, 5, ml::activation::relu()));
    ml::save_weights(m, tmp("wrong_shape"));

    ml::Model m2;
    m2.add(ml::make_dense(4, 5, ml::activation::relu()));
    EXPECT_THROW(ml::load_weights(m2, tmp("wrong_shape")), std::runtime_error);
}

TEST(WeightIOTest, LoadFromNonExistentFileThrows) {
    ml::Model m;
    m.add(ml::make_dense(2, 2, ml::activation::relu()));
    EXPECT_THROW(ml::load_weights(m, "/tmp/mltest_no_such_file_xyz.bin"),
                 std::runtime_error);
}

}  // namespace
