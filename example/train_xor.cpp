#include "model.h"
#include "weight_io.h"

#include <cstdio>
#include <string>
#include <vector>

// XOR dataset: inputs and expected outputs
static const std::vector<std::vector<float>> kInputs  = {{0,0}, {0,1}, {1,0}, {1,1}};
static const std::vector<std::vector<float>> kTargets = {{0},   {1},   {1},   {0}  };

static float epoch_loss(ml::Model& m) {
    float total = 0.0f;
    for (std::size_t i = 0; i < kInputs.size(); ++i)
        total += ml::train_step(m, kInputs[i], kTargets[i], 0.0f);  // lr=0: measure only
    return total / static_cast<float>(kInputs.size());
}

int main() {
    // ---- build model --------------------------------------------------------
    ml::Model m;
    m.add(ml::make_dense(2, 8, ml::activation::sigmoid(), 42))
     .add(ml::make_dense(8, 1, ml::activation::sigmoid(), 43));

    std::printf("Training 2→8→1 sigmoid network on XOR\n");
    std::printf("%s\n", std::string(50, '-').c_str());

    // ---- train --------------------------------------------------------------
    constexpr int   kEpochs    = 5000;
    constexpr float kLR        = 0.5f;
    constexpr int   kLogEvery  = 500;

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        float loss = 0.0f;
        for (std::size_t i = 0; i < kInputs.size(); ++i)
            loss += ml::train_step(m, kInputs[i], kTargets[i], kLR);
        loss /= static_cast<float>(kInputs.size());

        if (epoch % kLogEvery == 0)
            std::printf("epoch %5d  loss %.6f\n", epoch, loss);
    }

    // ---- evaluate -----------------------------------------------------------
    std::printf("%s\n", std::string(50, '-').c_str());
    std::printf("%-6s %-6s  %8s  %8s  %s\n", "x0", "x1", "output", "expected", "");
    std::printf("%s\n", std::string(50, '-').c_str());

    int correct = 0;
    for (std::size_t i = 0; i < kInputs.size(); ++i) {
        const auto out      = ml::forward(m, kInputs[i]);
        const int  pred     = out[0] >= 0.5f ? 1 : 0;
        const int  expected = static_cast<int>(kTargets[i][0]);
        correct += (pred == expected);
        std::printf("%-6.0f %-6.0f  %8.4f  %8d  %s\n",
                    kInputs[i][0], kInputs[i][1], out[0], expected,
                    pred == expected ? "OK" : "WRONG");
    }

    std::printf("%s\n", std::string(50, '-').c_str());
    std::printf("Accuracy: %d/%zu\n\n", correct, kInputs.size());

    // ---- save & reload ------------------------------------------------------
    const std::string weights_path = "xor_weights.bin";
    ml::save_weights(m, weights_path);
    std::printf("Weights saved to %s\n", weights_path.c_str());

    ml::Model loaded;
    loaded.add(ml::make_dense(2, 8, ml::activation::sigmoid()))
          .add(ml::make_dense(8, 1, ml::activation::sigmoid()));
    ml::load_weights(loaded, weights_path);
    std::printf("Weights reloaded — verifying forward pass matches:\n");

    bool mismatch = false;
    for (const auto& x : kInputs) {
        const auto o1 = ml::forward(m,      x);
        const auto o2 = ml::forward(loaded, x);
        if (o1[0] != o2[0]) { mismatch = true; break; }
    }
    std::printf("%s\n", mismatch ? "MISMATCH — load failed!" : "All outputs match.");

    return 0;
}
