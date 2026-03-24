
#include "activation.h"
#include "linear.h"

#include <cstdio>
#include <vector>
#include <string>

static const float kW1[8 * 2] = {
     4.56f,  4.57f,
    -4.58f, -4.61f,
     5.12f, -5.09f,
    -5.08f,  5.13f,
     3.91f,  3.88f,
    -3.92f, -3.87f,
     4.21f, -4.19f,
    -4.20f,  4.22f,
};

static const float kB1[8] = {
    -2.28f,  6.89f,  0.03f,  0.02f,
    -1.94f,  5.83f,  0.01f,  0.01f,
};

static const float kW2[1 * 8] = {
     5.32f, -4.98f,  6.41f,  6.38f,
    -5.30f, -6.12f, -5.01f, -5.03f,
};

static const float kB2[1] = { -2.71f };


static float run_forward(float x0, float x1) {
    constexpr int in_dim     = 2;
    constexpr int hidden_dim = 8;
    constexpr int out_dim    = 1;

    float x_in[in_dim] = { x0, x1 };
    float h_pre[hidden_dim];

    ispc::matmul(const_cast<float*>(kW1),
                 x_in,
                 h_pre,
                 hidden_dim, 1, in_dim);

    for (int i = 0; i < hidden_dim; ++i) h_pre[i] += kB1[i];

    float h[hidden_dim];
    ispc::sigmoid(h_pre, h, hidden_dim);

    float out_pre[out_dim];
    ispc::matmul(const_cast<float*>(kW2),
                 h,
                 out_pre,
                 out_dim, 1, hidden_dim);

    out_pre[0] += kB2[0];

    float out[out_dim];
    ispc::sigmoid(out_pre, out, out_dim);

    return out[0];
}

int main() {
    std::printf("XOR inference  (threshold 0.5)\n");
    std::printf("%-6s %-6s  %8s  %8s  %s\n",
                "x0", "x1", "output", "expected", "");
    std::printf("%s\n", std::string(40, '-').c_str());

    const float inputs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    const float expected[4]  = {  0,    1,    1,    0  };

    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float y    = run_forward(inputs[i][0], inputs[i][1]);
        int   pred = y >= 0.5f ? 1 : 0;
        bool  ok   = pred == static_cast<int>(expected[i]);
        correct   += ok;

        std::printf("%-6.0f %-6.0f  %8.4f  %8.0f  %s\n",
                    inputs[i][0], inputs[i][1],
                    y, expected[i],
                    ok ? "OK" : "WRONG");
    }

    std::printf("%s\n", std::string(40, '-').c_str());
    std::printf("Accuracy: %d/4\n", correct);
    return 0;
}
