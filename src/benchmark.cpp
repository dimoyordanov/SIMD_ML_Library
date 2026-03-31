#include "activation.h"
#include "backprop.h"
#include "linear.h"
#include "loss.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

template <typename Fn>
double time_us(Fn&& fn, int reps = 10) {
    auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

static std::vector<float> random_vec(int n, float lo = -1.0f, float hi = 1.0f,
                                     unsigned long long seed = 2) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

static void print_speedup_header(int N = 1 << 20) {
    std::printf("\n=== ISPC vs scalar speedup  (N=%d) ===\n\n", N);
    std::printf("| %-22s | %10s | %10s | %7s |\n",
                "function", "scalar(us)", "ispc(us)", "speedup");
    std::printf("| %s | %s | %s | %s |\n", std::string(22, '-').c_str(), std::string(10, '-').c_str(),
                std::string(10, '-').c_str(), std::string(8, '-').c_str());
}

static void print_cache_sensitivity_header() {
    std::printf("\n=== Cache-size sensitivity: ReLU throughput ===\n");
    std::printf("(reads input[] + writes output[] = 2 arrays × float = 8 bytes/elem)\n\n");
    std::printf("| %-14s | %10s | %12s | %12s |\n",
                "size (floats)", "bytes", "ispc (us)", "ispc (GB/s)");
    std::printf("| %s | %s | %s | %s |\n", std::string(14, '-').c_str(), std::string(10, '-').c_str(),
                std::string(12, '-').c_str(), std::string(12, '-').c_str());
}

static void print_cache_sensitivity_row(int size, const char* name,
                                        double bytes, double ispc_us, double gbs) {
    std::printf("| %8d %-5s | %10.2f | %12.2f | %12.2f |\n",
                size, name, bytes, ispc_us, gbs);
}

static void print_speedup_row(const char* name,
                               double scalar_us, double ispc_us) {
    std::printf("| %-22s | %10.2f | %10.2f | %7.2fx |\n",
                name, scalar_us, ispc_us, scalar_us / ispc_us - 1);
}

static void benchmark_speedup() {
    constexpr int N = 1 << 20;

    auto inp  = random_vec(N, -1.0f, 1.0f, time(0));
    auto inp2 = random_vec(N, -1.0f, 1.0f, time(0) + 1);
    std::vector<float> out(N);

    print_speedup_header();

    // relu
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i) out[i] = std::max(inp[i], 0.0f);
        });
        auto ispc_us = time_us([&] {
            ispc::relu(inp.data(), out.data(), N);
        });
        print_speedup_row("relu", scalar_us, ispc_us);
    }

    // sigmoid
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i)
                out[i] = 1.0f / (1.0f + std::exp(-inp[i]));
        });
        auto ispc_us = time_us([&] {
            ispc::sigmoid(inp.data(), out.data(), N);
        });
        print_speedup_row("sigmoid", scalar_us, ispc_us);
    }

    // tanh
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i) out[i] = std::tanh(inp[i]);
        });
        auto ispc_us = time_us([&] {
            ispc::tanh_activation(inp.data(), out.data(), N);
        });
        print_speedup_row("tanh", scalar_us, ispc_us);
    }

     // softmax
     {
        auto scalar_us = time_us([&] {
            float sum_exp = 0.0f;
            for (int i = 0; i < N; ++i) {
                out[i] = std::exp(inp[i]);
                sum_exp += out[i];
            }
            for (int i = 0; i < N; ++i) out[i] /= sum_exp;
        });
        auto ispc_us = time_us([&] {
            ispc::softmax(inp.data(), out.data(), N);
        });
        print_speedup_row("softmax", scalar_us, ispc_us);
    }

    // dot_product
    {
        float s = 0.0f;
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i) s += inp[i] * inp2[i];
        });
        volatile float x = s;
        
        auto ispc_us = time_us([&] {
            ispc::dot_product(inp.data(), inp2.data(), N);
        });
        print_speedup_row("dot_product", scalar_us, ispc_us);
    }

    // matmul 512x512
    {
        constexpr int M = 512, K = 512;
        auto A = random_vec(M * K, -1.0f, 1.0f, time(0));
        auto B = random_vec(K * M, -1.0f, 1.0f, time(0) + 1);
        std::vector<float> C(M * M);

        auto scalar_us = time_us([&] {
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < M; ++c) {
                    float s = 0.0f;
                    for (int k = 0; k < K; ++k)
                        s += A[r * K + k] * B[k * M + c];
                    C[r * M + c] = s;
                }
        });
        auto ispc_us = time_us([&] {
            ispc::matmul(A.data(), B.data(), C.data(), M, M, K);
        });
        print_speedup_row("matmul 512x512", scalar_us, ispc_us);
    }

    // matmul2 512x512
    {
        constexpr int M = 512, K = 512;
        auto A = random_vec(M * K, -1.0f, 1.0f, time(0));
        auto B = random_vec(K * M, -1.0f, 1.0f, time(0) + 1);
        std::vector<float> C(M * M);

        auto scalar_us = time_us([&] {
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < M; ++c) {
                    float s = 0.0f;
                    for (int k = 0; k < K; ++k)
                        s += A[r * K + k] * B[k * M + c];
                    C[r * M + c] = s;
                }
        });
        auto ispc_us = time_us([&] {
            ispc::matmul2(A.data(), B.data(), C.data(), M, M, K);
        });
        print_speedup_row("matmul2 512x512", scalar_us, ispc_us);
    }

    // mse_loss
    {
        float result{};
        auto scalar_us = time_us([&] {
            float s = 0.0f;
            for (int i = 0; i < N; ++i) {
                float d = inp[i] - inp2[i];
                s += d * d;
            }
            result = s / N;
        });
        auto ispc_us = time_us([&] {
            ispc::mse_loss(inp.data(), inp2.data(), N, &result);
        });
        print_speedup_row("mse_loss", scalar_us, ispc_us);
    }
}


static void benchmark_backprop_speedup() {
    constexpr int N = 1 << 20;

    auto inp  = random_vec(N, -1.0f, 1.0f, time(0));
    auto inp2 = random_vec(N,  0.0f, 1.0f, time(0) + 1);  // post-activation range
    std::vector<float> out(N);

    std::printf("\n=== Backprop ISPC vs scalar speedup  (N=%d) ===\n\n", N);
    print_speedup_header();

    // relu_backward
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i)
                out[i] = inp[i] > 0.0f ? inp2[i] : 0.0f;
        });
        auto ispc_us = time_us([&] {
            ispc::relu_backward(inp2.data(), inp.data(), out.data(), N);
        });
        print_speedup_row("relu_backward", scalar_us, ispc_us);
    }

    // sigmoid_backward
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i)
                out[i] = inp2[i] * inp[i] * (1.0f - inp[i]);
        });
        auto ispc_us = time_us([&] {
            ispc::sigmoid_backward(inp2.data(), inp.data(), out.data(), N);
        });
        print_speedup_row("sigmoid_backward", scalar_us, ispc_us);
    }

    // tanh_backward
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i)
                out[i] = inp2[i] * (1.0f - inp[i] * inp[i]);
        });
        auto ispc_us = time_us([&] {
            ispc::tanh_backward(inp2.data(), inp.data(), out.data(), N);
        });
        print_speedup_row("tanh_backward", scalar_us, ispc_us);
    }

    // mse_backward
    {
        auto scalar_us = time_us([&] {
            for (int i = 0; i < N; ++i)
                out[i] = 2.0f * (inp[i] - inp2[i]) / static_cast<float>(N);
        });
        auto ispc_us = time_us([&] {
            ispc::mse_backward(inp.data(), inp2.data(), out.data(), N);
        });
        print_speedup_row("mse_backward", scalar_us, ispc_us);
    }

    // matmul_backward_A  512×512
    {
        constexpr int M = 512, K = 512;
        auto dC = random_vec(M * 1,   -1.0f, 1.0f, time(0));
        auto B  = random_vec(K * 1,   -1.0f, 1.0f, time(0) + 1);
        std::vector<float> dA(M * K);

        auto scalar_us = time_us([&] {
            for (int r = 0; r < M; ++r)
                for (int c = 0; c < K; ++c)
                    dA[r * K + c] = dC[r] * B[c];
        });
        auto ispc_us = time_us([&] {
            ispc::matmul_backward_A(dC.data(), B.data(), dA.data(), M, 1, K);
        });
        print_speedup_row("matmul_backward_A 512", scalar_us, ispc_us);
    }

    // matmul_backward_B  512×512
    {
        constexpr int M = 512, K = 512;
        auto A  = random_vec(M * K, -1.0f, 1.0f, time(0));
        auto dC = random_vec(M * 1, -1.0f, 1.0f, time(0) + 1);
        std::vector<float> dB(K * 1);

        auto scalar_us = time_us([&] {
            for (int r = 0; r < K; ++r) {
                float s = 0.0f;
                for (int m = 0; m < M; ++m) s += A[m * K + r] * dC[m];
                dB[r] = s;
            }
        });
        auto ispc_us = time_us([&] {
            ispc::matmul_backward_B(A.data(), dC.data(), dB.data(), M, 1, K);
        });
        print_speedup_row("matmul_backward_B 512", scalar_us, ispc_us);
    }
}

static void benchmark_cache_sensitivity() {
    const std::vector<int> sizes = {
        1 << 10,
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
    };

    print_cache_sensitivity_header();

    for (int n : sizes) {
        auto inp = random_vec(n, -1.0f, 1.0f, time(0));
        std::vector<float> out(n);

        ispc::relu(inp.data(), out.data(), n);

        const int reps = std::max(10, 1 << 24 >> static_cast<int>(std::log2(n)));
        double us = time_us([&] {
            ispc::relu(inp.data(), out.data(), n);
        }, reps);

        double bytes_transferred = 2.0 * n * sizeof(float);
        double gb_per_s = (bytes_transferred / 1e9) / (us / 1e6);

        const char* unit = "floats";
        print_cache_sensitivity_row(n, unit, bytes_transferred, us, gb_per_s);
    }
}


int main() {
    benchmark_speedup();
    benchmark_backprop_speedup();
    benchmark_cache_sensitivity();
    std::printf("\n");
    return 0;
}
