#include <gtest/gtest.h>
#include "linear.h"

#include <vector>

namespace {

constexpr float kTol = 1e-4f;

// ------------------------------------------------------------------ dot_product
TEST(DotProductTest, OrthogonalVectorsGiveZero) {
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{0.0f, 1.0f};
    
    float result = ispc::dot_product(a.data(), b.data(), static_cast<int>(a.size()));

    EXPECT_NEAR(result, 0.0f, kTol);
}

TEST(DotProductTest, KnownVectorsGiveCorrectInnerProduct) {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{4.0f, 5.0f, 6.0f};
    float result = ispc::dot_product(a.data(), b.data(), static_cast<int>(a.size()));

    EXPECT_NEAR(result, 32.0f, kTol);
}

TEST(DotProductTest, UnitVectorWithItselfGivesOne) {
    std::vector<float> a{1.0f, 0.0f, 0.0f};
    std::vector<float> b{1.0f, 0.0f, 0.0f};
    float result = ispc::dot_product(a.data(), b.data(), static_cast<int>(a.size()));

    EXPECT_NEAR(result, 1.0f, kTol);
}

// ------------------------------------------------------------------ matmul
TEST(MatmulTest, IdentityMatrixLeavesInputUnchanged) {
    const int M = 2, N = 2, K = 2;
    std::vector<float> identity{1.0f, 0.0f,
                                0.0f, 1.0f};
    std::vector<float> A{3.0f, 7.0f,
                         1.0f, 5.0f};
    std::vector<float> C(M * N, 0.0f);

    ispc::matmul(identity.data(), A.data(), C.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], A[i], kTol) << "at flat index " << i;
    }
}

TEST(MatmulTest, KnownTwoByTwoProduct) {
    const int M = 2, N = 2, K = 2;
    std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B{5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> expected{19.0f, 22.0f, 43.0f, 50.0f};
    std::vector<float> C(M * N, 0.0f);

    ispc::matmul(A.data(), B.data(), C.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], expected[i], kTol) << "at flat index " << i;
    }
}

TEST(MatmulTest, NonSquareMatrix) {
    const int M = 2, N = 2, K = 3;
    std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> B{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const std::vector<float> expected{58.0f, 64.0f, 139.0f, 154.0f};
    std::vector<float> C(M * N, 0.0f);

    ispc::matmul(A.data(), B.data(), C.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], expected[i], kTol) << "at flat index " << i;
    }
}



// ------------------------------------------------------------------ matmul
TEST(Matmul2Test, IdentityMatrixLeavesInputUnchanged) {
    const int M = 2, N = 2, K = 2;
    std::vector<float> identity{1.0f, 0.0f,
                                0.0f, 1.0f};
    std::vector<float> A{3.0f, 7.0f,
                         1.0f, 5.0f};
    std::vector<float> C(M * N, 0.0f);

    ispc::matmul2(identity.data(), A.data(), C.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], A[i], kTol) << "at flat index " << i;
    }
}

TEST(Matmul2Test, KnownTwoByTwoProduct) {
    const int M = 2, N = 2, K = 2;
    std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B{5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> expected{19.0f, 22.0f, 43.0f, 50.0f};
    std::vector<float> C(M * N, 0.0f);

    ispc::matmul2(A.data(), B.data(), C.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], expected[i], kTol) << "at flat index " << i;
    }
}

TEST(Matmul2Test, NonSquareMatrix) {
    const int M = 2, N = 2, K = 3;
    std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> B{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const std::vector<float> expected{58.0f, 64.0f, 139.0f, 154.0f};
    std::vector<float> C(M * N, 0.0f);

    ispc::matmul2(A.data(), B.data(), C.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], expected[i], kTol) << "at flat index " << i;
    }
}



} // namespace
