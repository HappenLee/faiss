#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/simd_levels.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using Distance = void (*)(float*, const float*, const float*, size_t, size_t);

struct Metric {
    const char* name;
    Distance compute;
    bool l2;
};

constexpr Metric metrics[] = {
        {"L2", faiss::fvec_L2sqr_ny, true},
        {"IP", faiss::fvec_inner_products_ny, false},
};

void fill(std::vector<float>& values, std::mt19937& rng) {
    for (auto& value : values) {
        value = static_cast<float>(static_cast<int>(rng() % 65537) - 32768) /
                32768.0f;
    }
}

// Use a double-precision reference independent of both library builds.
void check(std::mt19937& rng) {
    size_t checks = 0;
    double max_abs_error = 0;
    for (auto metric : metrics) {
        for (size_t d : {0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 32}) {
            for (size_t ny : {0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 255, 256, 257}) {
                // Offset data and output by one float to exercise unaligned NEON loads.
                std::vector<float> x(d + 1), y(d * ny + 1), out(ny + 2, 12345);
                fill(x, rng);
                fill(y, rng);
                metric.compute(out.data() + 1, x.data() + 1, y.data() + 1, d, ny);
                if (out.front() != 12345 || out.back() != 12345) {
                    throw std::runtime_error("out-of-range output write");
                }
                for (size_t i = 0; i < ny; ++i) {
                    double expected = 0;
                    double magnitude = 0;
                    for (size_t j = 0; j < d; ++j) {
                        const double a = x[1 + j];
                        const double b = y[1 + i * d + j];
                        const double term = metric.l2 ? (a - b) * (a - b) : a * b;
                        expected += term;
                        magnitude += std::abs(term);
                    }
                    const double error = std::abs(expected - out[1 + i]);
                    if (!(error <= 1e-7 + 2e-6 * magnitude)) {
                        throw std::runtime_error("distance differs from reference");
                    }
                    ++checks;
                    max_abs_error = std::max(max_abs_error, error);
                }
            }
        }
    }
    std::cout << "Checked " << checks << " outputs, max absolute error "
              << max_abs_error << '\n';
}

template <class Fn>
double time_ns(const Fn& run, const float* output) {
    using Clock = std::chrono::steady_clock;
    for (size_t count = 64;; count *= 2) {
        const auto start = Clock::now();
        for (size_t i = 0; i < count; ++i) {
            run();
            // Make the outputs observable even with link-time optimization.
            asm volatile("" : : "r"(output) : "memory");
        }
        const auto elapsed = Clock::now() - start;
        if (elapsed >= std::chrono::milliseconds(30)) {
            return std::chrono::duration<double, std::nano>(elapsed).count() / count;
        }
    }
}

void bench(std::mt19937& rng) {
    std::cout << std::fixed << std::setprecision(3);
    for (auto metric : metrics) {
        for (size_t d : {1, 2, 4, 8, 12, 16, 32}) {
            for (size_t ny : {4, 17, 255, 256, 1024}) {
                std::vector<float> x(d), y(d * ny), out(ny);
                fill(x, rng);
                fill(y, rng);
                const double ns = time_ns(
                        [&] { metric.compute(out.data(), x.data(), y.data(), d, ny); },
                        out.data());
                std::cout << "batch," << metric.name << ',' << d << ',' << ny
                          << ",1," << ns << '\n';
            }
        }
        // 8-bit PQ lookup-table construction. No index training or list scan.
        constexpr size_t subquantizers = 96;
        for (size_t dsub : {1, 2, 4, 8, 12}) {
            faiss::ProductQuantizer pq(subquantizers * dsub, subquantizers, 8);
            std::vector<float> x(pq.d), out(pq.M * pq.ksub);
            fill(x, rng);
            fill(pq.centroids, rng);
            const double ns = time_ns(
                    [&] {
                        if (metric.l2) {
                            pq.compute_distance_table(x.data(), out.data());
                        } else {
                            pq.compute_inner_prod_table(x.data(), out.data());
                        }
                    },
                    out.data());
            std::cout << "pq_table," << metric.name << ',' << dsub << ',' << pq.ksub
                      << ',' << pq.M << ',' << ns << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (faiss::SIMDConfig::get_dispatched_level() != faiss::SIMDLevel::ARM_NEON) {
        throw std::runtime_error("benchmark requires the ARM_NEON dispatch path");
    }
    std::mt19937 rng(20261123);
    if (argc == 2 && std::string_view(argv[1]) == "--check") {
        check(rng);
    } else {
        bench(rng);
    }
}
