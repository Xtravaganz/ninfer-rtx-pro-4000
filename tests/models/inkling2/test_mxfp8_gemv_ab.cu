// Inkling 2 Mini Stage 3C: same-device, same-stream CUDA A/B evaluation.
// Tests Stage 3B (unchanged) and all Stage 3C candidates against independent
// Stage 3A C++ FP64 numerical oracle, with per-output error bounds.
#include "models/inkling2/mxfp8_gemv.h"
#include "models/inkling2/mxfp8_gemv_ab.h"
#include "models/inkling2/mxfp8_reference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace nk = ninfer::models::inkling2;
namespace oracle = ninfer::models::inkling2::mxfp8_reference;
namespace {

constexpr int kVariantCount = 4;

void require(bool truth, const char* context) {
    if (!truth) throw std::runtime_error(context);
}
void cuda_check(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

struct Stream {
    cudaStream_t handle = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "stream"); }
    ~Stream() { if (handle) (void)cudaStreamDestroy(handle); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) (void)cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& host, cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(ptr, host.data(), sizeof(T) * host.size(),
                                   cudaMemcpyHostToDevice, stream), "H2D");
    }
    void download(std::vector<T>& host, cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(host.data(), ptr, sizeof(T) * host.size(),
                                   cudaMemcpyDeviceToHost, stream), "D2H");
    }
};

struct Event {
    cudaEvent_t handle = nullptr;
    Event() { cuda_check(cudaEventCreate(&handle), "cudaEventCreate"); }
    ~Event() { if (handle) (void)cudaEventDestroy(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct Fixture {
    int rows, cols;
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> x;
    Fixture(int n, int k) : rows(n), cols(k),
        codes(static_cast<std::size_t>(n) * k),
        scales(static_cast<std::size_t>(n) * (k / 32)), x(k) {
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; ++c) {
                auto w = static_cast<std::uint8_t>((r * 73 + c * 19 + (c / 32) * 29) & 255);
                if ((w & 0x7f) == 0x7f) w = 0x7e;
                codes[static_cast<std::size_t>(r) * k + c] = w;
            }
            for (int g = 0; g < k / 32; ++g) {
                scales[static_cast<std::size_t>(r) * (k / 32) + g] =
                    static_cast<std::uint8_t>(118 + (r * 7 + g * 5) % 24);
            }
        }
        for (int c = 0; c < k; ++c)
            x[c] = static_cast<float>((c % 41) - 20) * 0.03125f;
    }
};

const char* variant_name(int id) {
    if (id == 0) return "stage3b_warp_scalar";
    return nk::mxfp8_ab_variant_name(static_cast<nk::Mxfp8AbVariant>(id - 1));
}

cudaError_t launch(int id,
                   const std::uint8_t* codes, const std::uint8_t* scales,
                   const float* x, float* y, int rows, int cols,
                   cudaStream_t stream) {
    if (id == 0) return nk::launch_mxfp8_gemv(codes, scales, x, y, rows, cols, stream);
    if (id > 0 && id < kVariantCount) {
        return nk::launch_mxfp8_gemv_ab(
            codes, scales, x, y, rows, cols,
            static_cast<nk::Mxfp8AbVariant>(id - 1), stream);
    }
    return cudaErrorInvalidValue;
}

void check_outputs(const Fixture& fixture, const std::vector<double>& expected,
                   const std::vector<double>& absolute_terms,
                   const std::vector<float>& observed, int id) {
    double max_error = 0, max_fraction = 0;
    int worst = 0;
    for (int r = 0; r < fixture.rows; ++r) {
        const double error = std::abs(static_cast<double>(observed[r]) - expected[r]);
        // Per-row condition-aware bound, unlike a global max(output) tolerance.
        // Slightly tighter than Stage 3B's original 100*eps; all variants use
        // exactly the same budget calculated from the independent FP64 oracle.
        const double bound = 2e-5 + 24.0 *
            static_cast<double>(std::numeric_limits<float>::epsilon()) * absolute_terms[r];
        if (!std::isfinite(observed[r]) || error > bound) {
            throw std::runtime_error(std::string(variant_name(id)) +
                " row=" + std::to_string(r) + " shape=" +
                std::to_string(fixture.rows) + "x" + std::to_string(fixture.cols) +
                " got=" + std::to_string(observed[r]) +
                " expected=" + std::to_string(expected[r]) +
                " error=" + std::to_string(error) + " bound=" + std::to_string(bound));
        }
        if (error / bound > max_fraction) worst = r;
        max_fraction = std::max(max_fraction, error / bound);
        max_error = std::max(max_error, error);
    }
    std::cout << "PASS " << variant_name(id) << " " << fixture.rows << "x"
              << fixture.cols << " max_abs_error=" << std::setprecision(9) << max_error
              << " max_error/bound=" << max_fraction << " worst_row=" << worst << '\n';
}

std::array<double, kVariantCount> bench(
    const Fixture& f, const DeviceBuffer<std::uint8_t>& dc,
    const DeviceBuffer<std::uint8_t>& ds, const DeviceBuffer<float>& dx,
    const DeviceBuffer<float>& dy, cudaStream_t stream) {
    constexpr int kWarmups = 48;
    constexpr int kIterations = 320;
    constexpr int kRounds = 5;
    for (int id = 0; id < kVariantCount; ++id) {
        for (int i = 0; i < kWarmups; ++i)
            cuda_check(launch(id, dc.ptr, ds.ptr, dx.ptr, dy.ptr,
                              f.rows, f.cols, stream), "warmup");
    }
    cuda_check(cudaStreamSynchronize(stream), "post-warmup synchronize");
    Event start, stop;
    std::array<std::vector<double>, kVariantCount> measurements;
    // Rotate order each round to reduce frequency/thermal-order bias.
    for (int round = 0; round < kRounds; ++round) {
        for (int position = 0; position < kVariantCount; ++position) {
            const int id = (position + round) % kVariantCount;
            cuda_check(cudaEventRecord(start.handle, stream), "record start");
            for (int iter = 0; iter < kIterations; ++iter) {
                cuda_check(launch(id, dc.ptr, ds.ptr, dx.ptr, dy.ptr,
                                  f.rows, f.cols, stream), "benchmark launch");
            }
            cuda_check(cudaEventRecord(stop.handle, stream), "record stop");
            cuda_check(cudaEventSynchronize(stop.handle), "benchmark synchronize");
            float elapsed_ms = 0.f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start.handle, stop.handle),
                       "cudaEventElapsedTime");
            measurements[id].push_back(static_cast<double>(elapsed_ms) * 1000 / kIterations);
        }
    }
    std::array<double, kVariantCount> medians{};
    const double bytes = static_cast<double>(f.rows) * f.cols * (1.0 + 1.0 / 32.0)
                         + 4.0 * (f.rows + f.cols);
    for (int id = 0; id < kVariantCount; ++id) {
        std::sort(measurements[id].begin(), measurements[id].end());
        const double us = measurements[id][kRounds / 2];
        medians[id] = us;
        const double effective_gibs = bytes / us * 1e6 / (1024.0 * 1024.0 * 1024.0);
        std::cout << "BENCH " << f.rows << "x" << f.cols << ' ' << variant_name(id)
                  << " median=" << std::fixed << std::setprecision(4) << us
                  << " us/call bandwidth=" << effective_gibs << " GiB/s"
                  << " speedup_vs_baseline=" << medians[0] / us << "x\n";
    }
    return medians;
}

void run_case(int rows, int cols, bool measure) {
    Fixture fixture(rows, cols);
    oracle::MatrixView ref{static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                           fixture.codes, fixture.scales};
    ref.validate();
    const auto expected = ref.matvec(fixture.x);
    std::vector<double> absolute_terms(rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            absolute_terms[r] += std::abs(ref.value(r, c) * static_cast<double>(fixture.x[c]));
    }
    Stream stream;
    DeviceBuffer<std::uint8_t> codes(fixture.codes.size()), scales(fixture.scales.size());
    DeviceBuffer<float> x(fixture.x.size()), y(static_cast<std::size_t>(rows));
    codes.upload(fixture.codes, stream.handle);
    scales.upload(fixture.scales, stream.handle);
    x.upload(fixture.x, stream.handle);
    std::vector<float> observed(rows);
    for (int id = 0; id < kVariantCount; ++id) {
        cuda_check(launch(id, codes.ptr, scales.ptr, x.ptr, y.ptr,
                          rows, cols, stream.handle), "candidate launch");
        y.download(observed, stream.handle);
        cuda_check(cudaStreamSynchronize(stream.handle), "candidate sync");
        check_outputs(fixture, expected, absolute_terms, observed, id);
    }
    if (measure) (void)bench(fixture, codes, scales, x, y, stream.handle);
}

float edge_result(int id, const std::vector<std::uint8_t>& codes,
                  const std::vector<std::uint8_t>& scales,
                  const std::vector<float>& x) {
    const int cols = static_cast<int>(codes.size());
    Stream stream;
    DeviceBuffer<std::uint8_t> dc(codes.size()), ds(scales.size());
    DeviceBuffer<float> dx(x.size()), dy(1);
    dc.upload(codes, stream.handle);
    ds.upload(scales, stream.handle);
    dx.upload(x, stream.handle);
    cuda_check(launch(id, dc.ptr, ds.ptr, dx.ptr, dy.ptr, 1, cols,
                      stream.handle), "edge launch");
    std::vector<float> result(1);
    dy.download(result, stream.handle);
    cuda_check(cudaStreamSynchronize(stream.handle), "edge sync");
    return result[0];
}

void check_edge_cases() {
    for (int id = 0; id < kVariantCount; ++id) {
        const float scale_min = edge_result(id, std::vector<std::uint8_t>(32, 0x38),
                                            {0x00}, std::vector<float>(32, 1.f));
        require(scale_min == std::ldexp(1.f, -122), "0x00 UE8M0 subnormal lost");
        std::vector<std::uint8_t> cancel(64, 0x38);
        for (int k = 32; k < 64; ++k) cancel[k] = 0xb8;
        require(edge_result(id, cancel, {0x7f, 0x7f},
                            std::vector<float>(64, 1.f)) == 0.f,
                "balanced +/- terms must cancel exactly");
        auto bad_code = std::vector<std::uint8_t>(32, 0x38);
        bad_code[7] = 0x7f;
        require(std::isnan(edge_result(id, bad_code, {0x7f},
                                        std::vector<float>(32, 1.f))), "NaN code lost");
        require(std::isnan(edge_result(id, std::vector<std::uint8_t>(32, 0x38),
                                        {0xff}, std::vector<float>(32, 1.f))), "NaN scale lost");
    }
    require(nk::launch_mxfp8_gemv_ab(nullptr, nullptr, nullptr, nullptr, 1, 32,
                                     nk::Mxfp8AbVariant::warp_vector4) ==
                cudaErrorInvalidValue, "null pointer accepted");
    Stream stream;
    DeviceBuffer<std::uint8_t> codes(64), scales(2);
    DeviceBuffer<float> x(65), y(1);
    require(nk::launch_mxfp8_gemv_ab(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                     0, 32, nk::Mxfp8AbVariant::split_k4,
                                     stream.handle) == cudaErrorInvalidValue,
            "zero N accepted");
    require(nk::launch_mxfp8_gemv_ab(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                     1, 33, nk::Mxfp8AbVariant::split_k4,
                                     stream.handle) == cudaErrorInvalidValue,
            "non-K32 accepted");
    require(nk::launch_mxfp8_gemv_ab(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                     1, 32, static_cast<nk::Mxfp8AbVariant>(99),
                                     stream.handle) == cudaErrorInvalidValue,
            "invalid variant accepted");
    require(nk::launch_mxfp8_gemv_ab(codes.ptr, scales.ptr, x.ptr + 1, y.ptr,
                                     1, 32, nk::Mxfp8AbVariant::warp_vector4,
                                     stream.handle) == cudaErrorInvalidValue,
            "misaligned vector4 input accepted");
    require(nk::launch_mxfp8_gemv_ab(codes.ptr + 1, scales.ptr, x.ptr, y.ptr,
                                     1, 32, nk::Mxfp8AbVariant::split_k4_vector4,
                                     stream.handle) == cudaErrorInvalidValue,
            "misaligned vector4 codes accepted");
    std::cout << "PASS Stage3C edge cases: UE8M0=0, cancellation, NaN, input contract\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        bool measure = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--bench") measure = true;
            else throw std::invalid_argument("usage: test_mxfp8_gemv_ab [--bench]");
        }
        int device = -1;
        cuda_check(cudaGetDevice(&device), "get device");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "get device properties");
        std::cout << "GPU " << device << ": " << prop.name << " sm_"
                  << prop.major << prop.minor << '\n';
        check_edge_cases();
        run_case(1, 32, false);
        run_case(7, 96, false);        // last vector tile partial
        run_case(129, 1152, false);   // partial final CTA
        run_case(128, 2304, false);
        run_case(2304, 1152, measure); // Inkling MXFP8 w2
        run_case(2304, 2304, measure); // Inkling MXFP8 w13
        std::cout << "OK Stage3C: baseline + vector4 + Split-K4 + Split-K4 vector4, "
                     "each tested against FP64 oracle\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL Stage3C: " << e.what() << '\n';
        return 1;
    }
}
