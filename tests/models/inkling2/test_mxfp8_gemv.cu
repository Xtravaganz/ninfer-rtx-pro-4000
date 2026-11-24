// Stage 3B: native CUDA executable, compared against independent Stage-3A
// C++ double-precision MatrixView oracle. No Torch or model download required.
#include "models/inkling2/mxfp8_gemv.h"
#include "models/inkling2/mxfp8_reference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = ninfer::models::inkling2::mxfp8_reference;
namespace nk = ninfer::models::inkling2;
namespace {

void require(bool condition, const char* why) {
    if (!condition) throw std::runtime_error(why);
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

template <class T> struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) (void)cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& host, cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(ptr, host.data(), sizeof(T) * host.size(),
                                   cudaMemcpyHostToDevice, stream), "upload");
    }
    void download(std::vector<T>& host, cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(host.data(), ptr, sizeof(T) * host.size(),
                                   cudaMemcpyDeviceToHost, stream), "download");
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
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;
    std::vector<float> x;

    Fixture(int n, int k) : rows(n), cols(k), codes(static_cast<std::size_t>(n)*k),
        scales(static_cast<std::size_t>(n)*(k/32)), x(k) {
        for (int row=0; row<n; ++row) {
            for (int col=0; col<k; ++col) {
                auto word = static_cast<std::uint8_t>(
                    (row*73 + col*19 + (col/32)*29) & 0xff);
                if ((word & 0x7fu)==0x7fu) word = 0x7eu;
                codes[static_cast<std::size_t>(row)*k+col] = word;
            }
            for (int block=0; block<k/32; ++block) {
                scales[static_cast<std::size_t>(row)*(k/32)+block] =
                    static_cast<std::uint8_t>(118+(row*7+block*5)%24);
            }
        }
        for (int col=0; col<k; ++col) {
            x[col] = static_cast<float>((col%41)-20) * 0.03125f;
        }
    }
};

void check_native_case(int rows, int cols, bool benchmark) {
    Fixture fixture(rows, cols);
    mx::MatrixView oracle{static_cast<std::size_t>(rows),
                          static_cast<std::size_t>(cols),
                          fixture.codes, fixture.scales};
    oracle.validate();
    const auto expected = oracle.matvec(fixture.x);

    Stream stream;
    DeviceBuffer<std::uint8_t> codes(fixture.codes.size());
    DeviceBuffer<std::uint8_t> scales(fixture.scales.size());
    DeviceBuffer<float> x(fixture.x.size());
    DeviceBuffer<float> y(static_cast<std::size_t>(rows));
    codes.upload(fixture.codes, stream.handle);
    scales.upload(fixture.scales, stream.handle);
    x.upload(fixture.x, stream.handle);
    cuda_check(nk::launch_mxfp8_gemv(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                      rows, cols, stream.handle), "launch");
    std::vector<float> observed(static_cast<std::size_t>(rows));
    y.download(observed, stream.handle);
    cuda_check(cudaStreamSynchronize(stream.handle), "synchronize");

    double max_abs_error = 0, max_relative_to_bound = 0;
    int worst_row = 0;
    for (int row=0; row<rows; ++row) {
        double absolute_terms = 0;
        for (int col=0; col<cols; ++col) {
            absolute_terms += std::abs(
                oracle.value(static_cast<std::size_t>(row), static_cast<std::size_t>(col))
                * static_cast<double>(fixture.x[col]));
        }
        const double error = std::abs(static_cast<double>(observed[row]) - expected[row]);
        // Per-output, input-conditioned FP32 accumulation bound. NOT one global
        // threshold over all rows. Cancellation is tested separately below.
        const double budget = 2e-5 + 100.0 *
            static_cast<double>(std::numeric_limits<float>::epsilon()) * absolute_terms;
        if (!std::isfinite(observed[row]) || error > budget) {
            throw std::runtime_error("GEMV row=" + std::to_string(row) +
                " N=" + std::to_string(rows) + " K=" + std::to_string(cols) +
                " got=" + std::to_string(observed[row]) +
                " reference=" + std::to_string(expected[row]) +
                " abs_error=" + std::to_string(error) +
                " budget=" + std::to_string(budget));
        }
        max_abs_error = std::max(max_abs_error, error);
        if (error/budget > max_relative_to_bound) worst_row=row;
        max_relative_to_bound = std::max(max_relative_to_bound, error/budget);
    }
    std::cout << "PASS native CUDA " << rows << "x" << cols
              << " max_abs_error=" << std::setprecision(9) << max_abs_error
              << " max_error/bound=" << max_relative_to_bound
              << " worst_row=" << worst_row << '\n';

    if (benchmark) {
        constexpr int kWarmups = 25;
        constexpr int kIterations = 300;
        for (int i=0; i<kWarmups; ++i) {
            cuda_check(nk::launch_mxfp8_gemv(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                              rows, cols, stream.handle), "warmup");
        }
        Event start, stop;
        cuda_check(cudaEventRecord(start.handle, stream.handle), "record start");
        for (int i=0; i<kIterations; ++i) {
            cuda_check(nk::launch_mxfp8_gemv(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                              rows, cols, stream.handle), "bench launch");
        }
        cuda_check(cudaEventRecord(stop.handle, stream.handle), "record stop");
        cuda_check(cudaEventSynchronize(stop.handle), "bench sync");
        float ms = 0.f;
        cuda_check(cudaEventElapsedTime(&ms, start.handle, stop.handle), "elapsed");
        const double us = static_cast<double>(ms) * 1000 / kIterations;
        const double bytes = static_cast<double>(rows)*cols +
            static_cast<double>(rows)*(cols/32) + 4.0*cols + 4.0*rows;
        const double gib_s = bytes/us*1e6/(1024.0*1024.0*1024.0);
        std::cout << "BENCH native fused MXFP8 GEMV " << rows << "x" << cols
                  << " " << us << " us/call, approx " << gib_s
                  << " GiB/s (device resident; includes kernel launch)\n";
    }
}

float one_row(std::vector<std::uint8_t> words,
              std::vector<std::uint8_t> scale_words,
              std::vector<float> input) {
    const int cols = static_cast<int>(words.size());
    Stream stream;
    DeviceBuffer<std::uint8_t> codes(words.size());
    DeviceBuffer<std::uint8_t> scales(scale_words.size());
    DeviceBuffer<float> x(input.size());
    DeviceBuffer<float> y(1);
    codes.upload(words, stream.handle);
    scales.upload(scale_words, stream.handle);
    x.upload(input, stream.handle);
    cuda_check(nk::launch_mxfp8_gemv(codes.ptr, scales.ptr, x.ptr, y.ptr,
                                      1, cols, stream.handle), "edge launch");
    std::vector<float> output(1);
    y.download(output, stream.handle);
    cuda_check(cudaStreamSynchronize(stream.handle), "edge sync");
    return output[0];
}

void check_edge_cases() {
    require(nk::launch_mxfp8_gemv(nullptr, nullptr, nullptr, nullptr, 1, 32)
                == cudaErrorInvalidValue, "null pointer not rejected");
    // The geometry checks are also exercised with nonnull DEVICE pointers.
    {
        Stream stream;
        DeviceBuffer<std::uint8_t> wc(32), ws(1);
        DeviceBuffer<float> vx(32), vy(1);
        require(nk::launch_mxfp8_gemv(wc.ptr, ws.ptr, vx.ptr, vy.ptr,
                                      1, 33, stream.handle) == cudaErrorInvalidValue,
                "invalid K not rejected");
        require(nk::launch_mxfp8_gemv(wc.ptr, ws.ptr, vx.ptr, vy.ptr,
                                      0, 32, stream.handle) == cudaErrorInvalidValue,
                "invalid N not rejected");
    }

    // UE8M0 0x00 = 2^-127. Summing 32 values of 1*2^-127 = 2^-122.
    const float smallest = one_row(std::vector<std::uint8_t>(32, 0x38),
                                    {0x00}, std::vector<float>(32, 1.f));
    require(smallest == std::ldexp(1.f, -122), "UE8M0 byte zero lost/FTZ");

    // Separate strict cancellation test, not subject to sum(|terms|) tolerance.
    std::vector<std::uint8_t> signs(64, 0x38);
    for (int i=32; i<64; ++i) signs[i] = 0xb8;
    const float canceled = one_row(signs, {0x7f, 0x7f},
                                    std::vector<float>(64, 1.f));
    require(canceled == 0.f, "balanced +/- groups must cancel exactly");

    // Invalid source words are propagated as NaN by the kernel. The caller must
    // reject them during artifact validation (Stage-3A validator) before use.
    auto bad_code = std::vector<std::uint8_t>(32, 0x38);
    bad_code[7] = 0x7f;
    require(std::isnan(one_row(bad_code, {0x7f}, std::vector<float>(32, 1.f))),
            "E4M3FN NaN not propagated");
    require(std::isnan(one_row(std::vector<std::uint8_t>(32, 0x38), {0xff},
                               std::vector<float>(32, 1.f))),
            "UE8M0 NaN not propagated");
    std::cout << "PASS UE8M0 0x00, cancellation, invalid-word NaN, input contract\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool bench = false;
        for (int i=1; i<argc; ++i) {
            if (std::string(argv[i]) == "--bench") bench = true;
            else throw std::invalid_argument("usage: mxfp8_gemv_test [--bench]");
        }
        int device = -1;
        cuda_check(cudaGetDevice(&device), "CUDA device");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "device properties");
        std::cout << "GPU " << device << ": " << prop.name
                  << " sm_" << prop.major << prop.minor << '\n';
        check_edge_cases();
        check_native_case(1, 32, false);
        check_native_case(7, 96, false);
        check_native_case(129, 1152, false);  // partial CTA at row boundary
        check_native_case(128, 2304, false);
        check_native_case(2304, 1152, bench); // Inkling w2, one expert
        check_native_case(2304, 2304, bench); // Inkling w13, one expert
        std::cout << "OK Inkling2 Stage3B: native CUDA MXFP8 GEMV against Stage3A double oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling2 Stage3B: " << error.what() << '\n';
        return 1;
    }
}
