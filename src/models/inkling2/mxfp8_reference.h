#pragma once

// Inkling 2 Mini Stage 3A: independent CPU oracle for checkpoint MXFP8
// E4M3FN × UE8M0 (per 32 contiguous K-axis values).
// No CUDA dependencies. NOT a production inference operator.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::inkling2::mxfp8_reference {

inline double ue8m0(std::uint8_t word) {
    if (word == 0xff) {
        throw std::invalid_argument("UE8M0 0xff is NaN, not a finite scale");
    }
    // Importantly, 0x00 denotes 2^-127, NOT zero.
    return std::ldexp(1.0, static_cast<int>(word) - 127);
}

inline double e4m3fn(std::uint8_t word) {
    if ((word & 0x7f) == 0x7f) {
        throw std::invalid_argument("E4M3FN 0x7f/0xff is NaN");
    }
    const int exponent = (word >> 3) & 0xf;
    const int mantissa = word & 0x7;
    const double magnitude = (exponent == 0)
        ? std::ldexp(static_cast<double>(mantissa), -9)
        : std::ldexp(static_cast<double>(8 + mantissa), exponent - 10);
    return std::copysign(magnitude, (word & 0x80) ? -1.0 : 1.0);
}

struct MatrixView {
    std::size_t rows;
    std::size_t cols;
    std::span<const std::uint8_t> codes;
    std::span<const std::uint8_t> scales;

    void validate() const {
        if (rows == 0 || cols == 0 || (cols % 32) != 0 ||
            rows > std::numeric_limits<std::size_t>::max() / cols ||
            codes.size() != rows * cols ||
            scales.size() != rows * (cols / 32)) {
            throw std::invalid_argument("invalid row-major MXFP8 matrix geometry");
        }
        for (const auto word : codes) {
            if ((word & 0x7f) == 0x7f) {
                throw std::invalid_argument("MXFP8 weight contains E4M3FN NaN");
            }
        }
        for (const auto word : scales) {
            if (word == 0xff) {
                throw std::invalid_argument("MXFP8 scale contains UE8M0 NaN");
            }
        }
    }

    double value(std::size_t row, std::size_t col) const {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("MXFP8 matrix index outside geometry");
        }
        if (cols % 32 != 0 || rows > std::numeric_limits<std::size_t>::max() / cols ||
            codes.size() != rows * cols || scales.size() != rows * (cols / 32)) {
            throw std::invalid_argument("invalid MXFP8 matrix storage");
        }
        return e4m3fn(codes[row * cols + col]) *
               ue8m0(scales[row * (cols / 32) + col / 32]);
    }

    std::vector<double> matvec(std::span<const float> x) const {
        validate();
        if (x.size() != cols) {
            throw std::invalid_argument("MXFP8 matvec input length mismatch");
        }
        std::vector<double> out(rows, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (std::size_t k = 0; k < cols; ++k) {
                sum += value(r, k) * static_cast<double>(x[k]);
            }
            out[r] = sum;
        }
        return out;
    }
};

}  // namespace ninfer::models::inkling2::mxfp8_reference
