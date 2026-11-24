#include "models/inkling2/mxfp8_reference.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace mx = ninfer::models::inkling2::mxfp8_reference;

namespace {
void require(bool condition, const char* why) {
    if (!condition) throw std::runtime_error(why);
}

template <class Function> void require_throws(Function f, const char* why) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(why);
}
}  // namespace

int main() {
    try {
        require(mx::ue8m0(0x00) == std::ldexp(1.0, -127), "UE8M0 0x00 is nonzero");
        require(mx::ue8m0(0x7e) == 0.5, "UE8M0 0x7e");
        require(mx::ue8m0(0x7f) == 1.0, "UE8M0 0x7f");
        require(mx::ue8m0(0x80) == 2.0, "UE8M0 0x80");
        require(mx::ue8m0(0xfe) == std::ldexp(1.0, 127), "UE8M0 0xfe");
        require_throws([] { (void)mx::ue8m0(0xff); }, "UE8M0 NaN must fail");
        require(mx::e4m3fn(0x38) == 1.0, "E4M3 normal 1");
        require(mx::e4m3fn(0xb8) == -1.0, "E4M3 normal -1");
        require(mx::e4m3fn(0x01) == std::ldexp(1.0, -9), "E4M3 subnormal");
        require(mx::e4m3fn(0x7e) == 448.0, "E4M3 max");
        require(std::signbit(mx::e4m3fn(0x80)), "E4M3 negative zero");
        require_throws([] { (void)mx::e4m3fn(0x7f); }, "E4M3 NaN must fail");
        require_throws([] { (void)mx::e4m3fn(0xff); }, "signed E4M3 NaN must fail");

        constexpr std::size_t rows = 3, cols = 64;
        std::vector<std::uint8_t> codes(rows * cols, 0x38);
        // Every second 32-value block is -1, each row uses distinct scales.
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 32; col < cols; ++col) {
                codes[row * cols + col] = 0xb8;
            }
        }
        std::array<std::uint8_t, rows * cols / 32> scales = {
            0x7f, 0x80, 0x80, 0x7e, 0x7e, 0x7f
        };
        mx::MatrixView view{rows, cols, codes, scales};
        view.validate();
        require(view.value(0, 0) == 1.0, "row 0 first block");
        require(view.value(0, 32) == -2.0, "row 0 second block");
        require(view.value(1, 0) == 2.0, "row 1 first block");
        require(view.value(1, 63) == -0.5, "row 1 second block");
        require(view.value(2, 0) == 0.5, "row 2 first block");
        require(view.value(2, 32) == -1.0, "row 2 second block");
        std::array<float, cols> x{};
        x.fill(1.0f);
        const auto y = view.matvec(x);
        require(y.size() == rows && y[0] == -32.0 && y[1] == 48.0 && y[2] == -16.0,
                "CPU matvec row/column group mapping");
        require_throws([&] { (void)view.matvec(std::span<const float>{x.data(), 12}); },
                       "wrong vector shape must fail");
        scales[1] = 0xff;
        require_throws([&] { view.validate(); }, "bad scale must fail");
        scales[1] = 0x80;
        codes[2] = 0x7f;
        require_throws([&] { view.validate(); }, "bad weight must fail");

        std::cout << "OK Inkling2 MXFP8 Stage3A: C++ scalar, G32 geometry, matvec, invalid words\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL Inkling2 MXFP8 Stage3A: " << e.what() << '\n';
        return 1;
    }
}
