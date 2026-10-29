#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace instinct {

std::array<std::uint8_t, 32> sha256(std::span<const std::byte> input);

}  // namespace instinct

