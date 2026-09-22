#pragma once

#include <stdexcept>
#include <string>

namespace forge {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline void check(bool condition, const std::string& message) {
  if (!condition) throw Error(message);
}

}  // namespace forge

