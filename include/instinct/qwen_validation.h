#pragma once

#include "instinct/model_file.h"

namespace instinct {

// Validates the exact tensor contract consumed by the Qwen2 execution path.
// CUDA retains the strict FP16 default until its own quantized backend is validated.
void validate_qwen2_weights(const ModelFile& file, bool allow_int8 = false);

}  // namespace instinct
