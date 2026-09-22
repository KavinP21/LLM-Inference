#pragma once

#include "forge/model_file.h"

namespace forge {

// Validates the exact tensor contract consumed by the v1 Qwen2 execution path.
void validate_qwen2_weights(const ModelFile& file);

}  // namespace forge

