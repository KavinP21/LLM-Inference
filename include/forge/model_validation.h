#pragma once

#include "forge/model_file.h"

namespace forge {

// Validate the exact tensor contract for any registered model family.
void validate_model_weights(const ModelFile& file);
void validate_gemma3_weights(const ModelFile& file);

}  // namespace forge
