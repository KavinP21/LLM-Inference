#pragma once

#include <nvToolsExt.h>

#include <cstdint>

namespace forge {

class NvtxRange {
 public:
  explicit NvtxRange(const char* name, std::uint32_t color = 0xff4c78a8U) {
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = color;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    nvtxRangePushEx(&attributes);
  }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace forge

