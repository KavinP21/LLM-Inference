#include "forge/engine.h"
#include "forge/scheduler.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cuda_runtime_api.h>

namespace py = pybind11;

PYBIND11_MODULE(_forge, module) {
  module.doc() = "Forge LLM C++/CUDA runtime";
  module.def("build_info", [] {
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    py::dict result;
#ifdef NDEBUG
    result["build_type"] = "Release";
#else
    result["build_type"] = "Debug";
#endif
    result["compiler"] = __VERSION__;
    result["cuda_compile_version"] = CUDART_VERSION;
    result["cuda_runtime_version"] = runtime_version;
    result["cuda_driver_version"] = driver_version;
    return result;
  });
  py::enum_<forge::SequenceState>(module, "SequenceState")
      .value("WAITING", forge::SequenceState::waiting)
      .value("PREFILLING", forge::SequenceState::prefilling)
      .value("RUNNING", forge::SequenceState::running)
      .value("COMPLETED", forge::SequenceState::completed)
      .value("CANCELLED", forge::SequenceState::cancelled)
      .value("REJECTED", forge::SequenceState::rejected);
  py::class_<forge::TokenEvent>(module, "TokenEvent")
      .def_readonly("request_id", &forge::TokenEvent::request_id)
      .def_readonly("token", &forge::TokenEvent::token)
      .def_readonly("finished", &forge::TokenEvent::finished);
  py::class_<forge::Engine>(module, "Engine")
      .def(py::init<const std::filesystem::path&, std::uint32_t, std::uint32_t, std::uint64_t>(),
           py::arg("model_path"), py::arg("max_num_sequences") = 16,
           py::arg("max_model_length") = 2048,
           py::arg("kv_cache_bytes") = 512ULL << 20U)
      .def("submit", [](forge::Engine& engine, const std::vector<std::int32_t>& tokens,
                         std::uint32_t maximum, const std::vector<std::int32_t>& eos) {
        return engine.submit(tokens, maximum, eos);
      }, py::arg("input_ids"), py::arg("max_new_tokens"), py::arg("eos_token_ids"))
      .def("step", &forge::Engine::step, py::call_guard<py::gil_scoped_release>())
      .def("cancel", &forge::Engine::cancel)
      .def("stats", [](const forge::Engine& engine) {
        const auto stats = engine.stats();
        py::dict scheduler;
        scheduler["submitted"] = stats.scheduler.submitted;
        scheduler["completed"] = stats.scheduler.completed;
        scheduler["cancelled"] = stats.scheduler.cancelled;
        scheduler["rejected"] = stats.scheduler.rejected;
        scheduler["waiting"] = stats.scheduler.waiting;
        scheduler["running"] = stats.scheduler.running;
        py::dict cache;
        cache["total_blocks"] = stats.kv_cache.total_blocks;
        cache["allocated_blocks"] = stats.kv_cache.allocated_blocks;
        cache["reserved_blocks"] = stats.kv_cache.reserved_blocks;
        cache["free_blocks"] = stats.kv_cache.free_blocks;
        cache["live_tokens"] = stats.kv_cache.live_tokens;
        cache["internal_fragmentation_tokens"] = stats.kv_cache.internal_fragmentation_tokens;
        cache["allocation_failures"] = stats.kv_cache.allocation_failures;
        cache["occupancy"] = stats.kv_cache.occupancy();
        py::dict result;
        result["scheduler"] = scheduler;
        result["kv_cache"] = cache;
        result["model_bytes"] = stats.model_bytes;
        result["kv_device_bytes"] = stats.kv_device_bytes;
        return result;
      })
      .def("generate", [](forge::Engine& engine, const std::vector<std::int32_t>& tokens,
                           std::uint32_t maximum, const std::vector<std::int32_t>& eos) {
        py::gil_scoped_release release;
        return engine.generate(tokens, maximum, eos);
      }, py::arg("input_ids"), py::arg("max_new_tokens"), py::arg("eos_token_ids"))
      .def("debug_prefill_logits", [](forge::Engine& engine,
                                       const std::vector<std::int32_t>& tokens) {
        py::gil_scoped_release release;
        return engine.debug_prefill_logits(tokens);
      }, py::arg("input_ids"));
}
