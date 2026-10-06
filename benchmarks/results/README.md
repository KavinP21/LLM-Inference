# Saved experiment evidence

Each directory contains recorded observations. Start with the report, then inspect
raw metrics, registered inputs, source identity, and acceptance result. Unit,
resource, and numerical checks do not waive a failed generation-quality gate.

| Run | Analysis |
| --- | --- |
| [October 6 repository QA](repository-qa-2026-10-06/) | [365-test regression, host tests, and README smoke](repository-qa-2026-10-06/summary.json) |
| [FP16 baseline/full matrix](resume-m3-max-2026-09-23/) | [Performance](../../docs/resume-benchmark-results.md) |
| [Initial INT8](int8-m3-max-2026-10-01/) | [Rejected quality](../../docs/mlx-int8-results.md) |
| [INT8 hardening](int8-hardening-m3-max-2026-10-01/) | [Mixed precision](../../docs/mlx-int8-hardening-results.md) |
| [Batch consistency](batch-invariant-m3-max-2026-10-01/) | [Rowwise tradeoffs](../../docs/mlx-batch-numerics-results.md) |
| [Second-order fitting](second-order-m3-max-2026-10-01/) | [Mixed results](../../docs/mlx-second-order-results.md) |
| [Cached calibration](cached-calibration-m3-max-2026-10-01/) | [Rejected quality](../../docs/mlx-cached-calibration-results.md) |
| [Scale fitting](scale-aware-m3-max-2026-10-01/) | [Stopped calibration](../../docs/mlx-scale-aware-results.md) |
| [Refinement](refined-m3-max-2026-10-02/) | [Family-specific gates](../../docs/mlx-refined-results.md) |
| [Qwen held-out](qwen-validation-m3-max-2026-10-02/) | [Rejected quality](../../docs/mlx-qwen-validation-results.md) |
| [Initial prefix run](prefix-cache-m3-max-2026-10-02/) | [Dispatch failure](../../docs/mlx-prefix-cache-results.md) |
| [Fresh prefix validation](prefix-cache-m3-max-2026-10-02-dispatch-fixed/) | [Passed validation; timing pending](../../docs/mlx-prefix-cache-results.md) |

Frozen runs bind input files and source hashes. Several include `sources.zip` and
checksum indices; preserve those bytes for old-result audits. Recorded paths may
refer to the original local checkout. Full replay requires matching exported
models, calibration artifacts, and registered filesystem/input identity. Model
weights and calibration archives under `models/` are excluded from Git.

The sources recorded by the latest prefix validation are preserved at commit
`d0a069d`. Later style and CI changes do not constitute a new official-model
experiment. Register fresh sources and inputs in a new directory for new runs.
