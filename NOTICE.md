# Sources and terms

This repository snapshots the existing LVR experiment code and adds portable local orchestration.
The owner has not selected a license for the project's own code or exported model weights.
No new license is granted by assembling this repository.

- Base weights: Qwen/Qwen2.5-VL-7B-Instruct, pinned in `manifests/qwen_model.json`.
- Frozen image generator / VAE: black-forest-labs/FLUX.2-klein-base-4B, pinned in `manifests/flow_model.json`.
- Dataset origins, immutable revisions and checksums: `lvr_pipeline_new/data/sources.json`,
  `lvr_pipeline_new/data/reasoning_sources.json`, `manifests/benchmark_downloads.json`, and the release source locks.
- Existing LVR component attribution: `lvr_pipeline_new/NOTICE.md`; original third-party license files are preserved.
- Benchmark scorer source files and the licenses recorded by the original accepted suite travel in the metadata release.

The data release is distributed separately at https://huggingface.co/datasets/Wing22/qwen25-flow-repro-data-v1.
The README and publication manifest record upload progress and the verified immutable snapshot.
Upstream data and model terms remain applicable when transferring or publishing artifacts.
