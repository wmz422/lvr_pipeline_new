# Sources and attribution

- Base model: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- LAM components are adapted from [AdaWorld](https://github.com/Little-Podi/AdaWorld) to Qwen visual features. The upstream Apache-2.0 license is preserved in `third_party/AdaWorld-LICENSE`.
- Rotary embedding utilities derive from [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch). Its MIT license is preserved in `third_party/rotary-embedding-torch-LICENSE`.
- Training images: [TreeVGR-RL-37K](https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K), [Monet-SFT-125K / Visual_CoT](https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K/tree/main/Visual_CoT), [Visual-CoT](https://huggingface.co/datasets/deepcs233/Visual-CoT), COCO, GQA and VisDrone.
- COCO/GQA LAM boxes were derived from [SEAL VQA](https://huggingface.co/datasets/craigwu/seal_vqa_data).
- Benchmark sources and pinned revisions are listed in `data/sources.json` and `benchmark/README.md`.

Original datasets and the base model retain their upstream terms. No original image archives are redistributed in the GitHub repository. The owner has not yet selected a license for the project's own code and exported weights; this document does not grant or replace a license.
