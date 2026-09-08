# Benchmark preparation

Run from the repository root:

```bash
python scripts/download_benchmarks.py --list
python scripts/download_benchmarks.py --benchmarks vstar mmvp blink hr_bench_4k hr_bench_8k mme_realworld_lite
```

Downloads are pinned in `data/sources.json`. Files are checked against recorded sizes and available SHA256 hashes. Existing complete datasets are reused. No benchmark images or raw datasets are included in Git.

| Adapter | Source | Evaluation split | Samples |
| --- | --- | --- | ---: |
| vstar | craigwu/vstar_bench | test_questions.jsonl | 191 |
| mmvp | MMVP/MMVP | Questions.csv | 300 |
| blink | BLINK-Benchmark/BLINK | val, 5 selected configurations | 697 |
| hr_bench_4k | DreamMr/HR-Bench | hr_bench_4k.parquet | 800 |
| hr_bench_8k | DreamMr/HR-Bench | hr_bench_8k.parquet | 800 |
| mme_realworld_lite | yifanzhang114/MME-RealWorld-lite-lmms-eval | train | 1,919 |

MMVP's upstream directory is named `MMVP Images`; the script provides the `MMVP_Images` layout expected by the evaluator. BLINK is stored as a DatasetDict for each configuration. HR-Bench and MME are stored with `datasets.save_to_disk`.

VSTAR's additional OCR and GPT4V-hard files are downloaded for reconstruction of the recorded LAM training set, but the VSTAR evaluator still uses only the 191 test questions. See `data/README.md` for benchmark exposure in LAM training.

The released scoring reproduces the existing per-question accuracy. In particular, MMVP results are not the official pair accuracy. BLINK covers five configurations rather than the entire BLINK collection.

The current release verification used already available local datasets; the download and extraction operations were not exercised.
