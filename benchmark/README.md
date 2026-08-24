# LVR Benchmark Viewer

Streamlit viewer for the four benchmark result files produced by
`/private/wmz/project/lvr_pipeline/configs/eval/plain.yaml`:

- `vstar`
- `hr_bench_4k`
- `hr_bench_8k`
- `mme_realworld_lite`

The default result directory is the `plain_v1_p99_len4096` step-7000 run from
the task. The default benchmark root is the old repository's `external/bench`
symbolic link, which currently resolves to the local benchmark data under
`/private/wmz/projects/lvr/evaluation/bench`.

## Start

From `/private/wmz/project/lvr_pipeline_new`:

```bash
/private/wmz/tool/miniforge3/envs/lvr/bin/python -m pip install -r benchmark/requirements.txt
/private/wmz/tool/miniforge3/envs/lvr/bin/python -m streamlit run benchmark/app.py --server.address 0.0.0.0
```

Open the URL printed by Streamlit. Use the sidebar to switch benchmark, show
only errors or correct samples, filter categories, search by question/ID, or
point the viewer at another compatible result directory.
