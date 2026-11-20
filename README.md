# Under the hood

Code, results and charts behind the field tests on **[Load Bearing](https://blog.negmbots.com/field-tests/)**, a blog on platform and infrastructure leadership in the AI era.

Each field test takes a new technology for a spin on my own hardware, starting from a question a platform leader would ask. The post says what I found and what I'd decide; this repository holds what you need to check it or run it again.

| Field test | Post |
|---|---|
| [`2026-11-ncm-on-amd-gpu`](2026-11-ncm-on-amd-gpu/) | [Taking NCM for a spin: predictable AI cost is a design choice, not a breakthrough](https://blog.negmbots.com/field-tests/2026-11-ncm-on-amd-gpu/) |

Each folder has its own README: the question, what's included, how to rerun it, and the design choices behind the measurements. `tools/` holds the two shared tools every run uses: `collect_env.py` records the software and hardware a run used, `sample_hw.py` samples clocks, power and temperatures every second.

## How this was built

<!-- TODO(you): one or two sentences on how you work. -->

## License

<!-- TODO(you): e.g. MIT for the code, CC BY 4.0 for results and charts. -->
