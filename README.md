# GPT-OSS Responses logprobs validation

These are sanitized validation artifacts for the GPT-OSS Responses logprobs PR.
This branch contains evidence only; the implementation and CI tests are on the
PR's feature branch.

- [Final review and current-base results](final-review/REPORT.md)
- [Detailed earlier hardware experiments and limitations](experiments/REPORT.md)
- [Repeated serving benchmark](experiments/benchmark-summary.md)
- [Raw engine mapping and boundary results](experiments/engine-pipeline-final.json)
- [H200 compiled/CUDA-graph HTTP requests](experiments/h200-api-final.json)

The earlier experiment report identifies its historical base and patch hash.
Final-review tests run on the rebased PR implementation. Host-specific paths
are replaced with `/path/to/workspace`, and private addresses with `H200_HOST`.
The standalone scripts accept CLI paths/URLs; adjust local paths in the shell
wrapper. Download/setup logs and credentials are not included.

The adjacent `CHECKSUMS.sha256` identifies these sanitized files. Historical
manifests identify the original tested implementation, not the sanitized logs.
