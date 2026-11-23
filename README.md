# Faiss ARM64 batch distance benchmark

This benchmark compares the Faiss `main` revision
`16a95d63f2980b90c19e5308b4baec8f050b9b5b` with the low-dimensional
NEON batch-distance patch. It builds the two CPU libraries with the same
Release configuration and calls the public Faiss distance APIs. The
optimization applies to the `ARM_NEON` kernels for dimensions 2, 4, and 8.

Run on a clean Ubuntu 24.04 AArch64 checkout with these dependencies:

```sh
sudo apt-get update
sudo apt-get install -y g++ cmake ninja-build libopenblas-dev libgtest-dev
bash run.sh
```

The workflow uses GitHub's `ubuntu-24.04-arm` runner. `compare.py` pins
both binaries to one logical CPU and alternates their order across seven
paired rounds. Each timed case doubles the iteration count until the
final sample lasts at least 30 ms. The summary reports the median time
for each binary and the median of the seven paired speedups.

- `batch` measures one public distance API call for a query vector and
  a contiguous array of candidates. Dimensions 16 and 32 exercise the
  generic path. The batch of 255 candidates includes a remainder.
- `pq_table` measures one `ProductQuantizer` table construction for 96
  subquantizers and 256 centroids each. The dimension column is the
  subvector dimension.
- `samples.csv` contains every measured sample. `summary.csv` includes
  the paired range. The environment and correctness logs accompany the
  measurements in `results/`.
