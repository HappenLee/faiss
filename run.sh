#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
src="$root/faiss-src"
build="$root/build"
results="$root/results"
base=16a95d63f2980b90c19e5308b4baec8f050b9b5b

test "$(uname -m)" = aarch64
mkdir -p "$results"
{
    lscpu
    c++ --version
    cmake --version
    printf 'Faiss base commit: %s\n' "$base"
    sha256sum "$root/neon-batch-distances.patch"
} | tee "$results/environment.txt"

git init -q "$src"
git -C "$src" fetch -q --depth 1 https://github.com/facebookresearch/faiss.git "$base"
git -C "$src" checkout -q --detach FETCH_HEAD
git -C "$src" apply --check "$root/neon-batch-distances.patch"

cmake -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF -DFAISS_ENABLE_MKL=OFF -DBUILD_TESTING=OFF

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for variant in baseline neon; do
    if [[ "$variant" == neon ]]; then
        git -C "$src" apply "$root/neon-batch-distances.patch"
    fi
    cmake --build "$build" --target faiss --parallel "$(nproc)"
    c++ -O3 -DNDEBUG -std=c++20 -fopenmp -I"$src" "$root/bench.cpp" \
        "$build/faiss/libfaiss.a" -lopenblas -o "$build/$variant"
    "$build/$variant" --check | tee "$results/$variant-correctness.txt"
done

c++ -O3 -DNDEBUG -std=c++20 -fopenmp -I"$src" \
    "$src/tests/test_distances_simd.cpp" "$build/faiss/libfaiss.a" \
    -lopenblas -lgtest_main -lgtest -pthread -o "$build/distances_test"
"$build/distances_test" | tee "$results/distances-test.txt"

python3 "$root/compare.py" "$build" "$results"
