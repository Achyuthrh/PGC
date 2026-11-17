# Experiment 6 — Memory Optimization in CUDA

![CUDA](https://img.shields.io/badge/CUDA-13.4-76B900?style=flat-square&logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RTX%203050%206GB%20(sm__86)-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Language](https://img.shields.io/badge/Language-CUDA%20C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20MSVC-0078D6?style=flat-square&logo=windows&logoColor=white)

## Overview

This experiment benchmarks GPU global-memory access patterns and shared-memory optimization in CUDA C++ ([`exp6.cu`](exp6.cu)) across two workloads:

1. **Matrix Transpose (`4096 x 4096` `float`)**: Compares a naive uncoalesced kernel (`transposeNaive`) against a shared-memory tiled coalesced kernel (`transposeCoalesced`).
2. **Naive Matrix Multiplication (`1024 x 1024` `float`)**: Evaluates a baseline global-memory `matMulNaive` kernel ($C = A \times B$).

---

## Hardware & Environment

| Component | Specification |
| --- | --- |
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| **Compute Capability** | 8.6 (`sm_86`, Ampere) |
| **CUDA Toolkit** | 13.4 |
| **OS & Toolchain** | Windows PowerShell + MSVC (`nvcc`) |

---

## Benchmark Results

Measured with CUDA events after a warm-up launch (averaged over 20 runs for transpose and 5 runs for matrix multiplication):

| Kernel / Workload | Matrix Size | Avg. Time | Speedup | Throughput | Verification |
| --- | --- | --- | --- | --- | --- |
| **Transpose (Uncoalesced)** | `N = 4096` | `2.704 ms` | `1.00x` (baseline) | `49.6 GB/s` | `PASS` |
| **Transpose (Coalesced)** | `N = 4096` | `1.084 ms` | `2.49x` | `123.8 GB/s` | `PASS` |
| **Naive Matrix Multiplication** | `N = 1024` | `4.466 ms` | — | `480.8 GFLOPS` | `PASS` |

---

## Key Takeaways

- **Coalesced Global Memory Access (`2.49x` Speedup)**:
  - In `transposeNaive`, threads in a warp read `in[y * N + x]` contiguously (coalesced) but write `out[x * N + y]` with a stride of `N` floats. Each warp issues scattered memory transactions across cache lines, saturating memory-controller overhead and dropping effective bandwidth to `49.6 GB/s`.
  - In `transposeCoalesced`, each `32 x 32` thread block stages a tile in on-chip `__shared__` memory: warps perform a coalesced row-wise read into `tile[threadIdx.y][threadIdx.x]`, synchronize via `__syncthreads()`, and then write `tile[threadIdx.x][threadIdx.y]` to transposed block coordinates `out[y * N + x]` so that consecutive `threadIdx.x` threads again write contiguous global addresses (`123.8 GB/s`).
- **Bank-Conflict Elimination with `tile[32][33]`**:
  - CUDA shared memory is divided into 32 banks (4-byte stride). With an unpadded `tile[32][32]`, reading column-wise (`tile[threadIdx.x][threadIdx.y]`) maps all 32 threads of a warp to the **same** memory bank, causing a 32-way bank conflict that serializes shared-memory reads.
  - Adding `+1` column padding (`__shared__ float tile[TILE][TILE + 1]`, i.e., `tile[32][33]`) shifts each row by one bank, mapping consecutive column elements across all 32 distinct banks and eliminating bank conflicts completely.
- **Why Naive MatMul Remains Global-Memory Bound**:
  - In `matMulNaive` (`16 x 16` thread blocks), every thread computes one element of `C` by iterating `k = 0 .. N-1` and reading `A[row * N + k]` and `B[k * N + col]` directly from global memory.
  - Each `float` load from `A` and `B` is used for only a single FMA (`2` FLOPs per `8` bytes loaded, an arithmetic intensity of `0.25 FLOP/byte`), leaving the SMs stalled on DRAM/L2 latency and bandwidth (`480.8 GFLOPS`) rather than compute throughput—motivating shared-memory tiled matrix multiplication in subsequent optimizations.

---

## Build & Run

Compile and run from PowerShell (with the MSVC / CUDA environment configured):

```powershell
nvcc -O2 -arch=sm_86 -o exp6.exe exp6.cu
.\exp6.exe
```

Optional CLI arguments override matrix sizes (`.\exp6.exe <N_transpose> <N_matmul>`), defaulting to `4096` and `1024`.

---

## Execution Output

![PowerShell Output](C:\Users\user\Pictures\Screenshots\Screenshot 2026-11-17 103424.png)
