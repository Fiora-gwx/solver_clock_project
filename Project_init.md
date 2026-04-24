# CLAUDE CODE IMPLEMENTATION BRIEF

## Purpose

This file is **for coding and experiment implementation**.

If you are Codex (or any coding agent), use this file as the operational spec for:

- how to organize the repo,
- what the method is,
- what not to misframe,
- what experiments to implement,
- what metrics to produce,
- how to prioritize work.


Do **not** invent solver-side innovations that are not part of the method.

The goal is to help me build a clean, reproducible codebase and a correct experimental pipeline.

---

# 1. Server Profile

# Basic Identity

- Hostname: `pgroup-Rack-Server`
- User: `gwx`
- Project root: `/home/gwx/solver_clock_project`
- OS: `Ubuntu 24.04.3 LTS (Noble Numbat)`
- Kernel: `Linux 6.14.0-37-generic`
- Python: `3.10.20`
- Active Python: `/home/gwx/miniconda3/envs/sc-diff/bin/python`

## Conda Environments
- `base` -> `/home/gwx/miniconda3`
- `flow_matching` -> `/home/gwx/miniconda3/envs/flow_matching`
- `sana` -> `/home/gwx/miniconda3/envs/sana`
- `sc-diff` -> `/home/gwx/miniconda3/envs/sc-diff` **(active)**
- `sc-pndm` -> `/home/gwx/miniconda3/envs/sc-pndm`
- `sc-sana` -> `/home/gwx/miniconda3/envs/sc-sana`

---

## GPU / CUDA / Driver

### GPU Summary
- GPU count: **4**
- GPU model: **NVIDIA GeForce RTX 4090**
- VRAM per GPU: **49140 MiB (~48 GB)**
- Driver version: **535.288.01**
- CUDA version reported by `nvidia-smi`: **12.2**
- `nvcc` version: **12.0.140**

### Current GPU Memory Snapshot
- GPU 0: `15 MiB / 49140 MiB` used
- GPU 1: `15 MiB / 49140 MiB` used
- GPU 2: `15 MiB / 49140 MiB` used
- GPU 3: `14 MiB / 49140 MiB` used



### GPU UUIDs
- GPU 0: `GPU-483b664c-27f9-a51a-1bc5-8d4381d7b71b`
- GPU 1: `GPU-072f9497-5736-c8b8-b081-432b4d2f6b2f`
- GPU 2: `GPU-371e43b5-e1a4-e799-48ca-b7ba7bc0c8b5`
- GPU 3: `GPU-fce6e17e-0284-7482-fdac-26517137cffe`

### GPU Topology / NUMA
- GPU0 <-> GPU1: `NODE`
- GPU2 <-> GPU3: `NODE`
- Cross-group communication (e.g. GPU0 <-> GPU2): `SYS`
- NUMA node 0 CPU affinity: `0-31,64-95`
- NUMA node 1 CPU affinity: `32-63,96-127`

### Practical Notes
- Multi-GPU experiments should preferably use:
  - `(GPU0, GPU1)` as one local pair
  - `(GPU2, GPU3)` as one local pair
- GPU 0 is currently partially occupied by vLLM, so avoid scheduling large experiments there unless those processes are stopped.

---

## CPU / Memory / Storage

### CPU
- CPU model: `Intel(R) Xeon(R) Gold 6454S`
- Sockets: `2`
- Cores per socket: `32`
- Threads per core: `2`
- Total logical CPUs: **128**
- NUMA nodes: **2**

### RAM / Swap
- Total RAM: **503 GiB**
- Used RAM: **48 GiB**
- Free RAM: **14 GiB**
- Buff/Cache: **449 GiB**
- Available RAM: **455 GiB**
- Swap: **8.0 GiB**
- Used Swap: **2.5 GiB**
- Free Swap: **5.5 GiB**

### Disk
- Root disk: `/dev/nvme0n1p2`
- Root size: **1.8T**
- Used: **1.5T**
- Available: **225G**
- Usage: **88%**

### Other Storage
- Additional disk present: `sda` with size **14.6T** (not shown as mounted in the provided output)

### Practical Notes
- Main system disk free space is only **225G**, so large-scale image generation / cached downloads / repeated benchmarks should be monitored carefully.
- If available, large experiment outputs should ideally be redirected to a larger mounted storage volume instead of the root partition.

---

## Repository State

### Main Repo
- Root: `/home/gwx/solver_clock_project`
- Branch: `main`
- Commit SHA: `9baaba9f49c817e497ea1ccd43131fc399a27496`

### Submodules
- `third_party/STORK`
  - SHA: `0793ded71498aafd04cb17987e66b7d2a9b1d715`
  - Ref hint: `heads/main`
- `third_party/diffusers`
  - SHA: `431066e96762442aad5b675893a91bb8c5bfb3b9`
  - Ref hint: `v0.16.0-4163-g431066e96`



---

## Raw Hardware Facts for Config Files

```yaml
server:
  hostname: pgroup-Rack-Server
  user: gwx
  project_root: /home/gwx/solver_clock_project
  os: Ubuntu 24.04.3 LTS
  kernel: Linux 6.14.0-37-generic
  python: 3.10.20
  active_env: sc-diff

hardware:
  gpu_count: 4
  gpu_model: NVIDIA GeForce RTX 4090
  vram_per_gpu_mib: 49140
  driver_version: 535.288.01
  cuda_driver_version: "12.2"
  nvcc_version: "12.0.140"

cpu:
  model: Intel(R) Xeon(R) Gold 6454S
  sockets: 2
  cores_per_socket: 32
  threads_per_core: 2
  logical_cpus: 128
  numa_nodes: 2

memory:
  ram_total: 503Gi
  ram_available: 455Gi
  swap_total: 8Gi
  swap_free: 5.5Gi

storage:
  root_mount: /
  root_device: /dev/nvme0n1p2
  root_total: 1.8T
  root_available: 225G
  root_usage_percent: 88

repo:
  branch: main
  main_sha: 9baaba9f49c817e497ea1ccd43131fc399a27496
  stork_sha: 0793ded71498aafd04cb17987e66b7d2a9b1d715
  diffusers_sha: 431066e96762442aad5b675893a91bb8c5bfb3b9
```

