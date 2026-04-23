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

# 2. Correct framing of the method

## 2.1 What the method is

The method is a **continuous offline clock / schedule construction method** for diffusion / flow ODE sampling.

The method works by:

1. defining a continuous time reparameterization,
2. estimating a continuous clock density from calibration trajectories,
3. discretizing this continuous clock into timestep grids for any desired NFE,
4. plugging the resulting grid into multiple existing solvers.

## 2.2 What the method is NOT

It is **not**:

- a new solver,
- a solver-clock joint design,
- a custom adaptive step-size controller,
- a method that only works for one solver,
- a STORK-specific modification.



## 2.3 Correct core message

The correct message is:

> We design a **continuous, solver-agnostic offline clock**.
> The clock can then be discretized and tested across multiple standard solvers.
> The innovation is in the **clock**, not in redesigning the solver.

## 2.4 Relation to AYS

So the key conceptual difference is:

- AYS: optimize a list of steps for a particular NFE.
- Ours: optimize a reusable continuous clock and then instantiate it at many NFEs.

This “continuous and reusable” property is central.

---

# 3. Mathematical method specification

This section is here so the coding agent does not guess the method.

---

## 3.1 Base ODE

We consider a deterministic sampling ODE:

\[
\frac{dx}{dt} = v_\theta(x(t), t; c), \qquad t\in[0,1].
\]

This covers:

- diffusion probability-flow ODEs,
- flow-matching / rectified-flow ODEs.

The primary scope is **ODE sampling**, not stochastic SDE sampling.

---

## 3.2 Continuous clock

Define a new time coordinate \(\tau\in[0,1]\) with

\[
\tau = \tau(t), \qquad m(t) := \frac{d\tau}{dt} > 0,
\qquad \int_0^1 m(t)\,dt = 1.
\]

Then with inverse map \(t=t(\tau)\), the reparameterized ODE becomes

\[
\frac{dx}{d\tau} = \frac{1}{m(t(\tau))} v_\theta(x(\tau), t(\tau); c).
\]

So the clock rescales traversal speed in time.

Equivalent useful forms:

\[
\tau(t)=\int_0^t m(s)\,ds,
\qquad
\frac{dt}{d\tau} = \frac{1}{m(t)}.
\]

---

## 3.3 Discretization for arbitrary NFE

Once the continuous clock is built, any NFE-specific schedule is obtained by:

1. taking a uniform grid in \(\tau\)-space,

\[
\tau_n = \frac{n}{N}, \qquad n=0,1,\dots,N,
\]

2. mapping back to physical time,

\[
t_n = t(\tau_n).
\]

This is the core reason the method is NFE-independent during construction.

---

## 3.4 Theoretical families: V and A

There are two theoretical clock families.

### V-family
Based on the variation of the reparameterized vector field.

Define

\[
\widetilde v(\tau) := \frac{1}{m(t(\tau))} v(x(t(\tau)), t(\tau)).
\]

Then formally,

\[
\frac{d\widetilde v}{d\tau}
= \frac{1}{m^2} D_t v - \frac{m_t}{m^3} v,
\]

where \(D_t v\) denotes the along-trajectory time derivative / material derivative.

This motivates a continuous objective of the form

\[
\mathcal J_V[m]
= \mathbb E\left[
\int_0^1
\left\|
\frac{1}{m^2} D_t v - \frac{m_t}{m^3} v
\right\|^2 m(t)\,dt
\right].
\]

### A-family
Based on a generator / Jacobian proxy.

Let

\[
A(t) := J_x v_\theta(x(t), t; c).
\]

Under reparameterization,

\[
B(\tau) := \frac{A(t(\tau))}{m(t(\tau))}.
\]

Then

\[
\frac{dB}{d\tau}
= \frac{1}{m^2} D_t A - \frac{m_t}{m^3} A.
\]

This motivates

\[
\mathcal J_A[m]
= \mathbb E\left[
\int_0^1
\left\|
\frac{1}{m^2} D_t A - \frac{m_t}{m^3} A
\right\|_F^2 m(t)\,dt
\right].
\]

---

## 3.5 a / b variants

The project currently considers:

- V-a,
- V-b,
- A-a,
- A-b.

But these are **not equal-priority methods**.

### Current practical conclusion

The most important current empirical conclusion is:

> **V-a is the best variant on small-model experiments and should be the primary method.**

So implementation priority is:

1. V-a first,
2. V-b / A-a / A-b only as secondary exploratory variants,
3. do not spend excessive time forcing weak variants to look competitive.

### Recommended coding convention

Represent these variants explicitly in config files as:

- `family: V | A`
- `variant: a | b`

And treat:

- `V-a` as `method_primary`,
- everything else as `method_ablation`.

---
读一下现有代码里V-a的具体实现，确认数学和代码之间的细节。代码已经在之前的文档里，直接对照 `shared_clock.py` 逐函数溯源，写下面这个完整规格。

---


# Section 3.6: V-a Clock — Complete Mathematical Specification

This section is the single authoritative reference for V-a implementation.
Every implementation decision below has a corresponding code location noted.
If code and this document conflict, fix the code.

---

## 3.6.1 Overview

V-a (velocity-variation equalization) constructs the clock density m(t) by:

1. Sampling K pilot trajectories from the model on calibration data.
2. Computing the material-derivative norm ||D_t v_θ(x^(k)(t_i), t_i)|| at each grid point.
3. Aggregating over K trajectories using RMS to get κ̂₁(t_i).
4. Using ᾱ(t_i) = sqrt(max(κ̂₁(t_i), ε)) and normalizing ᾱ to produce a valid probability density m₀(t).
5. Building the cumulative τ-map and exporting the clock profile.

Each step is fully specified below with no ambiguity.

---

## 3.6.2 Physical Time Grid

### Definition

Choose a uniform physical time grid of T+1 nodes:

  t_i = i / T,  i = 0, 1, ..., T

where T = physical_grid_size − 1.

### Recommended value

T = 64  →  physical_grid_size = 65

This gives time spacing Δt = 1/64 ≈ 0.0156.

Finer grids (T = 128) cost proportionally more compute in pilot sampling
but give more accurate density estimation. Coarser grids (T = 32) are
acceptable for smoke tests. The final clock profile is stored on this grid;
downstream discretization to any NFE uses monotone interpolation, so the
grid resolution sets an upper bound on clock precision.

### Code

  physical_grid = torch.linspace(0, 1, T+1)  # shape [T+1]
  # implemented in: training/nonuniform_nodes.py :: build_uniform_time_grid

---

## 3.6.3 Pilot Trajectory Sampling

### Purpose

Collect K representative state trajectories x^(k)(t_i) that reflect the
typical behavior of the model on the calibration distribution.

### Procedure

For k = 1, ..., K:

  1. Sample initial noise:  x^(k)(0) ~ N(0, I)
  2. Sample calibration label (or use unconditional):  c^(k) ~ D_cal
  3. Run a baseline solver on the physical time grid {t_i} to obtain:
       x^(k)(t_i),  i = 0, 1, ..., T

### Baseline solver choice

Use a solver that is cheap but gives trajectories representative of the
model's behavior. Heun2 is the default choice.

  pilot_solver = "heun2"
  pilot_nfe_budget = 2 × T   (Heun2 costs 2 NFE per step)

Euler is acceptable for speed. Do not use the same solver that will be
evaluated — the pilot solver is only for clock construction, not for the
final quality metric.

### K: number of trajectories

K = pilot_batch_size × pilot_num_batches

Recommended defaults:
  pilot_batch_size = 16
  pilot_num_batches = 4
  → K = 64 total trajectories

Sensitivity rule: if K ≥ 32 and FID results are stable across two random
seeds of the pilot, K is sufficient. Increasing K beyond 128 rarely helps
because V-a only uses per-time-step statistics, not trajectory correlations.

### Calibration data source

Use the training data loader (not the eval loader) as the calibration
source. This avoids any distributional leakage into the metric computation.

### Code

  # implemented in: training/ge_stork/shared_clock.py
  # function: sample_pilot_trajectories(...)
  # output: PilotTrajectoryArtifacts
  #   .trajectories: shape [K, T+1, C, H, W]
  #   .labels:       shape [K]
  #   .physical_grid: shape [T+1]

---

## 3.6.4 Velocity Norm Computation

### Definition

For each trajectory k and each time step i, compute the L2 norm of the
velocity field evaluated at the pilot state:

  Q_V^(k)(t_i) := || v_θ(x^(k)(t_i), t_i; c^(k)) ||₂

where the norm is taken over all pixel/channel dimensions, i.e. the
velocity tensor of shape [C, H, W] is flattened to a vector before
computing the Euclidean norm:

  Q_V^(k)(t_i) = sqrt( Σ_{c,h,w} [v_θ]_{c,h,w}² )

This is the Frobenius / L2 norm of the full velocity vector.
Do NOT use the L∞ norm, L1 norm, or per-channel norms.

### Why L2

The V-a objective J_V[m] has a squared L2 norm inside the integral.
The amplitude proxy must be consistent with the local variation term in the objective.
For the practical V-a clock, the core observable is the material derivative
D_t v, not the velocity v itself.

### Numerical note

In practice, v_θ is the physical velocity field of the model, already converted
from base_velocity if model_output_type = "base_velocity":

  v_θ(x, t) = ds/dr × v_base(x, r)

The clock should use the material derivative of this physical velocity field,
D_t v_θ = ∂_t v_θ + (v_θ · ∇_x)v_θ, not the raw velocity norm.
This is because the one-step local error kernel is controlled by D_t v_θ.

### Code

  # in: training/ge_stork/shared_clock.py
  # function: extract_local_objects(...)
  # material_derivative_tensor shape: [K, T+1, C, H, W]
  # material_derivative_norms computed as:
  #   _flatten_norm(material_derivative_tensor)
  #   = material_derivative_tensor.flatten(start_dim=2).norm(dim=-1)
  #   shape: [K, T+1]

---

## 3.6.5 Aggregation Over Trajectories

### Definition

Given the per-trajectory, per-time-step norms Q_V^(k)(t_i)=||D_t v_θ(X_i^(k),t_i)||,
aggregate over the K trajectories to get the shared one-step error-strength profile κ̂₁(t_i):

  κ̂₁(t_i) = sqrt( (1/K) Σ_{k=1}^{K} Q_V^(k)(t_i)² )

Then use the first-order clock density proxy:

  ᾱ(t_i) = sqrt(max(κ̂₁(t_i), ε))

This is the **root mean square (RMS)** aggregation, computed independently
at each time step t_i.

### Why RMS, not mean

The clock density m(t) appears squared in the V-b objective J_V[m].
Using RMS for aggregation means the analytic density m₀ ∝ ᾱ is
geometrically consistent with the E[α²] term that dominates the objective.

Concretely: if Q_V^(k)(t_i) has high variance across trajectories, RMS
gives more weight to large-norm trajectories than arithmetic mean does.
This is conservative — it ensures the clock allocates sufficient time to
the "hard" trajectories at each t_i.

Alternative aggregations and their drawbacks:
- Arithmetic mean: underweights high-norm outlier trajectories.
- Median: not differentiable and not consistent with squared objective.
- 90th percentile: too conservative, produces unstable clocks when K is small.

RMS is the correct choice. Do not change this aggregation without
re-deriving the connection to the squared objective.

### ε: numerical floor

ε = 1e-6  (default, configurable as shared_clock_eps)

Purpose: prevents division by zero if the velocity is exactly zero at any
grid point (possible at t=0 or t=1 near boundary conditions).
ε should be much smaller than the typical Q_V values (order 0.1-10 for
standard models) so it does not materially affect the clock shape.

### Code

  # in: training/ge_stork/shared_clock.py
  # function: _analytic_alpha_profile(observations, clock_family="va", eps=...)
  #
  # norms = observations.material_derivative_norms   # shape [K, T+1]
  # kappa = sqrt( mean_k[ norms^2 ] )
  # alpha = sqrt(max(kappa, eps))
  # implemented as:
  #   kappa = torch.sqrt(torch.mean(norms.pow(2), dim=0))
  #   alpha = torch.sqrt(torch.clamp(kappa, min=float(eps)))
  # shape: [T+1]

---

## 3.6.6 Clock Density Normalization

### Definition

Normalize ᾱ(t_i) to a valid probability density on [0,1] using the
trapezoidal quadrature weights of the physical grid:

  m₀(t_i) = ᾱ(t_i) / Z

where the normalization constant uses trapezoidal weights w_i:

  Z = Σᵢ ᾱ(t_i) × w_i

  w_0     = Δt / 2
  w_T     = Δt / 2
  w_i     = Δt   for 0 < i < T   (uniform grid → all interior weights equal)

This guarantees ∫₀¹ m₀(t) dt = 1 (up to quadrature error of order Δt²).

### Why trapezoidal, not rectangle rule

The trapezoidal rule correctly handles the endpoint half-weights.
Using uniform weights (rectangle rule) would systematically overcount the
endpoints by 2× relative to interior points, shifting the clock density
toward t=0 and t=1.

For a uniform grid with T=64, the difference between trapezoidal and
rectangle rule is small (≈1.5% at endpoints), but the principle matters
for non-uniform grids or when comparing implementations.

### Code

  # in: training/ge_stork/shared_clock.py
  # function: _normalize_density_from_profile(alpha_profile, physical_grid)
  #
  # quad_weights = _quadrature_weights(physical_grid)  # trapezoidal
  # total = sum(alpha_profile * quad_weights)
  # density = alpha_profile / total
  # shape: [T+1]

---

## 3.6.7 Cumulative τ-Map Construction

### Definition

Given the discrete density {m₀(t_i)}, build the cumulative τ-map by
numerical integration using the trapezoidal rule:

  τ(t₀) = 0
  τ(t_{i+1}) = τ(t_i) + 0.5 × (m₀(t_i) + m₀(t_{i+1})) × (t_{i+1} - t_i)

Then renormalize to enforce τ(t_T) = 1 exactly:

  τ_grid[i] ← τ_grid[i] / τ_grid[T]

Set endpoints exactly:
  τ_grid[0] = 0.0   (overwrite any numerical drift)
  τ_grid[T] = 1.0

### Monotonicity guarantee

Since m₀(t_i) > 0 for all i (guaranteed by ε > 0), the cumulative sum is
strictly increasing. This ensures the inverse map t(τ) is well-defined.

### Code

  # in: training/ge_stork/shared_clock.py
  # function: _build_tau_grid(physical_grid, density)
  # output shape: [T+1]

---

## 3.6.8 Schedule Materialization for a Given NFE

### Definition

Given the stored clock profile (physical_grid, density, tau_grid), and a
target step count N (equal to NFE for Euler/STORK; NFE/2 for Heun2):

Step 1: Uniform τ-grid
  τ_n = n / N,  n = 0, 1, ..., N    shape [N+1]

Step 2: Map τ → t via monotone interpolation
  For each τ_n, find the unique t_n such that τ(t_n) = τ_n.
  Use piecewise-linear interpolation on (tau_grid, physical_grid):
    t_n = interp1d(tau_grid, physical_grid, kind='linear')(τ_n)

Step 3: Compute g_grid (reparameterized field scaling)
  For each t_n, interpolate the density to get m(t_n), then:
    g_n = 1 / m(t_n)
  Clamp from below: g_n = max(g_n, ε) to prevent division by zero.

Step 4: Assemble ReparameterizedSchedule
  tau_grid: [τ_0, ..., τ_N]    uniform, spacing 1/N
  t_grid:   [t_0, ..., t_N]    non-uniform, monotone increasing
  g_grid:   [g_0, ..., g_N]    positive
  dtau:     1/N

### Validity checks that must pass

  tau_grid[0] == 0.0,  tau_grid[N] == 1.0
  t_grid[0] == 0.0,    t_grid[N] == 1.0
  all(tau_grid[n+1] - tau_grid[n] > 0)
  all(t_grid[n+1] - t_grid[n] > 0)
  all(g_grid > 0)

These are enforced at runtime by _validate_reparameterized_schedule in
fixed_step_solver.py.

### Why t_grid must also start at 0 and end at 1

t_grid[0] = t(τ=0) = t(0) = 0  by τ(0) = 0 and continuity.
t_grid[N] = t(τ=1) = t(1) = 1  by τ(1) = 1 and continuity.
These are structural constraints of the ODE formulation, not heuristics.
If the interpolation produces endpoints slightly off due to floating point,
snap them: t_grid[0] = 0.0, t_grid[-1] = 1.0.

### Code

  # in: training/ge_stork/shared_clock.py
  # function: SharedClockProfile.make_schedule(nfe, step_count, device, dtype)
  # calls: _linear_interpolate_monotone twice (τ→t, then t→density)

---

## 3.6.9 How the Schedule Enters the Solver

### The reparameterized ODE

After clock reparameterization, the ODE becomes:

  dx/dτ = g(t(τ)) × v_θ(x, t(τ); c)
        = (1/m(t(τ))) × v_θ(x, t(τ); c)

A solver on this ODE uses a uniform step Δτ = 1/N in τ-space,
but evaluates v_θ at physical time t_n = t(τ_n), scaled by g_n.

### Euler step example

  k₁ = g_n × v_θ(x_n, t_n; c)
  x_{n+1} = x_n + Δτ × k₁

### Heun2 step example

  k₁ = g_n   × v_θ(x_n, t_n; c)
  k₂ = g_{n+1} × v_θ(x_n + Δτ k₁, t_{n+1}; c)
  x_{n+1} = x_n + 0.5 × Δτ × (k₁ + k₂)

Note: both g_n and g_{n+1} are used for Heun2, so g_grid must have
length N+1 (one entry per τ-grid node, including the endpoint).

### Code

  # in: training/fixed_step_solver.py
  # function: solve_fixed_budget(..., reparameterized_schedule=schedule)
  # key line: _evaluate_reparameterized_field applies g before velocity

---

## 3.6.10 Complete Pipeline Summary

The full V-a offline clock construction pipeline in pseudocode:


INPUT:
  velocity_model v_θ            (pretrained, frozen)
  calibration_loader D_cal      (training set loader)
  T = 64                        (physical grid size - 1)
  K = 64                        (pilot trajectory count)
  pilot_solver = "heun2"
  ε = 1e-6

STEP 1: Build physical grid
  t_grid = linspace(0, 1, T+1)

STEP 2: Sample K pilot trajectories
  for k in 1..K:
    x0 = randn(C, H, W)
    c = sample from D_cal
    x^(k) = run pilot_solver on {t_i} with v_θ and (x0, c)
    # x^(k) has shape [T+1, C, H, W]

STEP 3: Compute material-derivative norms
  for k in 1..K, i in 0..T:
    Q[k, i] = || D_t v_θ(x^(k)[i], t_i; c^(k)) ||₂
    # Q shape: [K, T+1]

STEP 4: RMS aggregation
  kappa[i] = sqrt( mean_k( Q[k,i]² ) )
  alpha[i] = sqrt(max(kappa[i], ε))
  # alpha shape: [T+1]

STEP 5: Trapezoidal normalization
  w = trapezoidal_weights(t_grid)
  Z = sum(alpha * w)
  m0 = alpha / Z
  # m0 shape: [T+1], satisfies sum(m0 * w) = 1

STEP 6: Build τ-map
  tau[0] = 0
  tau[i+1] = tau[i] + 0.5*(m0[i] + m0[i+1]) * (t_grid[i+1] - t_grid[i])
  tau /= tau[T]
  tau[0] = 0.0;  tau[T] = 1.0
  # tau shape: [T+1], monotone from 0 to 1

STEP 7: Store SharedClockProfile
  profile = SharedClockProfile(
      physical_grid = t_grid,
      density       = m0,
      tau_grid      = tau,
      alpha_profile = alpha,
      ...
  )

OUTPUT: profile  (saved to disk, NFE-agnostic)

---

FOR EACH TARGET NFE N:

STEP 8: Materialize schedule
  tau_uniform = linspace(0, 1, N+1)
  t_sched = interp(tau_uniform; tau, t_grid)    # τ → t
  m_sched = interp(t_sched;    t_grid, m0)     # t → m
  g_sched = 1 / max(m_sched, ε)
  dtau    = 1 / N

  schedule = ReparameterizedSchedule(
      tau_grid = tau_uniform,
      t_grid   = t_sched,
      g_grid   = g_sched,
      dtau     = dtau,
  )

STEP 9: Run solver with schedule
  sample = solve_fixed_budget(v_θ, x_init, solver, NFE, schedule=schedule)


---

## 3.6.11 Hyperparameter Sensitivity Guide

| Parameter | Default | Effect on clock | When to change |
|---|---|---|---|
| T (physical grid size-1) | 64 | Grid resolution of m(t) | Increase to 128 for large models with sharp alpha profiles |
| K (pilot trajectories) | 64 | Statistical quality of ᾱ(t) | K≥32 is usually sufficient; K=128 for publication runs |
| pilot_solver | heun2 | Quality of pilot trajectories | euler acceptable for speed; avoid rk4 (too expensive) |
| ε | 1e-6 | Numerical floor for g_grid | Only change if material-derivative norms are unusually small (<1e-4) |
| aggregation | RMS | Shape of ᾱ(t) | Do not change without re-deriving theoretical consistency |
| norm type | L2 (Euclidean) | Scale of Q values | Do not change |

---

## 3.6.12 What V-a Does NOT Do

To prevent scope creep in implementation:

- V-a does NOT use second-order derivatives of v_θ.
- V-a does NOT compute Jacobians (that is A-family).
- V-a does NOT optimize any loss function (that is V-b).
- V-a does NOT depend on the solver used for final generation.
- V-a does NOT adapt per sample at generation time.
- V-a does NOT require model fine-tuning or retraining.

The clock is built once from K forward passes of v_θ on calibration
noise inputs, then frozen. All of this runs with torch.no_grad().


---

# 4. Scope of applicability

## 4.1 Primary scope

The primary scope is:

- deterministic diffusion ODE samplers,
- deterministic flow-matching ODE samplers,
- solvers that can consume a monotone timestep/sigma schedule.

## 4.2 Secondary scope

EDM-type deterministic samplers are acceptable if used in their deterministic ODE form.

## 4.3 Not primary scope

Do not treat these as main-theory targets:

- ancestral samplers,
- stochastic SDE samplers,
- one-step distilled samplers,
- consistency/LCM/TCD-type distilled methods.

---

# 5. Repo structure and coding rules

## 5.1 Repo purpose

The repo is a **code and experiment repo**.
It should help with:

- clock construction,
- clock discretization,
- running multiple backends,
- collecting metrics,
- keeping server assets documented.

## 5.2 Recommended structure

```text
solver_clock_project/
├── third_party/
│   ├── STORK/
│   └── diffusers/
├── checkpoints/
│   ├── hf/
│   ├── pndm/
│   └── sana/
├── data/
│   ├── pndm/
│   │   ├── prompts/
│   │   ├── fid_stats/
│   │   └── refs/
│   ├── sana/
│   └── cache/
├── schedules/
│   ├── baseline/
│   ├── V_a/
│   ├── V_b/
│   ├── A_a/
│   └── A_b/
├── configs/
│   ├── clocks/
│   ├── solvers/
│   ├── datasets/
│   ├── experiments/
│   └── assets_manifest.yaml
├── docs/
│   ├── SERVER_ASSETS.md
│   └── CHECKPOINT_TREE.txt
├── src/
│   ├── clock/
│   ├── adapters/
│   ├── runners/
│   ├── metrics/
│   └── utils/
├── scripts/
│   ├── setup/
│   ├── run/
│   └── eval/
└── outputs/
    ├── samples/
    ├── metrics/
    └── logs/
```

## 5.3 Asset rule

Real checkpoints stay under `checkpoints/`.
Legacy code paths inside `third_party/` should use **symlinks**, not duplicated files.

## 5.4 Documentation rule

Because the experiment server and coding assistant are not on the same machine, all server assets must be documented in:

- `docs/SERVER_ASSETS.md`
- `configs/assets_manifest.yaml`

No code should guess checkpoint paths.

---

# 6. Server / local workflow assumptions

## 6.1 Development split

- Mac: main coding, repo editing, Claude/Codex interaction
- Server: heavy experiments and checkpoint storage
- GitHub: synchronization layer

## 6.2 Consequence

The coding agent must assume:

- checkpoints are often **not stored in git**,
- server paths must be read from repo documents,
- experiment code should use config/manifest files rather than hard-coded machine-specific paths.

---

# 7. Experiment design principles

This section is the most important implementation section.

The experiments should be built to answer **specific method questions**, not to create many disconnected tables.

## 7.1 What we must prove experimentally

### Claim A: the clock is useful
Fix the solver, compare schedules.

### Claim B: the clock is reusable across solvers
Fix the clock, compare multiple solvers.

### Claim C: the continuous clock is reusable across NFEs
Construct one continuous clock, discretize it at many NFEs.

### Claim D: V-a is the best practical variant
Compare V-a, V-b, A-a, A-b on smaller models first.



---

# 8. What to mirror from STORK experiments

Use STORK as a reference for **benchmark discipline**, not for method framing.

## 8.1 From STORK classic benchmark scripts

The public STORK repo gives concrete benchmark patterns:

### CIFAR-10 benchmark pattern
From `external/PNDM/scripts/nips/cifar/diffusers-deis.sh`:

- dataset: `cifar10`
- config: `ddim_cifar10.yml`
- model path pattern: `models/ddim_cifar10.ckpt`
- FID stats: `inception_stats/fid_cifar10_train.npz`
- number of samples: **50k**
- NFE loop example: **10,20,...,100**
- metric: FID using `pytorch_fid`

Operational lesson:

- for classic small-model benchmarks, use large sample counts,
- use precomputed reference FID stats,
- save generated images then compute FID from folders/stat files.

### COCO / large-model benchmark pattern
From `external/PNDM/scripts/nips/coco/diffusers-euler.sh` and `rock4.sh`:

Common settings include:

- model: `stabilityai/stable-diffusion-3.5-medium`
- dataset: `coco-30k_512`
- samples: **30k**
- precision: **bfloat16**
- seed: **0**
- image size: **512**
- CFG scale: **3.5**
- multi-GPU process count: **8**
- FID computed from generated sample folders

The script patterns also show that different solvers may be scanned over different NFE grids. For example:

- Euler-style script: NFE loop like `8,16,24,32,40`
- ROCK/STORK-like script: NFE loop like `7,15,23,31,39`

Operational lesson:

- keep benchmark settings fixed across schedule comparisons,
- only change one factor at a time,
- use the same sample count and metric protocol for fair comparison.

### MJHQ-30K / Sana benchmark pattern
STORK README explicitly says MJHQ-30K benchmarking is based on `external/Sana`.
Sana metric toolkit supports:

- FID,
- CLIP-Score,
- GenEval,
- DPG-Bench,
- ImageReward.

Operational lesson:

- for large text-to-image models, do not use only FID,
- collect at least one alignment metric and one structured prompt-following metric when feasible.

---

# 9. What to mirror from AYS experiments

Use AYS as a reference for **schedule baselines and low-step evaluation style**.

Even if the exact appendix scripts are not fully reproduced here, the implementation should preserve the core AYS-style comparison structure:

## 9.1 AYS-style comparison pattern

For a fixed model and solver, compare:

- baseline default schedule,
- linear / evenly spaced schedule,
- AYS discrete optimized schedule,
- our continuous-clock-derived schedule.

## 9.2 Low-step regime is essential

AYS is fundamentally about low-step schedules. Therefore we must test:

- 6 NFE,
- 8 NFE,
- 10 NFE,
- 12 NFE,
- 16 NFE,
- 20 NFE,
- optionally 24 and 32.



## 9.3 AYS difference that must be coded and documented

When implementing the baseline families, keep this distinction explicit in configs and results:

- `AYS`: one discrete optimized grid per NFE,
- `OURS`: one continuous clock, then discretize to many NFEs.

This distinction should show up in:

- experiment names,
- result tables,
- plotting scripts,
- README comments.

---

# 10. Required experiment matrix

This is the actual coding target.

## 10.1 Stage 0: smoke tests

Purpose: make sure every backend works end-to-end.

### Small smoke tests
- 100 samples only
- one seed
- save images
- make sure metric scripts run

### Backends to smoke-test
- PNDM small-model line
- Diffusers large-model line
- Sana line

---

## 10.2 Stage 1: variant selection on small models

Purpose: determine whether V-a is indeed the best practical variant.

### Dataset / model
Use the small-model benchmark first, e.g. CIFAR-10 probability-flow model.

### Variants
- V-a
- V-b
- A-a
- A-b

### Solvers
Use a small set first:
- Euler
- Heun2
- one stronger multistep solver if available in backend

### NFEs
- 6
- 8
- 10
- 12
- 16
- 20

### Metrics
- FID
- optional runtime

### Decision rule
If V-a is consistently the strongest or most stable, lock it as the primary method.
Other variants become ablations only.

---

## 10.3 Stage 2: main clock-vs-schedule experiments

Purpose: prove the clock itself is useful.

### Fix solver, compare schedules
For each solver separately, compare:

- default/base schedule
- linear schedule
- AYS schedule
- V-a (main)
- optionally V-b / A-a / A-b in appendix

### Recommended solvers
For diffusion / deterministic ODE:
- Euler
- Heun2
- DPMSolverMultistep
- UniPC

For flow models:
- FlowMatchEuler
- FlowMatchHeun
- flow-DPM-Solver
- flow-UniPC

### NFEs
Main low-step regime:
- 6
- 8
- 10
- 12
- 16
- 20
- 24
- 32

### Metrics
Small models:
- FID

Large models:
- FID if available
- CLIP-Score / GenEval / DPG where feasible
- runtime / throughput

---

## 10.4 Stage 3: solver transfer experiments

Purpose: prove one clock transfers across solvers.

### Fix the clock to V-a
Then compare multiple solvers using the same clock discretization rule.

### What to report
- each solver under base schedule
- the same solver under V-a schedule
- improvement delta per solver

This is essential because the paper claim is solver-agnostic clock usefulness.

---

## 10.5 Stage 4: NFE transfer experiments

Purpose: prove one continuous clock transfers across NFEs.

### Procedure
- construct one continuous V-a clock once,
- discretize it at NFE = 6,8,10,12,16,20,24,32,
- report one curve.

This is one of the most important experiments, because it directly distinguishes us from AYS.

---

## 10.6 Stage 5: large-model experiments

Purpose: show practical usefulness outside toy models.

### Diffusers models to test first
These are supported in STORK-style diffusers code references:

- `stabilityai/stable-diffusion-3.5-medium`
- `black-forest-labs/FLUX.1-dev`
- `Alpha-VLLM/Lumina-Image-2.0`

### Sana line
At minimum:
- `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`

### Main purpose
Do not explode the matrix here.
The large-model section is mainly to show:

- the clock is practical,
- it plugs into real diffusion/flow pipelines,
- V-a still gives useful gains above ultra-tiny NFE.

---

# 11. Concrete experiment settings to implement

This section should directly inform YAMLs and run scripts.

## 11.1 Small-model CIFAR protocol

### Development protocol
Use for debugging:
- samples: 10k
- one seed
- NFE subset: 6, 8, 12, 20

### Full protocol
Use for final tables:
- samples: 50k
- reference FID stats: `fid_cifar10_train.npz`
- one fixed seed for primary tables
- optionally a second seed for robustness check

### Solvers for final CIFAR table
- Euler
- Heun2
- DEIS or DPM-Solver if available in the backend

### Schedules for final CIFAR table
- base/default
- linear
- AYS
- V-a

### Ablation schedules
- V-b
- A-a
- A-b

---

## 11.2 COCO-30K / SD3.5-medium protocol

### Match STORK-style setup as closely as practical
- samples: 30k
- image size: 512
- CFG: 3.5
- precision: bfloat16
- seed: 0
- multi-GPU: 8 processes if available

### Main comparisons
For each solver:
- base/default
- AYS
- V-a

### NFEs
Use the low-step to mid-low-step regime:
- 6
- 8
- 10
- 12
- 16
- 20
- 24
- 32

If some solver only supports a slightly different parity pattern, document it clearly.

---

## 11.3 Sana / MJHQ-30K protocol

### Development protocol
- smaller prompt subset first
- one seed
- a few NFEs only

### Full protocol
- MJHQ-30K full evaluation if feasible
- use official/familiar metric scripts where possible

### Metrics priority
1. FID
2. CLIP-Score
3. one structured metric such as GenEval or DPG-Bench

### Main comparisons
- base schedule
- AYS-style schedule if applicable in the chosen pipeline
- V-a

### Main purpose
This section is for practical value, not for exhausting every variant.

---

# 12. Plotting and table requirements

The repo must be able to produce the following automatically.

## 12.1 Required plots

### Plot 1: FID vs NFE
For each solver, compare:
- base
- AYS
- V-a

### Plot 2: per-solver improvement
For fixed NFE values (e.g. 8, 12, 20), show gain of V-a over base.

### Plot 3: NFE transfer of one clock
One continuous V-a clock discretized at many NFEs.

### Plot 4: variant ablation
V-a / V-b / A-a / A-b on small model.

## 12.2 Required tables

### Table A: main small-model results
Rows: solver × NFE
Columns: base / AYS / V-a

### Table B: large-model practical results
Rows: model × solver × NFE
Columns: quality + runtime

### Table C: variant ablation
Rows: NFE
Columns: V-a / V-b / A-a / A-b

---

# 13. Coding priorities

## Priority 1
Implement a robust V-a clock pipeline:

- calibration trajectory collection,
- continuous proxy estimation,
- clock normalization,
- monotone cumulative map,
- discretization to NFE-specific timesteps/sigmas.

## Priority 2
Implement reusable schedule export:

- `timesteps.npy`
- `sigmas.npy`
- `meta.json`

## Priority 3
Implement backend adapters:

- PNDM adapter,
- Diffusers adapter,
- Sana adapter.

## Priority 4
Implement result collection:

- FID runner,
- CLIP/GenEval/DPG wrappers,
- CSV summary output,
- plotting scripts.

## Priority 5
Only after the above, do broader ablations.

---

# 14. Things Claude must NOT do

- Do not reframe the method as solver-clock co-design.
- Do not make STORK-style solver innovation the main contribution.
- Do not over-focus on weak variants if V-a is already strongest.
- Do not hide poor 6-NFE behavior.
- Do not hardcode server checkpoint paths into multiple files.
- Do not assume checkpoints are present in git.
- Do not write a paper-first artifact before building the experiment pipeline.

---
