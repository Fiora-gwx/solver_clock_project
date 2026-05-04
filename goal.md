# /goal: Implement and validate GOES — Geometry-aware Oracle Edge Scheduling

## 0. Purpose

Implement a training-free schedule optimizer for diffusion / flow ODE samplers in this repository. The method is **GOES: Geometry-aware Oracle Edge Scheduling**.

The implementation must be suitable for paper-grade experiments: deterministic, reproducible, cached, clearly logged, and able to produce schedules, metrics, ablations, and result tables that can be directly used in a paper if the method works.

Do **not** implement this as a training method. Do **not** change model weights. The only optimized object is the sampling schedule.

---

## 1. Core method in one paragraph

GOES first constructs a **solver-independent high-accuracy oracle trajectory** for each calibration sample by solving the underlying deterministic probability-flow / flow ODE with a sufficiently high-NFE reference integrator. This oracle is cached once and reused across all target solvers. For a target solver `S`, GOES evaluates the solver-specific local edge defect by starting from the oracle state `x_star(a)`, applying one solver transition from `a` to `b`, and comparing the output to the shared oracle endpoint `x_star(b)` with a metric-aware mixed normal defect. The schedule is obtained by solving a discrete min-max problem over an admissible candidate grid: choose `N` intervals that minimize the maximum oracle edge defect.

The clean separation is:

```text
oracle trajectory: model-specific, data-specific, coordinate-specific, solver-independent
edge defect:       solver-specific
schedule:          solver-specific
```

---

## 2. Mathematical specification that must be implemented

### 2.1 Deterministic ODE setting

Assume a deterministic sampling ODE

```math
dx/du = f_theta(x, u; c)
```

where:

- `x` is the latent / state.
- `u` is a unified monotone sampling coordinate, such as `t`, `sigma`, `log_sigma`, or log-SNR.
- `c` is the condition, such as prompt, class label, CFG information, or other conditioning data.
- `f_theta` is the model-induced ODE drift.

The implementation must expose or infer a `CoordinateAdapter` that maps between the repository's native sampler coordinate and the unified increasing coordinate `u`.

DP and edge search should operate in increasing `u`. If the repository's native sampler runs in the reverse direction, convert schedules before and after optimization.

### 2.2 Universal solver-independent oracle

For each calibration sample `k`, fixed initial noise `z_k`, and fixed condition `c_k`, construct a high-accuracy reference trajectory:

```math
x_k^*(u) \approx x_k^{true}(u),  u in [u_min, u_max].
```

This oracle must be computed once and cached. It must not be recomputed separately for each target solver.

The oracle cache should contain, when available:

```text
x_star[k, j]        = oracle state at grid point u_j
v_star[k, j]        = tangent / derivative dx_star/du at u_j
model_out[k, j]     = optional model output / drift cache if useful
u_grid[j]           = reference grid in unified coordinate
condition[k]        = enough metadata to replay sample k
noise_seed[k]       = seed or initial noise identifier
metadata            = model id, coordinate id, ref integrator, ref_nfe, cfg, dtype, device, hashes
```

If the ODE drift can be evaluated at oracle states, use

```math
v_k^*(u) = f_theta(x_k^*(u), u; c_k).
```

If direct drift evaluation is not available, estimate tangent by smoothed central finite differences on a dense grid:

```math
v_k^*(u_j) = (x_k^*(u_{j+1}) - x_k^*(u_{j-1})) / (u_{j+1} - u_{j-1}).
```

If `||v_star||` is numerically too small at a point, fall back to full residual at that point and log the fallback count.

### 2.3 Oracle reuse requirement

The oracle must be reusable across solvers when all solvers approximate the same underlying deterministic ODE and use the same unified coordinate mapping.

The code must verify and log:

```text
model identifier
ODE / sampler family identifier if available
coordinate mapping
reference NFE
reference grid hash
condition / prompt split hash
initial noise hash
CFG / guidance parameters
```

If two solvers use incompatible dynamics, stochastic ancestral steps, dynamic thresholding that changes the effective ODE, or unhandled postprocessing, the code must mark the solver as not strictly covered by the deterministic-oracle theory. It may still run as an empirical experiment, but the log must say so.

### 2.4 Metric-aware mixed normal defect

For a target solver `S` and candidate edge `(a, b)`, query the oracle:

```math
x_k^*(a), x_k^*(b), v_k^*(b).
```

Run the target solver for a single transition starting from the oracle state:

```math
\hat{x}_{S,k}^{a->b} = Phi_S(x_k^*(a); a -> b).
```

Then define residual:

```math
r_{S,k}(a,b) = \hat{x}_{S,k}^{a->b} - x_k^*(b).
```

Given a positive metric `G(u)`, define the metric-normalized tangent:

```math
T_{k,G}(u) = v_k^*(u) / sqrt( v_k^*(u)^T G(u) v_k^*(u) + eps ).
```

Define mixed normal defect:

```math
||r||_{rho,G,k,u}^2
= r^T G(u) r - (1-rho) * ( T_{k,G}(u)^T G(u) r )^2
```

Equivalent form:

```math
||r||_{rho,G,k,u}^2
= ||P_perp,G r||_G^2 + rho * ||P_parallel,G r||_G^2.
```

Required behavior:

- `rho = 1` must exactly reduce to full metric residual.
- `rho = 0` is pure normal residual, but it must not be the default.
- Default `rho` should be `0.1` unless the experiment config overrides it.
- Clamp tiny negative values caused by floating-point roundoff to zero.
- Log the fraction of points where full-residual fallback was used due to tiny tangent norm.

### 2.5 Metric choices

Implement at least these metrics:

1. `identity`

```math
G(u) = I.
```

2. `edm_scalar`, when `sigma(u)` is available

```math
G(u) = 1 / (sigma(u)^2 + sigma_data^2) * I.
```

3. `channel_whitened`

Estimate channel variances from calibration oracle latents and use a diagonal channel metric:

```math
G_c(u) = 1 / (Var_c[x^*(u)] + eps).
```

For images / latents with shape `[B, C, H, W]`, channel variance should aggregate over calibration samples and spatial dimensions. Clamp the metric weights to a safe range to avoid outliers.

The metric module must expose a consistent API such as:

```python
metric.dot(x, y, u) -> scalar per sample
metric.norm_sq(x, u) -> scalar per sample
metric.apply(x, u) -> tensor with same shape as x if needed
```

### 2.6 Solver-specific edge cost

For a calibration set of `K` oracle trajectories, define per-sample edge cost:

```math
D_{S,k}(a,b) = ||r_{S,k}(a,b)||_{rho,G,k,b}^2.
```

Aggregate over calibration samples:

```math
D_S(a,b) = RobustAgg_k D_{S,k}(a,b).
```

Implement robust aggregation options:

```text
mean
median
trimmed_mean with default trim ratio 0.10
cvar with configurable alpha, default 0.80
```

Default aggregation should be `trimmed_mean_10pct`.

### 2.7 Discrete min-max schedule optimization

For target solver `S`, define an admissible candidate grid:

```math
A_S = {a_0, a_1, ..., a_M},  a_0 = u_min, a_M = u_max.
```

Precompute upper-triangular edge costs:

```math
D[j, l] = D_S(a_j, a_l),  j < l.
```

Solve:

```math
U_S^* = argmin_{0=i_0<i_1<...<i_N=M} max_m D[i_m, i_{m+1}].
```

Required DP recurrence:

```math
dp[n, l] = min_{j<l} max( dp[n-1, j], D[j,l] ).
```

with:

```math
dp[0,0] = 0,
dp[0,l>0] = inf.
```

The returned schedule must include exactly `N+1` points and must be strictly monotone in unified coordinate `u`.

Tie-breaking:

- Primary objective: minimize maximum edge cost.
- Optional secondary objective: minimize total selected edge cost.
- Optional tertiary objective: minimize smoothness penalty.
- Tie-breaking must not change the primary min-max optimum except within a documented tolerance.

### 2.8 Multistep solver support

Implement two modes.

#### Mode A: oracle-consistent history, when supported

If a multistep solver allows explicit history injection, construct its history from the shared oracle states and, if needed, oracle model outputs.

For a `p`-step solver, the local cost may depend on context:

```math
D_S(a_{i-p+1}, ..., a_i, b).
```

Implement this only if the repository's solver API supports it cleanly. Add tests if implemented.

#### Mode B: black-box replay refinement, default fallback

If the solver does not allow history injection, do not pretend that prefix replay is a prefix-free local edge cost.

Instead:

1. Use the one-step proxy edge DP to get an initial schedule.
2. Run the actual black-box solver with the candidate schedule.
3. Evaluate replay errors against the same universal oracle at schedule endpoints.
4. Refine the internal schedule points using local grid / coordinate search.

Replay loss:

```math
L_replay(U)
= SmoothMax_i E_i(U) + lambda_final * E_final(U) + mu_smooth * R(U).
```

where:

```math
E_i(U) = RobustAgg_k || x_{S,k}^{replay}(u_{i+1}; U) - x_k^*(u_{i+1}) ||_{rho,G,k,u_{i+1}}^2.
```

and `R(U)` can be a log-step-ratio smoothness penalty:

```math
R(U) = sum_i ( log((u_{i+1}-u_i)/(u_i-u_{i-1})) )^2.
```

This replay refinement is empirical and solver-specific. It still uses the same universal oracle and must not recompute the oracle.

---

## 3. Implementation deliverables

Codex should first inspect the repository structure and identify existing sampler, scheduler, model, dataset, and evaluation entry points. Then implement the smallest clean extension that fits the project style.

### 3.1 New modules or equivalent functionality

Create modules with names adapted to the repository conventions. Suggested structure:

```text
<repo>/goes/
  __init__.py
  config.py
  coordinate.py
  oracle.py
  oracle_cache.py
  interpolation.py
  metrics.py
  mixed_defect.py
  edge_evaluator.py
  dp_minimax.py
  replay_refinement.py
  schedules.py
  logging_utils.py
  experiment_runner.py
```

If the repository already has a scheduler or experiment framework, integrate there instead of duplicating infrastructure.

### 3.2 Config schema

Add a config object / YAML schema with at least:

```yaml
method: goes
model:
  name: null
  checkpoint: null
  dtype: float32
  device: cuda
coordinate:
  name: log_sigma        # or t, sigma, logsnr
  direction: increasing  # internal unified direction
oracle:
  ref_integrator: high_order_or_repo_default
  ref_nfe: 1000
  ref_grid_size: 2048
  interpolation: linear_or_cubic
  cache_dir: ./outputs/goes/oracle_cache
  reuse: true
calibration:
  num_samples: 32
  seed: 123
  prompt_file: null
  split: calibration
candidate_grid:
  size: 512
  type: uniform_in_u
solver:
  name: null
  target_nfe: 10
  mode: one_step_or_blackbox_multistep
metric:
  name: edm_scalar
  sigma_data: 0.5
  eps: 1.0e-12
mixed_defect:
  rho: 0.1
  fallback_full_residual_on_tiny_tangent: true
aggregation:
  name: trimmed_mean
  trim_ratio: 0.10
optimizer:
  name: dp_minimax
  tie_break_sum_cost: true
replay_refinement:
  enabled: false
  rounds: 3
  local_window: 8
  smoothmax_alpha: 10.0
  lambda_final: 0.0
  mu_smooth: 0.0
output:
  root: ./outputs/goes
  save_edge_table: true
  save_schedule: true
  save_images: false
  save_plots: true
```

### 3.3 CLI / script entry points

Add command-line entry points if the project uses scripts. Suggested commands:

```bash
# Build or load universal oracle
python -m goes.experiment_runner build-oracle --config configs/goes/default.yaml

# Search one solver schedule using cached oracle
python -m goes.experiment_runner search-schedule --config configs/goes/euler_10nfe.yaml

# Evaluate a schedule on held-out samples
python -m goes.experiment_runner evaluate --config configs/goes/euler_10nfe.yaml --schedule outputs/goes/.../schedule.json

# Run ablations
python -m goes.experiment_runner ablate-rho --config configs/goes/base.yaml
python -m goes.experiment_runner ablate-metric --config configs/goes/base.yaml
python -m goes.experiment_runner oracle-convergence --config configs/goes/base.yaml
```

Adapt these to the repository's CLI style if needed.

### 3.4 Outputs that must be saved

Every run must create a unique run directory:

```text
outputs/goes/<timestamp>_<method>_<solver>_<nfe>_<short_hash>/
  config.resolved.yaml
  schedule.json
  schedule_native.json
  edge_costs.npz
  selected_edges.csv
  calibration_metrics.csv
  heldout_metrics.csv
  oracle_metadata.json
  run_metadata.json
  plots/
    schedule.png
    edge_cost_heatmap.png
    selected_edge_costs.png
    nfe_quality_curve.png        # if applicable
  paper_tables/
    main_results.csv
    ablations.csv
    oracle_reuse_cost.csv
```

`schedule.json` must include:

```json
{
  "method": "GOES",
  "solver": "...",
  "target_nfe": 10,
  "coordinate": "...",
  "u_schedule": [...],
  "native_schedule": [...],
  "rho": 0.1,
  "metric": "...",
  "aggregation": "trimmed_mean_10pct",
  "oracle_cache_key": "...",
  "edge_objective": 0.0,
  "selected_edge_costs": [...]
}
```

---

## 4. Required tests

Add unit tests and small integration tests. Use the repository's existing test framework.

### 4.1 Mixed defect tests

Test the formula exactly.

1. `rho=1` equals full residual.
2. `rho=0` equals normal residual for identity metric.
3. Sign flip of tangent does not change the value.
4. Result is non-negative up to numerical tolerance.
5. If tangent norm is tiny, fallback behavior is triggered and logged.
6. For a straight line with pure tangential error, `rho>0` detects nonzero error.

### 4.2 DP min-max tests

Use a small manually constructed edge matrix where the optimal path is known.

Verify:

1. Schedule starts at index `0` and ends at index `M`.
2. It contains exactly `N+1` indices.
3. Indices are strictly increasing.
4. The DP objective equals brute-force enumeration for small `M`.
5. Tie-breaking does not break the primary min-max optimum.

### 4.3 Oracle cache tests

Verify:

1. Same oracle key loads cached data instead of recomputing.
2. Different model / coordinate / ref_nfe / calibration seed produces different cache key.
3. Oracle can be reused by at least two dummy solvers without rebuilding.
4. Interpolation returns correct shapes and finite values.
5. Coordinate conversion round-trip is accurate.

### 4.4 Edge evaluator tests

Use toy ODEs where exact trajectories are known.

Toy A: straight line trajectory with pure tangential solver bias.

Expected:

- `rho=0` can be zero.
- `rho>0` is nonzero.
- `rho=1` equals full residual.

Toy B: circle trajectory with normal bias.

Expected:

- Normal-sensitive defects detect the error.
- Tangent sign flip does not change cost.

Toy C: prefix independence.

Expected:

- Local edge evaluator must use `x_star(a)` and not a replayed approximate prefix state.
- Test by injecting a fake prefix error and confirming local edge cost is unchanged.

### 4.5 Smoke integration test

Run a tiny calibration with small grid and small NFE on CPU or a minimal model if available. It should finish quickly and produce:

```text
schedule.json
edge_costs.npz
selected_edges.csv
run_metadata.json
```

---

## 5. Experiments for paper-grade validation

Implement experiment scripts/configs that generate clean result tables. Experiments should separate calibration samples from held-out evaluation samples.

### 5.1 Universal oracle convergence

Purpose: verify that reference NFE is sufficient and that the oracle is not the bottleneck.

Sweep:

```text
ref_nfe in {100, 200, 500, 1000}
```

For each `ref_nfe`, search a schedule for the same solver and target NFE.

Report:

```text
schedule L1 distance to highest-ref-NFE schedule
selected edge objective
rank correlation of edge costs
final latent oracle MSE
image metrics if available
wall-clock oracle build time
```

Success criterion:

```text
Schedules and final metrics stabilize as ref_nfe increases.
Use the smallest ref_nfe after the stability plateau as the default.
```

### 5.2 Cross-solver oracle reuse

Purpose: prove that one oracle supports many solver-specific schedules.

Use the same oracle cache for:

```text
Euler
Heun
DPM-Solver++ or available multistep solver
UniPC or available corrector solver
DEIS or any existing high-order solver in the repo
```

Use only solvers actually present in the repository. If a solver is not available, skip it and log the skip reason.

Report:

```text
oracle cache key reused by all solvers
per-solver GOES schedule
per-solver edge objective
per-solver final metrics
oracle build time once
edge evaluation time per solver
search time per solver
total amortized cost
```

Must include an amortized cost table:

```text
separate oracle per solver vs shared universal oracle
```

### 5.3 Main NFE sweep

Purpose: measure quality-vs-NFE.

Sweep target NFE:

```text
N in {4, 5, 6, 8, 10, 12, 15, 20, 30, 50}
```

Compare schedules:

```text
uniform_t if available
uniform_sigma if available
uniform_logsnr if available
Karras / EDM if available
GOES
GOES + replay refinement for black-box multistep solvers
```



Report:

```text
final latent oracle MSE
FID / KID if dataset evaluation exists
CLIPScore / HPS / PickScore if text-to-image evaluation exists
generation wall-clock
NFE
solver name
schedule name
confidence intervals or bootstrap standard errors across held-out samples
```

### 5.4 Mixed normal rho ablation

Sweep:

```text
rho in {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}
```

Default expected choice is `rho=0.1`, but the experiment should determine the best value.

Report:

```text
edge objective
selected schedule
final latent oracle MSE
image metrics
held-out generalization gap
failure cases
```

### 5.5 Metric ablation

Compare:

```text
identity
edm_scalar
channel_whitened
```

Report:

```text
which metric produces schedules best correlated with held-out final quality
edge objective
final latent oracle MSE
image metrics
schedule shape
```

### 5.6 Calibration size ablation

Sweep:

```text
K in {4, 8, 16, 32, 64, 128}
```

Use the same held-out set for evaluation.

Report:

```text
calibration objective
held-out objective
held-out image metrics
schedule stability
runtime
```

### 5.7 Candidate grid density ablation

Sweep:

```text
M in {64, 128, 256, 512, 1024}
```

Report:

```text
DP objective
schedule stability
final metrics
edge table construction time
DP time
```

### 5.8 Multistep black-box replay refinement

For each available multistep solver, compare:

```text
baseline schedule
GOES one-step proxy DP
GOES one-step proxy DP + black-box replay refinement
```

Report:

```text
local edge objective
actual replay endpoint errors
final metrics
runtime
number of refinement rounds
```

### 5.9 Held-out generalization

Calibration and evaluation samples must be disjoint.

For image generation, record:

```text
prompt id
seed
initial noise id
CFG scale
solver
schedule
NFE
all metric values
```

Run at least three random calibration seeds if feasible, or bootstrap across prompts. Save confidence intervals.

### 5.10 Failure analysis

Automatically save the worst examples where GOES underperforms a baseline.

For each failure case, save:

```text
prompt / condition
seed
baseline image if available
GOES image if available
metric deltas
selected schedule
edge costs
replay endpoint errors
notes about tiny tangent fallback / metric outliers / interpolation warnings
```

---

## 6. Baseline policy

Use only baselines that are either already implemented in the repository or easy to define unambiguously.

Allowed unambiguous baselines:

```text
uniform in native t
uniform in sigma
uniform in log_sigma / logSNR if coordinate mapping exists
Karras / EDM schedule if already implemented or formula exists in the repo
existing project default schedule
```

Conditional baselines:

```text
AYS: only if a trusted explicit schedule table or implementation exists in the repository/config.
Other learned or optimized schedules: only if already implemented.
```

When a baseline is unavailable, write a skip reason to `run_metadata.json`. Do not silently replace unavailable baselines with approximations.

---

## 7. Reproducibility requirements

All experiments must use common random numbers where possible.

Required logging:

```text
Git commit hash if available
Python version
PyTorch / JAX / CUDA versions if applicable
model checkpoint hash or path
config resolved values
calibration split hash
held-out split hash
initial noise seeds
oracle cache key
schedule hash
```

Set deterministic seeds for:

```text
Python random
NumPy
PyTorch / JAX / framework RNG
DataLoader workers if applicable
```

Do not mix calibration samples and held-out evaluation samples.

---

## 8. Acceptance criteria

The task is complete only when all applicable items below are satisfied.

### 8.1 Method correctness

- Universal oracle is computed once and reused across multiple solvers.
- Edge cost starts from `x_star(a)`, not from a prefix-contaminated replay state.
- Mixed normal defect is implemented exactly according to the formula.
- Default `rho` is nonzero, preferably `0.1`.
- DP solves the discrete min-max problem and is tested against brute force for small cases.
- Candidate schedules are strictly monotone and have exactly `N+1` points.
- Coordinate direction and native coordinate conversion are handled explicitly.
- Stochastic / ancestral samplers are either excluded from theory mode or clearly marked as empirical-only.

### 8.2 Engineering correctness

- Unit tests pass.
- Smoke integration test produces the required output files.
- No model training is introduced.
- Existing sampler behavior remains unchanged unless GOES is explicitly selected.
- The implementation supports CPU smoke tests and GPU full experiments if the repo supports GPU.
- Edge table and oracle cache are saved in reusable formats.

### 8.3 Experiment usefulness

- At least one main benchmark can be run end-to-end.
- Results are saved as CSV/JSON with enough metadata for paper tables.
- Calibration and held-out results are separated.
- The experiment runner can generate plots or at least raw data for:
  - NFE-quality curve
  - rho ablation
  - metric ablation
  - oracle convergence
  - cross-solver oracle reuse cost

### 8.4 Paper-ready reporting

Final output should include:

```text
implementation summary
files changed
commands run
unit/integration test results
experiment commands
where results are saved
known limitations
which solvers are theory-covered vs empirical-only
```

---

## 9. Suggested first milestone

If the repository is complex, implement in this order:

1. `metrics.py` and `mixed_defect.py` with tests.
2. `dp_minimax.py` with brute-force tests.
3. Toy oracle and toy edge evaluator with tests.
4. Repository-integrated oracle cache for one simple solver.
5. One complete GOES schedule search run for a simple solver.
6. Main benchmark on the smallest feasible model / dataset.
7. Cross-solver oracle reuse.
8. Ablations and paper tables.

Do not start with large text-to-image experiments before the toy and small-solver tests pass.

---

## 10. Important non-negotiables

- Do not optimize a continuous monitor by equalization as the main algorithm.
- Do not use prefix replay error as a local edge defect.
- Do not default to pure normal residual.
- Do not construct a different oracle for every solver when the underlying ODE is shared.
- Do not claim theory for stochastic ancestral sampling unless Brownian coupling / SDE handling is explicitly implemented.
- Do not hide skipped baselines.
- Do not mix calibration and evaluation samples.
- Do not report only calibration performance.
- Do not leave results without machine-readable metadata.

---

## 11. Minimal pseudocode reference

```python
def build_universal_oracle(model, calibration_samples, coord, ref_integrator, ref_nfe, ref_grid):
    cache_key = make_oracle_key(model, calibration_samples, coord, ref_integrator, ref_nfe, ref_grid)
    if cache_exists(cache_key):
        return load_oracle(cache_key)

    states = []
    tangents = []
    for sample in calibration_samples:
        traj = solve_reference_ode(model, sample, coord, ref_integrator, ref_nfe, ref_grid)
        tangent = compute_tangent_or_drift(model, traj, sample, coord)
        states.append(traj)
        tangents.append(tangent)

    oracle = OracleCache(states=states, tangents=tangents, u_grid=ref_grid, metadata=...)
    save_oracle(cache_key, oracle)
    return oracle


def mixed_normal_defect_sq(r, v, metric, u, rho, eps):
    vGv = metric.dot(v, v, u)
    if vGv < eps:
        return metric.dot(r, r, u), True
    TG = v / sqrt(vGv + eps)
    full = metric.dot(r, r, u)
    tang = metric.dot(TG, r, u) ** 2
    value = full - (1.0 - rho) * tang
    return max(value, 0.0), False


def evaluate_edge_table(solver, oracle, candidate_grid, metric, rho, aggregation):
    M = len(candidate_grid) - 1
    D = full((M + 1, M + 1), inf)
    for j in range(M + 1):
        for l in range(j + 1, M + 1):
            a, b = candidate_grid[j], candidate_grid[l]
            costs = []
            for k in range(oracle.num_samples):
                x_a = oracle.state(k, a)
                x_b = oracle.state(k, b)
                v_b = oracle.tangent(k, b)
                x_hat = solver.single_edge_step_from_state(x_a, a, b, oracle.condition(k))
                r = x_hat - x_b
                cost, fallback = mixed_normal_defect_sq(r, v_b, metric, b, rho, eps=1e-12)
                costs.append(cost)
            D[j, l] = robust_aggregate(costs, aggregation)
    return D


def dp_minimax_schedule(D, N):
    M = D.shape[0] - 1
    dp = full((N + 1, M + 1), inf)
    prev = full((N + 1, M + 1), -1)
    dp[0, 0] = 0.0

    for n in range(1, N + 1):
        for l in range(1, M + 1):
            best = inf
            best_j = -1
            for j in range(n - 1, l):
                candidate = max(dp[n - 1, j], D[j, l])
                if candidate < best:
                    best = candidate
                    best_j = j
            dp[n, l] = best
            prev[n, l] = best_j

    indices = backtrack(prev, N, M)
    assert indices[0] == 0 and indices[-1] == M and len(indices) == N + 1
    return indices, dp[N, M]
```

---

## 12. Deliverable expected from Codex

After implementation, provide a concise final report with:

```text
1. What was implemented.
2. Exact files changed.
3. Tests added and test results.
4. Commands to build oracle, search schedule, and run evaluation.
5. Where outputs are saved.
6. Which repository solvers are currently supported.
7. Which experiments are ready for paper data collection.
8. Known limitations and next steps.
```