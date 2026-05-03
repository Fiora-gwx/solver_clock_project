
核心思想是：**不再用 Euler proxy 估计所有 solver 的 defect，而是让每个 solver 用自己的 base trajectory，在自己的 native 坐标 (u_S) 上，通过 16/32/64 多分辨率轨迹差异估计自己的 horizontal defect density。**

这能同时覆盖单步法和多步历史法，因为它把 solver 当成黑盒轨迹生成器，不需要手动 clone 每一步的 history state。

---

# 1. 你的想法总体是否成立？

我认为成立，但要注意一个关键修改：

不要直接比较 16、32、64 的 endpoint state 差异：

[
x_{16}(u_i)-x_{32}(u_i).
]

因为这个差异包含前面所有区间累积的误差，不是当前区间产生的局部 defect。

应该比较 **区间位移差异** 或 **window 位移差异**：

[
\Delta x_{N,i}
==============

x_N(u_{i+1})-x_N(u_i),
]

[
\Delta x_{2N,i}
===============

x_{2N}(u_{i+1})-x_{2N}(u_i),
]

然后定义：

[
\Psi_{S,i}^{(N)}
================

P_{\perp,i}
\left(
\Delta x_{N,i}
--------------

\Delta x_{2N,i}
\right).
]

这才是当前 interval 的 solver discrepancy，而不是全局 accumulated discrepancy。

对于多步法，例如 STORK、DEIS、DPM multistep，更推荐 window 版本：

[
\Delta_W x_{N,i}
================

x_N(u_{i+W})-x_N(u_i),
]

[
\Delta_W x_{2N,i}
=================

x_{2N}(u_{i+W})-x_{2N}(u_i),
]

[
\boxed{
\Psi_{S,i,W}^{(N)}
==================

P_{\perp,i,W}
\left(
\Delta_W x_{N,i}
----------------

\Delta_W x_{2N,i}
\right).
}
]

其中 (W) 至少覆盖 solver 的历史长度：

[
W\geq \text{solver order}.
]

这样就自然兼容单步和多步：

* Euler / EDM Euler：(W=1)
* DEIS / DPM-Solver multistep：(W=2) 或 (3)
* STORK：(W=4) 或根据 derivative/history order 设置

这个版本工程上更容易，也更符合最终方法的“target-solver defect”原则。

---

# 2. 为什么 16/32/64 多分辨率方案满足方法原理？

最终 FP 方法需要每个 solver 的水平误差系数：

[
\mathcal C_{S}^{\perp}(s).
]

理想定义是 local Richardson：

[
r_i^S
=====

## \Psi_h^S(z_i)

\Psi_{h/2}^S\circ\Psi_{h/2}^S(z_i),
]

然后投影：

[
r_{i,\perp}^S=P_{\perp,i}r_i^S.
]

但对多步法，(z_i=(x_i,H_i)) 包含复杂 history。直接 clone/restore history 很麻烦。你的方案换成：

[
\text{run } S \text{ at } N,2N,4N
]

并用同一个 solver 的 coarse/fine base trajectories 构造多分辨率 residual。这样得到的是：

[
\boxed{
\Psi_{S,i}^{(N)}
================

\text{target solver } S
\text{ 在该区域的 empirical horizontal discrepancy。}
}
]

如果 (N,2N,4N) 轨迹足够对齐，并且 (2N/4N) 足够细，则：

[
|\Psi_{S,i}^{(N)}|
\approx
\rho(q_i)
\mathcal C_{S}^{\perp}(s_i)
(\Delta s_i)^{q_i}.
]

所以：

[
\boxed{
D_{S,i}^{\perp}
===============

\frac{
|\Psi_{S,i}^{(N)}|
}{
\rho(q_i)(\Delta s_i)^{q_i}+\epsilon
}
\approx
\mathcal C_S^\perp(s_i).
}
]

这正是我们最终方法需要的量。

---

# 3. 需要避免的一个陷阱：官方 base schedule 不一定嵌套

你的设想是对每个 solver 分别跑：

[
N=16,\quad 32,\quad 64.
]

如果这三个 base schedule 的节点满足嵌套关系：

[
u_{16}\subset u_{32}\subset u_{64},
]

那么 Richardson 估计最干净。

但很多 scheduler 的官方 base schedule 不一定严格嵌套。比如 Karras/EDM、DPM、DEIS 的时间点可能会随 NFE 变化，而不是 16 的节点刚好包含在 32 里。

因此有两个实现模式：

## 模式 A：Official-base multiresolution，最实用

每个 solver 用自己的官方 base schedule at 16/32/64。
然后把 32/64 轨迹插值到 16/32 的节点上。

例如：

[
\widehat x_{32}(u_i^{16})
=========================

\mathcal I[x_{32}](u_i^{16}).
]

再比较：

[
\Delta x_{16,i}
===============

x_{16}(u_{i+1}^{16})-x_{16}(u_i^{16}),
]

[
\Delta \widehat x_{32,i}
========================

\widehat x_{32}(u_{i+1}^{16})-\widehat x_{32}(u_i^{16}).
]

这个模式最贴近真实 solver base 行为，适合第一版。

## 模式 B：Nested calibration grid，更严谨

先生成 (N=64) 的 native base grid，然后抽取 nested 子网格作为 32 和 16：

[
u_{32}=u_{64}[::2],
]

[
u_{16}=u_{64}[::4].
]

这样 Richardson 关系更严格，但它可能不再完全等于 solver 官方 16/32 base schedule。

建议第一阶段用模式 A，第二阶段做模式 B 作为 sanity/ablation。

---

# 4. Heun 是否删除？

可以。你说“官方 Heun 必须奇数，所以先删除不做保留”，我同意。

Heun 的 odd effective NFE 会破坏 16/32/64 这种干净 refinement ratio：

[
16 \to 32 \to 64.
]

如果硬做 Heun，要改成：

[
17,33,65
]

或者：

[
15,31,63.
]

但这会让实验和工程复杂化。当前目标是为所有主要 solver 建立统一 target-defect oracle，Heun 可以先排除。

在项目里可以标记：

```text
heun2: excluded_from_multiresolution_fp_clock_due_to_odd_effective_nfe
```

不要让它阻塞主线。

---

# 5. 对每个 solver 的 native coordinate (u_S)

你这个方案的关键是“各自的 (u)”。这是对的。

项目的 PNDM adapter 里已经有 solver registry，并且支持 Euler、DEIS、DPM-Solver、DPM-Solver++、UniPC、STORK 等 solver；其中 STORK、Heun 等被标成 sigma-native，DPM/DEIS/Euler 等可以走 timestep 或对应 scheduler representation【turn91file0】。当前旧 RI 文档也强调 clock measure 应该通过

[
d\tau=n(s)ds
]

拉回物理坐标

[
\alpha(u)=n(s(u))|x'(u)|
]

而不是固定在某一个 (\sigma) 参数化上。

我建议统一定义：

| solver family      | native coordinate (u_S)         | 备注                                             |
| ------------------ | ------------------------------- | ---------------------------------------------- |
| EulerDiscrete      | scheduler timesteps 或 sigma     | 以 scheduler 实际 set_timesteps 产生的 coordinate 为准 |
| EDM / Karras       | (\log\sigma) 或 (\sigma)         | 建议优先 (\log\sigma)，因为 EDM 常按 sigma 几何尺度组织       |
| DPM-Solver / DPM++ | (\lambda=\log\alpha-\log\sigma) | 最推荐，用 log-SNR/lambda                           |
| DEIS               | lambda / timestep               | 视 scheduler 实现，优先和 solver coefficients 一致      |
| STORK              | sigma                           | STORK 通常 sigma-native                          |
| UniPC              | timestep/lambda                 | 优先 scheduler-native                            |

实现上不需要一次完美。可以让每个 solver adapter 暴露：

```python
native_coordinate_name
native_base_grid(nfe)
record_trajectory(nfe, seed, prompts)
interpolate_state_to_u(...)
recommended_window_len
```

---

# 6. 多分辨率 horizontal residual 的最终公式

对目标 solver (S)，运行：

[
N_0=16,\quad N_1=32,\quad N_2=64.
]

得到三条轨迹：

[
x_{S,N_0}(u),
\quad
x_{S,N_1}(u),
\quad
x_{S,N_2}(u).
]

令 coarse grid 为 (u_i^{N_0})。将 (x_{S,N_1}) 插值到 (u_i^{N_0})：

[
\widehat x_{S,N_1}(u_i^{N_0})
=============================

\mathcal I[x_{S,N_1}](u_i^{N_0}).
]

定义 window interval：

[
[u_i,u_{i+W}].
]

coarse window displacement：

[
\Delta_W x_{S,N_0,i}
====================

x_{S,N_0}(u_{i+W})-x_{S,N_0}(u_i).
]

fine window displacement：

[
\Delta_W \widehat x_{S,N_1,i}
=============================

\widehat x_{S,N_1}(u_{i+W})-\widehat x_{S,N_1}(u_i).
]

midpoint tangent 用最高分辨率轨迹 (N_2=64) 估计：

[
T_{S,i,W}
=========

\frac{
\widehat x_{S,N_2}(u_{i+W})-\widehat x_{S,N_2}(u_i)
}{
\left|
\widehat x_{S,N_2}(u_{i+W})-\widehat x_{S,N_2}(u_i)
\right|
+\epsilon
}.
]

投影算子：

[
P_{\perp,S,i,W}
===============

I-T_{S,i,W}T_{S,i,W}^{\top}.
]

定义第一层 multiresolution residual：

[
\boxed{
\Psi_{S,i,W}^{(16)}
===================

P_{\perp,S,i,W}
\left(
\Delta_W x_{S,16,i}
-------------------

\Delta_W \widehat x_{S,32,i}
\right).
}
]

第二层 residual：

[
\boxed{
\Psi_{S,i,W}^{(32)}
===================

P_{\perp,S,i,W}
\left(
\Delta_W x_{S,32,i}
-------------------

\Delta_W \widehat x_{S,64,i}
\right),
}
]

其中 (x_{S,32}) 和 (x_{S,64}) 都需要在同一 32-grid 或 aligned grid 上比较。

估计 order：

[
\boxed{
q_{S,i}
=======

1+
\log_2
\frac{
|\Psi_{S,i,W}^{(16)}|+\epsilon
}{
|\Psi_{S,i,W}^{(32)}|+\epsilon
}.
}
]

clamp：

[
q_{S,i}\in[q_{\min},q_{\max}].
]

window arc length：

[
\Delta S_{S,i,W}
================

\sum_{j=i}^{i+W-1}\Delta s_{S,j}.
]

Richardson factor：

[
\rho(q)=|1-2^{1-q}|.
]

defect：

[
\boxed{
D_{S,i,W}^{\perp}
=================

\frac{
|\Psi_{S,i,W}^{(16)}|
}{
\rho(q_{S,i})(\Delta S_{S,i,W}+\epsilon)^{q_{S,i}}
+\epsilon
}.
}
]

将 window defect 分配回 interval：

[
\omega_{j|i,W}
==============

\frac{
\Delta s_{S,j}
}{
\Delta S_{S,i,W}+\epsilon
},
\qquad
j=i,\ldots,i+W-1.
]

[
D_{S,j}^{\perp}
\leftarrow
\sum_{i:j\in[i,i+W-1]}
\omega_{j|i,W}D_{S,i,W}^{\perp}.
]

最后构造 density：

[
w_{S,j}
=======

\left(
(q_{S,j}-1)D_{S,j}^{\perp}
\right)^{1/q_{S,j}}.
]

[
n_{S,j}
=======

\frac{
w_{S,j}
}{
\sum_k w_{S,k}\Delta s_{S,k}+\epsilon
}.
]

pullback：

[
\boxed{
\alpha_{S,j}
============

n_{S,j}
\frac{
\Delta s_{S,j}
}{
|\Delta u_{S,j}|+\epsilon
}.
}
]

这就是每个 solver 自己的 FP clock。

---

# 7. 为什么它同时满足单步和多步？

单步法：

[
W=1.
]

此时：

[
\Psi_{S,i}^{(16)}
]

就是 interval displacement 的 coarse/fine discrepancy，基本等价于 local Richardson residual。

多步法：

[
W\geq \text{history length}.
]

此时 defect 不再强行拆成单步，而是在完整 solver history 自然作用下观察 window-level discrepancy。

这就绕开了：

[
z_i=(x_i,H_i)
]

难以 clone 的问题。因为历史状态 (H_i) 是 trajectory rollout 自然生成的，不需要手动恢复。

所以你的方案满足多步法，只要用 window displacement 而不是单点 endpoint residual。

---

# 8. 需要制定的修改 plan

下面是可以直接发给 Codex 的计划。

---

## Codex Plan: Implement Target-Solver Multi-Resolution FP-Clock

### Goal

Refactor FP-Clock calibration so every solver has its own target-solver defect oracle. Do not use Euler proxy for DPM, DEIS, EDM, STORK. For each solver (S), run its own base solver at NFE (16,32,64), record trajectories in its native coordinate (u_S), and estimate Frenet-projected multiresolution defect (\Psi_S).

Heun2 is excluded from this stage because its vendor implementation requires odd effective NFE and breaks the clean (16\to32\to64) refinement ratio.

---

## 1. Add solver-native registry

Create:

```text
src/clock/solver_registry.py
```

or extend existing adapter.

Each solver entry must provide:

```python
@dataclass
class SolverNativeSpec:
    name: str
    family: str
    native_coordinate: Literal["timestep", "sigma", "log_sigma", "lambda"]
    supports_base_trajectory_recording: bool
    recommended_window_len: int
    stochastic: bool
    exclude_from_multiresolution: bool = False
    exclusion_reason: str = ""
```

Initial specs:

```yaml
euler:
  native_coordinate: timestep_or_sigma_from_scheduler
  window_len: 1

deis:
  native_coordinate: lambda_or_timestep_from_scheduler
  window_len: 2

dpm_solver_lu:
  native_coordinate: lambda
  window_len: 2

dpm_solver_default:
  native_coordinate: lambda
  window_len: 2

dpm_solver_pp:
  native_coordinate: lambda
  window_len: 2

edm:
  native_coordinate: log_sigma
  window_len: 1

stork4_1st:
  native_coordinate: sigma
  window_len: 4

stork4_2nd:
  native_coordinate: sigma
  window_len: 4

heun2:
  exclude_from_multiresolution: true
  exclusion_reason: "vendor Heun requires odd effective NFE; excluded from 16/32/64 FP calibration"
```

---

## 2. Add trajectory recorder

Create:

```text
src/clock/trajectory_recording.py
```

Main API:

```python
@dataclass
class SolverTrajectory:
    solver: str
    nfe: int
    native_coordinate: str
    u: np.ndarray                  # shape [num_nodes]
    x: torch.Tensor | np.ndarray   # shape [num_samples, num_nodes, ...]
    seed: int
    metadata: dict
```

Function:

```python
record_base_trajectory(
    solver: str,
    nfe: int,
    model,
    scheduler,
    initial_noise,
    prompts_or_labels,
    native_coordinate: str,
    save_intermediates: bool = True,
) -> SolverTrajectory
```

Requirements:

* Same prompts / labels across 16, 32, 64.
* Same initial noise across 16, 32, 64.
* For stochastic solvers, use common random numbers or mark as `drift_only` if not supported.
* Save all intermediate states, not only final samples.
* Save native coordinate nodes (u_j).

---

## 3. Add trajectory alignment

Create:

```text
src/clock/trajectory_alignment.py
```

Functions:

```python
interpolate_trajectory_to_grid(
    trajectory: SolverTrajectory,
    target_u: np.ndarray,
    method: Literal["linear"] = "linear",
) -> np.ndarray | torch.Tensor
```

Rules:

* Interpolate in solver-native coordinate.
* If coordinate is descending, handle descending order correctly.
* Do interpolation per sample and per latent dimension.
* Use float32/float64 carefully; output should match latent dtype if possible.
* For strict mode, assert target nodes lie inside source coordinate range.

---

## 4. Add multiresolution FP defect oracle

Create:

```text
src/clock/fp_multiresolution.py
```

Main API:

```python
@dataclass(frozen=True)
class MultiresolutionFPStats:
    solver: str
    native_coordinate: str
    coarse_nfe: int
    mid_nfe: int
    fine_nfe: int
    window_len: int
    u_grid: np.ndarray
    delta_s: np.ndarray
    psi_coarse_norm: np.ndarray
    psi_mid_norm: np.ndarray
    projected_order: np.ndarray
    horizontal_defect: np.ndarray
    residual_parallel_ratio: np.ndarray
    metadata: dict
```

Implement:

```python
collect_multiresolution_fp_stats(
    traj_16: SolverTrajectory,
    traj_32: SolverTrajectory,
    traj_64: SolverTrajectory,
    window_len: int,
    q_min: float = 1.05,
    q_max: float = 6.0,
    eps: float = 1e-12,
) -> MultiresolutionFPStats
```

For each window ([u_i,u_{i+W}]):

[
\Delta_W x_{16,i}
=================

x_{16}(u_{i+W})-x_{16}(u_i).
]

[
\Delta_W \hat x_{32,i}
======================

\hat x_{32}(u_{i+W})-\hat x_{32}(u_i).
]

[
\Psi_{i,W}^{(16)}
=================

P_{\perp,i,W}
\left(
\Delta_W x_{16,i}
-----------------

\Delta_W \hat x_{32,i}
\right).
]

Similarly:

[
\Psi_{i,W}^{(32)}
=================

P_{\perp,i,W}
\left(
\Delta_W x_{32,i}
-----------------

\Delta_W \hat x_{64,i}
\right).
]

Use high-resolution trajectory to compute tangent:

[
T_{i,W}
=======

\frac{
\hat x_{64}(u_{i+W})-\hat x_{64}(u_i)
}{
|\hat x_{64}(u_{i+W})-\hat x_{64}(u_i)|+\epsilon
}.
]

Projected order:

[
q_i
===

1+
\log_2
\frac{
|\Psi_i^{(16)}|+\epsilon
}{
|\Psi_i^{(32)}|+\epsilon
}.
]

Horizontal defect:

[
D_i^\perp
=========

\frac{
|\Psi_i^{(16)}|
}{
|1-2^{1-q_i}|(\Delta S_i+\epsilon)^{q_i}+\epsilon
}.
]

Distribute window defect back to interval by arc-length weights.

---

## 5. Build FP profile from multiresolution stats

Add:

```python
build_fp_clock_profile_from_multiresolution_stats(
    physical_grid: np.ndarray,
    stats: MultiresolutionFPStats,
    target_steps: int,
    smoothing_window: int = 1,
    aggregation: str = "mean_after_pullback",
) -> FPClockArtifacts
```

Density:

[
w_i=
\left(
(q_i-1)D_i^\perp
\right)^{1/q_i}.
]

[
n_i=
\frac{
w_i
}{
\sum_j w_j\Delta s_j+\epsilon
}.
]

[
\alpha_i=
n_i
\frac{
\Delta s_i
}{
|\Delta u_i|+\epsilon
}.
]

Convert interval alpha to node alpha and call existing `build_clock_profile_from_alpha`.

---

## 6. Exporter integration

Add clock family:

```yaml
clock:
  family: FP_CLOCK_MULTIREZ
  calibration_mode: target_solver_multiresolution
  calibration_nfes: [16, 32, 64]
  calibration_solver: target
  exclude_solvers: [heun2]
  window_len: auto
  native_coordinate: auto
```

Exporter behavior:

```python
if family == "FP_CLOCK_MULTIREZ":
    spec = get_solver_native_spec(target_solver)
    if spec.exclude_from_multiresolution:
        raise or record unsupported
    record trajectories at 16,32,64
    align trajectories
    collect_multiresolution_fp_stats
    build profile
    export bundle
```

Bundle metadata:

```json
{
  "schedule_family": "FP_CLOCK_MULTIREZ",
  "method": "Target-Solver Multi-Resolution Frenet-Projected Clock",
  "calibration_nfes": [16, 32, 64],
  "calibration_solver": "target",
  "native_coordinate": "...",
  "window_len": ...,
  "heun_excluded": true,
  "formula_version": 1
}
```

---

## 7. Smoke tests

### Test A: nested synthetic curve

Use a known curve with nested 16/32/64 samples.

Assert:

```text
psi_coarse_norm finite
psi_mid_norm finite
q finite
D_perp finite
alpha positive
tau monotone
```

### Test B: straight-line nonuniform parameterization

[
x(u)=(u^3,0).
]

Expected:

```text
horizontal defect approximately zero
parallel discrepancy may be nonzero
```

### Test C: circle

[
x(s)=(\cos s,\sin s).
]

Expected horizontal defect tracks curvature uniformly.

### Test D: solver registry

Assert:

```text
euler available
dpm_solver_pp available if backend supports it
deis available
stork available
heun2 marked excluded
```

### Test E: CIFAR-10 trajectory recording

For euler and stork:

```text
record 16/32/64 trajectories
build FP multirez schedule
run tiny generation
```

---

## 8. First experiments

### CIFAR-10 PNDM

Run:

```text
solvers = [euler, deis, dpm_solver_lu, dpm_solver_default, dpm_solver_pp, stork4_1st]
exclude = [heun2]
calibration_nfes = [16,32,64]
target_nfe = 10
num_samples = 2048 first, then 5000
```

Compare:

```text
base
LEGACY_SADB
FP_CLOCK_MULTIREZ
```

Report:

```text
solver | base FID | legacy SADB FID | FP multirez FID | FP-base | FP-legacy
```

### SD/SDXL text-to-image

Run:

```text
models = [SD1.5, SDXL]
solvers = [dpm_solver_pp, sde_dpm_solver_pp]
calibration_nfes = [16,32,64]
target_nfe = 10
prompts >= 64
seeds >= 3
```

Compare:

```text
base
AYS
FP_EULER_PROXY_SIGMA  # old proxy, ablation only
FP_TARGET_MULTIREZ_DPM
```

For SDE solver, either:

```text
drift_only=true
```

or implement matched stochastic trajectory recording.

---

# 9. Important caveats

## 9.1 This estimates a window trajectory defect, not exact local LTE

The multiresolution method gives a target-solver trajectory discrepancy:

[
\Psi_{S,i,W}.
]

It is not identical to clone/restore local LTE, but it is much more practical and history-aware.

This is acceptable if the paper states it as:

[
\boxed{
\text{empirical target-solver horizontal defect from multiresolution base trajectories}.
}
]

## 9.2 Need same initial noise and prompts

Otherwise the difference between 16/32/64 is not numerical defect; it is sample mismatch.

## 9.3 Need interpolation diagnostics

For non-nested official base schedules, interpolation error must be logged:

```text
alignment_mode=interpolated
max_alignment_gap
mean_alignment_gap
```

## 9.4 SDE / stochastic solvers need common randomness

If not implemented, mark:

```text
stochastic_mode=drift_only
```

Do not claim full SDE target defect.

---
