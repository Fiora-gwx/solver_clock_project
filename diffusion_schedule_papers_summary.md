# 扩散模型采样 / 离散化日程优化：两篇论文汇总与排版笔记

> 用途：论文写作参考、Related Work 梳理、方法部分公式引用、实验表格引用。  
> 排版策略：保留两篇论文的英文题名、作者、章节结构、核心公式编号语境、图表/表格信息与原文参考文献块；正文采用中文归纳，方便直接迁移到论文笔记或综述草稿中。

---

## 0. 两篇论文速览

| 项目 | Paper A | Paper B |
|---|---|---|
| 英文题名 | **Score-Optimal Diffusion Schedules** | **Align Your Steps: Optimizing Sampling Schedules in Diffusion Models** |
| 作者 | Christopher Williams, Andrew Campbell, Arnaud Doucet, Saifuddin Syed | Amirmojtaba Sabour, Sanja Fidler, Karsten Kreis |
| 机构 | University of Oxford | NVIDIA / University of Toronto / Vector Institute |
| 会议 / 版本 | NeurIPS 2024 | ICML 2024 / arXiv:2404.14507v1 |
| 核心问题 | 如何在扩散路径上自动选择最优**离散化时间表** | 如何针对给定数据集、模型和采样器优化**采样噪声水平表** |
| 关键思想 | 用校正器需要做的“工作量”定义路径代价；最优日程使每一步局部代价近似均衡 | 用 Girsanov 定理构造真实生成 SDE 与线性化求解器 SDE 的 KL 上界（KLUB），再优化 schedule |
| 主要对象 | DDM 的 diffusion path；corrector-optimized 与 predictor-optimized cost | 以 SDE solvers 为主，也验证到 ODE solvers 的泛化 |
| 优点 | 理论几何解释清晰；可在线训练；仅用 score evaluation 的低成本版本可扩展 | 低 NFE/few-step 采样收益明显；给出了大模型可直接使用的 schedules |
| 局限 | predictor cost 需要二阶信息，较贵；理论假设 perfect score estimation | KLUB 是上界，需要 early stopping；优化需额外 Monte Carlo 与 GPU 资源 |

**共同结论：** 两篇论文都把“采样/离散化日程”从手工 heuristic 超参数提升为可优化对象。它们都认为，在低采样步数（low NFE / few-step synthesis）下，schedule 的影响尤其大；高步数时，不同 schedule 的差异会随离散化误差减小而弱化。

---

# Part I. Score-Optimal Diffusion Schedules

## 1. 元信息与一句话总结

**一句话总结：** 这篇论文把扩散模型从参考高斯分布到数据分布的采样过程看作沿 diffusion path 的运动，并用“Langevin corrector 为纠正分布偏差所做的工作量”定义局部代价，从而构造一个无需手动调参、可在线更新的 score-optimal discretisation schedule。

### 1.1 原文结构保留

```text
1 Introduction
2 The Cost of Traversing the Diffusion Path
  2.1 Predictor/Corrector Decomposition of the Diffusion Update
  2.2 The Incremental Cost of Correction
  2.3 Corrector and Predictor Optimised Cost
3 Score-Optimal Schedules
  3.1 Diffusion Schedule Path Length and Energy
  3.2 Estimation of Score-Optimal Schedules
  3.3 Choice of Velocity Scaling
  3.4 Related Work
4 Computational Experiments
  4.1 Sampling the Mollified Cantor Distribution
  4.2 Adaptive Schedule Learning for Bimodal Example
  4.3 Scalable Schedule Learning Diffusion
  4.4 Sampling Pre-Trained Models
5 Discussion
Appendix A Analysis of incremental cost
Appendix B Training Algorithms
Appendix C Experiment Details
```

---

## 2. 研究动机与问题设定

扩散模型通常先定义 forward noising process，把数据分布 $p_0$ 逐步变成接近高斯参考分布 $p_1$；采样时再模拟 reverse process。问题在于：为了数值模拟 reverse diffusion，需要选择一个离散时间表

$$
\mathcal{T}=\{t_i\}_{i=0}^{T},\quad t_0=0,\;t_T=1.
$$

传统方法常把 noising schedule 与 discretisation schedule 绑定，或者使用人工设计的 polynomial / cosine / log-linear schedule。论文指出：schedule 选择会显著影响样本质量和推理效率，尤其对复杂数据分布更敏感。

**论文要解决的问题：** 给定扩散路径 $\{p_t\}_{t\in[0,1]}$，怎样自动选择一组离散化时间点，使采样过程的总“运输/校正代价”最小？

---

## 3. 核心公式与方法排版

### 3.1 Forward diffusion 与 backward diffusion

Forward process:

$$
dX_t = f(t)X_t\,dt + g(t)dW_t,\qquad X_0\sim p_0.
$$

对应的条件分布写作：

$$
p_t(x_t)=\int p_0(x_0)p_{t|0}(x_t|x_0)dx_0,
\qquad
p_{t|0}(x_t|x_0)=\mathcal{N}(x_t;s(t)x_0,\sigma^2(t)I).
$$

Backward diffusion:

$$
dX_t=\left[f(t)X_t-g(t)^2\nabla_x\log p_t(X_t)\right]dt+g(t)d\widetilde{W}_t,
\qquad X_1\sim p_1.
$$

---

### 3.2 Predictor / Corrector decomposition

论文把 backward update 拆成两部分：

1. **Probability Flow Prediction ODE**：确定性预测，把样本向目标分布方向推。
2. **Langevin Correction SDE**：随机校正，修正 predictor 造成的残余误差。

直观上，如果 predictor 已经把 $p_t$ 精确运输到 $p_{t'}$，corrector 不需要额外工作；如果 predictor 很差，则 corrector 需要做更多工作。

---

### 3.3 Incremental cost of correction

定义一个预测分布 $F_{t,t'}^{\sharp}p_t$，并用 Stein/Fisher divergence 衡量其与目标分布 $p_{t'}$ 的 score 差异：

$$
\mathcal{L}(t,t')
=v(t')^2D\left(p_{t'}\Vert F_{t,t'}^{\sharp}p_t\right),
$$

其中

$$
D(p\Vert q)=\mathbb{E}_{X\sim q}\left[\left\|\nabla\log p(X)-\nabla\log q(X)\right\|^2\right].
$$

论文进一步区分两种代价：

$$
\mathcal{L}_c(t,t')=v(t')^2D(p_{t'}\Vert p_t),
$$

$$
\mathcal{L}_p(t,t')=v(t')^2D(p_{t'}\Vert F_{t,t'}^{\sharp}p_t).
$$

| 代价 | 含义 | 计算特点 |
|---|---|---|
| Corrector-optimised cost $\mathcal{L}_c$ | 不考虑 predictor，只衡量 $p_t$ 到 $p_{t'}$ 的分布变化 | 只需要 score evaluation，较便宜 |
| Predictor-optimised cost $\mathcal{L}_p$ | 衡量 predictor 后的残余误差 | 更精细，但可能涉及 Hessian / Jacobian trace，较贵 |

---

### 3.4 局部代价与最优 schedule

当 $\Delta t=t'-t$ 很小时，增量代价具有局部二次形式：

$$
\mathcal{L}(t,t')=\delta(t)\Delta t^2+O(\Delta t^3).
$$

由此定义 schedule generator $\phi:[0,1]\to[0,1]$，令 $t_i=\phi(i/T)$。论文引入 path energy 与 path length：

$$
E(\phi)=\int_0^1 \delta(\phi(s))\dot{\phi}(s)^2ds,
$$

$$
\Lambda=\int_0^1 \sqrt{\delta(t)}dt.
$$

最优 schedule generator 满足：

$$
\phi^*(s)=\Lambda^{-1}(\Lambda s),
\qquad
\Lambda(t)=\int_0^t\sqrt{\delta(u)}du.
$$

**解释：** 最优 schedule 等价于在由 $\delta(t)$ 定义的几何度量下，以常速穿越 diffusion path。换句话说，每个离散步长应让局部 correction cost 尽可能均衡。

---

## 4. Algorithm 排版保留

### Algorithm 1: UpdateSchedule

```text
Require:
  Schedule T = {t_i}_{i=0}^T
  Incremental costs {L(t_{i+1}, t_i)}_{i=0}^{T-1}

1. Compute cumulative length:
   \hat{\Lambda}(t_i) = \sum_{j=0}^{i-1}\sqrt{L(t_{j+1}, t_j)},    i=0,...,T

2. Compute total estimated length:
   \hat{\Lambda}=\hat{\Lambda}(t_T)

3. Interpolate inverse length map:
   \hat{\Lambda}^{-1}(.) = Interpolate({(\hat{\Lambda}(t_0),t_0),..., (\hat{\Lambda}(t_T),t_T)})

4. Update schedule:
   t_i^* = \hat{\Lambda}^{-1}(i\hat{\Lambda}/T),   i=0,...,T

5. Return:
   T^* = {t_i^*}_{i=0}^T
```

### Algorithm 2: AdaptiveScheduleTraining

```text
Require:
  Initial schedule T = {t_i}_{i=0}^T
  Learning rate gamma in (0, 1)
  Score estimate s_theta

while not converged:
  for each batch B from data:
    1. Fix T and assign theta <- argmin_theta L_training(theta, B, T)
    2. Fix s_theta and estimate incremental costs L(t_{i+1}, t_i)
    3. Assign T^* <- UpdateSchedule(T, L(t_{i+1}, t_i))
    4. Update time locations:
       t_i <- gamma * t_i^* + (1 - gamma) * t_i
```

---

## 5. 实验结果整理

### 5.1 1D / toy examples

| 实验 | 目的 | 结论 |
|---|---|---|
| Mollified Cantor distribution | 验证复杂多模态 / fractal-like 数据下的 schedule 自适应能力 | 线性 schedule 不能清晰分离 8 个模式；optimized schedule 能恢复多模态结构 |
| Bimodal Gaussian | 比较线性 schedule 与在线学习 schedule | 学习 schedule 后 likelihood 增长，score error 降低 |
| CIFAR-10 / MNIST online schedule | 验证高维图像场景可在线更新 schedule | learned schedule 能把 incremental costs 近似均衡，并更关注高频细节区域 |

### 5.2 Pre-trained image models: Table 1 FID

| Schedule | CIFAR-10 | FFHQ | AFHQv2 | ImageNet |
|---|---:|---:|---:|---:|
| Eq. (22), $\rho=3$ | 5.47 | 2.80 | **2.05** | 1.46 |
| Eq. (22), $\rho=7$ | **1.96** | 2.46 | **2.05** | **1.42** |
| LogLinear (Lu et al., 2022) | 2.05 | **2.42** | 2.06 | 1.45 |
| Convex Schedule | 22.1 | 2.43 | 2.48 | 1.64 |
| Corrector optimised | 1.99 | 2.46 | **2.05** | 1.44 |
| Predictor optimised | 1.99 | 2.48 | **2.05** | - |

**写作可用解读：** corrector-optimised 与 predictor-optimised schedule 能自动恢复接近 $\rho=7$ 的高性能 hand-tuned schedule；其中 corrector cost 更便宜，因此在图像数据上更实用。

### 5.3 CIFAR-10 不同步数下的 FID: Table 2

| # points $T$ | 10 | 20 | 30 | 50 | 100 |
|---|---:|---:|---:|---:|---:|
| CO (ours) | **2.46** | 2.02 | **2.04** | 2.06 | 2.07 |
| $\rho=3$ | 50.75 | 3.92 | 2.09 | **2.01** | **2.05** |
| $\rho=7$ | 2.70 | **2.00** | 2.06 | 2.05 | 2.07 |
| $\rho=100$ | 3.09 | 2.06 | 2.05 | 2.06 | 2.07 |

### 5.4 CIFAR-10 不同步数下的 sFID: Table 3

| # points $T$ | 10 | 20 | 30 | 50 | 100 |
|---|---:|---:|---:|---:|---:|
| CO (ours) | **3.94** | 3.78 | 3.80 | 3.81 | 3.81 |
| $\rho=3$ | 24.08 | 4.90 | 3.80 | **3.77** | 3.80 |
| $\rho=7$ | 4.02 | **3.76** | **3.78** | 3.80 | 3.81 |
| $\rho=100$ | 4.31 | 3.81 | 3.81 | 3.81 | 3.81 |

---

## 6. 局限与可引用表述

**局限：**

- $\mathcal{L}_p$ 需要估计与 predictor Jacobian / Hessian 有关的项，因此在高维图像模型上可能较贵。
- 理论推导假设 score estimation 足够好或接近 perfect score；真实模型的 score error 可能影响 schedule 的最优性。
- 论文主要展示 schedule 对 sampling quality 的影响，进一步的信息几何解释和更广泛 solver 兼容性仍可扩展。

**可放入论文 Related Work 的句式：**

> Williams et al. (2024) formulate diffusion discretisation scheduling as a path-geometry problem and define a score-based correction cost along the diffusion path. Their optimal schedule equalizes local costs and can be estimated either for pre-trained models or online during training.

---

# Part II. Align Your Steps: Optimizing Sampling Schedules in Diffusion Models

## 7. 元信息与一句话总结

**一句话总结：** AYS 将采样 schedule 优化表述为最小化真实 generative SDE 与 solver-specific 线性化 SDE 输出分布之间的 KL 上界（KLUB），并通过 Monte Carlo 估计与 zeroth-order search 找到针对数据集、模型和求解器的 optimized schedule。

### 7.1 原文结构保留

```text
1 Introduction
2 Background
3 Optimizing Sampling Schedules
  3.1 The Need for Optimized Schedules
  3.2 Analyzing the Discretization Errors
  3.3 Practical Considerations of KLUB Estimation
4 Related Work
5 Experiments
  5.1 Toy Experiments
  5.2 CIFAR10, FFHQ, ImageNet
  5.3 Text-to-Image
  5.4 Video Generation Models
6 Conclusions and Future Work
Appendix A Theoretical Details
Appendix B Experiment Details
Appendix C Additional Results
```

---

## 8. 研究动机与问题设定

扩散模型采样可以看作在 $[t_{\min}, t_{\max}]$ 上求解 reverse-time SDE/ODE。数值求解时需要选择

$$
t_{\min}=t_0<t_1<\cdots<t_n=t_{\max},
$$

这组 noise levels / timesteps 被称为 sampling schedule。过去工作主要优化 solver 本身，而 schedule 多使用 EDM、LogSNR、time-uniform、cosine 等 hand-crafted heuristics。

AYS 的核心主张是：**schedule 本身是低维但影响很大的优化对象；在 few-step sampling 中，优化 schedule 往往能用相同 NFE 获得更高质量输出。**

---

## 9. 核心公式与方法排版

### 9.1 Forward process 与 reverse SDE/ODE

采用 Karras et al. (2022) 的记号，数据加噪分布为 $p(x;\sigma)$，noising schedule 由 $s(t)$ 与 $\sigma(t)$ 给出。Forward SDE:

$$
dx_t = \frac{\dot{s}(t)}{s(t)}x_t\,dt + s(t)\sqrt{2\sigma(t)\dot{\sigma}(t)}\,dw_t.
$$

Reverse-time diffusion process:

$$
dx_t = \left[\frac{\dot{s}(t)}{s(t)}x_t - 2s(t)^2\sigma(t)\dot{\sigma}(t)\nabla_x\log p\left(\frac{x_t}{s(t)},\sigma(t)\right)\right]dt
+s(t)\sqrt{2\sigma(t)\dot{\sigma}(t)}d\bar{w}_t.
$$

Probability Flow ODE:

$$
dx_t = \left[\frac{\dot{s}(t)}{s(t)}x_t - s(t)^2\sigma(t)\dot{\sigma}(t)\nabla_x\log p\left(\frac{x_t}{s(t)},\sigma(t)\right)\right]dt.
$$

---

### 9.2 Gaussian toy example: 最优 schedule 依赖数据分布

当 $p_{data}(x)=\mathcal{N}(0,c^2I)$，且 $s(t)=1,\sigma(t)=t$ 时，若使用 forward Euler / DDIM 求解 probability flow ODE，最优 schedule 满足：

$$
\alpha_{\min}=\arctan(t_{\min}/c),\qquad
\alpha_{\max}=\arctan(t_{\max}/c),
$$

$$
t_i^*=c\tan\left((1-i/n)\alpha_{\min}+(i/n)\alpha_{\max}\right).
$$

**写作要点：** 数据方差 $c$ 改变会显著改变最优 schedule，因此固定 heuristic schedule 难以适配所有数据集。

---

### 9.3 KLUB: 用 Girsanov 定理分析离散化误差

若两个 SDE 共享 diffusion term：

$$
\begin{cases}
dx_t=f_1(x_{0\to t},t)dt+g(t)dw_t,\\
dx_t=f_2(x_{0\to t},t)dt+g(t)dw_t,
\end{cases}
$$

则输出分布 $P_1,P_2$ 的 KL divergence 有上界：

$$
D_{KL}(P_1\Vert P_2)
\le
\frac{1}{2}\mathbb{E}_{P_1^{paths}}\left[\int_0^T
\frac{\|f_1(x_{0\to t},t)-f_2(x_{0\to t},t)\|^2}{g(t)^2}dt\right].
$$

AYS 将真实 learnt SDE 与求解器对应的线性化 SDE 代入该上界，得到 KLUB，并优化：

$$
(t_1^*,\ldots,t_{n-1}^*)
=\arg\min_{t_1,\ldots,t_{n-1}}
\sum_{i=1}^n KLUB(t_{i-1},t_i).
$$

---

### 9.4 Stochastic-DDIM 对应的 KLUB 形式

在 $D_\theta(x,\sigma)$ 为 denoiser 的情况下，AYS 对 Stochastic-DDIM 得到：

$$
KLUB(t_0,t_1,\ldots,t_n)
=
\sum_{i=1}^n\int_{t_{i-1}}^{t_i}
\frac{s(t)^2\dot{\sigma}(t)}{\sigma(t)^3}
\mathbb{E}_{x_t\sim p_t',\;x_{t_i}\sim p_{t_i|t}'}
\left\|
D_\theta\left(\frac{x_t}{s(t)},\sigma(t)\right)-
D_\theta\left(\frac{x_{t_i}}{s(t_i)},\sigma(t_i)\right)
\right\|^2dt.
$$

实际估计 KLUB 时，论文使用基于 Gaussian data assumption 的 time-importance sampling 来降低 Monte Carlo 方差。

---

## 10. Algorithm 排版保留

### Algorithm 1: KLUB optimization with $\sigma(t)=t$ and $s(t)=1$

```text
Input:
  denoiser D_theta(x, sigma)
  schedule t_i, i in {0,1,...,n}

repeat:
  noChange <- True
  for i = 1 to n-1:
    candidates[0,...,r-1] <- neighbourhood around t_i
    for j = 0 to r-1:
      KLUB[j] <- EstimateKLUB(D_theta, {t_{i-1}, candidates_j, t_{i+1}})
    minIdx <- argmin KLUB[0,...,r-1]
    if candidate_minIdx != t_i:
      t_i <- candidate_minIdx
      noChange <- False
until noChange
```

### Algorithm 2: Monte Carlo estimation of KLUB with $\sigma(t)=t$ and $s(t)=1$

```text
Input:
  denoiser D_theta(x, sigma)
  interval points t_min, t_mid, t_max
  Monte Carlo samples n

for i = 1 to n:
  sample x_0 ~ p_data(x)
  t <- ImportanceSample(pi, t_min, t_mid, t_max)
  t_upper <- (t < t_mid) ? t_mid : t_max
  x_t <- x_0 + t * N(0, I)
  x_tupper <- x_t + sqrt(t_upper^2 - t^2) * N(0, I)
  KLUB[i] <- ||D_theta(x_t, t) - D_theta(x_tupper, t_upper)||^2
             / (1/(t^2+c^2) - 1/(t_upper^2+c^2))
return mean(KLUB[0,...,n-1])
```

---

## 11. 实验结果整理

### 11.1 实验范围

| 数据 / 模型 | 评价内容 | 主要结论 |
|---|---|---|
| 2D toy data | 负对数似然 / outliers | AYS 样本更接近真实分布、离群点更少 |
| CIFAR10 / FFHQ / ImageNet | FID, sFID, IS | 多个 stochastic 与 deterministic solver 上 AYS 优于 baseline |
| Stable Diffusion 1.5 / SDXL / DeepFloyd-IF | 人评、FID-CLIP Pareto、定性图 | 低 NFE 下细节和文本对齐更好 |
| Stable Video Diffusion | 人评与稳定性 | AYS 改善时序稳定性，减少颜色/物体伪影 |

### 11.2 Large-scale optimized schedules: Table 3

> 表中数值是 noise levels，顺序为 $\sigma(t_n),\sigma(t_{n-1}),...,\sigma(t_0)$。

| Model | Optimized schedule |
|---|---|
| Stable Diffusion 1.5 | [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029] |
| SDXL | [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029] |
| DeepFloyd-IF / Stage 1 | [160.41, 8.081, 3.315, 1.885, 1.207, 0.785, 0.553, 0.293, 0.186, 0.030, 0.006] |
| Stable Video Diffusion | [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002] |

### 11.3 CIFAR10 FID: Table 5

| Sampling method | Schedule | NFE=10 | NFE=20 | NFE=30 | NFE=50 |
|---|---|---:|---:|---:|---:|
| Stochastic DDIM | EDM | 51.45 | 23.67 | 14.19 | 7.75 |
| Stochastic DDIM | AYS | **33.52** | **14.16** | **8.78** | **5.45** |
| SDE-DPM-Solver++ (2M) | EDM | 15.32 | 4.64 | 3.15 | 2.64 |
| SDE-DPM-Solver++ (2M) | AYS | **8.16** | **3.23** | **2.55** | **2.40** |
| ER-SDE-Solver 3 | EDM | 9.47 | 3.15 | 2.39 | 2.13 |
| ER-SDE-Solver 3 | AYS | **7.55** | **3.07** | **2.36** | 2.13 |
| DDIM | LogSNR | 16.44 | 6.01 | 3.97 | 2.82 |
| DDIM | AYS | **10.73** | **4.67** | **3.30** | **2.56** |
| DPM-Solver++ (2M) | LogSNR | 5.07 | 2.37 | 2.12 | 2.04 |
| DPM-Solver++ (2M) | AYS | **2.98** | **2.10** | **2.02** | **2.01** |

### 11.4 FFHQ FID: Table 6

| Sampling method | Schedule | NFE=10 | NFE=20 | NFE=30 | NFE=50 |
|---|---|---:|---:|---:|---:|
| Stochastic DDIM | EDM | 53.83 | 31.97 | 22.14 | 13.42 |
| Stochastic DDIM | AYS | **42.03** | **22.73** | **14.90** | **9.135** |
| SDE-DPM-Solver++ (2M) | EDM | 23.04 | 9.67 | 5.96 | 3.85 |
| SDE-DPM-Solver++ (2M) | AYS | **14.79** | **5.65** | **3.97** | **3.13** |
| ER-SDE-Solver 3 | EDM | 11.97 | 4.18 | 3.06 | **2.61** |
| ER-SDE-Solver 3 | AYS | **8.71** | **3.92** | **2.97** | 2.65 |
| DDIM | EDM | 18.37 | 8.19 | 5.60 | 3.96 |
| DDIM | AYS | **12.83** | **6.05** | **4.41** | **3.38** |
| DPM-Solver++ (2M) | LogSNR | 7.07 | 3.41 | 2.87 | 2.62 |
| DPM-Solver++ (2M) | AYS | **5.43** | **3.29** | 2.87 | 2.62 |

### 11.5 Video generation user study: Table 2

| Model | EDM | AYS |
|---|---:|---:|
| Stable Video Diffusion | 42% | **58%** |

---

## 12. 局限与可引用表述

**局限：**

- KLUB 是上界，不等价于真实误差；论文指出 early stopping 是必要的，否则可能过度优化上界。
- Schedule optimization 需要额外 Monte Carlo 估计和多次 denoiser forward pass。
- AYS 主要从 stochastic SDE solver 出发推导，虽然实验上也能泛化到 ODE solver，但理论上仍以 SDE solver 更直接。
- guidance scale 会改变 score model，因此严格意义上不同 guidance value 可能对应不同 optimal schedule。

**可放入论文 Related Work 的句式：**

> Sabour et al. (2024) propose Align Your Steps, a principled framework for optimizing diffusion sampling schedules by minimizing a Girsanov-based KL upper bound between the true generative SDE and the solver-specific linearized SDE. Their optimized schedules consistently improve few-step generation quality across image, text-to-image, and video diffusion models.

---

# Part III. 两篇论文的关系与写作参考

## 13. 方法对比

| 维度 | Score-Optimal Diffusion Schedules | Align Your Steps |
|---|---|---|
| 理论出发点 | predictor-corrector dynamics 与 Stein/Fisher divergence | Girsanov theorem 与 KL divergence upper bound |
| 优化对象 | diffusion path 上的 discretisation schedule | solver-specific sampling schedule / noise levels |
| 代价定义 | correction work / score discrepancy | true SDE 与 discretized solver SDE 的 mismatch |
| 是否 solver-specific | 相对更 path-intrinsic，可用于不同 sampler | 明确依赖 solver 的线性化 SDE；但实验显示可迁移 |
| 是否需要二阶信息 | $L_p$ 需要，$L_c$ 不需要 | 通常不需要二阶梯度，但需要 Monte Carlo 估计 KLUB |
| 在线训练 | 支持 online schedule learning | 主要针对训练后 schedule optimization |
| 实验重点 | 1D complex distributions、online training、pretrained image models | toy data、CIFAR/FFHQ/ImageNet、T2I、video |
| 写作定位 | “geometric / score-based schedule optimization” | “KLUB / solver-aware schedule optimization” |

---

## 14. Related Work 可直接改写的中文段落

### 14.1 研究空白段落

扩散模型采样的效率通常受到离散化误差和网络函数评估次数的共同限制。已有工作大量关注更高阶或更稳定的 SDE/ODE solver，例如 DDIM、DPM-Solver、DPM-Solver++、EDM sampler 等，但 sampling schedule 本身长期依赖人工设计的启发式函数。近期工作开始表明，schedule 不只是附属超参数，而是影响 few-step generation 质量的关键因素。

### 14.2 两篇论文并列引用段落

Williams et al. (2024) 从扩散路径几何出发，用 corrector 所需的 score-based correction work 定义局部代价，并推导出使路径长度均匀化的 score-optimal schedule。Sabour et al. (2024) 则从数值 SDE 求解误差出发，通过 Girsanov 定理构造 KL divergence upper bound，并针对给定模型、数据集和 solver 优化 schedule。二者都说明，优化 sampling / discretisation schedule 能在不改变模型权重的情况下提升低步数采样质量。

### 14.3 方法差异段落

两篇工作的主要差异在于代价函数的来源。Score-Optimal Diffusion Schedules 更强调扩散路径自身的几何结构，其代价由相邻分布之间的 score discrepancy 决定；Align Your Steps 更强调具体 solver 的离散化误差，其目标函数由真实 generative SDE 与 solver-induced linearized SDE 的 KL upper bound 给出。因此，前者更适合描述为 path-intrinsic / score-geometric scheduling，后者更适合描述为 solver-aware / KLUB-based scheduling。

### 14.4 实验结论段落

在实验上，两篇论文都观察到 schedule optimization 在 low NFE regime 中最有效。Williams et al. (2024) 的方法能自动恢复 Karras et al. (2022) 中通过手工搜索得到的高性能 polynomial schedule，并在 CIFAR-10 等数据集上获得竞争性 FID；Sabour et al. (2024) 的 AYS schedule 则在 CIFAR10、FFHQ、ImageNet、Stable Diffusion、SDXL 和 Stable Video Diffusion 等场景中普遍优于 EDM、LogSNR 或 time-uniform baseline。

---

## 15. 写论文时推荐引用的关键文献关系

| 引用位置 | 推荐引用 | 用法 |
|---|---|---|
| 扩散模型基础 | Sohl-Dickstein et al. (2015); Ho et al. (2020); Song et al. (2021) | 介绍 diffusion / score-based generative modeling |
| 经典采样与 noising/sampling design | Song et al. (2020a); Karras et al. (2022); Lu et al. (2022a,b) | 介绍 DDIM、EDM、DPM-Solver 系列 |
| Schedule optimization | Watson et al. (2021, 2022); Sabour et al. (2024); Williams et al. (2024) | 综述 schedule 从 heuristic 到 learnable/optimizable 的发展 |
| Few-step acceleration | Salimans & Ho (2022); Song et al. (2023); Sauer et al. (2023); Luo et al. (2023); Yin et al. (2023) | 与 distillation / consistency / one-step 方法对比 |
| 图像与视频大模型 | Rombach et al. (2021); Podell et al. (2023); Blattmann et al. (2023a,b) | 作为实验模型或应用背景 |

---

# Part IV. 原文 References 保留区

> 说明：以下参考文献块按 PDF 提取顺序保留，用于论文写作时核对 author-year citation 与 bibliography。由于 PDF 原文为双栏排版，少数词可能保留了换行断词；正式投稿前建议按目标会议模板或 BibTeX 再统一格式。

## 16. References from **Score-Optimal Diffusion Schedules**

```text
Arbel, M., Matthews, A., and Doucet, A. (2021).
Annealed flow transport Monte Carlo.
In
International Conference on Machine Learning.
Cantor, G. (1884). De la puissance des ensembles parfaits de points: Extrait d’une lettre adressée à
l’éditeur. Acta Mathematica, 4:381–392. Reprinted in: E. Zermelo (Ed.), Gesammelte Abhandlun-
gen Mathematischen und Philosophischen Inhalts, Springer, New York, 1980.
Choi, Y., Uh, Y., Yoo, J., and Ha, J.-W. (2020). Stargan v2: Diverse image synthesis for multiple
domains. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
Das, A., Fotiadis, S., Batra, A., Nabiei, F., Liao, F., Vakili, S., Shiu, D.-s., and Bernacchia, A. (2023).
Image generation with shortest path diffusion. In International Conference on Machine Learning.
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). Imagenet: A large-scale
hierarchical image database. IEEE Conference on Computer Vision and Pattern Recognition.
Fritsch, F. N. and Carlson, R. E. (1980). Monotone piecewise cubic interpolation. SIAM Journal on
Numerical Analysis, 17(2):238–246.
Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. In Advances in
Neural Information Processing Systems.
Hutchinson, M. (1989). A stochastic estimator of the trace of the influence matrix for Laplacian
smoothing splines. Communications in Statistics - Simulation and Computation, 18(3):1059–1076.
Johnson, O. (2004). Information Theory and the Central Limit Theorem. World Scientific.
Karras, T., Aittala, M., Aila, T., and Laine, S. (2022). Elucidating the design space of diffusion-based
generative models. In Advances in Neural Information Processing Systems.
Karras, T., Laine, S., and Aila, T. (2018). A style-based generator architecture for generative
adversarial networks. arxiv e-prints. In Conference on Computer Vision and Pattern Recognition
(CVPR).
Kingma, D., Salimans, T., Poole, B., and Ho, J. (2021). Variational diffusion models. In Advances in
Neural Information Processing Systems.
Krizhevsky, A., Hinton, G., et al. (2009). Learning multiple layers of features from tiny images.
Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. (2022). DPM-solver: A fast ODE solver for
diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information
Processing Systems.
Nichol, A. Q. and Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In
International Conference on Machine Learning.
Poincaré, H. (1890). Sur les équations aux dérivées partielles de la physique mathématique. American
Journal of Mathematics, pages 211–294.
Sabour, A., Fidler, S., and Kreis, K. (2024). Align your steps: Optimizing sampling schedules in
diffusion models. In International Conference on Machine Learning.
Santos, J. E., Fox, Z. R., Lubbers, N., and Lin, Y. T. (2023). Blackout diffusion: generative diffusion
models in discrete-state spaces. In International Conference on Machine Learning.
Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. (2015). Deep unsupervised
learning using nonequilibrium thermodynamics. In International Conference on Machine Learning.
Song, Y. and Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution.
In Advances in Neural Information Processing Systems.
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). Score-based
generative modeling through stochastic differential equations. In International Conference on
Learning Representations.
Syed, S., Bouchard-Côté, A., Deligiannidis, G., and Doucet, A. (2021). Non-reversible parallel
tempering: a scalable highly parallel MCMC scheme. Journal of the Royal Statistical Society
(Series B), 84:321–350.
Watson, D., Ho, J., Norouzi, M., and Chan, W. (2021). Learning to efficiently sample from diffusion
probabilistic models. arXiv preprint arXiv:2106.03802.
Xue, S., Liu, Z., Chen, F., Zhang, S., Hu, T., Xie, E., and Li, Z. (2024). Accelerating diffusion
sampling with optimized time steps. arXiv preprint arXiv:2402.17376.
```

---

## 17. References from **Align Your Steps: Optimizing Sampling Schedules in Diffusion Models**

```text
Deepfloyd.
if.
https://github.com/
deep-floyd/IF, 2023.
Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E.
Stochastic interpolants: A unifying framework for flows
and diffusions. arXiv preprint arXiv:2303.08797, 2023.
Atkinson, K., Han, W., and Stewart, D. E. Numerical Solu-
tion of Ordinary Differential Equations. John Wiley &
Sons, Ltd, 2009.
Bain, M., Nagrani, A., Varol, G., and Zisserman, A. Frozen
in time: A joint video and image encoder for end-to-end
retrieval. In IEEE International Conference on Computer
Vision, 2021.
Binkowski, M., Sutherland, D. J., Arbel, M., and Gretton, A.
Demystifying MMD GANs. In International Conference
on Learning Representations (ICLR), 2018.
Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D.,
Kilian, M., and Lorenz, D. Stable video diffusion: Scaling
latent video diffusion models to large datasets. ArXiv,
abs/2311.15127, 2023a.
Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim,
S. W., Fidler, S., and Kreis, K. Align your latents: High-
resolution video synthesis with latent diffusion models.
In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2023b.
Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix:
Learning to follow image editing instructions. In CVPR,
2023.
Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., and Zhang,
A. R. Sampling is as easy as learning the score: theory for
diffusion models with minimal data assumptions. ArXiv,
abs/2209.11215, 2022.
Cui, Q., Zhang, X., Lu, Z., and Liao, Q. Elucidating the
solution space of extended reverse-time sde for diffusion
models. ArXiv, abs/2309.06169, 2023.
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,
L. Imagenet: A large-scale hierarchical image database.
In 2009 IEEE conference on computer vision and pattern
recognition, pp. 248–255. Ieee, 2009.
Dhariwal, P. and Nichol, A. Diffusion models beat gans on
image synthesis, 2021.
Dockhorn, T., Vahdat, A., and Kreis, K. GENIE: Higher-
Order Denoising Diffusion Solvers. In Advances in Neu-
ral Information Processing Systems (NeurIPS), 2022.
Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller,
J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F.,
Podell, D., Dockhorn, T., English, Z., Lacey, K., Good-
win, A., Marek, Y., and Rombach, R. Scaling rectified
flow transformers for high-resolution image synthesis.
arXiv preprint arXiv:2403.03206, 2024.
Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and
Hochreiter, S. Gans trained by a two time-scale update
rule converge to a local nash equilibrium, 2017.
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion proba-
bilistic models. ArXiv, abs/2006.11239, 2020.
Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi,
M.,
and Fleet,
D. J.
Video diffusion models.
arXiv:2204.03458, 2022.
Hoogeboom, E., Heek, J., and Salimans, T. simple diffu-
sion: End-to-end diffusion for high resolution images. In
International Conference on Machine Learning, 2023.
Huang, C.-W., Lim, J. H., and Courville, A. C. A variational
perspective on diffusion-based generative models and
score matching. ArXiv, abs/2106.02808, 2021.
Janner, M., Du, Y., Tenenbaum, J., and Levine, S. Plan-
ning with diffusion for flexible behavior synthesis. In
International Conference on Machine Learning, 2022.
Jolicoeur-Martineau, A., Li, K., Piche-Taillefer, R., Kach-
man, T., and Mitliagkas, I. Gotta go fast when generating
data with score-based models. ArXiv, abs/2105.14080,
2021.
Karras, T., Laine, S., and Aila, T. A style-based generator
architecture for generative adversarial networks. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pp. 4401–4410, 2019.
Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating
the design space of diffusion-based generative models.
ArXiv, abs/2206.00364, 2022.
Kim, D., Lai, C.-H., Liao, W.-H., Murata, N., Takida, Y.,
Uesaka, T., He, Y., Mitsufuji, Y., and Ermon, S. Consis-
tency trajectory models: Learning probability flow ode
trajectory of diffusion. arXiv preprint arXiv:2310.02279,
2023.
Krizhevsky, A., Hinton, G., et al. Learning multiple layers
of features from tiny images. 2009.
Lin, C.-H., Gao, J., Tang, L., Takikawa, T., Zeng, X.,
Huang, X., Kreis, K., Fidler, S., Liu, M.-Y., and Lin, T.-
Y. Magic3d: High-resolution text-to-3d content creation.
In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2023.
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ra-
manan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco:
Common objects in context. In Computer Vision–ECCV
2014: 13th European Conference, Zurich, Switzerland,
September 6-12, 2014, Proceedings, Part V 13, pp. 740–
755. Springer, 2014.
Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and
Le, M. Flow matching for generative modeling. arXiv
preprint arXiv:2210.02747, 2022.
Liu, L., Ren, Y., Lin, Z., and Zhao, Z. Pseudo numerical
methods for diffusion models on manifolds. In Interna-
tional Conference on Learning Representations (ICLR),
2022.
Liu, X., Zhang, X., Ma, J., Peng, J., and Liu, Q. Instaflow:
One step is enough for high-quality diffusion-based text-
to-image generation. arXiv preprint arXiv:2309.06380,
2023.
Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-
solver: A fast ode solver for diffusion probabilistic model
sampling in around 10 steps. ArXiv, abs/2206.00927,
2022a.
Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-
solver++: Fast solver for guided sampling of diffusion
probabilistic models. ArXiv, abs/2211.01095, 2022b.
Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timo-
fte, R., and Gool, L. V. Repaint: Inpainting using de-
noising diffusion probabilistic models. 2022 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 11451–11461, 2022.
Luo, S., Tan, Y., Huang, L., Li, J., and Zhao, H. Latent
consistency models: Synthesizing high-resolution images
with few-step inference. ArXiv, abs/2310.04378, 2023.
Lyu, Z., Xudong, X., Yang, C., Lin, D., and Dai, B. Accel-
erating diffusion models via early stop of the diffusion
process. ArXiv, abs/2205.12524, 2022.
Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., Vanden-
Eijnden, E., and Xie, S. Sit: Exploring flow and diffusion-
based generative models with scalable interpolant trans-
formers. arXiv preprint arXiv:2401.08740, 2024.
Meng, C., Gao, R., Kingma, D. P., Ermon, S., Ho, J., and
Salimans, T. On distillation of guided diffusion models.
2023 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pp. 14297–14306, 2022.
Mirsky, Y. and Lee, W. The Creation and Detection of
Deepfakes: A Survey. ACM Comput. Surv., 54(1), 2021.
Nguyen, T. T., Nguyen, Q. V. H., Nguyen, C. M., Nguyen,
D., Nguyen, D. T., and Nahavandi, S.
Deep Learn-
ing for Deepfakes Creation and Detection: A Survey.
arXiv:1909.11573, 2021.
Nichol, A. and Dhariwal, P. Improved denoising diffusion
probabilistic models. ArXiv, abs/2102.09672, 2021.
Oksendal, B. Stochastic Differential Equations (3rd Ed.):
An Introduction with Applications.
Springer-Verlag,
Berlin, Heidelberg, 1992. ISBN 3387533354.
Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn,
T., Muller, J., Penna, J., and Rombach, R. Sdxl: Im-
proving latent diffusion models for high-resolution image
synthesis. ArXiv, abs/2307.01952, 2023.
Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dream-
fusion: Text-to-3d using 2d diffusion. arXiv, 2022.
Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen,
M. Hierarchical text-conditional image generation with
clip latents. ArXiv, abs/2204.06125, 2022.
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and
Ommer, B.
High-resolution image synthesis with la-
tent diffusion models. 2022 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp.
10674–10685, 2021.
Saharia, C., Chan, W., Chang, H., Lee, C. A., Ho, J., Sali-
mans, T., Fleet, D. J., and Norouzi, M. Palette: Image-to-
image diffusion models. ACM SIGGRAPH 2022 Confer-
ence Proceedings, 2021a.
Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D. J.,
and Norouzi, M. Image super-resolution via iterative
refinement. arXiv:2104.07636, 2021b.
Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Den-
ton, E. L., Ghasemipour, S. K. S., Ayan, B. K., Mah-
davi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet,
D. J., and Norouzi, M. Photorealistic text-to-image dif-
fusion models with deep language understanding. ArXiv,
abs/2205.11487, 2022.
Salimans, T. and Ho, J. Progressive distillation for fast
sampling of diffusion models, 2022.
Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R.
Adversarial diffusion distillation. ArXiv, abs/2311.17042,
2023a.
Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R.
Adversarial diffusion distillation. ArXiv, abs/2311.17042,
2023b.
Song, J., Meng, C., and Ermon, S. Denoising diffusion
implicit models. ArXiv, abs/2010.02502, 2020a.
Song, Y., Sohl-Dickstein, J. N., Kingma, D. P., Kumar,
A., Ermon, S., and Poole, B. Score-based generative
modeling through stochastic differential equations. ArXiv,
abs/2011.13456, 2020b.
Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consis-
tency models, 2023.
Vaccari, C. and Chadwick, A. Deepfakes and Disinforma-
tion: Exploring the Impact of Synthetic Political Video
on Deception, Uncertainty, and Trust in News. Social
Media + Society, 6(1):2056305120903408, 2020.
Vahdat, A., Kreis, K., and Kautz, J. Score-based gener-
ative modeling in latent space. In Neural Information
Processing Systems (NeurIPS), 2021.
Wang, Y., Wang, X., Dinh, A.-D., Du, B., and Xu, C. Learn-
ing to schedule in diffusion probabilistic models.
In
Proceedings of the 29th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, pp. 2478–2488,
2023.
Watson, D., Ho, J., Norouzi, M., and Chan, W. Learning
to efficiently sample from diffusion probabilistic models.
ArXiv, abs/2106.03802, 2021.
Watson, D., Chan, W., Ho, J., and Norouzi, M. Learning fast
samplers for diffusion models by differentiating through
sample quality. ArXiv, abs/2202.05830, 2022.
Xia, M., Shen, Y., Lei, C., Zhou, Y., Yi, R., Zhao, D.,
Wang, W., and Liu, Y.-j. Towards more accurate diffusion
model acceleration with a timestep aligner. arXiv preprint
arXiv:2310.09469, 2023.
Xiao, Z., Kreis, K., and Vahdat, A. Tackling the generative
learning trilemma with denoising diffusion GANs. In
International Conference on Learning Representations
(ICLR), 2022.
Xu, Y., Deng, M., Cheng, X., Tian, Y., Liu, Z., and Jaakkola,
T. Restart sampling for improving generative processes.
ArXiv, abs/2306.14878, 2023a.
Xu, Y., Zhao, Y., Xiao, Z., and Hou, T. Ufogen: You forward
once large scale text-to-image generation via diffusion
gans. ArXiv, abs/2311.09257, 2023b.
Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F.,
Freeman, W. T., and Park, T. One-step diffusion with
distribution matching distillation. ArXiv, abs/2311.18828,
2023.
Zhang, Q. and Chen, Y. Fast sampling of diffusion models
with exponential integrator. ArXiv, abs/2204.13902, 2022.
Zhao, W., Bai, L., Rao, Y., Zhou, J., and Lu, J. Unipc: A
unified predictor-corrector framework for fast sampling
of diffusion models. arXiv preprint arXiv:2302.04867,
2023.
Zheng, H., He, P., Chen, W., and Zhou, M. Truncated
diffusion probabilistic models. ArXiv, abs/2202.09671,
2022a.
Zheng, H., Nie, W., Vahdat, A., Azizzadenesheli, K., and
Anandkumar, A. Fast sampling of diffusion models via
operator learning. In International Conference on Ma-
chine Learning, 2022b.
Zheng, K., Lu, C., Chen, J., and Zhu, J. Dpm-solver-v3: Im-
proved diffusion ode solver with empirical model statis-
tics. Advances in Neural Information Processing Systems,
36, 2024.
```
