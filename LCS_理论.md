# 自校准缺陷均衡时间重参数化：第三方代码库复现指南

下面给出一个完整的复现流程。目标是把任意第三方 diffusion / flow sampler 改造成：

[
\textbf{training-free} + \textbf{solver-aware} + \textbf{order-free} + \textbf{parameterization-agnostic}
]

的时间重参数方法。

核心思想是：**不要手动判断 solver 是几阶，也不要手动决定应该看 (D_t v)、(D_\sigma v)、还是 (D_\lambda D_\theta)。直接通过 solver 自身的 step-refinement 行为估计它的局部误差缩放和缺陷强度，然后用这个缺陷强度生成新的时间表。**

---

# 0. 方法总览

给定一个第三方采样器，它通常包含：

1. 模型 `model(x, cond)`；
2. 时间变量，例如 `t`、`sigma`、`lambda`；
3. 一组采样节点，例如 `timesteps` 或 `sigmas`；
4. 一个 solver，例如 Euler、Heun、DPM-Solver、UniPC、STORK、EDM Euler、EDM Heun 等。

我们的目标不是改模型，也不是训练新网络，而是替换原始时间表：

[
u_0,u_1,\dots,u_N
]

为新的时间表：

[
\tilde u_0,\tilde u_1,\dots,\tilde u_N,
]

其中 (u) 是第三方代码库原生使用的采样变量，可以是：

[
u=t,\qquad u=\sigma,\qquad u=\lambda,\qquad u=\log\sigma,\qquad u=\log\mathrm{SNR}.
]

核心流程如下：

```text
第三方 sampler
    ↓
识别 native domain：t / sigma / lambda / logSNR
    ↓
封装 solver step operator ΨS(x, u_start, u_end)
    ↓
用少量 calibration samples 做 step-refinement
    ↓
估计局部有效阶数 q(u) 和缺陷强度 C(u)
    ↓
构造 defect-balanced density m(u)
    ↓
反解 CDF 得到新 schedule
    ↓
用新 schedule 运行原始 sampler
```

---

# 1. 第一步：识别第三方代码库的基本结构

进入一个第三方代码库时，先不要急着改 schedule。先回答下面几个问题。

---

## 1.1 模型输出是什么？

常见模型输出包括：

| 输出类型                | 代码里常见名字                          | 含义                                |
| ------------------- | -------------------------------- | --------------------------------- |
| noise prediction    | `eps`, `epsilon`, `noise_pred`   | 预测噪声 (\epsilon_\theta(x,u))       |
| data prediction     | `x0`, `pred_x0`, `denoised`, `D` | 预测干净样本 (x_{0,\theta}(x,u))        |
| score prediction    | `score`                          | 预测 (\nabla_x \log p_u(x))         |
| velocity prediction | `v`, `v_pred`, `velocity`        | 预测 DDPM / Imagen 风格 velocity      |
| flow velocity       | `u`, `dxdt`, `velocity`          | 直接预测 ODE velocity (\frac{dx}{dt}) |
| EDM denoiser        | `denoised`, `D(x, sigma)`        | 通常等价于 (x_0) 预测                    |

你需要先确认模型输出，否则很容易把 sampler 的 native derivative 搞错。

---

## 1.2 采样器在哪个 domain 里积分？

常见情况：

| 采样 domain       | 代码里常见变量               | 典型代码                           |
| --------------- | --------------------- | ------------------------------ |
| time domain     | `t`, `timesteps`      | DDPM / DDIM / VP sampler       |
| sigma domain    | `sigma`, `sigmas`     | EDM / k-diffusion              |
| log-SNR domain  | `lambda`, `logsnr`    | DPM-Solver 系列                  |
| normalized time | `t in [0,1]`          | Rectified Flow / Flow Matching |
| discrete index  | `i`, `timestep index` | 一些 diffusers scheduler         |

我们的建议：

[
\boxed{
\text{优先在第三方代码库的 native domain 里做 calibration。}
}
]

也就是说：

* 如果原代码用 `sigmas`，就把 (u) 设为 (\sigma)；
* 如果原代码用 `timesteps`，就把 (u) 设为 (t)；
* 如果原代码用 `lambda`，就把 (u) 设为 (\lambda)；
* 如果原代码用 `logSNR`，就把 (u) 设为 log-SNR。

这样做最稳，因为 solver 的阶数和误差结构是相对于它自己的积分变量定义的。

---

## 1.3 采样方向是什么？

很多 diffusion sampler 是反向积分：

[
\sigma_{\max}\to \sigma_{\min},
]

或者

[
t_{\max}\to t_{\min}.
]

因此 step size 可能是负的：

[
h = u_{\mathrm{end}} - u_{\mathrm{start}} < 0.
]

在估计阶数时，只使用

[
|h|.
]

在调用 solver 时，必须保留原始方向：

[
\Psi_S(x,u,u+h).
]

---

# 2. 第二步：统一模型输出为 native derivative

严格来说，我们的方法可以完全黑箱地使用 solver step，不一定需要显式写出 velocity。但是为了在第三方代码库中定位 sampler，你需要知道它内部到底在算什么。

---

## 2.1 通用线性噪声路径

很多 diffusion / flow 可以写成：

[
x_u = \alpha(u)x_0 + \sigma(u)\epsilon,
]

其中

[
\epsilon\sim \mathcal N(0,I).
]

如果代码里有 (\alpha(u))、(\sigma(u))，那么不同输出之间可以互相转换。

---

## 2.2 noise prediction

模型输出：

[
\epsilon_\theta(x,u).
]

则

[
\hat x_0(x,u)
=============

\frac{x-\sigma(u)\epsilon_\theta(x,u)}{\alpha(u)}.
]

score 近似为：

[
s_\theta(x,u)
\approx
-\frac{\epsilon_\theta(x,u)}{\sigma(u)}.
]

---

## 2.3 data prediction / denoiser

模型输出：

[
D_\theta(x,u)\approx x_0.
]

则

[
\epsilon_\theta(x,u)
====================

\frac{x-\alpha(u)D_\theta(x,u)}{\sigma(u)}.
]

score 近似为：

[
s_\theta(x,u)
\approx
\frac{\alpha(u)D_\theta(x,u)-x}{\sigma(u)^2}.
]

在 EDM / k-diffusion 代码中，经常使用：

[
d(x,\sigma)
===========

\frac{x-D_\theta(x,\sigma)}{\sigma}.
]

这个 (d) 就是 sigma-domain ODE 的 native derivative：

[
\frac{dx}{d\sigma}
==================

\frac{x-D_\theta(x,\sigma)}{\sigma}.
]

---

## 2.4 velocity prediction

如果使用 DDPM / Imagen 风格的 velocity：

[
v_\theta = \alpha\epsilon - \sigma x_0,
]

并且

[
\alpha^2+\sigma^2=1,
]

那么有：

[
\hat x_0
========

\alpha x-\sigma v_\theta,
]

[
\hat\epsilon
============

\sigma x+\alpha v_\theta.
]

如果代码里的 (v)-prediction 定义不同，必须先查该代码库的 conversion 函数，例如：

```python
pred_original_sample
prediction_type == "v_prediction"
convert_model_output(...)
```

不要假设所有库的 `v` 都完全相同。

---

## 2.5 score prediction

模型输出：

[
s_\theta(x,u)\approx \nabla_x\log p_u(x).
]

则

[
\epsilon_\theta(x,u)
\approx
-\sigma(u)s_\theta(x,u),
]

[
\hat x_0(x,u)
\approx
\frac{x+\sigma(u)^2 s_\theta(x,u)}{\alpha(u)}.
]

对于 VP SDE：

[
dx = -\frac12\beta(t)x,dt+\sqrt{\beta(t)},dw,
]

对应 probability flow ODE 通常写成：

[
\frac{dx}{dt}
=============

## -\frac12\beta(t)x

\frac12\beta(t)s_\theta(x,t).
]

代码中如果是 reverse-time 积分，符号由 step direction 自动处理。不要额外手动翻转符号，除非你完全确认原始实现的约定。

---

## 2.6 flow matching / rectified flow

如果模型直接输出：

[
v_\theta(x,t)\approx \frac{dx}{dt},
]

那么直接使用：

[
f(x,t)=v_\theta(x,t).
]

这种情况最简单。Euler / Heun / RK solver 通常直接作用在这个 velocity 上。

---

# 3. 第三步：选择 native coordinate (u)

我们的算法把所有 sampler 都抽象成：

[
\frac{dx}{du}=f_u(x,u).
]

如果第三方代码原来在 (t)-domain 采样：

[
\frac{dx}{dt}=f_t(x,t),
]

而你想改成 (\sigma)-domain，则需要：

[
\frac{dx}{d\sigma}
==================

\frac{dt}{d\sigma}
f_t(x,t(\sigma)).
]

如果从 (t) 改成 (\lambda)-domain：

[
\frac{dx}{d\lambda}
===================

\frac{dt}{d\lambda}
f_t(x,t(\lambda)).
]

一般地，

[
\boxed{
f_v(x,v)
========

\frac{du}{dv}
f_u(x,u(v)).
}
]

但是复现阶段建议：

[
\boxed{
\text{不要主动换 domain。先在原代码的 native domain 里做。}
}
]

原因是第三方 solver 的 order、stability、interpolation 都是相对于 native domain 写的。强行换 domain 会引入额外误差。

---

# 4. 第四步：封装 solver step operator

这是最关键的工程步骤。

你需要把第三方 sampler 的单步更新封装成：

```python
y = solver_step(x, u_start, u_end, extra_state=None)
```

数学上记作：

[
y
=

\Psi_S(x,u_{\mathrm{start}},u_{\mathrm{end}}).
]

---

## 4.1 一步法 solver

对于 Euler、Heun、RK、EDM Euler、EDM Heun、DPM-Solver 单步版本，可以直接封装。

### Euler example

如果 native ODE 是：

[
\frac{dx}{du}=f(x,u),
]

Euler step 是：

[
\Psi_S(x,u,u+h)
===============

x+h f(x,u).
]

代码形式：

```python
def solver_step_euler(x, u0, u1):
    h = u1 - u0
    f0 = model_to_derivative(x, u0)
    return x + h * f0
```

---

### Heun example

[
k_1=f(x,u),
]

[
x_{\mathrm{pred}}=x+h k_1,
]

[
k_2=f(x_{\mathrm{pred}},u+h),
]

[
\Psi_S(x,u,u+h)
===============

x+\frac{h}{2}(k_1+k_2).
]

代码形式：

```python
def solver_step_heun(x, u0, u1):
    h = u1 - u0
    k1 = model_to_derivative(x, u0)
    x_pred = x + h * k1
    k2 = model_to_derivative(x_pred, u1)
    return x + 0.5 * h * (k1 + k2)
```

---

### EDM Euler example

EDM / k-diffusion 中常见：

[
d(x,\sigma)
===========

\frac{x-D_\theta(x,\sigma)}{\sigma}.
]

Euler update：

[
x_{\mathrm{next}}
=================

x+(\sigma_{\mathrm{next}}-\sigma)d(x,\sigma).
]

代码形式：

```python
def solver_step_edm_euler(x, sigma0, sigma1):
    denoised = model(x, sigma0)
    d = (x - denoised) / sigma0
    return x + (sigma1 - sigma0) * d
```

---

## 4.2 多步法 solver

如果 solver 是 multistep，例如：

[
x_{n+1}
=======

\Psi_S(x_n,x_{n-1},\dots,x_{n-k+1};u_n,u_{n-1},\dots),
]

那么不能直接只传一个 (x_n)。

此时有两种选择。

### 选择 A：把 multistep solver 当成短区间 solver

封装：

```python
y = solve_segment(x_start, u_start, u_end, num_internal_steps)
```

也就是从 (u_{\mathrm{start}}) 积分到 (u_{\mathrm{end}})，内部使用原始 multistep solver。

然后用不同 internal step 数比较收敛阶。

这是最稳的 multistep 方案。

---

### 选择 B：构造一致历史

如果你想估计真正的 local defect，需要准备：

[
x_{n-k+1},\dots,x_n.
]

这些历史状态应该来自同一条高精度或原始轨迹。然后比较：

1. 一个大步 multistep update；
2. 两个半步 multistep update。

这更复杂，不建议第一版实现。

---

# 5. 第五步：采集 calibration states

我们需要估计：

[
x\sim \mu_u
]

处的 solver 缺陷。

实际做法：用原始 sampler 跑少量 pilot trajectories，保存若干时间点的中间状态。

---

## 5.1 选择 calibration grid

选择 (M) 个 native domain 点：

[
u_1,u_2,\dots,u_M.
]

建议：

```text
M = 32 或 64
```

如果原始 schedule 在 (\sigma)-domain 很不均匀，可以在 (\log\sigma) 上均匀选点，再映射回 (\sigma)。

如果原始 schedule 在 (t)-domain，可以直接在 (t) 上选点。

---

## 5.2 选择 calibration samples

选择 (K) 个随机种子：

```text
K = 4, 8, 16
```

对每个 seed，运行原始 sampler，并保存：

[
x_k(u_i),\qquad i=1,\dots,M.
]

得到：

[
{x_k(u_i)}_{k=1}^K.
]

这些样本近似来自：

[
x_k(u_i)\sim\mu_{u_i}.
]

---

## 5.3 注意事项

保存 calibration states 时，必须使用和最终采样一致的设置：

| 设置                      | 是否必须一致 |
| ----------------------- | ------ |
| CFG scale               | 必须一致   |
| prompt conditioning     | 最好一致   |
| model precision         | 必须一致   |
| thresholding / clipping | 必须一致   |
| solver type             | 必须一致   |
| prediction type         | 必须一致   |
| stochasticity           | 建议关闭   |

如果最终使用 classifier-free guidance，那么 calibration 时也必须用同样的 CFG scale，因为 guidance 会改变向量场的曲率和 Lipschitz 常数。

---

# 6. 第六步：用 step-refinement 估计局部有效阶数

这是我们方法的核心。

对于某个 solver (S)，假设它的局部误差满足：

[
\Phi_{u+h,u}(x)-\Psi_S(x,u,u+h)
===============================

h^{q(u)}\mathcal E_S(x,u)
+
O(h^{q(u)+1}).
]

这里：

[
q(u)=p(u)+1.
]

其中 (p(u)) 是有效全局阶数，(q(u)) 是有效局部阶数。

我们不假设 (p) 是多少，而是从数值缩放中估计 (q(u))。

---

## 6.1 coarse step 与 refined step

给定 refinement factor：

[
r=2.
]

计算 coarse step：

[
y_{\mathrm{coarse}}
===================

\Psi_S(x,u,u+h).
]

计算 refined step：

[
y_{\mathrm{fine}}
=================

\Psi_S(\cdot,u+h/2,u+h)
\circ
\Psi_S(x,u,u+h/2).
]

一般形式：

[
y_{\mathrm{fine}}
=================

\underbrace{
\Psi_S(\cdot,u+(r-1)h/r,u+h)
\circ\cdots\circ
\Psi_S(x,u,u+h/r)
}_{r\text{ 个小步}}.
]

然后定义：

[
\Delta_S(x,u,h)
===============

d(y_{\mathrm{fine}},y_{\mathrm{coarse}}).
]

推荐距离：

[
d(y,z)
======

\sqrt{
\frac{1}{D}
|y-z|_2^2
},
]

其中 (D) 是数据维度，例如图像的 (C\times H\times W)。

---

## 6.2 RMS 聚合

对 calibration samples 取 RMS：

[
A_S(u_i,h_j)
============

\left(
\frac1K
\sum_{k=1}^K
\Delta_S(x_k(u_i),u_i,h_j)^2
\right)^{1/2}.
]

理论上：

[
A_S(u,h)
\approx
B_S(u)|h|^{q(u)}.
]

其中

[
B_S(u)
\approx
\left|1-r^{1-q(u)}\right|
C_S(u).
]

所以

[
C_S(u)
\approx
\frac{B_S(u)}
{\left|1-r^{1-q(u)}\right|}.
]

在实际 schedule 构造中，直接用 (B_S(u)) 也可以，因为我们主要关心相对大小。

---

## 6.3 多个测试步长

对每个 (u_i)，选择几个小步长：

[
h_j\in{h_0,\ h_0/2,\ h_0/4,\ h_0/8}.
]

注意：

1. (u_i+h_j) 不能超出合法采样区间；
2. (h_j) 要沿采样方向；
3. 拟合时使用 (|h_j|)；
4. (h_j) 不能太大，否则不在渐近区；
5. (h_j) 不能太小，否则数值误差和神经网络不连续性会主导。

---

## 6.4 log-log 拟合

对每个 (u_i)，拟合：

[
\log A_S(u_i,h_j)
=================

\beta_i
+
q_i\log |h_j|.
]

得到：

[
\widehat q_i=q_i,
]

[
\widehat B_i=\exp(\beta_i).
]

修正后的 local defect coefficient：

[
\widehat C_i
============

\frac{
\widehat B_i
}{
\left|1-r^{1-\widehat q_i}\right|+\varepsilon
}.
]

其中 (\varepsilon) 是很小的常数，避免除零，例如：

[
\varepsilon=10^{-12}.
]

---

# 7. 第七步：构造 defect-balanced density

我们的目标是生成一个时间密度：

[
m(u)\ge 0,
]

满足：

[
\int_{u_{\min}}^{u_{\max}}m(u),du=1.
]

如果局部缺陷为：

[
C_S(u)|h|^{q(u)},
]

那么最优 density 近似满足：

[
\boxed{
m^*(u)
\propto
\left(
(q(u)-1)C_S(u)
\right)^{1/q(u)}
}
]

如果 (q(u)) 基本稳定，可以取：

[
\bar q
======

\mathrm{median}_i\widehat q_i,
]

然后使用：

[
\boxed{
m^*(u_i)
\propto
\widehat C_i^{1/\bar q}.
}
]

如果 (q(u)) 明显变化，则使用：

[
\boxed{
m^*(u_i)
\propto
\left(
(\widehat q_i-1)\widehat C_i
\right)^{1/\widehat q_i}.
}
]

---

## 7.1 稳定化处理

直接使用 (\widehat C_i) 可能会很 noisy，所以建议对 log-density 做平滑。

先定义 raw weight：

[
w_i
===

\left(
(\widehat q_i-1)\widehat C_i+\varepsilon
\right)^{1/\widehat q_i}.
]

然后取 log：

[
\ell_i=\log(w_i+\varepsilon).
]

对 (\ell_i) 做平滑，例如：

```python
ell_smooth = gaussian_filter1d(ell, sigma=1.0)
```

然后：

[
w_i^{\mathrm{smooth}}=\exp(\ell_i^{\mathrm{smooth}}).
]

---

## 7.2 clipping

为了避免某些点权重过大或过小，建议做 quantile clipping：

[
w_i
\leftarrow
\mathrm{clip}
\left(
w_i,
Q_{0.05}(w),
Q_{0.95}(w)
\right).
]

也可以加 density floor：

[
w_i
\leftarrow
(1-\rho)w_i+\rho \bar w,
]

其中：

[
\rho\in[0.02,0.1].
]

---

## 7.3 和原始 schedule 混合

第一版复现时，建议不要完全替换原 schedule，而是混合：

[
m_{\mathrm{final}}(u)
=====================

(1-\eta)m_{\mathrm{base}}(u)
+
\eta m_{\mathrm{defect}}(u).
]

其中：

[
\eta\in[0.5,1.0].
]

如果第三方代码的原 schedule 已经很强，例如 Karras schedule、DPM-Solver 默认 logSNR schedule，可以先用：

[
\eta=0.5.
]

如果希望测试我们方法的纯效果，可以用：

[
\eta=1.0.
]

---

# 8. 第八步：从 density 生成新的 schedule

假设 native domain 区间为：

[
u_{\mathrm{start}}\to u_{\mathrm{end}}.
]

注意这可能是降序，例如：

[
\sigma_{\max}\to\sigma_{\min}.
]

定义 oriented coordinate：

[
s\in[0,1].
]

可以令：

[
s(u)
====

\frac{
|u-u_{\mathrm{start}}|
}{
|u_{\mathrm{end}}-u_{\mathrm{start}}|
}.
]

在 calibration grid 上有 density values：

[
m_i=m(u_i).
]

构造 CDF：

[
F(u)
====

\frac{
\int_{u_{\mathrm{start}}}^{u}
m(v),|dv|
}{
\int_{u_{\mathrm{start}}}^{u_{\mathrm{end}}}
m(v),|dv|
}.
]

然后给定总步数 (N)，新的节点满足：

[
F(\tilde u_n)=\frac{n}{N},
\qquad
n=0,\dots,N.
]

数值上可以用插值反函数：

```python
new_u = inverse_cdf(np.linspace(0, 1, N + 1))
```

最后把：

```python
scheduler.timesteps = new_u
```

或者：

```python
scheduler.sigmas = new_u
```

替换进原始 sampler。

---

# 9. 伪代码：完整算法

## 9.1 一步法版本

```python
def calibrate_defect_schedule(
    solver_step,
    pilot_states,       # dict: u_i -> tensor [K, C, H, W]
    u_grid,             # shape [M]
    h_list_fn,          # function: u_i -> list of h_j
    refinement=2,
    eps=1e-12,
):
    """
    Estimate q(u), C(u), and defect-balanced weights.

    solver_step(x, u0, u1) should implement the original sampler's one-step update.
    """

    M = len(u_grid)
    q_hat = np.zeros(M)
    C_hat = np.zeros(M)

    for i, u in enumerate(u_grid):
        xs = pilot_states[u]  # [K, ...]
        h_list = h_list_fn(u)

        log_h = []
        log_A = []

        for h in h_list:
            deltas = []

            for x in xs:
                # coarse step
                y_coarse = solver_step(x, u, u + h)

                # refined step
                y_fine = x
                for r in range(refinement):
                    u_a = u + r * h / refinement
                    u_b = u + (r + 1) * h / refinement
                    y_fine = solver_step(y_fine, u_a, u_b)

                # per-dimension RMS displacement
                delta = rms(y_fine - y_coarse)
                deltas.append(delta)

            A = sqrt(mean([d ** 2 for d in deltas]))

            log_h.append(log(abs(h) + eps))
            log_A.append(log(A + eps))

        # robust linear regression:
        # log A = beta + q log |h|
        beta, q = linear_fit(log_h, log_A)

        q_hat[i] = q
        B = exp(beta)

        correction = abs(1.0 - refinement ** (1.0 - q))
        C_hat[i] = B / (correction + eps)

    return q_hat, C_hat
```

---

## 9.2 构造 density

```python
def build_defect_density(
    u_grid,
    q_hat,
    C_hat,
    use_local_q=True,
    smooth=True,
    clip=True,
    floor=0.05,
    eps=1e-12,
):
    if use_local_q:
        q_eff = np.maximum(q_hat, 1.05)
        weights = ((q_eff - 1.0) * C_hat + eps) ** (1.0 / q_eff)
    else:
        q_bar = np.median(q_hat)
        q_bar = max(q_bar, 1.05)
        weights = (C_hat + eps) ** (1.0 / q_bar)

    # smooth in log-space
    if smooth:
        log_w = np.log(weights + eps)
        log_w = gaussian_smooth(log_w, sigma=1.0)
        weights = np.exp(log_w)

    # quantile clipping
    if clip:
        lo = np.quantile(weights, 0.05)
        hi = np.quantile(weights, 0.95)
        weights = np.clip(weights, lo, hi)

    # density floor
    weights = (1.0 - floor) * weights + floor * np.mean(weights)

    # normalize as a density over u_grid
    density = normalize_density(u_grid, weights)

    return density
```

---

## 9.3 生成新 schedule

```python
def make_new_schedule(u_grid, density, num_steps):
    """
    u_grid may be increasing or decreasing.
    density is defined on u_grid.
    """

    # Work in oriented coordinate along sampling direction
    arc = np.zeros_like(u_grid)
    arc[1:] = np.cumsum(np.abs(np.diff(u_grid)))

    # cumulative integral of density along arc
    cdf = cumulative_trapezoid(density, arc)
    cdf = cdf / cdf[-1]

    target = np.linspace(0.0, 1.0, num_steps + 1)

    # inverse CDF interpolation
    new_arc = interp1d(cdf, arc)(target)
    new_u = interp1d(arc, u_grid)(new_arc)

    return new_u
```

---

# 10. 多步 solver 的版本

对于 multistep solver，不建议直接用单步 step-refinement，因为它依赖历史。

可以改成短区间自收敛。

---

## 10.1 短区间误差估计

选择一个短区间：

[
[u_i,u_i+H_i].
]

分别用 (n)、(2n)、(4n) 个内部步积分：

[
y_n,\qquad y_{2n},\qquad y_{4n}.
]

定义：

[
A(u_i,\Delta u)
===============

d(y_n,y_{2n}),
]

其中：

[
\Delta u=\frac{|H_i|}{n}.
]

如果 solver 的全局阶数是 (p)，则：

[
A(u_i,\Delta u)
\approx
\widetilde C(u_i)(\Delta u)^{p(u_i)}.
]

拟合：

[
\log A
======

\beta_i+p_i\log \Delta u.
]

得到：

[
\widehat p_i.
]

然后：

[
\widehat q_i=\widehat p_i+1.
]

density 使用：

[
m(u_i)
\propto
\left(
\widetilde C(u_i)
\right)^{1/(\widehat p_i+1)}.
]

---

## 10.2 multistep 伪代码

```python
def calibrate_multistep_segment(
    solve_segment,
    pilot_states,
    u_grid,
    H_fn,
    n_list=(2, 4, 8),
    eps=1e-12,
):
    M = len(u_grid)
    p_hat = np.zeros(M)
    C_hat = np.zeros(M)

    for i, u in enumerate(u_grid):
        xs = pilot_states[u]
        H = H_fn(u)

        log_du = []
        log_A = []

        for n in n_list:
            deltas = []

            for x in xs:
                y_n = solve_segment(x, u, u + H, num_steps=n)
                y_2n = solve_segment(x, u, u + H, num_steps=2 * n)

                delta = rms(y_2n - y_n)
                deltas.append(delta)

            A = sqrt(mean([d ** 2 for d in deltas]))
            du = abs(H) / n

            log_du.append(log(du + eps))
            log_A.append(log(A + eps))

        beta, p = linear_fit(log_du, log_A)

        p_hat[i] = p
        C_hat[i] = exp(beta)

    q_hat = p_hat + 1.0
    return q_hat, C_hat
```

---

# 11. 针对不同代码库类型的接入方式

## 11.1 EDM / k-diffusion / sigma-domain sampler

常见形式：

```python
sigmas = get_sigmas(...)
for i in range(len(sigmas) - 1):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    denoised = model(x, sigma)
    d = (x - denoised) / sigma
    x = x + (sigma_next - sigma) * d
```

对应数学形式：

[
\frac{dx}{d\sigma}
==================

\frac{x-D_\theta(x,\sigma)}{\sigma}.
]

接入步骤：

1. native domain 设为：

[
u=\sigma.
]

2. 封装：

```python
solver_step(x, sigma0, sigma1)
```

3. 用原始 `sigmas` 跑 pilot trajectories，保存 (x(\sigma_i))。

4. 估计：

[
q(\sigma),\quad C(\sigma).
]

5. 构造：

[
m(\sigma)\propto C(\sigma)^{1/q(\sigma)}.
]

6. 反解 CDF 得到：

```python
new_sigmas
```

7. 替换：

```python
sigmas = new_sigmas
```

---

## 11.2 DDIM / DDPM / VP time-domain sampler

常见形式：

```python
timesteps = scheduler.timesteps
for t in timesteps:
    noise_pred = model(x, t)
    x = scheduler.step(noise_pred, t, x)
```

接入步骤：

1. native domain 设为：

[
u=t.
]

2. 如果 `scheduler.step` 只能接受 discrete timestep，需要确认是否允许 continuous timestep。

3. 如果只允许 discrete index，则新 schedule 必须投影回合法 index：

[
\tilde t_i \mapsto \mathrm{round}(\tilde t_i).
]

4. 封装：

```python
solver_step(x, t0, t1)
```

如果 scheduler 的 step 函数强依赖当前 index 和 next index，需要传入两者：

```python
solver_step(x, t0, t1):
    model_out = model(x, t0)
    return scheduler.step_from_to(model_out, t0, t1, x)
```

5. 估计 (q(t))、(C(t))。

6. 生成新的 `timesteps`。

---

## 11.3 DPM-Solver / log-SNR-domain sampler

DPM-Solver 通常在 (\lambda)-domain 或与 (\lambda) 强相关的变量中构造。

其中：

[
\lambda
=======

\log\frac{\alpha}{\sigma}
]

或者有些代码使用：

[
\lambda
=======

\log\frac{\alpha^2}{\sigma^2}.
]

两者差一个 factor 2，必须检查代码约定。

接入步骤：

1. 找到代码里的 native variable：

```python
lambda_t
marginal_lambda(t)
inverse_lambda(lambda)
```

2. native domain 设为：

[
u=\lambda.
]

3. 封装 DPM-Solver 的一步更新：

```python
solver_step(x, lambda0, lambda1)
```

如果原始实现只接受 `t_start, t_end`，则内部做：

[
t_0=\lambda^{-1}(\lambda_0),
\qquad
t_1=\lambda^{-1}(\lambda_1).
]

4. 不要手动把 DPM-Solver 改成普通 RK。我们的 calibration 直接作用在 DPM-Solver step operator 上。

5. 估计：

[
q(\lambda),\quad C(\lambda).
]

6. 生成新的：

```python
new_lambdas
```

7. 再映射回模型需要的 conditioning：

```python
new_timesteps = inverse_lambda(new_lambdas)
```

---

## 11.4 Rectified Flow / Flow Matching

常见形式：

```python
for i in range(N):
    t0 = timesteps[i]
    t1 = timesteps[i + 1]
    v = model(x, t0)
    x = x + (t1 - t0) * v
```

对应：

[
\frac{dx}{dt}=v_\theta(x,t).
]

接入步骤：

1. native domain 设为：

[
u=t.
]

2. 封装 `solver_step`。

3. 估计 (q(t))、(C(t))。

4. 生成新的 `timesteps`。

对 rectified flow，如果原始 flow 已经比较直，(C(t)) 可能较平，此时新 schedule 会接近均匀时间表。这是正常现象。

---

# 12. 质量控制：如何判断 calibration 是否有效？

估计完成后，你应该检查以下指标。

---

## 12.1 log-log 线性关系

对每个 (u_i)，检查：

[
\log A(u_i,h)
\quad\text{vs.}\quad
\log |h|.
]

如果大致线性，说明处于有效渐近区。

如果曲线明显弯曲，说明 (h) 选择不合适。

---

## 12.2 有效阶数是否合理

常见参考：

| Solver       | 理想局部阶数 (q) | 理想全局阶数 (p=q-1) |
| ------------ | ---------: | -------------: |
| Euler        |          2 |              1 |
| Heun         |          3 |              2 |
| RK4          |          5 |              4 |
| DPM-Solver-1 |          2 |              1 |
| DPM-Solver-2 |          3 |              2 |
| DPM-Solver-3 |          4 |              3 |

注意：实际估计到的 (q) 可能低于理论值，原因包括：

1. 模型不够光滑；
2. CFG scale 太大；
3. thresholding / clipping 破坏光滑性；
4. (\sigma\to0) 附近奇异；
5. discrete timestep embedding 不支持连续时间；
6. solver startup 阶数低；
7. multistep 历史不一致；
8. mixed precision 误差。

所以我们关心的是：

[
\boxed{
q_{\mathrm{eff}}(u)
}
]

而不是论文里标称的阶数。

---

## 12.3 缺陷强度是否集中在合理区域

一般 diffusion sampler 的误差可能集中在：

* high-noise 到 mid-noise 的 transition 区域；
* low-noise 细节生成区域；
* guidance 强烈改变方向的区域；
* (\sigma) 很小导致 derivative 变大的区域；
* logSNR 变化剧烈的区域。

如果 (C(u)) 完全随机跳动，通常说明 calibration 不稳定。

可以增加：

```text
K：calibration sample 数
M：calibration time grid 数
h_list：测试步长数量
```

或者增强平滑。

---

# 13. 最终采样流程

最终 sampling 不需要再做 step-refinement。

只需要：

1. 加载模型；
2. 加载或计算好的新 schedule；
3. 用原始 solver；
4. 替换原始 `timesteps` 或 `sigmas`；
5. 正常采样。

伪代码：

```python
# 1. original scheduler
base_schedule = scheduler.get_schedule(num_steps=N)

# 2. calibration
pilot_states = collect_pilot_states(
    model=model,
    scheduler=scheduler,
    base_schedule=base_schedule,
    u_grid=u_grid,
    num_calib_samples=K,
)

q_hat, C_hat = calibrate_defect_schedule(
    solver_step=solver_step,
    pilot_states=pilot_states,
    u_grid=u_grid,
    h_list_fn=h_list_fn,
)

# 3. build new schedule
density = build_defect_density(
    u_grid=u_grid,
    q_hat=q_hat,
    C_hat=C_hat,
)

new_schedule = make_new_schedule(
    u_grid=u_grid,
    density=density,
    num_steps=N,
)

# 4. final sampling
scheduler.set_schedule(new_schedule)

samples = sample(
    model=model,
    scheduler=scheduler,
    num_steps=N,
    prompts=prompts,
)
```

---

# 14. 推荐的默认超参数

| 超参数                     |                推荐值 |
| ----------------------- | -----------------: |
| calibration grid (M)    |            32 或 64 |
| calibration samples (K) |             4 到 16 |
| refinement factor (r)   |                  2 |
| 测试步长数量                  |              3 或 4 |
| log-density smoothing   | Gaussian sigma = 1 |
| clipping quantile       |           5% 到 95% |
| density floor           |         0.02 到 0.1 |
| schedule mixing (\eta)  |          0.5 到 1.0 |
| (q) 下界                  |               1.05 |
| (q) 上界                  |              6 或 8 |

---

# 15. 实验对照设计

为了证明方法有效，建议做以下对照。

---

## 15.1 同 NFE 对照

固定采样步数 (N)，比较：

1. 原始 schedule；
2. Karras schedule；
3. uniform in (t)；
4. uniform in (\sigma)；
5. uniform in (\lambda)；
6. 我们的 defect-balanced schedule。

要求：

[
\text{NFE 完全一致。}
]

---

## 15.2 同 solver 对照

固定 solver，只换 schedule：

```text
Euler + base schedule
Euler + our schedule

Heun + base schedule
Heun + our schedule

DPM-Solver + base schedule
DPM-Solver + our schedule
```

这样可以证明我们的改进来自时间重参数，而不是 solver 改动。

---

## 15.3 跨 parameterization 对照

同一个模型或等价模型，分别测试：

| 模型输出                 | 是否能接入 |
| -------------------- | ----- |
| noise prediction     | 是     |
| data prediction      | 是     |
| velocity prediction  | 是     |
| score prediction     | 是     |
| direct flow velocity | 是     |

重点展示：

[
\boxed{
\text{我们的方法不依赖模型输出形式，只依赖 solver step 的自收敛缺陷。}
}
]

---

# 16. 常见失败情况和处理方式

## 16.1 (q(u)) 估计非常低

例如：

[
q(u)<1.
]

可能原因：

1. step size 太大；
2. step size 太小，被数值误差主导；
3. 模型 timestep embedding 是离散的；
4. solver 内部有 clipping；
5. dynamic thresholding 造成非光滑；
6. stochastic sampler 没有关掉随机性。

处理：

```text
扩大 h_list 的中间范围；
关闭 stochasticity；
关闭或固定 thresholding；
使用 median q；
对 q 做 clipping。
```

---

## 16.2 (\Delta(h)) 不随 (h) 下降

如果：

[
A(h/2)\not< A(h),
]

说明 step-refinement 没有进入渐近区。

处理：

1. 增大 calibration batch；
2. 改变 (h) 范围；
3. 检查 solver step 是否真的支持任意 (u_0\to u_1)；
4. 检查模型是否能接受 continuous timestep；
5. 检查是否有 stochastic noise。

---

## 16.3 新 schedule 在端点过密

低噪声端经常出现：

[
C(u)\to\infty.
]

处理：

1. density clipping；
2. density floor；
3. 保留原始最后一步 denoise；
4. 限制最小 step size；
5. 和原 schedule 混合。

---

## 16.4 第三方 scheduler 不允许任意 timestep

有些代码只允许整数 timestep index。

处理：

1. 先生成 continuous schedule；
2. 投影到最近合法 index；
3. 去重；
4. 如果去重后步数减少，重新插值补点；
5. 确保 schedule 单调。

---

# 17. 最终方法可以在论文中这样定义

我们的方法不假设 solver 阶数，而是定义：

[
\Delta_S(u,h)
=============

\left(
\mathbb E_{x\sim\mu_u}
\left[
d\left(
\Psi_S(x,u,u+h),
\Psi_S^{(r)}(x,u,u+h)
\right)^2
\right]
\right)^{1/2},
]

其中 (\Psi_S^{(r)}) 表示使用 (r) 个小步完成同一区间。

若：

[
\Delta_S(u,h)
\asymp
B_S(u)|h|^{q_S(u)},
]

则：

[
q_S(u)
======

\frac{
\partial\log \Delta_S(u,h)
}{
\partial\log |h|
},
]

[
C_S(u)
======

\frac{
B_S(u)
}{
|1-r^{1-q_S(u)}|
}.
]

最终 schedule density 为：

[
\boxed{
m_S^*(u)
\propto
\left(
(q_S(u)-1)C_S(u)
\right)^{1/q_S(u)}.
}
]

当 (q_S(u)) 近似常数时：

[
\boxed{
m_S^*(u)
\propto
C_S(u)^{1/q_S}.
}
]

这就是完整的：

[
\boxed{
\textbf{Order-Free Solver-Aware Defect-Balanced Time Reparameterization}
}
]

或者中文：

[
\boxed{
\textbf{免阶数假设的求解器感知缺陷均衡时间重参数化}
}
]

核心卖点是：

1. 不需要训练；
2. 不需要知道 solver 名称；
3. 不需要人工输入 solver 阶数；
4. 不依赖模型输出类型；
5. 不依赖 (t)、(\sigma)、(\lambda) 的具体选择；
6. 直接在第三方代码库的 native sampler 上估计有效局部误差；
7. 生成的新 schedule 可以直接替换原始 schedule。
