
---

# GPDE：几何感知预测缺陷极小极大调度

## 1. 问题设定

### 1.1 确定性 ODE 采样器

考虑由确定性 ODE 定义的采样过程：

[
\frac{dx}{du}=f_\theta(x,u),
\qquad
u\in[u_{\min},u_{\max}].
]

其中 (u) 是采样坐标，可以是时间、噪声水平、log-SNR，或其任意单调重参数化。为统一表述，本文假设调度在 (u) 上递增：

[
U=(u_0<u_1<\cdots<u_N),
\qquad
u_0=u_{\min},
\qquad
u_N=u_{\max}.
]

若实际采样方向与 (u) 的递增方向相反，可以通过 (u\mapsto -u) 或其他单调变换统一到上述形式。

本文方法针对 deterministic ODE sampler，例如 deterministic DDIM、probability-flow ODE sampler、DPM-Solver 类 ODE sampler、EDM ODE sampler、flow matching ODE sampler 等。

本文不直接覆盖 ancestral sampler 或 SDE sampler。若扩展到随机采样器，需要固定随机路径，或引入 strong-error coupling / weak-error objective。该情形不属于本文的基本理论范围。

---

### 1.2 Teacher oracle 与 student solver

给定 (K) 条校准轨迹。第 (k) 条轨迹的 high-accuracy teacher 解记为：

[
x_k^*(u),
\qquad
k=1,\ldots,K.
]

该 teacher 由高精度 ODE solver 离线求解，并通过 dense-output 插值在任意查询点 (u) 上提供：

[
x_k^*(u),
\qquad
\partial_u x_k^*(u),
\qquad
f_\theta(x_k^*(u),u).
]

给定一个固定的 student solver (\mathcal S) 和固定步数 (N)，目标是在不训练模型参数的情况下，寻找共享调度：

[
U=(u_0,\ldots,u_N)
]

使 few-step student sampler 尽量接近 high-accuracy teacher trajectory。

该目标优化的是 **teacher-alignment**，而不是直接优化 FID、人类偏好或感知质量。Teacher-distance 改善不必然推出 generation-quality 改善。因此实验必须同时报告 teacher-alignment 指标和 generation-quality 指标。

---

## 2. 度量感知几何

### 2.1 预条件状态空间度量

在状态空间上引入随采样坐标变化的正定度量：

[
G(u)\succ0.
]

对应内积和范数为：

[
\langle r_1,r_2\rangle_{G(u)}
=============================

r_1^\top G(u)r_2,
]

[
|r|_{G(u)}^2
============

r^\top G(u)r.
]

常见选择包括：

[
G(u)
====

\frac{1}{\sigma(u)^2+\sigma_{\text{data}}^2}I,
]

或基于校准集的 channel-wise whitening：

[
G(u)
====

\operatorname{diag}
\left(
\hat\sigma_1(u)^{-2},
\ldots,
\hat\sigma_d(u)^{-2}
\right).
]

(G(u)) 的作用是消除不同噪声水平、不同 channel、不同参数化下的尺度偏差，使 residual 的度量更接近预条件后的有效误差。

---

### 2.2 (G)-单位切向量

对 teacher 轨迹定义 (G)-单位切向量：

[
T_{k,G}(u)
==========

\frac{\partial_u x_k^*(u)}
{|\partial_u x_k^*(u)|_{G(u)}}.
]

实践中推荐直接使用 ODE RHS 计算切向量：

[
\partial_u x_k^*(u)
===================

f_\theta(x_k^*(u),u),
]

但必须确保 (f_\theta) 是当前 (u) 坐标下的 RHS。如果原始模型定义在 (\sigma)、(t)、(\lambda) 等其他坐标下，需要使用链式法则转换到 (u)。

若

[
|\partial_u x_k^*(u)|_{G(u)}
]

低于数值阈值，则切向方向不稳定。此时可退回 full residual，即令 (\rho=1)，或跳过切向投影。

---

### 2.3 endpoint 处的 (\rho)-混合残差范数

给定从 (a) 到 (b) 的单步预测 residual：

[
r
=

## \Phi_{\mathcal S}(x_k^*(a);a\to b)

x_k^*(b),
]

该 residual 是 endpoint (b) 处的状态空间误差。因此残差范数应在 (b) 处定义。

定义：

[
\boxed{
|r|_{\rho,k,b}^2
================

## r^\top G(b)r

(1-\rho)
\bigl(T_{k,G}(b)^\top G(b)r\bigr)^2.
}
]

其中：

[
\rho\in(0,1].
]

因为 (T_{k,G}(b)) 是 (G(b))-单位向量，可以将 (r) 分解为 (G(b))-正交的切向与法向部分：

[
r=r_\parallel^{G(b)}+r_\perp^{G(b)},
]

其中：

[
r_\parallel^{G(b)}
==================

\bigl(T_{k,G}(b)^\top G(b)r\bigr)T_{k,G}(b),
]

[
r_\perp^{G(b)}
==============

r-r_\parallel^{G(b)}.
]

于是：

[
\boxed{
|r|_{\rho,k,b}^2
================

|r_\perp^{G(b)}|*{G(b)}^2
+
\rho
|r*\parallel^{G(b)}|_{G(b)}^2.
}
]

解释如下：

[
\rho=1
]

对应 full residual；

[
0<\rho<1
]

降低切向误差权重，但不完全忽略切向误差；

[
\rho=0
]

退化为纯法向 residual，可能对直线或近直线轨迹上的切向误差失明。因此 (\rho=0) 只建议作为消融实验，不建议作为主方法。

(\rho) 不应设为理论固定常数，而应通过 held-out validation 选择。更严格地说，切向误差和法向误差对最终样本的影响应由后续 flow 的传播敏感度决定；固定 (\rho) 是一种低成本近似。

---

## 3. Oracle 起始预测缺陷

### 3.1 单步 solver 的 oracle-start local defect

对单步 solver，定义 oracle-start local defect：

[
\boxed{
d_k(a,b)
========

\left|
\Phi_{\mathcal S}(x_k^*(a);a\to b)
----------------------------------

x_k^*(b)
\right|_{\rho,k,b}^2.
}
]

该定义每次都从 teacher oracle state (x_k^*(a)) 出发，因此只测量当前区间 ([a,b]) 的 solver prediction defect，不混入前缀累计误差。

若使用从头 replay 得到的近似状态 (x_{i,k}) 作为起点，则 residual 会包含：

[
\text{prefix accumulated error}
+
\text{current local prediction error}.
]

这种混合误差不再是当前区间的局部 difficulty，也不具备可加 monitor 理论所需的局部性。

---

### 3.2 传播加权预测缺陷

局部 endpoint defect 不一定等价于最终样本误差，因为当前误差会被后续 ODE flow 放大或衰减。

可选地定义传播加权缺陷：

[
\boxed{
\tilde d_k(a,b)
===============

\left|
F_{b\to u_{\max}}^{\mathrm{ref}}
\left(
\Phi_{\mathcal S}(x_k^*(a);a\to b)
\right)
-------

x_k^*(u_{\max})
\right|*{G(u*{\max})}^2.
}
]

其中 (F_{b\to u_{\max}}^{\mathrm{ref}}) 是 high-accuracy teacher flow。

若 teacher flow 在 (b) 附近可微，则有一阶近似：

[
\tilde d_k(a,b)
\approx
\left|
S_k(b)
\left(
\Phi_{\mathcal S}(x_k^*(a);a\to b)
----------------------------------

x_k^*(b)
\right)
\right|*{G(u*{\max})}^2,
]

其中 (S_k(b)) 是从 (b) 到终点的线性化传播算子。

传播加权 defect 更接近最终误差目标，但计算成本更高。若不使用传播加权，则理论结论应理解为 local defect minimax，而不是最终样本质量最优。

---

### 3.3 多步 solver 的 oracle-consistent defect

对 (p)-步 multistep solver，当前预测不仅依赖 (u_i)，还依赖历史状态和历史模型评估。令历史调度状态为：

[
s_i=(u_{i-p+1},\ldots,u_i).
]

若 solver 支持 oracle-consistent history injection，则可构造：

[
H_k^{*,\mathcal S}(s_i),
]

即 solver 在 teacher oracle 轨迹和对应模型评估上形成的内部历史。

多步 solver 的局部缺陷定义为：

[
\boxed{
d_k(s_i,u_{i+1})
================

\left|
\Phi_{\mathcal S}
\bigl(
x_k^*(u_i),
H_k^{*,\mathcal S}(s_i);
u_i\to u_{i+1}
\bigr)
------

x_k^*(u_{i+1})
\right|*{\rho,k,u*{i+1}}^2.
}
]

因此，多步 solver 的 defect 一般不是 pairwise cost：

[
d_k(u_i,u_{i+1}),
]

而是 augmented-state cost：

[
d_k(s_i,u_{i+1}).
]

若 solver 不支持 oracle-consistent history injection，则只能使用 full replay defect：

[
d_k^{\mathrm{replay}}(U,i)
==========================

\left|
\Phi_{\mathcal S}(x_{i,k},H_{i,k};u_i\to u_{i+1})
-------------------------------------------------

x_k^*(u_{i+1})
\right|*{\rho,k,u*{i+1}}^2.
]

此量依赖整条调度 (U)，不再是局部 edge cost，也不能使用 monitor 等分理论或普通 edge DP 精确求解。此时方法降级为黑盒调度优化。

---

## 4. 局部幂律模型与 monitor 构造

### 4.1 局部 defect 幂律模型

假设单步 solver 在 (u) 附近满足局部幂律误差：

[
\ell_k(u,h)
===========

## \Phi_{\mathcal S}(x_k^*(u);u\to u+h)

# x_k^*(u+h)

C_k(u)h^\alpha
+
O(h^{\alpha+1}).
]

其中 (\alpha>0) 是有效局部 truncation defect 阶数。

由于 (d_k) 是 squared norm，因此：

[
d_k(u,u+h)
==========

# |\ell_k(u,h)|_{\rho,k,u+h}^2

a_k(u)h^q
+
O(h^{q+1}),
]

其中：

[
q=2\alpha,
]

[
a_k(u)
======

|C_k(u)|_{\rho,k,u}^2.
]

endpoint norm 中的 (u+h) 与 (u) 的差异只影响高阶项。

---

### 4.2 局部 defect coefficient 的估计

在 probe step (\eta) 足够小的情况下，可以估计：

[
\boxed{
a_k(u)
\approx
\frac{d_k(u,u+\eta)}{\eta^q}.
}
]

若 (q) 未知，可使用多个 probe step (\eta_j) 做 log-log regression：

[
\log d_k(u,u+\eta_j)
\approx
\log a_k(u)
+
q\log \eta_j.
]

得到全局有效阶数：

[
\hat q.
]

如果局部斜率随 (u) 明显变化，则全局 (q) 假设不再充分，应使用局部阶数扩展，见第 4.6 节。

---

### 4.3 多轨迹聚合

对每个 (u)，将不同校准轨迹的局部 defect coefficient 聚合。推荐使用 CVaR：

[
\boxed{
\bar a(u)
=========

\operatorname{CVaR}_{\beta,k}\bigl[a_k(u)\bigr].
}
]

其中 (\operatorname{CVaR}_\beta) 表示高分位尾部平均，常用：

[
\beta\in[0.8,0.95].
]

CVaR 的意义是关注 hard prompts，同时比 max 更稳健。

由于 CVaR 对正数具有正齐次性：

[
\operatorname{CVaR}_{\beta,k}
\left[
a_k(u)h^q
\right]
=======

h^q
\operatorname{CVaR}_{\beta,k}
\left[
a_k(u)
\right],
]

因此聚合后的局部 defect 仍满足：

[
D(u,u+h)
\approx
\bar a(u)h^q,
]

其中：

[
D(u,u+h)
========

\operatorname{CVaR}_{\beta,k}
\left[
d_k(u,u+h)
\right].
]

如果使用 median 或 trimmed mean，则方法更接近 robust-average schedule，而不是 trajectory-minimax schedule。若需要更严格的 minimax 语义，可使用：

[
\bar a(u)=\max_k a_k(u).
]

---

### 4.4 正确的 monitor 密度

考虑目标：

[
\min_{{h_i}}
\max_i
\bar a(u_i)h_i^q,
\qquad
\sum_i h_i=L.
]

在 leading-order 最优分配中，各区间 defect 应近似相等：

[
\bar a(u_i)h_i^q=C.
]

因此：

[
h_i
===

C^{1/q}
\bar a(u_i)^{-1/q}.
]

这说明应当等分的不是 (\bar a(u))，而是其 (q)-th root：

[
\boxed{
\omega(u)
=========

\bigl(\bar a(u)+\epsilon_a\bigr)^{1/q}.
}
]

其中 (\epsilon_a>0) 是数值稳定 floor。理论分析中若假设：

[
\bar a(u)>0,
]

则可以令：

[
\epsilon_a=0.
]

定义区间 monitor mass：

[
\boxed{
\mathcal M([a,b])
=================

\int_a^b\omega(u),du.
}
]

(\mathcal M) 是可加的：

[
\mathcal M([a,c])
=================

\mathcal M([a,b])
+
\mathcal M([b,c]).
]

---

### 4.5 monitor 的坐标协变性

设：

[
v=g(u)
]

为光滑单调重参数化。记同一个物理小区间在两个坐标下的步长为：

[
h_u=\Delta u,
\qquad
h_v=\Delta v.
]

由于：

[
h_v
===

\left|\frac{dv}{du}\right|h_u,
]

因此：

[
h_u
===

\left|\frac{du}{dv}\right|h_v.
]

假设同一个局部 defect 在 (u) 坐标中表示为：

[
D
\approx
a_u(u)h_u^q,
]

在 (v) 坐标中表示为：

[
D
\approx
a_v(v)h_v^q.
]

代入步长变换：

[
D
\approx
a_u(u)
\left|\frac{du}{dv}\right|^q
h_v^q.
]

因此：

[
\boxed{
a_v(v)
======

a_u(u)
\left|\frac{du}{dv}\right|^q
============================

a_u(u)
\left|\frac{dv}{du}\right|^{-q}.
}
]

于是：

[
\omega_v(v)
===========

# a_v(v)^{1/q}

a_u(u)^{1/q}
\left|\frac{du}{dv}\right|
==========================

\omega_u(u)
\left|\frac{du}{dv}\right|.
]

两边乘以 (dv)，得到：

[
\boxed{
\omega_v(v),dv
==============

\omega_u(u),du.
}
]

因此，(\omega(u)du) 是由局部 defect 诱导的 monitor 1-form。等分：

[
\int\omega(u),du
]

得到的调度不依赖于同一物理 defect 函数的单调坐标表达。

该结论只针对同一个局部 defect 函数的重参数化表达。若改变 solver 内部公式、改变 scheduler 的实际积分变量，或在新坐标下使用不同离散更新规则，则 (a(u)) 需要重新测量。

---

### 4.6 局部阶数 (q(u)) 的扩展

全局 (q) 假设在理论上简洁，但 diffusion ODE 在低噪声端、强 CFG、thresholding、guidance rescale 或模型误差放大区域中，可能表现出不同的有效局部阶数。

若观测到：

[
D(u,u+h)
\approx
a(u)h^{q(u)},
]

则等 defect 条件为：

[
a(u_i)h_i^{q(u_i)}
==================

C.
]

于是：

[
h_i
===

\left(\frac{C}{a(u_i)}\right)^{1/q(u_i)}.
]

对应连续采样密度为：

[
\boxed{
n_C(u)
======

# \frac{1}{h(u)}

\left(\frac{a(u)}{C}\right)^{1/q(u)}.
}
]

常数 (C) 由步数约束决定：

[
\boxed{
\int_{u_{\min}}^{u_{\max}}
\left(\frac{a(u)}{C}\right)^{1/q(u)}
du
==

N.
}
]

求出 (C) 后，定义：

[
\tau_C(u)
=========

\frac{1}{N}
\int_{u_{\min}}^u n_C(\xi)d\xi.
]

调度为：

[
u_m
===

\tau_C^{-1}
\left(
\frac{m}{N}
\right).
]

当 (q(u)\equiv q) 为常数时，该形式退化为：

[
n_C(u)\propto a(u)^{1/q},
]

即全局 (q) 的 monitor 等分。

实践中应报告 (\hat q(u)) 的稳定性。如果 (\hat q(u)) 在不同 (u) 区间明显变化，则应比较全局 (q)、局部 (q(u)) 和直接 DP 搜索三种方案。

---

## 5. 极小极大等分理论

### 5.1 可加 monitor 的精确等分定理

令：

[
\Omega
======

\int_{u_{\min}}^{u_{\max}}\omega(u),du.
]

对任意划分：

[
U=(u_0<u_1<\cdots<u_N),
]

定义第 (i) 个区间的 monitor mass：

[
\mathcal M_i
============

\int_{u_i}^{u_{i+1}}\omega(u),du.
]

则：

[
\sum_{i=0}^{N-1}\mathcal M_i
============================

\Omega.
]

因此：

[
\max_i \mathcal M_i
\geq
\frac{\Omega}{N}.
]

当且仅当所有区间 monitor mass 相等时达到下界：

[
\mathcal M_i
============

\frac{\Omega}{N},
\qquad
i=0,\ldots,N-1.
]

若 (\omega(u)>0) 且连续，则该划分唯一。

定义归一化时钟：

[
\tau(u)
=======

\frac{
\int_{u_{\min}}^u\omega(\xi)d\xi
}{
\Omega
}.
]

则 monitor 等分调度为：

[
\boxed{
u_m^*
=====

\tau^{-1}
\left(
\frac{m}{N}
\right),
\qquad
m=0,\ldots,N.
}
]

---

### 5.2 与局部 defect 的渐近 minimax 连接

令：

[
\omega(u)=a(u)^{1/q},
\qquad
\Omega=\int_{u_{\min}}^{u_{\max}}\omega(u),du.
]

对区间 (I=[s,t])，定义：

[
\mathcal M(I)
=============

\int_s^t\omega(u),du.
]

假设存在常数 (C_E>0)，使得对所有足够小的区间 (I=[s,t])，有一致近似：

[
\boxed{
\left|
D(s,t)
------

\mathcal M([s,t])^q
\right|
\leq
C_E |t-s|^{q+1}.
}
]

该条件可由以下局部模型推出：

[
D(u,u+h)
========

a(u)h^q
+
O(h^{q+1}),
]

并要求：

[
a\in C^1,
\qquad
0<a_{\min}\leq a(u)\leq a_{\max}<\infty.
]

证明直观如下：

[
\mathcal M([u,u+h])
===================

# \int_u^{u+h}\omega(\xi)d\xi

\omega(u)h
+
O(h^2).
]

因此：

[
\mathcal M([u,u+h])^q
=====================

\omega(u)^qh^q
+
O(h^{q+1})
==========

a(u)h^q
+
O(h^{q+1}).
]

所以：

[
D(u,u+h)
========

\mathcal M([u,u+h])^q
+
O(h^{q+1}).
]

---

### 5.3 渐近 minimax 最优性定理

设：

[
0<\omega_{\min}\leq \omega(u)\leq \omega_{\max}<\infty,
]

且第 5.2 节的一致近似成立。

令 (U^\Omega) 为等分 monitor mass 的调度：

[
\mathcal M([u_i^\Omega,u_{i+1}^\Omega])
=======================================

\frac{\Omega}{N}.
]

则当 (N) 足够大时：

[
\boxed{
\max_i
D(u_i^\Omega,u_{i+1}^\Omega)
\leq
\left(\frac{\Omega}{N}\right)^q
+
C_1N^{-(q+1)}.
}
]

此外，对任意满足局部网格条件：

[
h_{\max}
========

\max_i(u_{i+1}-u_i)
\leq
\frac{C_h}{N}
]

的 (N) 步划分 (U)，有：

[
\boxed{
\max_i
D(u_i,u_{i+1})
\geq
\left(\frac{\Omega}{N}\right)^q
-------------------------------

C_2N^{-(q+1)}.
}
]

因此，在局部渐近区内，monitor 等分调度是 minimax-optimal 到 (O(N^{-(q+1)})) 误差。

---

### 5.4 定理证明

对等分调度 (U^\Omega)，每个区间满足：

[
\mathcal M_i
============

\frac{\Omega}{N}.
]

又因为：

[
\omega(u)\geq\omega_{\min}>0,
]

所以：

[
h_i^\Omega
\leq
\frac{\mathcal M_i}{\omega_{\min}}
==================================

\frac{\Omega}{N\omega_{\min}}.
]

由局部近似：

[
D_i
===

\mathcal M_i^q
+
O((h_i^\Omega)^{q+1})
=====================

\left(\frac{\Omega}{N}\right)^q
+
O(N^{-(q+1)}).
]

因此：

[
\max_iD_i
\leq
\left(\frac{\Omega}{N}\right)^q
+
C_1N^{-(q+1)}.
]

对任意满足 (h_{\max}\leq C_h/N) 的划分，记：

[
\mathcal M_i
============

\mathcal M([u_i,u_{i+1}]).
]

由于：

[
\sum_i\mathcal M_i=\Omega,
]

必有：

[
\max_i\mathcal M_i
\geq
\frac{\Omega}{N}.
]

取达到最大 monitor mass 的区间 (I_j)，则：

[
\mathcal M_j
\geq
\frac{\Omega}{N}.
]

由局部近似：

[
D_j
\geq
\mathcal M_j^q
--------------

C_Eh_j^{q+1}.
]

又因为：

[
h_j
\leq
\frac{C_h}{N},
]

所以：

[
D_j
\geq
\left(\frac{\Omega}{N}\right)^q
-------------------------------

C_EC_h^{q+1}N^{-(q+1)}.
]

因此：

[
\max_iD_i
\geq
D_j
\geq
\left(\frac{\Omega}{N}\right)^q
-------------------------------

C_2N^{-(q+1)}.
]

证毕。

---

### 5.5 理论适用边界

该定理是局部渐近 minimax 定理，依赖：

[
h_{\max}=O(1/N).
]

实际 fast sampling 常关注：

[
N\in{4,8,10,15,20}.
]

在这种少步数区域，局部幂律近似：

[
D(u,u+h)\approx a(u)h^q
]

可能不准确，且 (O(N^{-(q+1)})) 修正项可能与主项：

[
\left(\frac{\Omega}{N}\right)^q
]

同量级。因此在少步数场景中，monitor 等分更适合作为 warm start 或可解释初始化，最终 schedule 应通过直接边际搜索或局部精化验证。

---

## 6. 估计误差与 inverse-CDF 稳定性

设真实累积函数为：

[
F(u)
====

\int_{u_{\min}}^u\omega(\xi)d\xi.
]

估计累积函数为：

[
\hat F(u)
=========

\int_{u_{\min}}^u\hat\omega(\xi)d\xi.
]

若：

[
\sup_u
|\hat F(u)-F(u)|
\leq
\varepsilon_F,
]

且：

[
\omega_{\min}
=============

\inf_u\omega(u)>0,
]

则 inverse-CDF 稳定性给出：

[
\boxed{
\max_m
|\hat u_m-u_m^*|
\leq
\frac{\varepsilon_F}{\omega_{\min}}
+
O(\varepsilon_F^2).
}
]

若 monitor 由 teacher oracle、有限校准集、finite-difference probe 和 quadrature 估计得到，则通常有：

[
\varepsilon_F
=============

O_p
\left(
\Delta_{\mathrm{ref}}^r
+
K^{-1/2}
+
\varepsilon_{\mathrm{fd}}
+
\varepsilon_{\mathrm{quad}}
\right).
]

其中：

[
\Delta_{\mathrm{ref}}
]

是 teacher oracle dense-output 网格间距；

[
r
]

是 oracle 插值阶数；

[
K
]

是校准轨迹数量；

[
\varepsilon_{\mathrm{fd}}
]

是 finite-difference probe 和 (q) 估计带来的误差；

[
\varepsilon_{\mathrm{quad}}
]

是数值积分误差。

若最终调度还需投影到 admissible grid (\mathcal A_{\mathcal S})，且该网格最大间距为 (\delta_{\mathcal A})，则额外产生：

[
O(\delta_{\mathcal A})
]

级别的 schedule perturbation。

---

## 7. 优化路径 A：monitor 等分

### 7.1 适用条件

路径 A 使用局部幂律模型：

[
D(u,u+h)
\approx
\bar a(u)h^q.
]

其理论保证主要适用于：

1. 单步 deterministic ODE solver；
2. 步数足够大或区间足够小的局部渐近区；
3. 有效阶数 (q) 在主要区间内稳定；
4. teacher oracle 足够精确；
5. (G)、(\rho)、CVaR 聚合方式在 held-out validation 上选定。

对少步数和 multistep solver，路径 A 仍可作为可解释 warm start，但不应单独声称全局最优。

---

### 7.2 算法

输入：

[
\mathcal S,
\quad
N,
\quad
{x_k^*(u)}_{k=1}^K,
\quad
G(u),
\quad
\rho,
\quad
\beta,
\quad
\epsilon_a.
]

步骤如下。

**第一步：构建 teacher oracle。**

用 high-accuracy teacher solver 在 dense grid 上求解：

[
x_k^*(u),
\qquad
k=1,\ldots,K.
]

并建立 dense-output 插值器。

---

**第二步：计算 oracle-start defect。**

在 probe grid (\mathcal P) 上，对每个 (u\in\mathcal P) 和 probe step (\eta)，计算：

[
d_k(u,u+\eta)
=============

\left|
\Phi_{\mathcal S}(x_k^*(u);u\to u+\eta)
---------------------------------------

x_k^*(u+\eta)
\right|_{\rho,k,u+\eta}^2.
]

---

**第三步：估计局部阶数和 coefficient。**

若使用全局 (q)，由多个 (\eta_j) 拟合：

[
\log d_k(u,u+\eta_j)
\approx
\log a_k(u)
+
q\log\eta_j.
]

得到：

[
\hat q.
]

然后估计：

[
a_k(u)
\approx
\frac{d_k(u,u+\eta)}{\eta^{\hat q}}.
]

---

**第四步：聚合不同轨迹。**

[
\bar a(u)
=========

\operatorname{CVaR}_{\beta,k}
[a_k(u)].
]

---

**第五步：构造 monitor。**

[
\omega(u)
=========

\bigl(\bar a(u)+\epsilon_a\bigr)^{1/\hat q}.
]

---

**第六步：计算累积时钟。**

[
\tau(u)
=======

\frac{
\int_{u_{\min}}^u\omega(\xi)d\xi
}{
\int_{u_{\min}}^{u_{\max}}\omega(\xi)d\xi
}.
]

---

**第七步：反演得到调度。**

[
u_m
===

\tau^{-1}
\left(
\frac{m}{N}
\right),
\qquad
m=0,\ldots,N.
]

---

**第八步：admissible snapping。**

若 solver 只允许某些 admissible 点：

[
\mathcal A_{\mathcal S}
=======================

{v_0<\cdots<v_M},
]

则将 (u_m) 投影到 (\mathcal A_{\mathcal S})，并修复可能出现的重复点，保持：

[
u_0<u_1<\cdots<u_N.
]

---


---

## 9. 多步 solver 的处理

### 9.1 多步 solver 的理论边界

对 (p)-步 multistep solver，当前更新一般依赖历史网格：

[
s_i=(u_{i-p+1},\ldots,u_i).
]

局部 defect 应写成：

[
d_k(s_i,u_{i+1}),
]

而不是：

[
d_k(u_i,u_{i+1}).
]

其 leading-order 形式通常为：

[
d_k(s_i,u_{i+1})
\approx
a_k(u_i;h_{i-p+1},\ldots,h_i)h_i^q.
]

因此局部误差系数依赖历史步长比。除非 mesh ratio 缓慢变化、历史比率近似固定，或 solver 退化为单步形式，否则不存在严格的 schedule-independent 一维 monitor：

[
a(u).
]

所以路径 A 的严格理论保证主要适用于单步 solver。对于 multistep solver，路径 A 可作为 warm start 或近似解释

---



## 10. 少步数采样策略

实际 fast sampling 常用：

[
N\in{4,8,10,15,20}.
]

在这种少步数区域：

1. 区间长度较大；
2. 局部幂律近似可能失效；
3. (q) 可能随 (u) 变化；
4. CFG、thresholding、guidance rescale 等非线性操作可能主导 defect；
5. multistep history-ratio 依赖更明显。



少步数实验中应至少比较：

1. uniform in (u)；
2. uniform in (\sigma)；
3. uniform in log-SNR；
4. Karras / EDM-style schedule；
5. 路径 A；



---




## 12. 实验协议

### 12.1 teacher oracle 稳定性

需要检查 teacher oracle 足够精确。建议至少比较：

[
\text{ref-NFE}=500,\ 1000,\ 2000
]

或等价高精度设置。

若不同 teacher 精度下得到的 (\omega(u)) 或最终 schedule 差异明显，说明 oracle 误差仍然影响方法，需要提高 teacher 精度或改进 dense-output 插值。

---

### 12.2 (\rho) 消融

报告：

[
\rho\in{0,0.05,0.1,0.3,1.0}.
]

主方法的 (\rho) 应通过 held-out validation 选择，而不是在 calibration set 上选择。

需要特别观察：

[
\rho=0
]

是否在低曲率或近直线轨迹段出现不稳定。

---

### 12.3 (q) 与 monitor 指数消融

需要比较：

[
\omega(u)=\bar a(u)^{1/q}
]

与错误指数形式：

[
\omega(u)=\bar a(u).
]

该实验用于验证 (q)-th root 对 schedule 形状的影响。

还应报告局部斜率：

[
\hat q(u)
=========

\frac{\partial \log D(u,u+\eta)}
{\partial \log \eta}.
]

若 (\hat q(u)) 明显变化，应比较：

1. 全局 (q)；
2. 局部 (q(u))；




### 12.5 local defect vs. propagated defect

比较：

[
d_k(a,b)
]

和：

[
\tilde d_k(a,b).
]

若传播加权版本明显更好，说明后续 flow sensitivity 对最终误差有重要影响，单纯 endpoint local defect 不足以描述最终样本误差。

---



### 12.7 指标报告

必须同时报告两类指标。

Teacher-alignment 指标：

[
|x_N^{\mathrm{student}}-x_N^{\mathrm{teacher}}|,
]

latent MSE-to-teacher，

[
\operatorname{LPIPS}
(x_N^{\mathrm{student}},
x_N^{\mathrm{teacher}}),
]

per-step oracle-start defect distribution。

Generation-quality 指标：

[
\operatorname{FID},
\qquad
\operatorname{CLIPScore},
\qquad
\operatorname{ImageReward/HPS},
\qquad
\operatorname{aesthetic\ score},
]

以及必要时的人类偏好评测。

若 teacher-distance 改善但 generation-quality 未改善，应将结论表述为 teacher-alignment schedule calibration，而不是 quality-optimal schedule optimization。

---


## 14. 核心数学链条

GPDE 的核心链条可以总结为：

[
\text{deterministic ODE sampler}
]

[
\Downarrow
]

[
\text{high-accuracy teacher oracle }x_k^*(u)
]

[
\Downarrow
]

[
\text{endpoint }G\text{-metric and }\rho\text{-mixed residual}
]

[
\Downarrow
]

[
\text{oracle-start local prediction defect }d_k(a,b)
]

[
\Downarrow
]

[
d_k(u,u+h)\approx a_k(u)h^q
]

[
\Downarrow
]

[
\bar a(u)
=========

\operatorname{CVaR}_{\beta,k}[a_k(u)]
]

[
\Downarrow
]

[
\boxed{
\omega(u)
=========

\bigl(\bar a(u)+\epsilon_a\bigr)^{1/q}
}
]

[
\Downarrow
]

[
\omega(u)du
\text{ is a coordinate-covariant monitor 1-form}
]

[
\Downarrow
]

[
\text{equalize }
\int\omega(u)du
\quad
\text{or solve discrete minimax DP}
]

[
\Downarrow
]

[
\text{training-free calibrated sampling schedule}.
]

---

## 15. 最终理论表述

可以将 GPDE 的理论贡献概括为：

**GPDE 是一个针对确定性 ODE sampler 的 training-free schedule calibration 框架。对单步 solver，在 oracle-start local defect 满足局部幂律模型且有效阶数 (q) 稳定的条件下，聚合 defect coefficient (\bar a(u)) 诱导出坐标协变的 monitor 1-form**

[
\omega(u)du
===========

\bar a(u)^{1/q}du.
]

**等分该 monitor mass 的调度在局部渐近区内达到 minimax-optimal，误差为 (O(N^{-(q+1)}))。对少步数和 multistep solver，monitor 路径主要作为可解释 warm start；最终调度应通过直接 edge defect 的 DP、augmented-state DP，或不可注入 history 时的黑盒 replay 优化获得。**

