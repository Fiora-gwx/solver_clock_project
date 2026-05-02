下面给出我认为最严谨、最优雅、最适合写成论文方法章节的最终版。它基于你现在这版 RI-SADB 的核心思想，但我做了三处关键修正：

1. 不再把融合叫 Fisher-Rao geodesic，而是叫 **KL-barycentric log-density fusion**，数学上更准确。
2. SADB residual 不只做弧长归一化，还做 **法向投影**，这样才能真正去掉参数化造成的切向污染。
3. 多条 pilot trajectory 不强行共享一个弧长坐标，而是每条轨迹先生成自己的 clock measure，再在公共物理坐标上聚合。

你上传的精炼版已经明确提出了原始 SADB defect 的参数化问题、弧长 pullback、曲率密度和最终 (\alpha_\eta(\sigma)=n_\eta(s(\sigma))|v(\sigma)|) 的结构；下面是我建议的最终严谨版。

---

# RI-SADB: Reparametrization-Invariant Defect-Balanced Clock

## 0. 核心思想

原始 SADB 的本质是：在某个物理坐标 (u) 上估计局部数值 defect，然后让采样节点向 defect 大的区域集中。

但是问题在于：
(u) 可以是 timestep、sigma、log-SNR、lambda。不同 (u) 描述的是同一条 denoising trajectory，但 (x'(u),x''(u),x^{(q)}(u)) 都会变化。因此，直接用 (u)-坐标下的 residual 或高阶导数定义 defect，会把“轨迹本身难走”和“参数化速度变化”混在一起。

RI-SADB 的核心改写是：

[
\boxed{
\text{把 timestep scheduling 变成在 denoising trajectory 的弧长测度上分配积分节点。}
}
]

最终 clock 不直接定义为某个坐标下的函数，而定义为一个 intrinsic measure：

[
\boxed{
d\tau = n_\eta(s),ds.
}
]

其中：

* (s) 是 trajectory arc length；
* (n_\eta(s)) 是单位弧长上的采样密度；
* (d\tau) 是真正的 clock measure。

如果最后需要在 (\sigma) 坐标下导出 schedule，只需要做 pullback：

[
\boxed{
\alpha_\eta(\sigma)
===================

# \frac{d\tau}{d\sigma}

n_\eta(s(\sigma))
\left|
\frac{dx}{d\sigma}
\right|.
}
]

这里的 (\left|\frac{dx}{d\sigma}\right|) 不是 defect，而是坐标变换的 Jacobian。

---

# 1. 问题设定

设 diffusion denoising trajectory 是一条正则曲线：

[
x : I \to \mathbb{R}^d,
\qquad
u \mapsto x(u),
]

其中 (u) 是任意单调采样坐标，例如：

[
u\in{\sigma,t,\lambda}.
]

假设：

[
\left|
\frac{dx}{du}
\right|>0.
]

定义速度和加速度：

[
v(u)=\frac{dx}{du},
\qquad
a(u)=\frac{d^2x}{du^2}.
]

原始 SADB 类方法通常使用 refinement residual：

[
r_i
===

## x_{i+1}^{\mathrm{coarse}}

x_{i+1}^{\mathrm{fine}},
]

并构造类似：

[
D_i^u
=====

\frac{|r_i|}
{|\Delta u_i|^{q_i}}.
]

这个量的问题是：它依赖 (u)。如果换一个严格单调重参数化：

[
\tilde u=\phi(u),
]

同一条几何轨迹没有变，但 (\Delta \tilde u_i)、(\frac{dx}{d\tilde u})、(\frac{d^2x}{d\tilde u^2}) 都会变，导致 defect 也变。

RI-SADB 的目标是构造一个 clock measure：

[
d\tau
]

使其满足：

[
\boxed{
d\tau \text{ 不依赖于选择 } u=\sigma,t,\lambda \text{ 中的哪一个作为参数。}
}
]

---

# 2. 弧长参数化：把“速度”与“形状”分离

定义弧长：

[
s(u)
====

\int_{u_0}^{u}
\left|
\frac{dx}{d\xi}
\right|
d\xi.
]

因此：

[
\frac{ds}{du}
=============

|v(u)|.
]

令：

[
x(u)=\gamma(s(u)).
]

则 (\gamma) 是弧长参数化的轨迹，满足：

[
\left|
\frac{d\gamma}{ds}
\right|
=1.
]

定义单位切向量：

[
T(s)=\frac{d\gamma}{ds}.
]

在任意参数 (u) 下：

[
v(u)
====

# \frac{dx}{du}

\frac{d\gamma}{ds}
\frac{ds}{du}
=============

T(s(u)),s'(u).
]

也就是：

[
v = |v|T.
]

加速度为：

[
a(u)
====

\frac{d}{du}
\left(
T(s(u))s'(u)
\right).
]

展开：

[
a(u)
====

\frac{dT}{ds}(s'(u))^2
+
T(s(u))s''(u).
]

这给出 Frenet 分解：

[
a=a_\perp+a_\parallel.
]

其中：

[
a_\parallel
===========

T,s''(u)
]

是切向项，只改变 traversal speed；

[
a_\perp
=======

\frac{dT}{ds}(s'(u))^2
]

是法向项，表示轨迹真的在弯。

用投影公式写：

[
\boxed{
a_\perp
=======

## a

\frac{\langle a,v\rangle}{|v|^2}v.
}
]

曲率定义为：

[
\boxed{
\kappa(u)
=========

\frac{|a_\perp(u)|}{|v(u)|^2}.
}
]

因为：

[
|v(u)|^2=(s'(u))^2,
]

而：

[
|a_\perp(u)|
============

\left|
\frac{dT}{ds}
\right|
(s'(u))^2,
]

所以：

[
\kappa(u)
=========

\left|
\frac{dT}{ds}
\right|
=======

\left|
\frac{d^2\gamma}{ds^2}
\right|.
]

因此：

[
\boxed{
\kappa \text{ 是轨迹的内禀几何量，不依赖于参数化。}
}
]

---

# 3. 几何密度 (n_G(s))：为什么是 (\sqrt{\kappa})

RI-SADB 的几何部分想回答：

> 如果只看轨迹几何，哪些弧长区域需要更多采样点？

答案是：

[
n_G(s)\propto \sqrt{\kappa(s)}.
]

这个结论可以从两条独立路径推出。

---

## 3.1 路径一：等几何弦误差原则

在弧长参数下，对 (\gamma(s+h)) 做 Taylor 展开：

[
\gamma(s+h)
===========

\gamma(s)
+
h\gamma'(s)
+
\frac{h^2}{2}\gamma''(s)
+
O(h^3).
]

因为：

[
\gamma'(s)=T(s),
]

[
|\gamma''(s)|=\kappa(s),
]

所以用直线段近似轨迹时的局部几何偏差为：

[
\left|
\gamma(s+h)
-----------

## \gamma(s)

hT(s)
\right|
=======

\frac{h^2}{2}\kappa(s)
+
O(h^3).
]

若希望每个 interval 的几何误差近似相等：

[
\kappa(s_i)h_i^2
================

\text{const},
]

则：

[
h_i
\propto
\kappa(s_i)^{-1/2}.
]

单位弧长上的节点密度是步长的倒数：

[
n_G(s_i)
\propto
\frac{1}{h_i}
\propto
\sqrt{\kappa(s_i)}.
]

因此：

[
\boxed{
n_G(s)\propto \sqrt{\kappa(s)}.
}
]

---

## 3.2 路径二：TOTR 风格的法向加速度最小化

设推理 clock 为 (\tau)，弧长 traversal speed 为：

[
w(s)=\frac{ds}{d\tau}.
]

则：

[
\frac{dx}{d\tau}
================

\frac{d\gamma}{ds}
\frac{ds}{d\tau}
================

w(s)T(s).
]

二阶导为：

[
\frac{d^2x}{d\tau^2}
====================

\frac{d}{d\tau}
(wT).
]

因为：

[
\frac{d}{d\tau}
===============

# \frac{ds}{d\tau}\frac{d}{ds}

w\frac{d}{ds},
]

所以：

[
\frac{d^2x}{d\tau^2}
====================

# w\frac{d}{ds}(wT)

w w_s T
+
w^2 \frac{dT}{ds}.
]

其中：

[
w w_s T
]

是切向加速度；

[
w^2\frac{dT}{ds}
================

\kappa(s)w^2N
]

是法向加速度。

RI-SADB 只优化法向加速度，因为切向加速度对应 traversal speed 的变化，而不是轨迹几何弯曲。

定义法向加速度能量：

[
\mathcal{A}_\perp[w]
====================

\int
\left|
\kappa(s(\tau))w(s(\tau))^2N
\right|^2
d\tau.
]

即：

[
\mathcal{A}_\perp[w]
====================

\int
\kappa(s(\tau))^2w(s(\tau))^4
d\tau.
]

由于：

[
d\tau=\frac{ds}{w(s)},
]

得到：

[
\mathcal{A}_\perp[w]
====================

\int_0^L
\kappa(s)^2w(s)^3
ds.
]

总推理时间归一化为：

[
\int_0^L
\frac{1}{w(s)}
ds
==

1.

]

构造拉格朗日泛函：

[
\mathcal{L}[w]
==============

\int_0^L
\left[
\kappa(s)^2w(s)^3
+
\lambda w(s)^{-1}
\right]
ds.
]

对 (w) 求变分：

[
\frac{\delta \mathcal{L}}{\delta w}
===================================

## 3\kappa(s)^2w(s)^2

# \lambda w(s)^{-2}

0.

]

因此：

[
3\kappa(s)^2w(s)^4=\lambda.
]

于是：

[
w(s)
\propto
\kappa(s)^{-1/2}.
]

而 clock density 是：

[
n_G(s)
======

# \frac{d\tau}{ds}

\frac{1}{w(s)}.
]

所以：

[
\boxed{
n_G(s)\propto \sqrt{\kappa(s)}.
}
]

这和等几何弦误差原则得到完全相同的结果。

---

## 3.3 带 floor 的完整几何权重

纯 (\sqrt{\kappa}) 在直线段 (\kappa=0) 处会变成 0，这不适合作为采样密度。直线段虽然没有曲率，但仍然需要沿弧长覆盖。

因此引入参考弧长尺度：

[
\ell>0.
]

定义 dimensionless curvature：

[
\ell\kappa(s).
]

最终几何权重取：

[
\boxed{
w_G(s)
======

\left(
1+
(\ell\kappa(s))^2
\right)^{1/4}.
}
]

其极限性质是：

当：

[
\ell\kappa(s)\ll 1,
]

有：

[
w_G(s)\approx 1.
]

这对应 uniform arc-length clock。

当：

[
\ell\kappa(s)\gg 1,
]

有：

[
w_G(s)
\approx
(\ell\kappa(s))^{1/2}.
]

这恢复：

[
n_G(s)\propto \sqrt{\kappa(s)}.
]

自然尺度选择为：

[
\boxed{
\ell=\frac{L}{N},
}
]

其中 (L) 是轨迹总弧长，(N) 是目标 sampling steps。此时：

[
\ell\kappa(s)\approx 1
]

表示：

[
\text{局部曲率半径} \approx \text{平均采样步长}.
]

这很自然：当平均步长已经接近曲率半径时，该区域需要更多节点。

归一化得到几何密度：

[
\boxed{
n_G(s)
======

\frac{
w_G(s)
}{
\int_0^L w_G(r),dr
}.
}
]

---

# 4. Solver-aware defect：弧长归一化 + 法向投影

纯几何密度只看轨迹是否弯曲，但 diffusion solver 的误差还包括：

* vector field stiffness；
* score field 的时间变化；
* 多步 solver 的 history error；
* CFG 导致的非线性放大；
* SDE solver 的 stochastic term。

所以不能完全丢掉 SADB 的 refinement defect。

但原始 residual：

[
r_i
===

## x_{i+1}^{\mathrm{coarse}}

x_{i+1}^{\mathrm{fine}}
]

也包含参数化污染。尤其是切向 residual。切向 residual 很多时候只是表示“同一条轨迹上走快或走慢”，不一定代表几何上偏离 trajectory。

因此我们对 residual 做切向/法向分解。

---

## 4.1 Residual 的法向投影

对第 (i) 个区间，定义单位切向量：

[
T_i
===

\frac{v_i}{|v_i|}.
]

将 residual 分解为：

[
r_i=r_{i,\parallel}+r_{i,\perp}.
]

其中：

[
r_{i,\parallel}
===============

\langle r_i,T_i\rangle T_i,
]

[
\boxed{
r_{i,\perp}
===========

## r_i

\langle r_i,T_i\rangle T_i.
}
]

严格的 reparametrization-invariant geometric residual 使用：

[
|r_{i,\perp}|.
]

但是，完全丢弃切向 residual 可能会损失一些 solver-specific 信息。因此定义一个带权 residual：

[
\boxed{
R_{i,\beta}
===========

\sqrt{
|r_{i,\perp}|^2
+
\beta|r_{i,\parallel}|^2
},
\qquad
0\le \beta\le 1.
}
]

其中：

* (\beta=0)：严格几何版本，只保留法向 residual；
* (\beta=1)：保留完整 residual，但做弧长归一化；
* (0<\beta<1)：削弱切向参数化污染，同时保留部分 solver-specific 信息。

我建议理论主版本使用：

[
\beta=0
]

作为 clean RI-SADB；实验里做 (\beta) ablation。

---

## 4.2 弧长归一化 defect

设区间弧长为：

[
\Delta s_i
==========

\int_{u_i}^{u_{i+1}}
\left|
\frac{dx}{du}
\right|
du.
]

若 solver refinement residual 的 leading order 为：

[
R_{i,\beta}
\approx
C_i^s(\Delta s_i)^{q_i},
]

则 arc-length defect strength 定义为：

[
\boxed{
D_{i,\beta}^s
=============

\frac{
R_{i,\beta}
}{
(\Delta s_i)^{q_i},\rho(q_i)
}.
}
]

其中 (\rho(q_i)) 是 Richardson refinement factor。对于 full/half comparison，若 leading error 为 (h^{q_i})，常见形式为：

[
\rho(q_i)
=========

\left|1-2^{1-q_i}\right|.
]

因此：

[
D_{i,\beta}^s
]

表示“每单位弧长上的 solver residual strength”。

---

# 5. 从 arc-length defect 推导 SADB density

现在我们在弧长域 ([0,L]) 上分配 (N) 个采样步。

令：

[
n(s)=\frac{d\tau}{ds}
]

是单位弧长上的归一化节点密度，满足：

[
\int_0^L n(s),ds=1.
]

若总步数为 (N)，那么局部弧长步长近似为：

[
h(s)
\approx
\frac{1}{N n(s)}.
]

假设局部 solver error per step 为：

[
e_D(s)
\approx
D^s(s)h(s)^{q(s)}.
]

在弧长微元 (ds) 中，interval 数量约为：

[
N n(s),ds.
]

因此总 defect contribution 为：

[
d\mathcal{E}_D
\approx
N n(s)D^s(s)
\left(
\frac{1}{N n(s)}
\right)^{q(s)}
ds.
]

忽略与优化无关的 (N^{1-q})，得到目标：

[
\mathcal{E}_D[n]
================

\int_0^L
D^s(s)n(s)^{1-q(s)}
ds.
]

求解：

[
\min_{n>0}
\int_0^L
D^s(s)n(s)^{1-q(s)}
ds
\quad
\text{s.t.}
\quad
\int_0^L n(s)ds=1.
]

构造拉格朗日泛函：

[
\mathcal{L}[n]
==============

\int_0^L
D^s(s)n(s)^{1-q(s)}
ds
+
\lambda
\left(
\int_0^L n(s)ds-1
\right).
]

若暂时把 (q(s)) 在局部视为常数，变分条件为：

[
(1-q)D^s(s)n(s)^{-q}
+
\lambda
=======

0.

]

所以：

[
n(s)^q
\propto
(q-1)D^s(s).
]

因此：

[
\boxed{
n_D(s)
\propto
\left(
(q(s)-1)D^s(s)
\right)^{1/q(s)}.
}
]

这和现有 SADB 的形式保持一致，只是把 defect 从 (u)-坐标改到了弧长坐标。当前代码中 `build_defect_balanced_profile` 的 interval alpha 也正是类似：

[
\alpha_i
========

\exp
\left(
\frac{\log(q_i-1)+\log D_i}{q_i}
\right)
=======

\left((q_i-1)D_i\right)^{1/q_i}
]

的结构。

于是定义 solver-aware 权重：

[
\boxed{
w_D(s_i)
========

\left(
(q_i-1)D_{i,\beta}^s
\right)^{1/q_i}.
}
]

归一化：

[
\boxed{
n_D(s)
======

\frac{
w_D(s)
}{
\int_0^L w_D(r),dr
}.
}
]

---

# 6. 两个密度的融合：KL-barycentric log-density fusion

现在我们有两个定义在同一弧长域上的密度：

[
n_D(s)
]

和：

[
n_G(s).
]

其中：

* (n_D)：solver-aware density，来自 arc-length normalized projected residual；
* (n_G)：geometry-aware density，来自 curvature / normal acceleration principle。

最终 density 定义为：

[
\boxed{
n_\eta(s)
=========

\frac{
n_D(s)^{1-\eta}
n_G(s)^\eta
}{
\int_0^L
n_D(r)^{1-\eta}
n_G(r)^\eta
dr
},
\qquad
\eta\in[0,1].
}
]

这里我建议称为：

[
\boxed{
\text{KL-barycentric log-density fusion}
}
]

而不是 Fisher-Rao geodesic。

原因是：这个 normalized geometric mean 精确对应下面的 KL barycenter 问题：

[
n_\eta
======

\arg\min_{n}
\left[
(1-\eta)\operatorname{KL}(n|n_D)
+
\eta\operatorname{KL}(n|n_G)
\right],
]

其中：

[
\operatorname{KL}(n|m)
======================

\int_0^L
n(s)\log\frac{n(s)}{m(s)}
ds.
]

证明如下。

构造：

[
\mathcal{J}[n]
==============

(1-\eta)
\int n\log\frac{n}{n_D}ds
+
\eta
\int n\log\frac{n}{n_G}ds
+
\lambda
\left(
\int n ds-1
\right).
]

对 (n) 求变分：

[
\frac{\delta \mathcal{J}}{\delta n}
===================================

(1-\eta)
\left(
\log n-\log n_D+1
\right)
+
\eta
\left(
\log n-\log n_G+1
\right)
+
\lambda.
]

整理：

[
\log n
======

(1-\eta)\log n_D
+
\eta\log n_G
+
\text{const}.
]

因此：

[
n(s)
\propto
n_D(s)^{1-\eta}
n_G(s)^\eta.
]

归一化后就是：

[
n_\eta(s)
=========

\frac{
n_D(s)^{1-\eta}
n_G(s)^\eta
}{
\int n_D^{1-\eta}n_G^\eta
}.
]

这说明最终融合不是 heuristic，而是两个 density 的 KL barycenter。

---

# 7. Pullback 到物理坐标并生成 schedule

RI-SADB 的 intrinsic clock measure 是：

[
d\tau
=====

n_\eta(s)ds.
]

由于：

[
ds
==

\left|
\frac{dx}{du}
\right|
du,
]

所以在任意物理坐标 (u) 下：

[
d\tau
=====

n_\eta(s(u))
\left|
\frac{dx}{du}
\right|
du.
]

定义物理坐标下的 clock density：

[
\boxed{
\alpha_\eta(u)
==============

n_\eta(s(u))
\left|
\frac{dx}{du}
\right|.
}
]

若 (u=\sigma)，则：

[
\boxed{
\alpha_\eta(\sigma)
===================

n_\eta(s(\sigma))
\left|
\frac{dx}{d\sigma}
\right|.
}
]

然后：

[
\tau(u)
=======

\frac{
\int_{u_0}^{u}
\alpha_\eta(\xi)d\xi
}{
\int_{u_0}^{u_1}
\alpha_\eta(\xi)d\xi
}.
]

对 (N) 步推理，均匀取：

[
\tau_j=\frac{j}{N},
\qquad
j=0,\dots,N.
]

反插值得到：

[
\boxed{
u_j
===

\tau^{-1}(\tau_j).
}
]

若 (u=\sigma)，得到：

[
\sigma_j
========

\tau^{-1}\left(\frac{j}{N}\right).
]

---

# 8. Reparametrization invariance 定理

设有另一个参数：

[
\tilde u=\phi(u),
]

其中 (\phi) 严格单调。则：

[
\frac{dx}{d\tilde u}
====================

\frac{dx}{du}
\frac{du}{d\tilde u}.
]

因此：

[
\left|
\frac{dx}{d\tilde u}
\right|
d\tilde u
=========

\left|
\frac{dx}{du}
\right|
du
==

ds.
]

由于 (n_\eta(s)) 是定义在弧长 (s) 上的 density，所以：

[
d\tilde\tau
===========

n_\eta(s(\tilde u))
\left|
\frac{dx}{d\tilde u}
\right|
d\tilde u.
]

代入上式：

[
d\tilde\tau
===========

n_\eta(s)
ds.
]

而原参数 (u) 下：

[
d\tau
=====

n_\eta(s(u))
\left|
\frac{dx}{du}
\right|
du
==

n_\eta(s)ds.
]

因此：

[
\boxed{
d\tilde\tau=d\tau.
}
]

也即：

[
\boxed{
\tilde\alpha_\eta(\tilde u)d\tilde u
====================================

\alpha_\eta(u)du.
}
]

这才是严格意义上的 reparametrization invariance。
注意：不是说 (\alpha_\eta(u)) 这个函数本身不变，而是说 clock measure 不变。

---

# 9. 多条 pilot trajectory 的最终形式

实际 calibration 中不是一条轨迹，而是多条 pilot trajectories：

[
x^{(k)}(u),
\qquad
k=1,\dots,K.
]

不同 trajectory 有不同弧长：

[
s_k(u)
======

\int_{u_0}^{u}
\left|
\frac{dx^{(k)}}{d\xi}
\right|
d\xi.
]

因此不应该强行构造一个共享的 (s)。更严谨的做法是：每条 trajectory 先生成自己的 intrinsic clock measure，再 pullback 到公共坐标 (u)，最后聚合这些 measures。

---

## 9.1 单条 trajectory 的 clock density

对第 (k) 条 trajectory，重复上述过程，得到：

[
n_{\eta,k}(s_k).
]

其 pullback density 是：

[
\alpha_{\eta,k}(u)
==================

n_{\eta,k}(s_k(u))
\left|
\frac{dx^{(k)}}{du}
\right|.
]

对应 measure：

[
d\tau_k
=======

\alpha_{\eta,k}(u)du.
]

---

## 9.2 聚合多个 clock measures

定义 ensemble clock measure：

[
d\bar\tau
=========

\frac{1}{K}
\sum_{k=1}^K d\tau_k.
]

于是：

[
d\bar\tau
=========

\left[
\frac{1}{K}
\sum_{k=1}^K
\alpha_{\eta,k}(u)
\right]du.
]

所以最终 ensemble density 为：

[
\boxed{
\bar\alpha_\eta(u)
==================

\frac{1}{K}
\sum_{k=1}^K
n_{\eta,k}(s_k(u))
\left|
\frac{dx^{(k)}}{du}
\right|.
}
]

然后归一化：

[
\bar\tau(u)
===========

\frac{
\int_{u_0}^{u}
\bar\alpha_\eta(\xi)d\xi
}{
\int_{u_0}^{u_1}
\bar\alpha_\eta(\xi)d\xi
}.
]

最终 schedule：

[
\boxed{
u_j
===

\bar\tau^{-1}
\left(
\frac{j}{N}
\right).
}
]

这是多 trajectory 版本的严格形式。

---

# 10. RI-SADB 的完整算法公式

给定第 (k) 条 pilot trajectory：

[
x_i^{(k)}=x^{(k)}(u_i),
\qquad
i=0,\dots,M.
]

---

## Step 1：估计弧长增量

[
\Delta s_{i,k}
==============

\int_{u_i}^{u_{i+1}}
\left|
\frac{dx^{(k)}}{du}
\right|du.
]

离散近似：

[
\Delta s_{i,k}
\approx
\frac{1}{2}
\left(
|v_{i,k}|+|v_{i+1,k}|
\right)
|\Delta u_i|.
]

总弧长：

[
L_k=\sum_i \Delta s_{i,k}.
]

---

## Step 2：估计曲率

速度：

[
v_{i,k}
=======

\frac{dx^{(k)}}{du}(u_i).
]

加速度：

[
a_{i,k}
=======

\frac{d^2x^{(k)}}{du^2}(u_i).
]

法向投影：

[
a_{\perp,i,k}
=============

## a_{i,k}

\frac{
\langle a_{i,k},v_{i,k}\rangle
}{
|v_{i,k}|^2+\epsilon
}
v_{i,k}.
]

曲率：

[
\boxed{
\kappa_{i,k}
============

\frac{
|a_{\perp,i,k}|
}{
|v_{i,k}|^2+\epsilon
}.
}
]

区间曲率可取节点平均：

[
\kappa_{i+\frac12,k}
====================

\frac{1}{2}
(\kappa_{i,k}+\kappa_{i+1,k}).
]

---

## Step 3：构造几何权重

选择参考尺度：

[
\ell_k=\frac{L_k}{N}.
]

定义：

[
\boxed{
w_{G,i,k}
=========

\left(
1+
(\ell_k\kappa_{i+\frac12,k})^2
\right)^{1/4}.
}
]

归一化：

[
\boxed{
n_{G,i,k}
=========

\frac{
w_{G,i,k}
}{
\sum_j w_{G,j,k}\Delta s_{j,k}
}.
}
]

---

## Step 4：构造法向投影 refinement defect

设 refinement residual 为：

[
r_{i,k}
=======

## x_{i+1,k}^{\mathrm{coarse}}

x_{i+1,k}^{\mathrm{fine}}.
]

切向单位向量：

[
T_{i,k}
=======

\frac{
v_{i,k}
}{
|v_{i,k}|+\epsilon
}.
]

切向 residual：

[
r_{\parallel,i,k}
=================

\langle r_{i,k},T_{i,k}\rangle T_{i,k}.
]

法向 residual：

[
r_{\perp,i,k}
=============

## r_{i,k}

r_{\parallel,i,k}.
]

带权 residual：

[
\boxed{
R_{i,k,\beta}
=============

\sqrt{
|r_{\perp,i,k}|^2
+
\beta|r_{\parallel,i,k}|^2
}.
}
]

arc-length defect：

[
\boxed{
D_{i,k,\beta}^{s}
=================

\frac{
R_{i,k,\beta}
}{
(\Delta s_{i,k}+\epsilon)^{q_{i,k}}
,
\rho(q_{i,k})
}.
}
]

其中：

[
\rho(q)
=======

|1-2^{1-q}|.
]

SADB 权重：

[
\boxed{
w_{D,i,k}
=========

\left(
(q_{i,k}-1)
D_{i,k,\beta}^{s}
\right)^{1/q_{i,k}}.
}
]

归一化：

[
\boxed{
n_{D,i,k}
=========

\frac{
w_{D,i,k}
}{
\sum_j w_{D,j,k}\Delta s_{j,k}
}.
}
]

---

## Step 5：KL-barycentric fusion

[
\boxed{
n_{\eta,i,k}
============

\frac{
n_{D,i,k}^{1-\eta}
n_{G,i,k}^{\eta}
}{
\sum_j
n_{D,j,k}^{1-\eta}
n_{G,j,k}^{\eta}
\Delta s_{j,k}
}.
}
]

其中：

[
\eta\in[0,1].
]

---

## Step 6：pullback 到物理坐标

区间速度 Jacobian：

[
J_{i,k}
=======

\frac{\Delta s_{i,k}}{|\Delta u_i|+\epsilon}.
]

单轨迹 clock density：

[
\boxed{
\alpha_{\eta,i,k}
=================

n_{\eta,i,k}J_{i,k}.
}
]

多轨迹聚合：

[
\boxed{
\bar\alpha_{\eta,i}
===================

\frac{1}{K}
\sum_{k=1}^{K}
\alpha_{\eta,i,k}.
}
]

最后构造：

[
\bar\tau_i
==========

\frac{
\sum_{j<i}
\bar\alpha_{\eta,j}|\Delta u_j|
}{
\sum_j
\bar\alpha_{\eta,j}|\Delta u_j|
}.
]

均匀采样：

[
\tau_m=\frac{m}{N}.
]

反插值：

[
\boxed{
u_m=\bar\tau^{-1}\left(\frac{m}{N}\right).
}
]

这就是最终 RI-SADB schedule。

---

# 11. 特殊情形

## 11.1 (\eta=0)：Arc-length SADB

当：

[
\eta=0,
]

有：

[
n_\eta=n_D.
]

这时 RI-SADB 退化为：

[
\boxed{
\text{arc-length normalized, normal-projected SADB}.
}
]

它仍然是 solver-aware 的，但已经去掉了原始 SADB 中最明显的参数化污染。

---

## 11.2 (\eta=1)：Pure geometric clock

当：

[
\eta=1,
]

有：

[
n_\eta=n_G.
]

于是：

[
\alpha_1(u)
===========

n_G(s(u))
\left|
\frac{dx}{du}
\right|.
]

若处于大曲率区域：

[
\ell\kappa\gg1,
]

则：

[
n_G(s)
\propto
\sqrt{\kappa(s)}.
]

因此：

[
\alpha_1(u)
\propto
\sqrt{\kappa(u)}
|v(u)|.
]

又因为：

[
\kappa(u)
=========

\frac{|a_\perp(u)|}{|v(u)|^2},
]

所以：

[
\alpha_1(u)
\propto
\sqrt{
\frac{|a_\perp(u)|}{|v(u)|^2}
}
|v(u)|.
]

得到：

[
\boxed{
\alpha_1(u)
\propto
\sqrt{|a_\perp(u)|}.
}
]

这说明在大曲率极限下，几何 clock 只看法向加速度，而不受切向速度变化污染。

---

## 11.3 (\beta=0)：严格几何 residual

当：

[
\beta=0,
]

有：

[
R_{i,k,\beta}
=============

|r_{\perp,i,k}|.
]

此时 solver defect 中的切向参数化污染被完全移除。

---

## 11.4 (\beta=1)：保留完整 solver residual

当：

[
\beta=1,
]

有：

[
R_{i,k,\beta}
=============

|r_{i,k}|.
]

这更接近原始 SADB，但仍然使用：

[
\Delta s_i
]

而不是：

[
\Delta u_i.
]

所以它是 arc-length normalized SADB。

---

# 12. 最终理论性质

## 性质 1：严格 clock-measure invariance

对任意单调重参数化：

[
\tilde u=\phi(u),
]

有：

[
\boxed{
\tilde\alpha_\eta(\tilde u)d\tilde u
====================================

\alpha_\eta(u)du.
}
]

因此最终采样节点由 intrinsic measure 决定，而不是由具体选择的物理坐标决定。

---

## 性质 2：几何密度有双重推导

[
n_G(s)
\propto
\sqrt{\kappa(s)}
]

同时来自：

1. 等几何弦误差原则；
2. TOTR-style normal acceleration relaxation。

这说明曲率密度不是经验设计，而是几何误差控制的自然结果。

---

## 性质 3：保留 solver-aware 信息

RI-SADB 不是用曲率替代 SADB，而是：

[
\boxed{
\text{RI-SADB}
==============

\text{arc-length normalized solver defect}
+
\text{intrinsic curvature density}.
}
]

具体通过：

[
n_\eta
\propto
n_D^{1-\eta}
n_G^\eta
]

融合。

---

## 性质 4：融合是 KL barycenter

最终 density 满足：

[
n_\eta
======

\arg\min_n
\left[
(1-\eta)\operatorname{KL}(n|n_D)
+
\eta\operatorname{KL}(n|n_G)
\right].
]

因此不是简单 heuristic 加权，而是信息几何意义下的 log-density barycenter。

---

# 13. 最终一句话定义

可以把方法定义成：

[
\boxed{
\textbf{RI-SADB constructs a reparametrization-invariant clock measure on the arc-length domain of denoising trajectories by fusing a normal-projected, arc-length-normalized solver defect density with an intrinsic Frenet-curvature density through KL-barycentric log-density interpolation.}
}
]

中文：

[
\boxed{
\text{RI-SADB 将 diffusion timestep scheduling 重新表述为在 denoising trajectory 的内禀弧长域上分配积分节点；它将采样难度分解为法向投影的弧长归一化 solver defect 和 Frenet 曲率诱导的几何密度，并通过 KL-barycentric log-density 融合得到严格参数化不变的 clock measure。}
}
]

---

# 14. 最简核心公式版

最终方法可以压缩成下面 6 个公式。

第一，弧长：

[
ds=|x'(u)|du.
]

第二，曲率：

[
\kappa
======

\frac{
\left|
a-
\frac{\langle a,v\rangle}{|v|^2}v
\right|
}{
|v|^2
}.
]

第三，几何密度：

[
n_G(s)
\propto
\left(
1+(\ell\kappa(s))^2
\right)^{1/4}.
]

第四，法向投影 SADB defect：

[
D_{i,\beta}^s
=============

\frac{
\sqrt{
|r_{i,\perp}|^2+\beta|r_{i,\parallel}|^2
}
}{
(\Delta s_i)^{q_i}\rho(q_i)
}.
]

第五，solver defect density：

[
n_D(s_i)
\propto
\left(
(q_i-1)D_{i,\beta}^s
\right)^{1/q_i}.
]

第六，最终 RI-SADB clock：

[
\boxed{
d\tau
=====

n_\eta(s)ds,
\qquad
n_\eta(s)
\propto
n_D(s)^{1-\eta}n_G(s)^\eta.
}
]

拉回任意坐标 (u)：

[
\boxed{
\alpha_\eta(u)
==============

# \frac{d\tau}{du}

n_\eta(s(u))|x'(u)|.
}
]

这就是最终严谨优雅版。
