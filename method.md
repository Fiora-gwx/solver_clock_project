
---

# D-GPDE: Distributional Geometric Prediction Defect Equalization

## 1. Problem Setup

We consider a deterministic diffusion ODE sampler:

[
\frac{dx}{du}=f_\theta(x,u),
\qquad
u\in[u_{\min},u_{\max}],
]

where (u) denotes a monotone sampling coordinate, such as time, noise level, log-SNR, or any smooth monotone reparameterization thereof.

Let the exact teacher flow be:

[
F_{a\to b}(x),
]

defined by the high-accuracy integration of the ODE from (a) to (b). Given an initial distribution

[
x_0\sim p_{u_{\min}},
]

the teacher marginal distribution at coordinate (u) is:

[
\boxed{
p_u=(F_{u_{\min}\to u})*#p*{u_{\min}}.
}
]

That is, (p_u) is the pushforward of the initial distribution under the teacher ODE flow.

We are given a fixed student solver (\mathcal S). For a single step from (u) to (v), the student update is denoted by:

[
\Phi_{\mathcal S}(x;u\to v).
]

The goal is to choose an (N)-step sampling schedule:

[
U=(u_0<u_1<\cdots<u_N),
\qquad
u_0=u_{\min},
\qquad
u_N=u_{\max},
]

such that the student solver approximates the teacher ODE sampling process as well as possible, without retraining the model.

---

# 2. Teacher Calibration Trajectories

In practice, we draw (K) calibration initial states:

[
x_{k,0}\sim p_{u_{\min}},
\qquad
k=1,\ldots,K.
]

For each (k), we compute a high-accuracy teacher trajectory:

[
x_k^*(u)
========

F_{u_{\min}\to u}(x_{k,0}).
]

Thus, for every (u),

[
x_k^*(u)\sim p_u,
\qquad
k=1,\ldots,K.
]

The calibration states

[
{x_k^*(u)}_{k=1}^K
]

are therefore Monte Carlo samples from the teacher marginal distribution (p_u).

This is the key distributional upgrade of D-GPDE:

[
\boxed{
\text{The calibration trajectories are not merely individual oracle paths; they sample }p_u.
}
]

---

# 3. Geometry: (G)-Metric and (\rho)-Mixed Residual

## 3.1 State-space metric

We introduce a positive definite state-space metric:

[
G(u)\succ0.
]

For any residual (r), define:

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

The role of (G(u)) is to normalize state-space errors across noise levels, channels, and parameterizations.

Common choices include:

[
G(u)=\frac{1}{\sigma(u)^2+\sigma_{\mathrm{data}}^2}I,
]

or channel-wise whitening:

[
G(u)=
\operatorname{diag}
\left(
\hat \sigma_1(u)^{-2},
\ldots,
\hat \sigma_d(u)^{-2}
\right).
]

---

## 3.2 (G)-unit tangent direction

For each teacher trajectory, define the (G)-unit tangent vector:

[
T_{k,G}(u)
==========

\frac{\partial_u x_k^*(u)}
{|\partial_u x_k^*(u)|_{G(u)}}.
]

Since (x_k^*(u)) solves the ODE,

[
\partial_u x_k^*(u)
===================

f_\theta(x_k^*(u),u).
]

Thus in practice:

[
T_{k,G}(u)
==========

\frac{f_\theta(x_k^*(u),u)}
{|f_\theta(x_k^*(u),u)|_{G(u)}}.
]

If the tangent norm is numerically unstable, one may fall back to the full residual norm by setting (\rho=1).

---

## 3.3 (\rho)-mixed residual norm

For a residual (r) measured at endpoint (b), define:

[
\boxed{
|r|_{\rho,k,b}^2
================

## r^\top G(b)r

(1-\rho)
\left(
T_{k,G}(b)^\top G(b)r
\right)^2.
}
]

Here:

[
\rho\in(0,1].
]

Equivalently, decompose (r) into (G(b))-orthogonal tangent and normal components:

[
r=r_\parallel^{G(b)}+r_\perp^{G(b)}.
]

Then:

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

Thus:

* (\rho=1): full residual;
* (0<\rho<1): downweight tangent residual;
* (\rho=0): pure normal residual, usually only for ablation.

The (\rho)-mixed norm is retained in D-GPDE as a low-cost geometric correction.

---

# 4. Oracle-Start Local Prediction Defect

For (x\sim p_u), define the oracle-start local defect of the student solver over ([u,v]):

[
\boxed{
d_{\mathcal S}(x;u,v)
=====================

\left|
\Phi_{\mathcal S}(x;u\to v)
---------------------------

F_{u\to v}(x)
\right|_{\rho,u,v}^{2}.
}
]

For calibration trajectory (k), this becomes:

[
\boxed{
d_k(u,v)
========

\left|
\Phi_{\mathcal S}(x_k^*(u);u\to v)
----------------------------------

x_k^*(v)
\right|_{\rho,k,v}^{2}.
}
]

This is an **oracle-start** defect: every step starts from the teacher state (x_k^*(u)). Therefore it measures the intrinsic local prediction error of the student solver on interval ([u,v]), without contamination from prefix accumulated error.

This distinction is important. If one instead starts from the student state after previous steps, the residual contains both:

[
\text{previous accumulated error}
+
\text{current local solver defect}.
]

Such a replay residual is useful for black-box validation, but it is not a local monitor quantity.

---

# 5. Local Power-Law Defect Model

Assume that for small step size (h>0), the student solver satisfies a local power-law defect model:

[
d_{\mathcal S}(x;u,u+h)
=======================

a_{\mathcal S}(x,u)h^q
+
O(h^{q+1}).
]

Here:

* (q>0) is the effective squared defect order;
* (a_{\mathcal S}(x,u)\ge0) is the local defect coefficient;
* the error term is assumed locally uniform over (x\sim p_u).

For the (k)-th calibration trajectory:

[
d_k(u,u+h)
==========

a_k(u)h^q
+
O(h^{q+1}),
]

where:

[
a_k(u)
======

a_{\mathcal S}(x_k^*(u),u).
]

Given a small probe step (\eta), estimate:

[
\boxed{
a_k(u)
\approx
\frac{d_k(u,u+\eta)}{\eta^q}.
}
]

If (q) is unknown, use multiple probe steps (\eta_j) and fit:

[
\log d_k(u,u+\eta_j)
\approx
\log a_k(u)+q\log \eta_j.
]

A global (\hat q) can be estimated by pooled log-log regression. If the local slope varies significantly with (u), one may use a local order (q(u)), but the base D-GPDE theory assumes constant (q).

---

# 6. Distributional Risk Defect

The main upgrade of D-GPDE is to aggregate local defect coefficients over the teacher marginal distribution (p_u).

Define a risk functional:

[
\mathcal R_{x\sim p_u}[\cdot].
]

Possible choices include:

[
\mathcal R=\mathbb E,
]

[
\mathcal R=\operatorname{CVaR}_\beta,
]

or a mixture:

[
\mathcal R_{\alpha,\beta}
=========================

(1-\alpha)\mathbb E
+
\alpha\operatorname{CVaR}_\beta.
]

Define the distributional defect coefficient:

[
\boxed{
\bar a_{\mathcal R}(u)
======================

\mathcal R_{x\sim p_u}
\left[
a_{\mathcal S}(x,u)
\right].
}
]

Using calibration trajectories:

[
\boxed{
\bar a_{\mathcal R}(u)
\approx
\mathcal R_{k=1}^K
\left[
a_k(u)
\right].
}
]

Special cases:

### Mean risk

[
\boxed{
\bar a_{\mathrm{mean}}(u)
=========================

\frac1K
\sum_{k=1}^K a_k(u).
}
]

This corresponds to average distributional local risk.

### CVaR risk

[
\boxed{
\bar a_{\mathrm{CVaR}}(u)
=========================

\operatorname{CVaR}_{\beta,k}
[
a_k(u)
].
}
]

This corresponds to hard-prompt or tail-risk local difficulty.

### Mixed risk

[
\boxed{
\bar a_{\alpha,\beta}(u)
========================

(1-\alpha)
\frac1K\sum_{k=1}^K a_k(u)
+
\alpha
\operatorname{CVaR}_{\beta,k}
[
a_k(u)
].
}
]

This interpolates between average distributional alignment and robust hard-prompt alignment.

---

# 7. Distributional Defect Over an Interval

Define the distributional local defect over a small interval:

[
\bar D_{\mathcal R}(u,u+h)
==========================

\mathcal R_{x\sim p_u}
\left[
d_{\mathcal S}(x;u,u+h)
\right].
]

Using the local power-law model:

[
d_{\mathcal S}(x;u,u+h)
=======================

a_{\mathcal S}(x,u)h^q
+
O(h^{q+1}),
]

we obtain:

[
\bar D_{\mathcal R}(u,u+h)
==========================

\mathcal R_{x\sim p_u}
[
a_{\mathcal S}(x,u)h^q
+
O(h^{q+1})
].
]

For positive homogeneous risk functionals such as expectation, CVaR, and max:

[
\mathcal R[cZ]=c\mathcal R[Z],
\qquad c\ge0.
]

Therefore:

[
\boxed{
\bar D_{\mathcal R}(u,u+h)
==========================

\bar a_{\mathcal R}(u)h^q
+
O(h^{q+1}).
}
]

This is the central local model of D-GPDE.

---

# 8. The D-GPDE Monitor

Given:

[
\bar D_{\mathcal R}(u,u+h)
\approx
\bar a_{\mathcal R}(u)h^q,
]

we seek a schedule that equalizes local distributional risk.

For a local interval ([u_i,u_{i+1}]) with step size (h_i=u_{i+1}-u_i), the leading-order defect is:

[
\bar D_{\mathcal R}(u_i,u_{i+1})
\approx
\bar a_{\mathcal R}(u_i)h_i^q.
]

To equalize this quantity across intervals, we require:

[
\bar a_{\mathcal R}(u_i)h_i^q
=============================

C.
]

Hence:

[
h_i
===

C^{1/q}
\bar a_{\mathcal R}(u_i)^{-1/q}.
]

Therefore, sampling density should be proportional to:

[
\bar a_{\mathcal R}(u)^{1/q}.
]

Define the D-GPDE monitor:

[
\boxed{
\omega_{\mathcal R}(u)
======================

\left(
\bar a_{\mathcal R}(u)+\epsilon_a
\right)^{1/q}.
}
]

Here (\epsilon_a>0) is a numerical floor.

The corresponding monitor mass of an interval ([a,b]) is:

[
\boxed{
\mathcal M_{\mathcal R}([a,b])
==============================

\int_a^b
\omega_{\mathcal R}(u),du.
}
]

The D-GPDE schedule is obtained by equalizing this monitor mass.

---

# 9. Inverse-CDF Schedule Construction

Let:

[
\Omega_{\mathcal R}
===================

\int_{u_{\min}}^{u_{\max}}
\omega_{\mathcal R}(u),du.
]

Define the normalized cumulative monitor clock:

[
\boxed{
\tau_{\mathcal R}(u)
====================

\frac{
\int_{u_{\min}}^u
\omega_{\mathcal R}(\xi),d\xi
}{
\Omega_{\mathcal R}
}.
}
]

The (N)-step D-GPDE schedule is:

[
\boxed{
u_m
===

\tau_{\mathcal R}^{-1}
\left(
\frac{m}{N}
\right),
\qquad
m=0,\ldots,N.
}
]

Equivalently, each interval satisfies:

[
\boxed{
\mathcal M_{\mathcal R}([u_m,u_{m+1}])
======================================

\frac{\Omega_{\mathcal R}}{N}.
}
]

---

# 10. Algorithm: D-GPDE

## Inputs

* Student solver (\mathcal S);
* target step count (N);
* calibration initial samples ({x_{k,0}}_{k=1}^K);
* high-accuracy teacher solver;
* metric (G(u));
* tangent weight (\rho);
* risk functional (\mathcal R);
* probe grid (\mathcal P={v_0,\ldots,v_M});
* probe step sizes ({\eta_j}_{j=1}^J);
* numerical floor (\epsilon_a).

---

## Step 1: Build teacher oracle

For each calibration sample (x_{k,0}), compute:

[
x_k^*(u)
========

F_{u_{\min}\to u}(x_{k,0}).
]

Use dense output to query (x_k^*(u)) at arbitrary (u).

---

## Step 2: Compute oracle-start defects

For each (u\in\mathcal P), probe step (\eta_j), and trajectory (k), compute:

[
d_k(u,u+\eta_j)
===============

\left|
\Phi_{\mathcal S}(x_k^*(u);u\to u+\eta_j)
-----------------------------------------

x_k^*(u+\eta_j)
\right|_{\rho,k,u+\eta_j}^{2}.
]

---

## Step 3: Estimate (q)

Fit:

[
\log d_k(u,u+\eta_j)
\approx
\log a_k(u)
+
q\log\eta_j.
]

This gives (\hat q).

---

## Step 4: Estimate local defect coefficients

For a chosen probe step (\eta), or by regression intercept:

[
a_k(u)
\approx
\frac{
d_k(u,u+\eta)
}{
\eta^{\hat q}
}.
]

---

## Step 5: Aggregate by distributional risk

Compute:

[
\bar a_{\mathcal R}(u)
======================

\mathcal R_{k=1}^K
[
a_k(u)
].
]

For the mean version:

[
\bar a_{\mathrm{mean}}(u)
=========================

\frac1K
\sum_{k=1}^K a_k(u).
]

For the CVaR version:

[
\bar a_{\mathrm{CVaR}}(u)
=========================

\operatorname{CVaR}_{\beta,k}
[
a_k(u)
].
]

For the mixed version:

[
\bar a_{\alpha,\beta}(u)
========================

(1-\alpha)\bar a_{\mathrm{mean}}(u)
+
\alpha\bar a_{\mathrm{CVaR}}(u).
]

---

## Step 6: Build monitor

[
\omega_{\mathcal R}(u)
======================

\left(
\bar a_{\mathcal R}(u)+\epsilon_a
\right)^{1/\hat q}.
]

Smooth (\log \omega_{\mathcal R}(u)) if necessary.

---

## Step 7: Compute inverse-CDF schedule

Numerically compute:

[
\tau_{\mathcal R}(u)
====================

\frac{
\int_{u_{\min}}^u
\omega_{\mathcal R}(\xi)d\xi
}{
\int_{u_{\min}}^{u_{\max}}
\omega_{\mathcal R}(\xi)d\xi
}.
]

Then output:

[
u_m
===

\tau_{\mathcal R}^{-1}
\left(
\frac{m}{N}
\right),
\qquad
m=0,\ldots,N.
]

---

# 11. Main Theoretical Results

Below are the core theorems you can use in the paper.

---

# Theorem 1: Distributional Local Power Law

**Theorem.**
Assume that for (x\sim p_u), the oracle-start defect satisfies:

[
d_{\mathcal S}(x;u,u+h)
=======================

a_{\mathcal S}(x,u)h^q
+
R(x,u,h),
]

where:

[
|R(x,u,h)|
\le
C_R(x,u)h^{q+1},
]

and:

[
\mathcal R_{x\sim p_u}[C_R(x,u)]<\infty.
]

Assume also that the risk functional (\mathcal R) is positive homogeneous and monotone. Then:

[
\boxed{
\bar D_{\mathcal R}(u,u+h)
==========================

\bar a_{\mathcal R}(u)h^q
+
O(h^{q+1}),
}
]

where:

[
\bar D_{\mathcal R}(u,u+h)
==========================

\mathcal R_{x\sim p_u}
[
d_{\mathcal S}(x;u,u+h)
],
]

and:

[
\bar a_{\mathcal R}(u)
======================

\mathcal R_{x\sim p_u}
[
a_{\mathcal S}(x,u)
].
]

---

## Proof

By the local defect model:

[
d_{\mathcal S}(x;u,u+h)
=======================

a_{\mathcal S}(x,u)h^q
+
R(x,u,h).
]

Apply (\mathcal R):

[
\bar D_{\mathcal R}(u,u+h)
==========================

\mathcal R_{x\sim p_u}
[
a_{\mathcal S}(x,u)h^q
+
R(x,u,h)
].
]

For expectation, this gives exactly:

[
\bar D_{\mathbb E}(u,u+h)
=========================

h^q
\mathbb E[a_{\mathcal S}(x,u)]
+
\mathbb E[R(x,u,h)].
]

Since:

[
|\mathbb E[R(x,u,h)]|
\le
\mathbb E[C_R(x,u)]h^{q+1},
]

we obtain:

[
\bar D_{\mathbb E}(u,u+h)
=========================

\bar a_{\mathbb E}(u)h^q
+
O(h^{q+1}).
]

For CVaR and other positive homogeneous monotone risk functionals, the same leading-order relation follows from:

[
\mathcal R[cZ]=c\mathcal R[Z],
\qquad c\ge0,
]

and the boundedness of the remainder risk.

Therefore:

[
\bar D_{\mathcal R}(u,u+h)
==========================

\bar a_{\mathcal R}(u)h^q
+
O(h^{q+1}).
]

[
\blacksquare
]

---

# Theorem 2: Correct Monitor Density

**Theorem.**
Suppose:

[
\bar D_{\mathcal R}(u,u+h)
==========================

\bar a_{\mathcal R}(u)h^q
+
O(h^{q+1}),
]

with:

[
\bar a_{\mathcal R}(u)>0.
]

Then the leading-order solution of:

[
\min_{{h_i}}
\max_i
\bar a_{\mathcal R}(u_i)h_i^q,
\qquad
\sum_i h_i=L,
]

is obtained by equalizing:

[
\bar a_{\mathcal R}(u_i)h_i^q.
]

The corresponding continuous monitor density is:

[
\boxed{
\omega_{\mathcal R}(u)
======================

\bar a_{\mathcal R}(u)^{1/q}.
}
]

---

## Proof

At the leading order, the local risk on interval (i) is:

[
E_i
===

\bar a_{\mathcal R}(u_i)h_i^q.
]

If an optimal allocation has two intervals (i,j) with:

[
E_i>E_j,
]

then one can decrease (h_i) slightly and increase (h_j) slightly, keeping:

[
\sum_i h_i=L,
]

while decreasing the maximum risk, unless (E_i=E_j) for all active intervals.

Thus the leading-order minimax condition is:

[
\bar a_{\mathcal R}(u_i)h_i^q=C.
]

Solving for (h_i):

[
h_i
===

C^{1/q}
\bar a_{\mathcal R}(u_i)^{-1/q}.
]

Therefore the local sampling density is:

[
\frac1{h_i}
\propto
\bar a_{\mathcal R}(u_i)^{1/q}.
]

Hence the monitor density is:

[
\omega_{\mathcal R}(u)
======================

\bar a_{\mathcal R}(u)^{1/q}.
]

[
\blacksquare
]

---

# Theorem 3: Coordinate Covariance of the D-GPDE Monitor

**Theorem.**
Let (v=g(u)) be a smooth monotone reparameterization. Suppose the same physical local defect can be written as:

[
\bar D_{\mathcal R}(u,u+h_u)
\approx
\bar a_u(u)h_u^q
]

in (u)-coordinates and:

[
\bar D_{\mathcal R}(v,v+h_v)
\approx
\bar a_v(v)h_v^q
]

in (v)-coordinates. Then:

[
\boxed{
\omega_v(v),dv
==============

\omega_u(u),du,
}
]

where:

[
\omega_u(u)=\bar a_u(u)^{1/q},
\qquad
\omega_v(v)=\bar a_v(v)^{1/q}.
]

Thus (\omega(u)du) is a coordinate-covariant monitor 1-form.

---

## Proof

Since:

[
v=g(u),
]

we have:

[
h_v
===

\left|
\frac{dv}{du}
\right|h_u.
]

Thus:

[
h_u
===

\left|
\frac{du}{dv}
\right|h_v.
]

The same physical defect satisfies:

[
\bar a_u(u)h_u^q
================

\bar a_v(v)h_v^q.
]

Substitute:

[
h_u=
\left|
\frac{du}{dv}
\right|h_v.
]

Then:

[
\bar a_u(u)
\left|
\frac{du}{dv}
\right|^q
h_v^q
=====

\bar a_v(v)h_v^q.
]

Therefore:

[
\bar a_v(v)
===========

\bar a_u(u)
\left|
\frac{du}{dv}
\right|^q.
]

Taking (q)-th roots:

[
\omega_v(v)
===========

# \bar a_v(v)^{1/q}

\bar a_u(u)^{1/q}
\left|
\frac{du}{dv}
\right|
=======

\omega_u(u)
\left|
\frac{du}{dv}
\right|.
]

Multiplying by (dv):

[
\omega_v(v)dv
=============

\omega_u(u)du.
]

[
\blacksquare
]

---

# Theorem 4: Exact Equalization of Monitor Mass

**Theorem.**
Let:

[
\omega_{\mathcal R}(u)>0
]

be continuous on ([u_{\min},u_{\max}]), and define:

[
\Omega_{\mathcal R}
===================

\int_{u_{\min}}^{u_{\max}}
\omega_{\mathcal R}(u),du.
]

For any (N)-interval schedule:

[
U=(u_0,\ldots,u_N),
]

define monitor masses:

[
M_i
===

\int_{u_i}^{u_{i+1}}
\omega_{\mathcal R}(u),du.
]

Then:

[
\max_i M_i
\ge
\frac{\Omega_{\mathcal R}}{N}.
]

Equality holds if and only if:

[
M_i=
\frac{\Omega_{\mathcal R}}{N},
\qquad
i=0,\ldots,N-1.
]

The unique equal-mass schedule is:

[
\boxed{
u_m
===

\tau_{\mathcal R}^{-1}
\left(
\frac{m}{N}
\right).
}
]

---

## Proof

Since:

[
\sum_{i=0}^{N-1}M_i
===================

\Omega_{\mathcal R},
]

the largest interval mass must satisfy:

[
\max_i M_i
\ge
\frac1N
\sum_{i=0}^{N-1}M_i
===================

\frac{\Omega_{\mathcal R}}{N}.
]

Equality holds exactly when all (M_i) are equal.

Since (\omega_{\mathcal R}(u)>0), the cumulative function:

[
\tau_{\mathcal R}(u)
====================

\frac{
\int_{u_{\min}}^u
\omega_{\mathcal R}(\xi)d\xi
}{
\Omega_{\mathcal R}
}
]

is strictly increasing and invertible. Therefore:

[
u_m
===

\tau_{\mathcal R}^{-1}
\left(
\frac{m}{N}
\right)
]

satisfies:

[
\int_{u_m}^{u_{m+1}}
\omega_{\mathcal R}(u),du
=========================

\frac{\Omega_{\mathcal R}}{N}.
]

[
\blacksquare
]

---

# Theorem 5: Asymptotic Minimax Optimality for Distributional Risk

**Theorem.**
Assume:

1. (\omega_{\mathcal R}(u)) is continuous and bounded:

[
0<\omega_{\min}
\le
\omega_{\mathcal R}(u)
\le
\omega_{\max}
<\infty.
]

2. The interval defect satisfies the uniform approximation:

[
\left|
\bar D_{\mathcal R}(s,t)
------------------------

\mathcal M_{\mathcal R}([s,t])^q
\right|
\le
C_E |t-s|^{q+1},
]

where:

[
\mathcal M_{\mathcal R}([s,t])
==============================

\int_s^t
\omega_{\mathcal R}(u),du.
]

Let (U^\Omega) be the equal-monitor-mass schedule. Then:

[
\boxed{
\max_i
\bar D_{\mathcal R}(u_i^\Omega,u_{i+1}^\Omega)
\le
\left(
\frac{\Omega_{\mathcal R}}{N}
\right)^q
+
C_1N^{-(q+1)}.
}
]

Moreover, for any schedule (U) satisfying:

[
h_{\max}
========

\max_i(u_{i+1}-u_i)
\le
\frac{C_h}{N},
]

we have:

[
\boxed{
\max_i
\bar D_{\mathcal R}(u_i,u_{i+1})
\ge
\left(
\frac{\Omega_{\mathcal R}}{N}
\right)^q
---------

C_2N^{-(q+1)}.
}
]

Therefore, D-GPDE is asymptotically minimax optimal for distributional local risk up to order (O(N^{-(q+1)})).

---

## Proof

For the equal-mass schedule:

[
\mathcal M_{\mathcal R}([u_i^\Omega,u_{i+1}^\Omega])
====================================================

\frac{\Omega_{\mathcal R}}{N}.
]

Since:

[
\omega_{\mathcal R}(u)\ge \omega_{\min},
]

each interval length satisfies:

[
h_i^\Omega
\le
\frac{
\Omega_{\mathcal R}/N
}{
\omega_{\min}
}
=

O(N^{-1}).
]

Using the approximation assumption:

[
\bar D_{\mathcal R}(u_i^\Omega,u_{i+1}^\Omega)
==============================================

\left(
\frac{\Omega_{\mathcal R}}{N}
\right)^q
+
O((h_i^\Omega)^{q+1}).
]

Since:

[
h_i^\Omega=O(N^{-1}),
]

we obtain:

[
\max_i
\bar D_{\mathcal R}(u_i^\Omega,u_{i+1}^\Omega)
\le
\left(
\frac{\Omega_{\mathcal R}}{N}
\right)^q
+
C_1N^{-(q+1)}.
]

Now consider any admissible schedule (U) with (h_{\max}\le C_h/N). Let:

[
M_i=
\mathcal M_{\mathcal R}([u_i,u_{i+1}]).
]

Because:

[
\sum_iM_i=\Omega_{\mathcal R},
]

there exists (j) such that:

[
M_j\ge\frac{\Omega_{\mathcal R}}{N}.
]

By the approximation assumption:

[
\bar D_{\mathcal R}(u_j,u_{j+1})
\ge
M_j^q
-----

C_E h_j^{q+1}.
]

Thus:

[
\bar D_{\mathcal R}(u_j,u_{j+1})
\ge
\left(
\frac{\Omega_{\mathcal R}}{N}
\right)^q
---------

C_E
\left(
\frac{C_h}{N}
\right)^{q+1}.
]

Therefore:

[
\max_i
\bar D_{\mathcal R}(u_i,u_{i+1})
\ge
\left(
\frac{\Omega_{\mathcal R}}{N}
\right)^q
---------

C_2N^{-(q+1)}.
]

[
\blacksquare
]

---

# Theorem 6: Monte Carlo Estimation of the Distributional Monitor

**Theorem.**
Assume (a_{\mathcal S}(x,u)) has finite variance under (p_u). For the mean-risk version:

[
\bar a_{\mathrm{mean}}(u)
=========================

\mathbb E_{x\sim p_u}
[
a_{\mathcal S}(x,u)
],
]

the empirical estimator:

[
\hat a_{\mathrm{mean}}(u)
=========================

\frac1K
\sum_{k=1}^K
a_k(u)
]

satisfies:

[
\boxed{
\hat a_{\mathrm{mean}}(u)
-------------------------

# \bar a_{\mathrm{mean}}(u)

O_p(K^{-1/2}).
}
]

If the estimate of (a_k(u)) has additional probe error (\varepsilon_{\mathrm{probe}}(u)), then:

[
\boxed{
\hat a_{\mathrm{mean}}(u)
-------------------------

# \bar a_{\mathrm{mean}}(u)

O_p(K^{-1/2})
+
O(\varepsilon_{\mathrm{probe}}(u)).
}
]

---

## Proof

Since:

[
x_k^*(u)\sim p_u,
]

we have:

[
a_k(u)=a_{\mathcal S}(x_k^*(u),u)
]

as i.i.d. samples from the distribution of (a_{\mathcal S}(x,u)) under (p_u).

By the central limit theorem:

[
\frac1K
\sum_{k=1}^K
a_k(u)
------

# \mathbb E[a_{\mathcal S}(x,u)]

O_p(K^{-1/2}).
]

If (a_k(u)) is estimated from finite probe steps, the deterministic or stochastic probe error adds:

[
O(\varepsilon_{\mathrm{probe}}(u)).
]

[
\blacksquare
]

---

# Theorem 7: Inverse-CDF Stability

**Theorem.**
Let the true cumulative monitor be:

[
F(u)
====

\int_{u_{\min}}^u
\omega_{\mathcal R}(\xi)d\xi,
]

and the estimated cumulative monitor be:

[
\hat F(u)
=========

\int_{u_{\min}}^u
\hat\omega_{\mathcal R}(\xi)d\xi.
]

Assume:

[
\sup_u
|\hat F(u)-F(u)|
\le
\varepsilon_F,
]

and:

[
\omega_{\mathcal R}(u)\ge\omega_{\min}>0.
]

Let:

[
u_m=
F^{-1}
\left(
\frac{m}{N}\Omega_{\mathcal R}
\right),
]

and (\hat u_m) be the corresponding estimated inverse-CDF schedule. Then:

[
\boxed{
\max_m
|\hat u_m-u_m|
\le
\frac{\varepsilon_F}{\omega_{\min}}
+
O(\varepsilon_F^2).
}
]

---

## Proof

Since:

[
F'(u)=\omega_{\mathcal R}(u)\ge\omega_{\min},
]

the inverse function (F^{-1}) is Lipschitz with constant at most:

[
\frac1{\omega_{\min}}.
]

Thus a perturbation of the cumulative monitor by at most (\varepsilon_F) induces a schedule perturbation bounded by:

[
|\hat u_m-u_m|
\le
\frac{\varepsilon_F}{\omega_{\min}}
+
O(\varepsilon_F^2).
]

[
\blacksquare
]

---

# 12. Optional Theorem: (W_2)-Type Distributional Coupling Bound

This theorem should be written carefully. It is useful for positioning D-GPDE as a distributional method, but it should not be oversold.

Let:

[
p_{*,T}
]

be the terminal teacher distribution, and:

[
p_{\mathcal S,T}
]

be the terminal student distribution induced by the schedule (U).

For the same initial sample (x_0), define:

[
x_T^*
=====

F_{u_{\min}\to T}(x_0),
]

[
x_T^{\mathcal S}
================

\Phi_{\mathcal S,U}(x_0).
]

This shared-initial-noise construction gives a coupling between (p_{*,T}) and (p_{\mathcal S,T}). Hence:

[
\boxed{
W_2^2(p_{\mathcal S,T},p_{*,T})
\le
\mathbb E_{x_0}
\left[
|x_T^{\mathcal S}-x_T^*|_{G_T}^2
\right].
}
]

Under a first-order error propagation expansion:

[
x_T^{\mathcal S}-x_T^*
======================

\sum_{i=0}^{N-1}
S_i e_i
+
\text{higher-order terms},
]

where (e_i) is the oracle-start local error at step (i), one can obtain a conservative bound:

[
\boxed{
W_2^2(p_{\mathcal S,T},p_{*,T})
\le
C_{\mathrm{stab}}(U)
\sum_{i=0}^{N-1}
\mathbb E_{x\sim p_{u_i}}
[
d_{\mathcal S}(x;u_i,u_{i+1})
]
+
\text{higher-order terms}.
}
]

Here (C_{\mathrm{stab}}(U)) absorbs:

1. downstream flow Lipschitz constants;
2. correlations between propagated step errors;
3. high-order nonlinear propagation terms.

This theorem does **not** claim that endpoint local defect is an exact terminal distribution error. It only states that distributional local defect controls a first-order coupling proxy for terminal Wasserstein error under stability assumptions.

This is the cleanest way to connect D-GPDE to distribution-level alignment without adding future-pullback.

---

# 13. Relation Between Mean, CVaR, and Minimax

D-GPDE can be understood as a risk-family method:

[
\boxed{
\bar a_{\mathcal R}(u)
======================

\mathcal R_{x\sim p_u}
[
a_{\mathcal S}(x,u)
].
}
]

Different choices of (\mathcal R) give different objectives.

---

## 13.1 Mean risk

[
\mathcal R=\mathbb E.
]

Then:

[
\bar a_{\mathbb E}(u)
=====================

\mathbb E_{x\sim p_u}
[
a_{\mathcal S}(x,u)
].
]

This yields average distributional local-risk equalization.

It is most suitable when the goal is average teacher-student distribution alignment.

---

## 13.2 CVaR risk

[
\mathcal R=\operatorname{CVaR}_\beta.
]

Then:

[
\bar a_{\mathrm{CVaR}}(u)
=========================

\operatorname{CVaR}*{\beta,x\sim p_u}
[
a*{\mathcal S}(x,u)
].
]

This yields tail-risk local-defect equalization.

It is suitable when one wants robustness to hard prompts, hard trajectories, or rare high-error states.

---

## 13.3 Worst-case risk

[
\mathcal R=\operatorname{ess,sup}.
]

Then:

[
\bar a_{\max}(u)
================

\sup_{x\in \operatorname{supp}(p_u)}
a_{\mathcal S}(x,u).
]

This gives a true worst-case local minimax monitor, but it is typically too conservative and unstable in high-dimensional diffusion models.

---

## 13.4 Mixed risk

[
\mathcal R_{\alpha,\beta}
=========================

(1-\alpha)\mathbb E
+
\alpha\operatorname{CVaR}_\beta.
]

Then:

[
\bar a_{\alpha,\beta}(u)
========================

(1-\alpha)
\mathbb E[a_{\mathcal S}(x,u)]
+
\alpha
\operatorname{CVaR}*\beta[a*{\mathcal S}(x,u)].
]

This is often the most practical default, because it balances average distributional risk and hard-case robustness.

Recommended defaults:

[
\alpha\in{0.25,0.5},
\qquad
\beta\in[0.8,0.95].
]

---

# 14. Complete Mathematical Chain

The final D-GPDE chain is:

[
\text{deterministic ODE sampler}
]

[
\Downarrow
]

[
p_u=(F_{u_{\min}\to u})*#p*{u_{\min}}
]

[
\Downarrow
]

[
x_k^*(u)\sim p_u
]

[
\Downarrow
]

[
d_k(u,u+h)
==========

\left|
\Phi_{\mathcal S}(x_k^*(u);u\to u+h)
------------------------------------

x_k^*(u+h)
\right|_{\rho,k,u+h}^2
]

[
\Downarrow
]

[
d_k(u,u+h)
\approx
a_k(u)h^q
]

[
\Downarrow
]

[
\bar a_{\mathcal R}(u)
======================

\mathcal R_{k=1}^K
[
a_k(u)
]
]

[
\Downarrow
]

[
\boxed{
\omega_{\mathcal R}(u)
======================

\left(
\bar a_{\mathcal R}(u)+\epsilon_a
\right)^{1/q}
}
]

[
\Downarrow
]

[
\omega_{\mathcal R}(u)du
\text{ is a coordinate-covariant monitor 1-form}
]

[
\Downarrow
]

[
u_m=
\tau_{\mathcal R}^{-1}
\left(
\frac{m}{N}
\right)
]

[
\Downarrow
]

[
\text{distributional local-risk equalized sampling schedule}.
]

---

# 15. Practical Implementation Notes

## 15.1 What changes compared with original GPDE?

Almost nothing.

Original:

[
\bar a(u)
=========

\operatorname{CVaR}_{\beta,k}
[
a_k(u)
].
]

D-GPDE:

[
\bar a_{\mathcal R}(u)
======================

\mathcal R_{k}
[
a_k(u)
].
]

If (\mathcal R=\operatorname{CVaR}_\beta), D-GPDE recovers the original robust version.

If (\mathcal R=\mathbb E), D-GPDE becomes the distribution-average version.

If (\mathcal R=(1-\alpha)\mathbb E+\alpha\operatorname{CVaR}_\beta), it becomes a mixed-risk method.

---

## 15.2 Computational cost

The dominant cost remains:

[
O(KMJ\cdot C_{\mathcal S}),
]

where:

* (K): number of calibration trajectories;
* (M): number of probe grid points;
* (J): number of probe step sizes;
* (C_{\mathcal S}): cost of one student probe.

Changing the aggregation from CVaR to mean or mixed risk adds no model evaluations.

Mean aggregation costs:

[
O(K)
]

per grid point.

CVaR aggregation costs:

[
O(K\log K)
]

if implemented by sorting, or approximately (O(K)) with selection algorithms.

Thus D-GPDE has essentially the same computational cost as the original GPDE.

---

# 16. Suggested Paper Section Structure

You can organize the method section as follows:

## 3. Distributional Geometric Prediction Defect

### 3.1 Teacher marginal distribution

Define:

[
p_u=(F_{u_{\min}\to u})*#p*{u_{\min}}.
]

Explain that calibration trajectories provide Monte Carlo samples from (p_u).

### 3.2 Oracle-start local defect

Define:

[
d_{\mathcal S}(x;u,v).
]

Then define empirical:

[
d_k(u,v).
]

### 3.3 Distributional risk coefficient

Define:

[
\bar a_{\mathcal R}(u)
======================

\mathcal R_{x\sim p_u}
[
a_{\mathcal S}(x,u)
].
]

Explain mean, CVaR, and mixed risk.

### 3.4 Monitor construction

Derive:

[
\omega_{\mathcal R}(u)
======================

\bar a_{\mathcal R}(u)^{1/q}.
]

### 3.5 Schedule by inverse-CDF

Define:

[
\tau_{\mathcal R}(u)
]

and:

[
u_m=\tau_{\mathcal R}^{-1}(m/N).
]

---

# 17. Suggested Theory Section Structure

## 4. Theory

### 4.1 Distributional local power law

State and prove Theorem 1.

### 4.2 Monitor 1-form and coordinate covariance

State and prove Theorem 3.

### 4.3 Asymptotic minimax distributional risk equalization

State and prove Theorem 5.

### 4.4 Estimation and stability

State Theorem 6 and Theorem 7.

### 4.5 Coupling interpretation

State the conservative (W_2)-type bound as interpretation, not as the main theorem.

---

# 18. Limitations to State Clearly

You should explicitly state the following limitations.

## Limitation 1: Local rather than terminal objective

D-GPDE equalizes distributional local solver defect. It does not explicitly model downstream amplification of local errors.

Thus it should be interpreted as:

[
\text{local distributional risk calibration},
]

not exact final quality optimization.

---

## Limitation 2: Asymptotic nature

The minimax theorem is local and asymptotic. It assumes:

[
h_{\max}=O(1/N).
]

For very small (N), such as:

[
N=4,8,10,
]

the local power-law approximation may be loose.

---

## Limitation 3: Teacher-alignment vs generation quality

Improved teacher alignment does not necessarily imply better FID, CLIPScore, or human preference.

Therefore experiments must report both:

[
\text{teacher-alignment metrics}
]

and:

[
\text{generation-quality metrics}.
]

This limitation is already consistent with your original framing .

---

## Limitation 4: Multistep solvers

For multistep solvers, the local defect may depend on historical states and step ratios. Therefore a one-dimensional monitor is exact mainly for single-step solvers or multistep solvers with stable history behavior.

For multistep solvers, D-GPDE should be treated as a warm-start or approximate schedule calibration method.

---

# 19. Final Method Summary for Paper

You can write the final method summary like this:

> We propose D-GPDE, a training-free schedule calibration method for deterministic diffusion ODE samplers. Instead of assigning timesteps according to a fixed noise schedule, D-GPDE estimates the local prediction defect of a student solver under the teacher marginal distribution (p_u). For each coordinate (u), calibration trajectories provide Monte Carlo samples (x_k^*(u)\sim p_u). We measure the oracle-start defect of the student solver against a high-accuracy teacher flow and fit a local power law (d_k(u,u+h)\approx a_k(u)h^q). A risk functional (\mathcal R), such as expectation, CVaR, or their mixture, aggregates the coefficients into a distributional risk density (\bar a_{\mathcal R}(u)). The correct monitor is the (q)-th root (\omega_{\mathcal R}(u)=\bar a_{\mathcal R}(u)^{1/q}), whose equal-mass inverse-CDF discretization yields the final sampling schedule. We show that (\omega_{\mathcal R}(u)du) is coordinate-covariant and that equalizing its mass is asymptotically minimax optimal for local distributional risk.

---

# 20. The Most Important Final Formulas

If you only keep a compact version, keep these:

[
p_u=(F_{u_{\min}\to u})*#p*{u_{\min}}.
]

[
d_{\mathcal S}(x;u,u+h)
=======================

\left|
\Phi_{\mathcal S}(x;u\to u+h)
-----------------------------

F_{u\to u+h}(x)
\right|_{\rho,u+h}^2.
]

[
d_{\mathcal S}(x;u,u+h)
\approx
a_{\mathcal S}(x,u)h^q.
]

[
\bar a_{\mathcal R}(u)
======================

\mathcal R_{x\sim p_u}
[
a_{\mathcal S}(x,u)
].
]

[
\bar a_{\mathcal R}(u)
\approx
\mathcal R_{k=1}^K
[
a_k(u)
].
]

[
\boxed{
\omega_{\mathcal R}(u)
======================

\left(
\bar a_{\mathcal R}(u)+\epsilon_a
\right)^{1/q}.
}
]

[
\tau_{\mathcal R}(u)
====================

\frac{
\int_{u_{\min}}^u
\omega_{\mathcal R}(\xi)d\xi
}{
\int_{u_{\min}}^{u_{\max}}
\omega_{\mathcal R}(\xi)d\xi
}.
]

[
\boxed{
u_m
===

\tau_{\mathcal R}^{-1}
\left(
\frac{m}{N}
\right).
}
]

