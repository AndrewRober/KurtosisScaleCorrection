#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.72in),
  columns: 2,
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 9.5pt)
#set par(justify: true, leading: 0.56em)
#set heading(numbering: "1.1")

#let theorem(title, body) = block(
  fill: rgb("#f8fafc"),
  stroke: (left: 2.5pt + rgb("#1d4ed8")),
  inset: (x: 8pt, y: 6pt),
  radius: 2pt,
  width: 100%,
)[
  *#title.* #emph(body)
]

#place(
  top + center,
  scope: "parent",
  float: true,
  block(width: 100%, below: 1.2em)[
    #align(center)[
      #text(size: 15.5pt, weight: "bold")[
        Beyond Bessel's Correction: Power-Scale Concavity–Convexity Duality and Kurtosis-Optimal Shrinkage for LLM Quantization and Group-Relative RL
      ]
      #v(0.5em)
      #text(size: 10.5pt, weight: "semibold")[Andrew Rober]
      #v(0.2em)
      #text(size: 9pt, fill: rgb("#475569"))[
        Independent AI & Systems Research • `https://github.com/AndrewRober/KurtosisScaleCorrection` • November 2026
      ]
    ]
    #v(0.6em)
    #block(
      fill: rgb("#f1f5f9"),
      inset: 9pt,
      radius: 3pt,
      width: 100%,
    )[
      #text(weight: "bold")[Abstract — ]
      Standard deep learning frameworks (`PyTorch`, `NumPy`, `.NET`) default to Bessel's $n - 1$ degrees-of-freedom divisor (`ddof=1`) under the widespread assumption that it yields an "unbiased" scale estimator. Revisiting an earlier empirical investigation into sample standard deviation bias (`DeviationBaisExperiment`), we prove that Bessel's $S_(n-1) = sqrt(W / (n - 1))$ is *never* unbiased for nonlinear scale powers $sigma^p$ ($p != 2$) and is strictly *inadmissible* in Mean Squared Error (MSE) across virtually all modern machine learning regimes. First, we establish the *Power-Scale Concavity–Convexity Duality Law*: for any distribution with standardized fourth moment (kurtosis) $kappa = mu_4 / sigma^4 >= 1$, the relative bias of $(S_(n-1)^2)^(p/2)$ is $(p(p - 2))/(8n) (kappa - 1) + O(n^(-2))$. Consequently, forward standard deviation ($p = +1$, used in block quantization) suffers universal downward concavity bias $-(kappa - 1)/(8n)$, whereas inverse standard deviation ($p = -1$, used in LayerNorm, micro-batch whitening, and advantage normalization) flips sign into a $3 times$ steeper *upward convexity explosion* $+(3(kappa - 1))/(8n)$, with inverse-variance expectation $(n - 1)/(n - 3)$ diverging at $n <= 3$ and inflating by $3.00 times$ at group size $G = 4$. Second, we derive the universal MSE-optimal divisor $c_p^*(n, kappa) = (n - 1) + ((3p - 2)/4)(kappa - 1) + O(n^(-1))$, unifying Stein's $n + 1$ Gaussian variance rule ($p = 2, kappa = 3$) with heavy-tailed LLM channel shrinkage ($kappa gt.double 3$). Third, in Group Relative Policy Optimization (`GRPO`) for Reinforcement Learning with Verifiable Rewards (RLVR), we prove via Stein's Lemma that coupling numerator and denominator on the *same* $G$ rollouts reverses convexity explosion into severe signal attenuation ($-60.1%$ at $G = 2$, $-30.9%$ at $G = 4$), whereas removing `std` entirely (`Dr. GRPO`) breaks multi-task scale invariance. Introducing *Kurtosis-Optimal Power Scaling* (`KPS`) and *Empirical-Bayes Group Normalization* (`EB-GRPO`), we validate all theorems symbolically via `SymPy` and empirically on an `NVIDIA GB10` Grace Blackwell superchip across `Qwen2.5-1.5B-Instruct`, `BAAI/bge-large-en-v1.5`, and 24-seed multi-task RLVR: `KPS` cuts heavy-tailed variance MSE by up to $55.4%$, slashes $G = 4$ whitening MSE by $76.1%$–$98.5%$, reduces live LLM sub-block inverse-scale MSE by $11.6%$–$27.5%$, and boosts hard-math pass\@1 in multi-task RLVR from $7.31%$ (`Dr. GRPO`) and $13.32%$ (`GRPO`) to $15.54%$.
    ]
  ]
)

= Introduction

Dividing the centered sum of squared deviations $W = sum_(i=1)^n (X_i - overline(X))^2$ by $n - 1$ rather than $n$ — Bessel's correction — is one of the earliest lessons in mathematical statistics @cochran1934distribution. Because $EE[W] = (n - 1) sigma^2$ for independent observations with variance $sigma^2$, modern numerical libraries including `torch.std(..., correction=1)` default to $S_(n-1) = sqrt(W / (n - 1))$ everywhere from activation statistics and microscaling quantization (`NVFP4` block size $n = 16$, `MXFP4` block size $n = 32$) @rouhani2023microscaling to Group Relative Policy Optimization (`GRPO`) in reasoning LLMs @shao2024deepseekmath @deepseek2025r1.

Yet in an earlier exploratory C\# study @rober2023deviation, empirical subsampling of finite datasets repeatedly showed $S_(n-1)$ producing *larger* estimation error than the "biased" divisor $n$ at small sample sizes, motivating a deeper mathematical inquiry: *when does Bessel's correction actually help, when does it hurt, and how does it behave under the extreme non-Gaussian kurtosis ($kappa in [20, 300]$) @sun2024massive @xiao2023smoothquant and tiny rollout groups ($G in {2, 4, 8}$) @liu2025drgrpo of modern deep learning?*

In this paper, we resolve both the empirical anomalies of the 2023 repository and the broader statistical foundation of small-sample scale estimation across arbitrary power transformations $sigma^p$:

1. *Forensic Audit & Finite-Population Law (Section 2):* Auditing @rober2023deviation uncovers five software defects (including `int / int` mean truncation that inflates variance on `{1..6}` by $+8.57%$, $O(n^2)$ inner-loop re-summation, and deferred `IEnumerable` re-sampling), alongside the genuine statistical Simple Random Sampling Without Replacement (SRSWOR) overshoot $EE[S_(n-1)^2] = N/(N - 1) sigma_"pop"^2$.
2. *Kurtosis-Optimal Minimum-MSE Divisor (Section 3):* Using exact fourth-moment cumulant algebra @oneill2014some, we prove that for *any* distribution with kurtosis $kappa = mu_4 / sigma^4 >= 1$, the unique MSE-minimizing variance divisor is $c_"var"^*(n, kappa) = (n - 1) + ((n - 1) kappa - (n - 3))/n$, recovering Stein's $n + 1$ @stein1964inadmissibility for Gaussians while increasing linearly with $kappa$ under heavy tails.
3. *Power-Scale Concavity–Convexity Duality (Section 4):* Generalizing @gurland1971simple to arbitrary $sigma^p$ ($p in RR$) and non-Gaussian kurtosis $kappa$, we prove a sharp sign-flip trichotomy at $p in {0, 2}$: forward standard deviation ($p = +1$) underestimates $sigma$, whereas whitening/normalization ($p = -1$) @ba2016layernorm explodes upward by $3 times$ the magnitude, governed by the universal MSE-optimal divisor $c_p^*(n, kappa) = (n - 1) + ((3p - 2)/4)(kappa - 1) + O(n^(-1))$.
4. *Coupled vs. Decoupled `GRPO` Duality & `EB-GRPO` (Section 5):* Recent work (`Dr. GRPO` @liu2025drgrpo) observed instability in `GRPO`'s $1 / S_(G-1)$ term and removed `std` division altogether. We prove via Stein's Lemma that self-normalizing on the *same* $G$ rollouts attenuates expected policy gradients by $(G - 1)/G dot c_4(G)$ ($-30.9%$ at $G = 4$), while `Dr. GRPO` sacrifices multi-task scale invariance — and solve both simultaneously via Empirical-Bayes cross-group shrinkage (`EB-GRPO`).

= Forensic Audit & Finite-Population Theory

Before extending the theory, scientific reproducibility requires auditing the original C\# implementation in `DeviationBaisExperiment` @rober2023deviation. Careful static and numerical inspection identified five distinct implementation bugs:

- *Bug 1 (`int / int` Mean Truncation & $O(n^2)$ Complexity):* `Extensions.cs` evaluated `(value - (data.Sum() / data.Length))` on `int[]` inside `.AsParallel().Sum(...)`. Integer division truncated fractional sample means $overline(X)$ to $floor(overline(X))$, adding strictly positive bias $n(overline(X) - floor(overline(X)))^2$ to $W$ ($+8.57%$ variance and $+4.20%$ std inflation on `{1,2,3,4,5,6}` where $overline(X) = 3.5 -> 3$), while recomputing `data.Sum()` $n$ times ($10^(12)$ additions at $N = 10^6$).
- *Bug 2 (Lazy `IEnumerable` Re-Execution):* Sample sequences wrapped `r2.Next(...)` inside unmaterialized `Select` queries, drawing *different* random subsets during preview, disk export, and metric evaluation (`MAD`, `MSE`, `MAPE`).
- *Bugs 3–5 (Tuple Swap, Wrong Reference Std, & Index Cap):* Line 168 swapped `(biased, nonbiased)` on the $10^6$ dataset; lines 159 & 169 subtracted `smallDsStd` instead of `mediumDsStd`/`largeDsStd`; and `Experiment2`/`3` capped `r2.Next(0, SMALL_DS_SIZE)` at index $1000$ even when $N = 200,000$.

All five defects are fixed in our upgraded `.NET 8` library (`0` warnings, `100%` unit-test pass rate). Once single-seed noise and truncation bugs are removed, the exact finite-population and non-Gaussian laws emerge cleanly.

#theorem("Theorem 1 (Non-Gaussian Variance of W & SRSWOR Law)")[
  (a) Let $X_1, dots, X_n$ be i.i.d. with mean $mu$, variance $sigma^2 > 0$, and standardized fourth moment $kappa = EE[(X - mu)^4] / sigma^4 >= 1$. For $W = sum_(i=1)^n (X_i - overline(X))^2$ and all $n >= 2$, skewness $mu_3$ cancels identically and:
  $ EE[W] = (n - 1) sigma^2, \ "Var"(W) = (n - 1)/n sigma^4 [(n - 1) kappa - (n - 3)]. $
  (b) Under Simple Random Sampling Without Replacement (SRSWOR) of size $n$ from a finite population ${x_1, dots, x_N}$ with population variance $sigma_"pop"^2 = 1/N sum_(i=1)^N (x_i - mu)^2$ @cochran1977sampling:
  $ EE_"SRSWOR" [S_(n-1)^2] = N/(N - 1) sigma_"pop"^2 > sigma_"pop"^2, \ EE_"SRSWOR" [S_n^2] = (n - 1)/n N/(N - 1) sigma_"pop"^2. $
]

*Proof Sketch.* Writing $W = Y^T A Y$ for $Y_i = X_i - mu$ and idempotent centering matrix $A = I_n - (1/n) bold(1) bold(1)^T$ ($A_(i i) = (n-1)/n$, $A_(i j) = -1/n$), expanding $EE[(Y^T A Y)^2]$ partitions index quadruples $(i, j, k, l)$ into $n$ diagonal terms $A_(i i)^2 kappa sigma^4$, $n(n-1)$ paired diagonal terms $A_(i i)^2 sigma^4$, and $2n(n-1)$ off-diagonal pairs $A_(i j)^2 sigma^4$; odd moments vanish because no partition has pattern $(3, 1)$ with zero-mean independent variables. Algebraic simplification yields $"Var"(W)$, verified symbolically in `SymPy`. Part (b) follows by averaging over all $binom(N, n)$ subsets. $square$

= Kurtosis-Optimal Variance Shrinkage

Unbiasedness ($EE[hat(theta)] = theta$) does not minimize squared estimation error $"MSE"(hat(theta)) = "Var"(hat(theta)) + ("Bias"(hat(theta)))^2$. Using Theorem 1(a), we obtain the exact global minimum-MSE variance divisor across *all* distributions with finite fourth moment:

#theorem("Theorem 2 (Exact Kurtosis-Optimal Variance Divisor)")[
  Among all scale-equivariant variance estimators $hat(sigma)_c^2 = W / c$ ($c > 0$), the unique global minimizer of $"MSE"(hat(sigma)_c^2, sigma^2) = EE[(W / c - sigma^2)^2]$ for sample size $n >= 2$ and kurtosis $kappa >= 1$ is:
  $ c_"var"^*(n, kappa) = (n - 1) + ((n - 1) kappa - (n - 3)) / n. $
  The resulting MSE strictly dominates Bessel's $n - 1$ for every non-degenerate distribution ($kappa > 1$ or $n > 3$), cutting relative MSE by factor $(n - 1) / c_"var"^*(n, kappa)$.
]

*Proof.* Differentiating $EE[W^2]/c^2 - 2 sigma^2 EE[W]/c + sigma^4$ with respect to $1/c$ yields $c_"var"^* = EE[W^2] / (sigma^2 EE[W]) = (n - 1) + "Var"(W) / ((n - 1) sigma^4)$. Substituting Theorem 1(a) gives the exact closed form. $square$

Theorem 2 immediately explains three distinct phenomena:
1. *Platykurtic Uniform Regime ($kappa = 9/5 = 1.8$):* In @rober2023deviation's uniform setup, $c_"var"^*(n, 1.8) = n - 0.2 + 1.2/n$. At $n = 6$, $c_"var"^*(6, 1.8) = 6.0 = n$ — meaning the "biased" divisor $n$ is *literally the exact minimum-MSE oracle*, strictly beating Bessel's $n - 1 = 5$!
2. *Gaussian Regime ($kappa = 3$):* $c_"var"^*(n, 3) = (n - 1) + (2n)/n = n + 1$, recovering Stein's classic inadmissibility theorem @stein1964inadmissibility without normality-specific integration.
3. *Leptokurtic LLM Outlier Regime ($kappa gt.double 3$):* On heavy-tailed Transformer channels where $kappa = 34.1$ at `NVFP4` block size $n = 16$, Bessel divides by $15$, whereas $c_"var"^*(16, 34.1) = 46.18$ ($> 3 times$ larger!), slashing variance MSE by $67.5%$ (oracle) and $55.4%$ (blind online kurtosis estimation; Figure 1B).

#figure(
  image("../figures/fig1_forensic_and_kurtosis_mse.png", width: 100%),
  caption: [*(A)* Forensic audit of the 2023 study ($N = 1000$): removing single-seed sampling noise and integer-mean truncation reveals smooth SRSWOR RMSE curves. *(B)* Relative variance MSE at `NVFP4` block size $n = 16$ across six distributions ($120,000$ trials): `KPS` adapts automatically from Gaussian ($c^* = 17$) to heavy-tailed outliers ($c^* = 46.2$, $-55.4%$ blind MSE vs. Bessel).],
  placement: top,
  scope: "parent",
)

= Power-Scale Concavity–Convexity Duality

In deep learning, variance ($p = 2$) is rarely used directly; quantization scales multiply by $sigma^(+1)$ ($p = 1$), while normalization layers (`LayerNorm`, `BatchNorm`, `GRPO`) multiply by $sigma^(-1)$ ($p = -1$) with gradient variance scaling as $sigma^(-2)$ ($p = -2$).

#theorem("Theorem 3 (Power-Scale Concavity–Convexity Duality Law)")[
  For any $p in RR backslash {0}$ and i.i.d. distribution with kurtosis $kappa >= 1$ and finite eighth moment, let $hat(theta)_(p,"Bessel") = (S_(n-1)^2)^(p/2)$. Then as $n -> infinity$:
  $ (EE[(S_(n-1)^2)^(p/2)] - sigma^p) / sigma^p = (p(p - 2)) / (8n) (kappa - 1) + O(n^(-2)). $
  Furthermore, the MSE-minimizing divisor in $hat(theta)_(p,c) = (W / c)^(p/2)$ satisfies the universal expansion:
  $ c_p^*(n, kappa) = (n - 1) + ((3p - 2) / 4) (kappa - 1) + O(n^(-1)). $
]

*Proof.* Write $S_(n-1)^2 = sigma^2 (1 + xi)$ where $EE[xi] = 0$, $EE[xi^2] = "Var"(W)/((n-1)^2 sigma^4) = (kappa - 1)/n + O(n^(-2))$, and higher cumulants satisfy $EE[xi^3], EE[xi^4] = O(n^(-2))$. Expanding $(1 + xi)^(p/2) = 1 + (p/2) xi + (p(p-2)/8) xi^2 + O(xi^3)$ yields the relative bias law. Minimizing $EE[(lambda Y - sigma^p)^2]$ for $Y = (S_(n-1)^2)^(p/2)$ gives optimal multiplier $lambda^* = sigma^p EE[Y] / EE[Y^2] = 1 - (p(3p-2)/(8n))(kappa - 1) + O(n^(-2))$. Equating $lambda^* = ((n - 1)/c_p^* )^(p/2)$ cancels $p$ identically and yields $c_p^*(n, kappa)$. $square$

Theorem 3 exposes a striking duality verified across $150,000$ Monte Carlo trials in Figure 2A:
- *Concave Regime ($0 < p < 2$, Forward Quantization Scale $p = +1$):* Because $p(p - 2) = -1 < 0$, Bessel's standard deviation *always underestimates* $sigma$ by $-(kappa - 1)/(8n)$, and its optimal MSE divisor shifts above $n - 1$ to $c_1^*(n, kappa) = (n - 1) + (kappa - 1)/4$ ($n - 0.5$ for Gaussians; $> n$ whenever $kappa > 5$).
- *Convex Regime ($p < 0$, Whitening & Advantage Gain $p = -1$):* Because $p(p - 2) = +3 > 0$, inverse standard deviation suffers a *positive upward explosion* $+(3(kappa - 1))/(8n)$ three times steeper than forward std bias! To counteract this upward explosion inside $(W / c)^(-1/2) = sqrt(c / W)$, the MSE-optimal numerator scale shrinks sharply to $c_(-1)^*(n, kappa) = (n - 1) - (5/4)(kappa - 1)$ ($n - 3.5$ for Gaussians), cutting $G = 4$ Laplace whitening MSE by $76.1%$ (`2.361 -> 0.564`; Figure 2B).

#figure(
  image("../figures/fig2_concavity_convexity_duality.png", width: 100%),
  caption: [*(A)* Empirical ($150,000$ MC trials, solid) vs. Theorem 3 asymptotic bias (dashed) at $n = 20$: Bessel plug-in $(S_(n-1)^2)^(p/2)$ flips sign cleanly at $p in {0, 2}$, transitioning from downward concavity bias at $p = +1$ to $3 times$ steeper upward convexity explosion at $p = -1$. *(B)* Decoupled inverse-scale ($p = -1$) MSE on Laplace groups ($G = 3..32$): `KPS` and `EB` shrinkage eliminate the small-$G$ explosion.],
  placement: top,
  scope: "parent",
)

= Coupled vs. Decoupled `GRPO` Duality

When $Z ~ cal(N)(0, sigma^2)$ is normalized by an *independent* sample scale $S_(n-1)$ ($W / sigma^2 ~ chi_(n-1)^2$), exact Gamma integration gives $EE[sigma^2 / S_(n-1)^2] = (n - 1)/(n - 3)$ for $n > 3$ (and $+infinity$ for $n <= 3$) — a *$3.00 times$ squared variance explosion* at $G = 4$. Surprisingly, however, in standard `GRPO` @shao2024deepseekmath, the rollout rewards $r_1, dots, r_G$ appear in *both* numerator and denominator of the policy gradient estimator $hat(g)_G = 1/G sum_(i=1)^G ((r_i - overline(r)_G) / S_(G-1)) nabla_theta log pi_theta (y_i | x)$. Our fourth theorem proves that this numerator-denominator coupling inverts the pathology:

#theorem("Theorem 4 (Gaussian Gamma Laws & Coupled Same-Group GRPO Attenuation)")[
  (a) Under $X_i ~ cal(N)(mu, sigma^2)$, $c_p^*(n, 3) = 2 [Gamma((n - 1 + 2p)/2) / Gamma((n - 1 + p)/2)]^(2/p)$ whenever $n - 1 + 2p > 0$, whose asymptotic expansion $c_p^*(n, 3) - (n - 1) -> (3p - 2)/2$ matches Theorem 3 identically at all $p$ ($+2$ at $p=2$, $+1/2$ at $p=1$, $-5/2$ at $p=-1$, $-4$ at $p=-2$). \
  (b) Let $(r_i, s_i)_(i=1)^G$ be i.i.d. bivariate Gaussian reward and score pairs with true difficulty-normalized gradient $g^* = "Cov"(r, s) / sigma_r$. Then coupled same-group `GRPO` satisfies:
  $ EE[hat(g)_G^"GRPO"] = (G - 1) / G dot c_4(G) dot g^*, \ c_4(G) = sqrt(2 / (G - 1)) Gamma(G / 2) / Gamma((G - 1) / 2) < 1. $
]

*Proof.* Conditioning on $r_(1:G)$, bivariate Gaussian regression (Stein's Lemma) gives $EE[s_i | r_(1:G)] = mu_s + (g^* / sigma_r)(r_i - mu_r)$. Substituting into $hat(g)_G^"GRPO"$ annihilates $mu_s$ via $sum (r_i - overline(r)_G) = 0$ and collapses $sum (r_i - overline(r)_G)(r_i - mu_r) = (G - 1) S_(G-1)^2$, cancelling one power of $S_(G-1)$ in the denominator to leave $((G - 1)/G)(S_(G-1)/sigma_r) g^*$, whose expectation is $((G - 1)/G) c_4(G) g^*$. $square$

At small rollout budgets, Theorem 4(b) proves severe *downward signal attenuation*: `GRPO` retains only $39.89%$ of $g^*$ at $G = 2$ ($-60.1%$ bias!) and $69.10%$ at $G = 4$ ($-30.9%$ bias!). Meanwhile, `Dr. GRPO` @liu2025drgrpo sets $hat(A)_i = r_i - overline(r)_G$, retaining $((G-1)/G) sigma_r g^*$ — which scales linearly with reward units $sigma_r$, allowing high-variance rubric tasks to drown out sparse `{0, 1}` binary math verifiers in multi-task RLVR!

*Empirical-Bayes `EB-GRPO` Solution:* Given prompt group variance $S_(b,G-1)^2 = W_b / (G - 1)$, `EB-GRPO` maintains a bias-corrected per-domain exponential moving average prior $hat(sigma)_("ema",d)^2$ and computes posterior scale $hat(s)_b^2 = (W_b + nu_0 hat(sigma)_("ema",d)^2) / (G - 1 + nu_0)$ ($nu_0 = 4$) alongside exact centering projection compensation $G / (G - 1)$:
$ hat(A)_(b,i)^"EB" = G / (G - 1) dot (r_(b,i) - overline(r)_b) / sqrt(hat(s)_b^2 + epsilon) bold(1)_[W_b > 0]. $
`EB-GRPO` eliminates the $G <= 3$ singularity, recovers $>90%$ gradient signal even at $G = 2$ (Figure 4A), and preserves exact invariance to $100 times$ reward rescaling.

= Experiments on `NVIDIA GB10`

All experiments were executed in `PyTorch 2.11.0+cu130` on an `NVIDIA GB10` Grace Blackwell system (`aarch64`, 121 GiB unified memory) alongside our `.NET 8` C\# suite.

#figure(
  image("../figures/fig3_llm_activations_gb10.png", width: 100%),
  caption: [Live Transformer benchmarks on `NVIDIA GB10`: *(A)* Median token kurtosis across all 28 layers of `Qwen2.5-1.5B-Instruct` ($kappa_"max" = 309.8$) and 24 layers of `BGE-Large-1024D` ($kappa_"max" = 201.3$). *(B)* 4-bit block-$32$ quantization SQNR across `Qwen2.5-1.5B` layers (`KPS` beats Bessel across every layer by $+1.83$ dB mean, while illustrating massive-activation clipping relative to `AbsMax`). *(C)* Token-stream sub-block ($n = 16$) scale ($p = +1$) and whitening ($p = -1$) MSE vs. PyTorch default Bessel.],
  placement: top,
  scope: "parent",
)

== Non-Gaussian Variance & Whitening Risk

Table 1 summarizes $120,000$-trial variance estimation at `NVFP4` block size $n = 16$ and $80,000$-trial Laplace whitening ($p = -1$). On heavy-tailed outlier channels ($kappa = 34.1$), blind online `KPS` cuts variance MSE by $55.35%$ (`2.078 -> 0.928`), while at $G = 4$ rollout size, single-group `KPS` ($p = -1$) reduces whitening MSE by $76.12%$ (`2.361 -> 0.564`) and `EB` cross-group shrinkage reduces it by $98.47%$ (`0.0361`).

#figure(
  placement: bottom,
  scope: "parent",
  table(
    columns: (1.5fr, 0.7fr, 1.0fr, 1.0fr, 1.0fr, 1.0fr, 1.4fr),
    align: (left, right, right, right, right, right, right),
    stroke: 0.5pt + rgb("#cbd5e1"),
    inset: 5pt,
    fill: (_, row) => if row == 0 { rgb("#e2e8f0") } else { none },
    [*Distribution ($n = 16$)*], [*Kurtosis $kappa$*], [*Oracle $c_"var"^*$*], [*Bessel ($n - 1$)*], [*MLE ($n$)*], [*Stein ($n + 1$)*], [*KPS Blind (Ours)*],
    [Uniform], [1.80], [15.88], [0.0584], [*0.0552*], [0.0592], [0.0607 (+4.0%)],
    [Gaussian], [3.00], [17.00], [0.1347], [0.1223], [*0.1186*], [0.1198 (-11.1%)],
    [Logistic], [4.20], [18.12], [0.2066], [0.1857], [0.1750], [*0.1628 (-21.2%)*],
    [Laplace], [6.00], [19.81], [0.3202], [0.2854], [0.2632], [*0.2301 (-28.2%)*],
    [Student-$t_5$], [9.00], [22.62], [0.4862], [0.4311], [0.3921], [*0.2736 (-43.7%)*],
    [Outlier Mixture], [34.13], [46.18], [2.0778], [1.8300], [1.6313], [*0.9277 (-55.4%)*],
  ),
  caption: [Variance estimation MSE across six canonical distributions at `NVFP4` block size $n = 16$ ($120,000$ Monte Carlo trials; bold = lowest MSE among blind estimators; percentages show KPS change vs. Bessel).],
)

== Live Transformer Activations (`Qwen2.5-1.5B` & `BGE-Large`)

Evaluating $3,776$–$4,288$ natural-language tokens across all layers of `Qwen/Qwen2.5-1.5B-Instruct` (28 layers) and `BAAI/bge-large-en-v1.5` (24 layers) on `NVIDIA GB10` confirms extreme departures from normality (Figure 3A): median per-token kurtosis reaches $309.8$ in `Qwen2.5-1.5B` (95th-percentile channel kurtosis $kappa_95 = 61.0$) and $201.3$ in `BGE-Large`. When estimating channel scale from $n = 16$ token micro-blocks (Figure 3C), `KPS` reduces `Qwen2.5-1.5B` forward std MSE by $9.83%$ and inverse-std whitening MSE by $11.64%$, and cuts `BGE-Large` inverse-std whitening MSE by $27.45%$. In 4-bit block quantization (`b = 16` and `b = 32`), `KPS` kurtosis-adaptive clipping consistently outperforms Bessel Gaussian clipping by $+1.49$ to $+1.83$ dB SQNR across both models (`10.25 -> 12.02` dB at `b=16` and `5.76 -> 7.59` dB at `b=32` on `Qwen2.5-1.5B`; Figure 3B).

#figure(
  image("../figures/fig4_grpo_rlvr_gb10.png", width: 100%),
  caption: [`GRPO` policy optimization on `NVIDIA GB10`: *(A)* Empirical verification of Theorem 4(b) coupled signal attenuation (`41.0%` at $G=2$, `69.3%` at $G=4$) vs. `EB-GRPO` (`90.3%`–`98.1%`). *(B, C)* 24-seed multi-task RLVR training ($G = 4$) combining sparse `{0,1}` Hard Math, `{0,1}` Medium Logic, and `[0,15]` Rubric rewards: `Dr. GRPO` starves Hard Math (`7.31%`), while `EB-GRPO` achieves highest Hard Math pass\@1 (`15.54%`) and overall score (`49.27%`).],
  placement: top,
  scope: "parent",
)

== Multi-Task `GRPO` RLVR Policy Training

Figure 4B–C evaluates a shared neural policy trained via `AdamW` across 24 random seeds at small rollout budget $G = 4$ on three simultaneous verifiable tasks: Hard Binary Math ($r in {0, 1}$, random pass\@1 $= 6.25%$), Medium Binary Logic ($r in {0, 1}$, random $= 18.75%$), and a high-variance `[0, 15]` continuous Rubric task ($approx 200 times$ reward variance). As predicted by Section 5:
- `Dr. GRPO` (no `std` normalization) is completely dominated by the `[0, 15]` rubric gradients, starving Hard Binary Math (`7.31%` final pass\@1, barely above `6.25%` chance; `40.85%` balanced score).
- Standard `GRPO` (`Bessel G-1`) preserves task balance (`48.27%`) and reaches `13.32%` Hard Math pass\@1, but suffers $-30.9%$ coupled small-$G$ attenuation.
- `EB-GRPO` achieves *`15.54%` Hard Math pass\@1* (`+112.6%` relative to `Dr. GRPO`, and $+31.4%$ larger gain above random chance than Standard `GRPO`) alongside the highest overall multi-task score (`49.27%`).

= Limitations & Boundary Conditions

Scientific candor requires highlighting two clear boundary regimes observed in our benchmarks:
1. *Platykurtic Blind-Kurtosis Estimation Noise ($kappa < 3$, Small $n$):* On strictly bounded Uniform data ($kappa = 1.8$, $n = 16$), oracle $c_"var"^*(16, 1.8) = 15.88$ beats Bessel by $5.6%$, whereas *blind* single-block kurtosis estimation incurs $+4.0%$ higher MSE than Bessel (`0.0607` vs. `0.0584`) due to fourth-moment sample variance when true tails are lighter than Gaussian. When distributions are known a priori to be bounded uniform, simple MLE ($n$) is preferable.
2. *Unrotated Massive Activations in 4-Bit Token Quantization ($kappa > 300$):* While `KPS` beats Bessel $sigma$-clipping by $+1.83$ dB on `Qwen2.5-1.5B` hidden states, *both* $sigma$-clipping schemes trail `AbsMax` (`24.24` dB; Figure 3B) when quantizing raw unrotated LLM hidden states directly. Because "massive activations" @sun2024massive concentrate $>90%$ of a token's $L_2$ norm into a single coordinate ($|x_"max"| approx 100 sigma$), any $3 sigma$–$5 sigma$ clip truncates the sink feature itself. Thus, `KPS` clipping should be applied *after* Hadamard rotation (`QuaRot`) or SmoothQuant migration @xiao2023smoothquant, or to weight/channel streams where outliers represent tail noise rather than sink coordinates.

= Conclusion

Bessel's $n - 1$ divisor solves only linear variance unbiasedness under infinite-population i.i.d. sampling. Across nonlinear power scales $sigma^p$, heavy-tailed Transformer representations, and small-group `GRPO` reinforcement learning, our Kurtosis-Optimal Power Scale (`KPS`) and `EB-GRPO` estimators provide exact, zero-overhead drop-in replacements in `PyTorch` and `.NET`.

#bibliography("references.bib", style: "ieee")
