"""
kps_scale.py: Kurtosis- & Power-Optimal Scale Estimation (KPS) & Empirical-Bayes GRPO (EB-GRPO)
===============================================================================================
NumPy + PyTorch (CPU / CUDA) reference library implementing:
  - Theorem 1: Finite-Population SRSWOR variance/scale correction & non-Gaussian 4th-moment variance law
  - Theorem 2: Exact Minimum-MSE Kurtosis-Adaptive Variance Estimator W / c_var*(n, kappa)
  - Theorem 3: Universal Power-Scale Concavity-Convexity Duality Estimator for sigma^p:
                 * mode="unbiased": cures -(kappa - 1)/(8n) std bias and +3(kappa - 1)/(8n) whitening bias
                 * mode="mse":      applies universal divisor c_p*(n, kappa) = (n - 1) + ((3p - 2)/4)*(kappa - 1)
  - Theorem 4: Coupled vs. Decoupled Normalization Duality & Empirical-Bayes GRPO (EB-GRPO)
               for small rollout groups G in {2, 4, 8, 16} across heterogeneous verifiable rewards.
"""

from __future__ import annotations
import math
import numpy as np
from scipy import special

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def stabilized_kurtosis_np(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean, W, kappa_hat) along `axis` in O(n) vectorized NumPy.
    Applies Fisher finite-sample excess-kurtosis adjustment shrunk smoothly toward Gaussian (3.0)
    at small sample sizes n, while respecting Cauchy-Schwarz kappa >= 1.0.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    if n < 2:
        raise ValueError(f"Sample size along axis={axis} must be >= 2, got {n}.")

    mean = np.mean(x, axis=axis, keepdims=True)
    diff = x - mean
    d2 = diff * diff
    w = np.sum(d2, axis=axis, keepdims=keepdims)
    m4 = np.sum(d2 * d2, axis=axis, keepdims=keepdims)

    denom = np.maximum(w * w, 1e-60)
    raw_kurt = np.where(w > 1e-30, (n * m4) / denom, 3.0)

    if n >= 4:
        excess = ((n + 1.0) * raw_kurt - 3.0 * (n - 1.0)) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
        weight = (n - 3.0) / (n + 5.0)
        kurt = 3.0 + weight * excess
    else:
        kurt = raw_kurt

    kurt = np.maximum(1.0, kurt)
    if not keepdims:
        mean = np.squeeze(mean, axis=axis)
    return mean, w, kurt


def optimal_variance_divisor(n: int, kappa: np.ndarray | float) -> np.ndarray | float:
    """Theorem 2: Exact minimum-MSE divisor c_var*(n, kappa) = (n - 1) + ((n - 1)*kappa - (n - 3)) / n."""
    kappa_c = np.maximum(1.0, kappa)
    return (n - 1.0) + ((n - 1.0) * kappa_c - (n - 3.0)) / float(n)


def exact_gaussian_power_divisor(n: int, power: float) -> float:
    """
    Theorem 4(a): Exact finite-n Gaussian MSE-optimal divisor c_p*(n, 3) via Gamma functions:
      c_p*(n, 3) = (E[Z^p] / E[Z^(p/2)])^(2/p) = 2 * (Gamma((n - 1 + 2p)/2) / Gamma((n - 1 + p)/2))^(2/p)
    when n - 1 + 2p > 0. Falls back smoothly to guarded positive floor at singular small-n negative powers.
    """
    nu = n - 1.0
    if abs(power) < 1e-12:
        return nu
    if nu + 2.0 * power <= 0.25:
        # Below inverse-chi-square 2nd-moment existence threshold (e.g. p = -1, n <= 3)
        return max(0.35, 0.22 * nu)
    log_ratio = (
        0.5 * power * math.log(2.0)
        + special.gammaln(0.5 * (nu + 2.0 * power))
        - special.gammaln(0.5 * (nu + power))
    )
    return math.exp((2.0 / power) * log_ratio)


def kps_estimate_np(
    x: np.ndarray,
    power: float = 1.0,
    axis: int = -1,
    known_kappa: float | np.ndarray | None = None,
    finite_pop_N: int | None = None,
    mode: str = "mse",
    eps: float = 1e-12,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Unified Kurtosis- & Power-Optimal Scale Estimator (KPS) for sigma^power.
      - power =  2.0 : Variance (sigma^2)
      - power =  1.0 : Standard Deviation (sigma)
      - power = -1.0 : Inverse Standard Deviation / Whitening Gain (sigma^-1)
      - power = -2.0 : Precision / Inverse Variance (sigma^-2)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    _, w, est_kappa = stabilized_kurtosis_np(x, axis=axis, keepdims=True)
    kappa = np.maximum(1.0, est_kappa if known_kappa is None else known_kappa)

    fpc = 1.0
    if finite_pop_N is not None and finite_pop_N > n:
        # Theorem 1(b): Exact SRSWOR factor targeting finite-population sigma_pop^2 = (1/N) sum(X_i - mu)^2
        fpc = (finite_pop_N - 1.0) / float(finite_pop_N)

    if mode == "unbiased":
        s2 = (w / (n - 1.0)) * fpc
        if abs(power - 2.0) < 1e-12:
            out = s2
        else:
            # Theorem 3(a): Second-order non-Gaussian concavity/convexity relative bias correction
            rel_bias = (power * (power - 2.0) * (kappa - 1.0)) / (8.0 * n)
            corr = np.clip(1.0 + rel_bias, 0.35, 3.0)
            out = np.power(np.maximum(s2, eps), 0.5 * power) / corr
    elif mode == "mse":
        if abs(power - 2.0) < 1e-12:
            c_star = optimal_variance_divisor(n, kappa) / fpc
            out = w / c_star
        else:
            # Anchor at exact finite-n Gaussian Gamma divisor + Theorem 3 non-Gaussian kurtosis shift!
            c_gauss = exact_gaussian_power_divisor(n, power)
            delta_non_gauss = 0.25 * (3.0 * power - 2.0) * (kappa - 3.0)
            min_div = max(0.35, 0.22 * (n - 1.0)) if power < 0 else 0.5
            c_star = np.maximum(min_div, c_gauss + delta_non_gauss) / fpc
            out = np.power((w + eps) / c_star, 0.5 * power)
    else:
        raise ValueError(f"Unknown mode='{mode}', expected 'mse' or 'unbiased'.")

    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


if HAS_TORCH:
    def stabilized_kurtosis_torch(x: "torch.Tensor", dim: int = -1, keepdim: bool = True):
        n = x.shape[dim]
        mean = x.mean(dim=dim, keepdim=True)
        diff = x - mean
        d2 = diff * diff
        w = d2.sum(dim=dim, keepdim=keepdim)
        m4 = (d2 * d2).sum(dim=dim, keepdim=keepdim)

        raw_kurt = torch.where(w > 1e-24, (n * m4) / torch.clamp(w * w, min=1e-48), torch.full_like(w, 3.0))
        if n >= 4:
            excess = ((n + 1.0) * raw_kurt - 3.0 * (n - 1.0)) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
            weight = (n - 3.0) / (n + 5.0)
            kurt = 3.0 + weight * excess
        else:
            kurt = raw_kurt
        kurt = torch.clamp(kurt, min=1.0, max=250.0)
        if not keepdim:
            mean = mean.squeeze(dim)
        return mean, w, kurt

    def kps_power_scale_torch(
        x: "torch.Tensor",
        power: float = 1.0,
        dim: int = -1,
        known_kappa: float | None = None,
        eps: float = 1e-12,
        keepdim: bool = True,
    ) -> "torch.Tensor":
        """PyTorch GPU implementation of Theorem 2 & Theorem 3 KPS scale estimator."""
        n = x.shape[dim]
        _, w, est_kappa = stabilized_kurtosis_torch(x, dim=dim, keepdim=True)
        kappa = est_kappa if known_kappa is None else torch.full_like(est_kappa, float(known_kappa))

        if abs(power - 2.0) < 1e-12:
            c_star = (n - 1.0) + ((n - 1.0) * kappa - (n - 3.0)) / float(n)
            out = w / c_star
        else:
            c_gauss = exact_gaussian_power_divisor(n, power)
            delta_non_gauss = 0.25 * (3.0 * power - 2.0) * (kappa - 3.0)
            min_div = max(0.35, 0.22 * (n - 1.0)) if power < 0 else 0.5
            c_star = torch.clamp(c_gauss + delta_non_gauss, min=min_div)
            out = torch.pow((w + eps) / c_star, 0.5 * power)

        if not keepdim:
            out = out.squeeze(dim)
        return out

    class EBGRPONormalizer:
        """
        Theorem 4: Empirical-Bayes Group Advantage Normalizer (EB-GRPO) in PyTorch.
        Supports per-task/domain or global bias-corrected variance EMA + exact G/(G-1) Stein gain recovery.
        """
        def __init__(self, num_domains: int = 1, nu0: float = 4.0, beta: float = 0.95, eps: float = 1e-8, device: str = "cpu"):
            self.nu0 = nu0
            self.beta = beta
            self.eps = eps
            self.ema_var = torch.zeros(num_domains, dtype=torch.float64, device=device)
            self.steps = torch.zeros(num_domains, dtype=torch.long, device=device)

        @torch.no_grad()
        def normalize(self, rewards: "torch.Tensor", domain_ids: "torch.Tensor | None" = None) -> "torch.Tensor":
            """
            rewards: shape (B, G) where B is prompt batch size and G is rollout group size.
            domain_ids: optional integer tensor of shape (B,) indexing reward domain/scale family.
            """
            orig_dtype = rewards.dtype
            r = rewards.to(torch.float64)
            B, G = r.shape
            if domain_ids is None:
                domain_ids = torch.zeros(B, dtype=torch.long, device=r.device)

            mean = r.mean(dim=-1, keepdim=True)
            diff = r - mean
            w = (diff * diff).sum(dim=-1)  # (B,)
            s2 = w / max(G - 1, 1)

            # Update per-domain bias-corrected EMA variance using batch-mean sample variance
            prior_vars = torch.zeros_like(s2)
            for d in torch.unique(domain_ids):
                mask = (domain_ids == d)
                batch_var_d = s2[mask].mean()
                self.steps[d] += 1
                self.ema_var[d] = self.beta * self.ema_var[d] + (1.0 - self.beta) * batch_var_d
                bc = 1.0 - (self.beta ** int(self.steps[d].item()))
                prior_vars[mask] = self.ema_var[d] / max(bc, 1e-12)

            post_var = (w + self.nu0 * prior_vars) / ((G - 1.0) + self.nu0)
            denom = torch.sqrt(post_var + self.eps).unsqueeze(-1)
            gain = G / max(G - 1.0, 1.0)

            active = (w > 1e-24).unsqueeze(-1)
            adv = torch.where(active, gain * diff / denom, torch.zeros_like(diff))
            return adv.to(orig_dtype)
