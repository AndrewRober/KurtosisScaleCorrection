"""
test_all.py: Unit & Mathematical Consistency Tests for KPS & EB-GRPO
"""

import math
import numpy as np
import torch
from kps_scale import (
    kps_estimate_np,
    kps_power_scale_torch,
    optimal_variance_divisor,
    exact_gaussian_power_divisor,
    EBGRPONormalizer,
)


def test_stein_and_kurtosis_divisors():
    for n in [2, 5, 16, 32, 100]:
        assert abs(optimal_variance_divisor(n, 3.0) - (n + 1.0)) < 1e-12
    assert abs(optimal_variance_divisor(6, 1.8) - 6.0) < 1e-12
    assert abs(exact_gaussian_power_divisor(16, 2.0) - 17.0) < 1e-10


def test_srswor_unbiased_variance():
    rng = np.random.default_rng(7)
    pop = rng.uniform(0.0, 100.0, size=40)
    true_pop_var = float(np.mean((pop - np.mean(pop)) ** 2))

    est_fpc = []
    est_bessel = []
    for _ in range(15000):
        sub = rng.choice(pop, size=10, replace=False)
        est_fpc.append(float(kps_estimate_np(sub, power=2.0, finite_pop_N=40, mode="unbiased")))
        est_bessel.append(float(np.var(sub, ddof=1)))

    # SRSWOR Bessel overestimates sigma_pop^2 by 40/39 = +2.56%, whereas FPC is unbiased!
    assert abs(np.mean(est_fpc) - true_pop_var) / true_pop_var < 0.006
    assert (np.mean(est_bessel) - true_pop_var) / true_pop_var > 0.018


def test_numpy_and_torch_parity():
    rng = np.random.default_rng(99)
    x_np = rng.laplace(0.0, 2.5, size=(12, 16))
    x_th = torch.tensor(x_np, dtype=torch.float64)

    for p in [2.0, 1.0, -1.0]:
        v_np = kps_estimate_np(x_np, power=p, axis=-1, mode="mse")
        v_th = kps_power_scale_torch(x_th, power=p, dim=-1, keepdim=False).numpy()
        assert np.allclose(v_np, v_th, rtol=1e-5, atol=1e-6), f"Mismatch at p={p}"


def test_eb_grpo_zero_collapse_and_scale_invariance():
    norm = EBGRPONormalizer(num_domains=2, nu0=4.0, beta=0.9)
    r1 = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    dom = torch.tensor([0, 1], dtype=torch.long)
    adv1 = norm.normalize(r1, dom)
    assert torch.all(adv1[0] == 0.0)
    assert abs(float(adv1[1].sum().item())) < 1e-5

    # Scaling rewards by 100x on a fresh normalizer yields identical normalized advantages!
    n_a = EBGRPONormalizer(num_domains=1, nu0=4.0, beta=0.9, eps=1e-12)
    n_b = EBGRPONormalizer(num_domains=1, nu0=4.0, beta=0.9, eps=1e-12)
    r_base = torch.tensor([[1.0, 2.0, 3.0, 6.0]], dtype=torch.float64)
    a_base = n_a.normalize(r_base)
    a_scaled = n_b.normalize(100.0 * r_base)
    assert torch.allclose(a_base, a_scaled, rtol=1e-6)


if __name__ == "__main__":
    test_stein_and_kurtosis_divisors()
    test_srswor_unbiased_variance()
    test_numpy_and_torch_parity()
    test_eb_grpo_zero_collapse_and_scale_invariance()
    print("ALL PYTHON & PYTORCH KPS / EB-GRPO TESTS PASSED!")
