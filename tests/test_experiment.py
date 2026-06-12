"""Tests for the experiment lines (T2-04 cf_gradient, T2-05 active_loop).
Small/fast by design; the full runs live under experiments/ with frozen prereg."""
import numpy as np
import pytest
from theone.experiment.cf_gradient import (build_dataset, DualHeadMLP,
                                           vital_sign_1, baseline_intervention_mse_ci)


class TestCfGradientToy:
    def test_dataset_has_confounding_signature(self):
        Phi, yf, yc = build_dataset(64, base_seed=1)
        assert Phi.shape == (64, 5)
        # observational slope (factual) > true ATE (counterfactual) under +confounding
        assert yf.mean() > yc.mean()

    def test_training_is_deterministic_under_seed(self):
        Phi, yf, yc = build_dataset(32, base_seed=2)
        mu, sd = Phi.mean(0), Phi.std(0) + 1e-9
        X = (Phi - mu) / sd
        a = DualHeadMLP(seed=5).train(X, yf, yc, lam=1.0, epochs=50)
        b = DualHeadMLP(seed=5).train(X, yf, yc, lam=1.0, epochs=50)
        assert a[0][-1] == pytest.approx(b[0][-1])
        assert a[1][-1] == pytest.approx(b[1][-1])

    def test_lambda1_learns_factual_lambda0_leaves_cf_unsupervised(self):
        Phi, yf, yc = build_dataset(48, base_seed=3)
        mu, sd = Phi.mean(0), Phi.std(0) + 1e-9
        X = (Phi - mu) / sd
        hf1, hc1 = DualHeadMLP(seed=9).train(X, yf, yc, lam=1.0, epochs=150)
        hf0, hc0 = DualHeadMLP(seed=9).train(X, yf, yc, lam=0.0, epochs=150)
        assert hf1[-1] < 0.25 * hf1[0]            # factual head learns
        assert hc1[-1] < 0.25 * hc1[0]            # cf head learns when lambda>0
        # lambda=0: cf head weights get ZERO gradient (frozen); any cf-loss change is
        # incidental trunk drift, NOT learning -> supervised cf loss is far lower.
        assert hc1[-1] < 0.2 * hc0[-1]

    def test_vital_sign_1_threshold_logic(self):
        # both heads drop >=1% for >=3 consecutive epochs -> True
        h = [1.0, 0.98, 0.96, 0.94, 0.92]
        assert vital_sign_1(h, h) is True
        # one head flat -> False (must be BOTH)
        flat = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert vital_sign_1(h, flat) is False
        # drops below 1% -> False
        slow = [1.0, 0.999, 0.998, 0.997, 0.996]
        assert vital_sign_1(slow, slow) is False

    def test_baseline_ci_lower_below_point(self):
        yf = np.array([0.6, 0.7, 0.5, 0.8, 0.55])
        yc = np.array([0.3, 0.2, 0.25, 0.4, 0.3])
        mse, lo = baseline_intervention_mse_ci(yf, yc, seed=1, B=300)
        assert lo <= mse and mse > 0
