import numpy as np
from fairness_audit import fairness_metrics


def test_fairness_metrics_known_case():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.1, 0.9, 0.1, 0.8, 0.7, 0.9])
    m = fairness_metrics(y, p, g, risk_cutoff=0.5)
    assert np.isclose(m["approval_rate_reference"], 0.75)
    assert np.isclose(m["approval_rate_protected"], 0.25)
    assert np.isclose(m["disparate_impact_ratio"], 1/3)
    assert np.isclose(m["equal_opportunity_difference"], -0.5)
