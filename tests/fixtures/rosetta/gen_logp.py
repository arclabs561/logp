# /// script
# requires-python = ">=3.10"
# dependencies = ["scipy", "scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for logp information-theory primitives.

Provenance for logp_infotheory.json. logp works in NATS (natural log), so every
reference uses base e.

Library oracles (external, trusted):
  entropy_nats              -> scipy.stats.entropy(p)            [base e]
  kl_divergence             -> scipy.stats.entropy(p, q)         [KL, base e]
  jensen_shannon_divergence -> scipy.spatial.distance.jensenshannon(p, q)**2
  mutual_information        -> sklearn.metrics.mutual_info_score(contingency)

Cross-implementation checks (numpy formula, no scipy/sklearn function exists):
  cross_entropy_nats, bhattacharyya_coeff, hellinger_squared,
  renyi_entropy, tsallis_entropy.

Deferred (no clean external oracle): Bregman, Csiszar f-divergence, alpha family.

Regenerate: uv run tests/fixtures/rosetta/gen_logp.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score

SEED = 0
rng = np.random.default_rng(SEED)

# Two strictly-positive distributions on the 5-simplex (no zeros, so KL and
# cross-entropy are finite).
p = rng.dirichlet(np.full(5, 2.0))
q = rng.dirichlet(np.full(5, 2.0))

alpha = 2.0

# Joint distribution for mutual information: an integer contingency table,
# normalized to a row-major joint. sklearn.mutual_info_score computes MI in nats
# from the contingency counts, which is logp's exact oracle.
n_x, n_y = 3, 4
counts = rng.integers(1, 12, size=(n_x, n_y))
p_xy = (counts / counts.sum()).reshape(-1)  # row-major flatten

expected = {
    # Library oracles.
    "entropy_p": float(scipy_entropy(p)),
    "kl_pq": float(scipy_entropy(p, q)),
    "js_pq": float(jensenshannon(p, q, base=None) ** 2),
    "mutual_information": float(mutual_info_score(None, None, contingency=counts)),
    # Cross-implementation (numpy formula).
    "cross_entropy_pq": float(-np.sum(p * np.log(q))),
    "bhattacharyya_pq": float(np.sum(np.sqrt(p * q))),
    "hellinger_squared_pq": float(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)),
    "renyi_p": float(np.log(np.sum(p**alpha)) / (1.0 - alpha)),
    "tsallis_p": float((1.0 - np.sum(p**alpha)) / (alpha - 1.0)),
}

fixture = {
    "provenance": {
        "generator": "gen_logp.py",
        "libraries": "scipy + scikit-learn",
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "units": "nats (natural log)",
        "note": "entropy/kl/js/mi are library oracles; the rest are numpy-formula cross-checks.",
    },
    "p": p.tolist(),
    "q": q.tolist(),
    "p_xy": p_xy.tolist(),
    "n_x": n_x,
    "n_y": n_y,
    "alpha": alpha,
    "expected": expected,
}

out = Path(__file__).parent / "logp_infotheory.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
for k, v in expected.items():
    print(f"{k:24s} {v:.12f}")
print(f"wrote {out}")
