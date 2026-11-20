from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gencast_da.cada_features import masked_obs_mse
from gencast_da.diagnostics import diagnostic_means
from gencast_da.random_utils import validation_key


@pytest.mark.parametrize(
    "module",
    (
        "gencast_da.train_cada_controller",
        "gencast_da.smoke_cada",
        "gencast_da.eval_cada_fixed_transition",
    ),
)
def test_entrypoint_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_unobserved_nan_does_not_poison_masked_loss() -> None:
    prediction = jnp.asarray([[[[[2.0, jnp.nan]]]]])
    observation = jnp.asarray([[[[[1.0, 0.0]]]]])
    mask = jnp.asarray([[[[[1.0, 0.0]]]]])

    loss = masked_obs_mse(prediction, observation, mask)

    assert np.isfinite(float(loss))
    assert float(loss) == 1.0


def test_diagnostic_state_is_required_and_count_is_exact() -> None:
    state = {
        "cada": {
            "score_sum": jnp.asarray(7.0),
            "control_sum": jnp.asarray(14.0),
            "u_l2_sum": jnp.asarray(21.0),
            "substep_obs_sum": jnp.asarray(28.0),
            "substep_count": jnp.asarray(7.0),
        }
    }
    diagnostics = diagnostic_means(state)

    assert float(diagnostics["substep_count"]) == 7.0
    assert float(diagnostics["score"]) == 1.0
    assert float(diagnostics["control"]) == 2.0
    with pytest.raises(ValueError, match="missing CADA diagnostics"):
        diagnostic_means({})


def test_validation_rng_depends_only_on_example_and_rollout_index() -> None:
    root = jax.random.PRNGKey(3)

    first = validation_key(root, 2, 1)
    second = validation_key(root, 2, 1)
    different_example = validation_key(root, 3, 1)
    different_rollout = validation_key(root, 2, 2)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_example)
    assert not np.array_equal(first, different_rollout)
