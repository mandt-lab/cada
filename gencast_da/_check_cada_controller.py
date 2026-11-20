"""CPU-only sanity check for `gencast_da.cada_controller.CADAController`.

Runs three quick checks that don't need GPU or GenCast:

  Test 1 (shape parity)
      Output shape = noisy_T shape, regardless of inputs.
  Test 2 (zero-init)
      With newly-initialised params (zero-init final projection), output
      MUST be identically zero for any inputs.
  Test 3 (gradient flow)
      A first MSE update must move the zero-initialized output projection;
      the following backward pass must reach upstream controller parameters.

Run:
    python -m gencast_da._check_cada_controller
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from gencast_da.cada_controller import CADAController


def main() -> None:
    B, L, H, W = 1, 13, 32, 64
    rng = jax.random.PRNGKey(0)

    def fwd(noisy_T, cond_T, obs_like, mask, last_u, sigma_t):
        net = CADAController(num_levels=L, hidden_channels=8, num_blocks=2)
        return net(noisy_T, cond_T, obs_like, mask, last_u, sigma_t)

    transformed = hk.transform(fwd)

    noisy_T = jnp.asarray(np.random.randn(B, L, H, W).astype(np.float32))
    cond_T = jnp.asarray(np.random.randn(B, L, H, W).astype(np.float32))
    obs_like = jnp.asarray(np.random.randn(B, L, H, W).astype(np.float32))
    mask = jnp.asarray((np.random.rand(B, L, H, W) > 0.5).astype(np.float32))
    last_u = jnp.zeros((B, L, H, W), dtype=jnp.float32)
    sigma_t = jnp.asarray(np.array([1.5], dtype=np.float32))

    params = transformed.init(rng, noisy_T, cond_T, obs_like, mask, last_u, sigma_t)
    n_params = sum(p.size for d in params.values() for p in d.values())
    print(f"[check_cada_ctrl] controller has {n_params:,} parameters")
    print(f"[check_cada_ctrl] scopes: {sorted(params.keys())}")

    # Test 1
    u = transformed.apply(
        params, None, noisy_T, cond_T, obs_like, mask, last_u, sigma_t
    )
    assert u.shape == noisy_T.shape, (
        f"Test 1 FAILED: output shape {u.shape} != input {noisy_T.shape}"
    )
    print(f"[check_cada_ctrl] Test 1 PASS  (output shape {u.shape})")

    # Test 2
    u_max = float(jnp.max(jnp.abs(u)))
    assert u_max == 0.0, (
        f"Test 2 FAILED: zero-init final projection but output max = {u_max}"
    )
    print("[check_cada_ctrl] Test 2 PASS  (zero-init -> u == 0)")

    # Test 3
    target = jnp.asarray(np.random.randn(*u.shape).astype(np.float32) * 0.1)

    def loss_fn(p):
        u_pred = transformed.apply(
            p, None, noisy_T, cond_T, obs_like, mask, last_u, sigma_t
        )
        return jnp.mean((u_pred - target) ** 2)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    leaves = [
        (scope, float(jnp.max(jnp.abs(value))))
        for scope, values in grads.items()
        for value in values.values()
    ]
    assert all(np.isfinite(value) for _, value in leaves)
    output_max = max(value for scope, value in leaves if "proj_out" in scope)
    assert np.isfinite(loss_val), "Test 3 FAILED: loss is non-finite"
    assert output_max > 0.0, "Test 3 FAILED: output projection has no gradient"

    updated = jax.tree_util.tree_map(
        lambda parameter, gradient: parameter - 0.1 * gradient,
        params,
        grads,
    )
    _, second_grads = jax.value_and_grad(loss_fn)(updated)
    upstream = [
        float(jnp.max(jnp.abs(value)))
        for scope, values in second_grads.items()
        if "proj_out" not in scope
        for value in values.values()
    ]
    assert upstream and all(np.isfinite(value) for value in upstream)
    upstream_max = max(upstream)
    assert upstream_max > 0.0, "Test 3 FAILED: gradient did not reach upstream layers"
    print(
        "[check_cada_ctrl] Test 3 PASS  "
        f"(loss={float(loss_val):.4f}, upstream max grad={upstream_max:.3e})"
    )

    print("[check_cada_ctrl] ALL THREE TESTS PASS")


if __name__ == "__main__":
    main()
