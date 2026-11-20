"""Reproducible, batch-aware metrics for CADA evaluation.

Unless stated otherwise, trajectory tensors use ``[B, T, C, ...]`` and
functions preserve the leading batch/time dimensions.  No function silently
aligns, truncates, or broadcasts prediction and truth trajectories.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Literal, ParamSpec, TypeVar

import torch

Reduction = Literal["none", "batch_mean", "mean"]
P = ParamSpec("P")
R = TypeVar("R")


def _check_pair(prediction: torch.Tensor, truth: torch.Tensor) -> None:
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction and truth must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(truth.shape)}"
        )
    if prediction.ndim < 3:
        raise ValueError("expected trajectory tensors shaped [B, T, ...]")
    if not prediction.is_floating_point() or not truth.is_floating_point():
        raise TypeError("metrics require floating-point tensors")


def _event_dimensions(
    tensor: torch.Tensor, event_dims: Sequence[int] | None
) -> tuple[int, ...]:
    if event_dims is None:
        dims = tuple(range(2, tensor.ndim))
    else:
        dims = tuple(
            dim if dim >= 0 else tensor.ndim + dim for dim in event_dims
        )
    if not dims:
        raise ValueError("at least one event dimension is required")
    if len(set(dims)) != len(dims) or any(
        dim < 0 or dim >= tensor.ndim for dim in dims
    ):
        raise ValueError(f"invalid event dimensions {event_dims}")
    if 0 in dims or 1 in dims:
        raise ValueError("event dimensions may not include batch (0) or time (1)")
    return dims


def _reduce_batch(metric: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return metric
    if reduction == "batch_mean":
        return metric.mean(dim=0)
    if reduction == "mean":
        return metric.mean()
    raise ValueError("reduction must be 'none', 'batch_mean', or 'mean'")


def rmse(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    event_dims: Sequence[int] | None = None,
    reduction: Reduction = "none",
) -> torch.Tensor:
    """Compute root-mean-square error with explicit reductions.

    By default the event dimensions are every dimension after ``B, T``.
    ``reduction='none'`` returns one RMSE per batch item and timestep
    (``[B, T]``); ``'batch_mean'`` averages those per-item RMSE values over
    batch (``[T]``).  ``'mean'`` is the conventional global RMSE,
    ``sqrt(mean((prediction - truth)**2))``; it is deliberately *not* the
    mean of the ``[B, T]`` RMSE values.
    """

    _check_pair(prediction, truth)
    dims = _event_dimensions(prediction, event_dims)
    squared_error = (prediction - truth).square()
    if reduction == "mean":
        return squared_error.mean().sqrt()
    per_item = squared_error.mean(dim=dims).sqrt()
    return _reduce_batch(per_item, reduction)


def pearson_correlation(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    event_dims: Sequence[int] | None = None,
    reduction: Reduction = "none",
) -> torch.Tensor:
    """Return Pearson correlation over event dimensions.

    The default output is ``[B, T]``. ``batch_mean`` returns the arithmetic
    mean correlation curve over batch and ``mean`` returns the arithmetic mean
    over batch and time. Population covariance/variance conventions are used;
    their common normalization cancels in the correlation coefficient.
    Correlation is undefined, and returned as ``NaN``, when either centered
    event has zero norm.
    """

    _check_pair(prediction, truth)
    dims = _event_dimensions(prediction, event_dims)
    pred_centered = prediction - prediction.mean(dim=dims, keepdim=True)
    truth_centered = truth - truth.mean(dim=dims, keepdim=True)
    numerator = (pred_centered * truth_centered).sum(dim=dims)
    pred_norm = pred_centered.square().sum(dim=dims).sqrt()
    truth_norm = truth_centered.square().sum(dim=dims).sqrt()
    denominator = pred_norm * truth_norm
    correlation = numerator / denominator
    correlation = torch.where(
        denominator > 0,
        correlation,
        torch.full_like(correlation, float("nan")),
    )
    return _reduce_batch(correlation, reduction)


def high_correlation_time(
    correlation: torch.Tensor,
    *,
    threshold: float = 0.9,
    dt: float = 1.0,
    time_origin: float = 0.0,
) -> torch.Tensor:
    """Return the last one-based time whose correlation is ``>= threshold``.

    Time is the final axis, so ``[B, T]`` input returns ``[B]`` and ``[T]``
    input returns a scalar tensor. Following the existing evaluation scripts,
    index ``i`` maps to ``time_origin + (i + 1) * dt``. A trajectory that never
    reaches the threshold returns ``NaN``. Non-finite correlations do not
    qualify. Equality qualifies, in particular a correlation of exactly 0.9
    at the default threshold.
    """

    if correlation.ndim < 1 or correlation.shape[-1] == 0:
        raise ValueError("correlation must have a non-empty time dimension")
    if not correlation.is_floating_point():
        raise TypeError("correlation must be floating point")
    if not torch.isfinite(torch.tensor(threshold)) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [-1, 1]")
    if not dt > 0:
        raise ValueError("dt must be positive")

    qualifies = torch.isfinite(correlation) & (correlation >= threshold)
    steps = torch.arange(
        1,
        correlation.shape[-1] + 1,
        device=correlation.device,
        dtype=correlation.dtype,
    )
    last_step = torch.where(qualifies, steps, torch.zeros_like(steps)).amax(
        dim=-1
    )
    times = time_origin + last_step * dt
    return torch.where(
        last_step > 0, times, torch.full_like(times, float("nan"))
    )


def periodic_total_variation_1d(
    trajectories: torch.Tensor,
    *,
    normalize: bool = False,
    reduction: Reduction = "none",
) -> torch.Tensor:
    """Compute periodic discrete total variation along the last dimension.

    For ``[B, T, C, W]`` input, the default result is ``[B, T, C]`` and equals
    ``sum_i |u[i+1 mod W] - u[i]|``. Set ``normalize=True`` for the mean
    absolute periodic difference instead of the sum. Batch reductions retain
    all non-batch dimensions.
    """

    if trajectories.ndim < 3:
        raise ValueError("expected at least [B, T, W]")
    if not trajectories.is_floating_point():
        raise TypeError("total variation requires a floating-point tensor")
    if trajectories.shape[-1] < 2:
        raise ValueError("the periodic spatial dimension needs at least 2 points")
    difference = torch.roll(trajectories, shifts=-1, dims=-1) - trajectories
    variation = (
        difference.abs().mean(dim=-1)
        if normalize
        else difference.abs().sum(dim=-1)
    )
    return _reduce_batch(variation, reduction)


def _positive_spacing(spacing: float, name: str) -> float:
    value = float(spacing)
    if not value > 0:
        raise ValueError(f"{name} must be positive")
    return value


def _check_velocity(
    velocity: torch.Tensor, u_channel: int, v_channel: int
) -> None:
    if velocity.ndim < 3:
        raise ValueError("velocity must be shaped [..., C, H, W]")
    if not velocity.is_floating_point():
        raise TypeError("velocity must be floating point")
    channels = velocity.shape[-3]
    if min(u_channel, v_channel) < 0 or max(u_channel, v_channel) >= channels:
        raise ValueError(
            f"velocity has {channels} channels, cannot select "
            f"u={u_channel}, v={v_channel}"
        )
    if velocity.shape[-2] < 3 or velocity.shape[-1] < 3:
        raise ValueError("both spatial dimensions need at least 3 points")


def _periodic_central_difference(
    field: torch.Tensor, spacing: float, dim: int
) -> torch.Tensor:
    return (
        torch.roll(field, shifts=-1, dims=dim)
        - torch.roll(field, shifts=1, dims=dim)
    ) / (2.0 * spacing)


def _nonperiodic_central_difference(
    field: torch.Tensor, spacing: float, dim: int
) -> torch.Tensor:
    """Second-order central interior and one-sided boundary differences."""

    result = torch.empty_like(field)
    middle = [slice(None)] * field.ndim
    ahead = [slice(None)] * field.ndim
    behind = [slice(None)] * field.ndim
    middle[dim] = slice(1, -1)
    ahead[dim] = slice(2, None)
    behind[dim] = slice(None, -2)
    result[tuple(middle)] = (
        field[tuple(ahead)] - field[tuple(behind)]
    ) / (2.0 * spacing)

    first = [slice(None)] * field.ndim
    second = [slice(None)] * field.ndim
    third = [slice(None)] * field.ndim
    first[dim], second[dim], third[dim] = 0, 1, 2
    result[tuple(first)] = (
        -3.0 * field[tuple(first)]
        + 4.0 * field[tuple(second)]
        - field[tuple(third)]
    ) / (2.0 * spacing)

    last = [slice(None)] * field.ndim
    penultimate = [slice(None)] * field.ndim
    antepenultimate = [slice(None)] * field.ndim
    last[dim], penultimate[dim], antepenultimate[dim] = -1, -2, -3
    result[tuple(last)] = (
        3.0 * field[tuple(last)]
        - 4.0 * field[tuple(penultimate)]
        + field[tuple(antepenultimate)]
    ) / (2.0 * spacing)
    return result


def vorticity_2d_finite_difference(
    velocity: torch.Tensor,
    *,
    dx: float,
    dy: float,
    u_channel: int = 0,
    v_channel: int = 1,
) -> torch.Tensor:
    """Compute periodic 2D vorticity with centered finite differences.

    Input is ``[..., C, H, W]`` with ``x`` on width and ``y`` on height;
    output is ``[..., H, W]``. The grid is uniform and endpoint-exclusive:
    samples represent a periodic domain of lengths ``Lx=W*dx`` and
    ``Ly=H*dy``. The returned convention is ``omega = dv/dx - du/dy``.
    """

    _check_velocity(velocity, u_channel, v_channel)
    dx = _positive_spacing(dx, "dx")
    dy = _positive_spacing(dy, "dy")
    u = velocity[..., u_channel, :, :]
    v = velocity[..., v_channel, :, :]
    return _periodic_central_difference(v, dx, -1) - _periodic_central_difference(
        u, dy, -2
    )


def vorticity_2d_spectral(
    velocity: torch.Tensor,
    *,
    dx: float,
    dy: float,
    u_channel: int = 0,
    v_channel: int = 1,
) -> torch.Tensor:
    """Compute periodic 2D vorticity using FFT derivatives.

    Input and output axes follow :func:`vorticity_2d_finite_difference`.
    Samples must lie on a uniform, endpoint-exclusive periodic grid
    ``[0, W*dx) x [0, H*dy)``. Frequencies are angular wavenumbers
    ``2*pi*fftfreq``. Nyquist modes use PyTorch's FFT ordering; for real input
    the real part of the inverse transform is returned.
    """

    _check_velocity(velocity, u_channel, v_channel)
    dx = _positive_spacing(dx, "dx")
    dy = _positive_spacing(dy, "dy")
    u = velocity[..., u_channel, :, :]
    v = velocity[..., v_channel, :, :]
    real_dtype = velocity.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(
        velocity.shape[-1], d=dx, device=velocity.device, dtype=real_dtype
    )
    ky = 2.0 * torch.pi * torch.fft.fftfreq(
        velocity.shape[-2], d=dy, device=velocity.device, dtype=real_dtype
    )
    dv_dx = torch.fft.ifft2(
        torch.fft.fft2(v, dim=(-2, -1)) * (1j * kx.view(1, -1)),
        dim=(-2, -1),
    ).real
    du_dy = torch.fft.ifft2(
        torch.fft.fft2(u, dim=(-2, -1)) * (1j * ky.view(-1, 1)),
        dim=(-2, -1),
    ).real
    return dv_dx - du_dy


def kolmogorov_dissipation(
    velocity: torch.Tensor,
    *,
    viscosity: float,
    dx: float,
    dy: float,
    u_channel: int = 0,
    v_channel: int = 1,
    reduction: Reduction = "none",
) -> torch.Tensor:
    """Return the periodic Kolmogorov-flow kinetic-energy dissipation rate.

    Centered finite differences use the same grid convention as
    :func:`vorticity_2d_finite_difference`. The pointwise rate is
    ``2*nu*(Sxx**2 + Syy**2 + 2*Sxy**2)`` and is spatially averaged, yielding
    the leading dimensions (normally ``[B, T]``). ``batch_mean`` returns
    ``[T]`` and ``mean`` returns one scalar.
    """

    _check_velocity(velocity, u_channel, v_channel)
    nu = float(viscosity)
    if not nu >= 0:
        raise ValueError("viscosity must be non-negative")
    dx = _positive_spacing(dx, "dx")
    dy = _positive_spacing(dy, "dy")
    u = velocity[..., u_channel, :, :]
    v = velocity[..., v_channel, :, :]
    du_dx = _periodic_central_difference(u, dx, -1)
    du_dy = _periodic_central_difference(u, dy, -2)
    dv_dx = _periodic_central_difference(v, dx, -1)
    dv_dy = _periodic_central_difference(v, dy, -2)
    shear = 0.5 * (du_dy + dv_dx)
    pointwise = 2.0 * nu * (du_dx.square() + dv_dy.square() + 2.0 * shear.square())
    return _reduce_batch(pointwise.mean(dim=(-2, -1)), reduction)


def era5_vorticity(
    fields: torch.Tensor,
    *,
    dx: float,
    dy: float,
    u_channel: int = 1,
    v_channel: int = 2,
    periodic_longitude: bool = True,
    periodic_latitude: bool = False,
) -> torch.Tensor:
    """Compute ERA5 relative vorticity ``dv/dx - du/dy`` from supplied spacing.

    Input is ``[..., C, H, W]`` with longitude/x on width and latitude/y on
    height. ``dx`` and ``dy`` are physical spacings in consistent units.
    Longitude is periodic by default; latitude uses second-order one-sided
    boundary differences unless explicitly marked periodic. This Cartesian
    diagnostic does not apply spherical metric factors, so callers must supply
    already projected fields/spacings when physical spherical vorticity is
    required.
    """

    _check_velocity(fields, u_channel, v_channel)
    dx = _positive_spacing(dx, "dx")
    dy = _positive_spacing(dy, "dy")
    u = fields[..., u_channel, :, :]
    v = fields[..., v_channel, :, :]
    derivative_x = (
        _periodic_central_difference(v, dx, -1)
        if periodic_longitude
        else _nonperiodic_central_difference(v, dx, -1)
    )
    derivative_y = (
        _periodic_central_difference(u, dy, -2)
        if periodic_latitude
        else _nonperiodic_central_difference(u, dy, -2)
    )
    return derivative_x - derivative_y


def synchronized_cuda_time(
    function: Callable[P, R],
    /,
    *args: P.args,
    device: torch.device | str | None = None,
    **kwargs: P.kwargs,
) -> tuple[R, float]:
    """Run ``function`` once and return ``(result, elapsed_seconds)``.

    CUDA is synchronized immediately before and after the call so asynchronous
    kernels are included. If CUDA is unavailable, this is a CPU wall-clock
    timer. Supplying a non-CUDA device also selects CPU timing.
    """

    if device is None:
        timing_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        timing_device = torch.device(device)
    if timing_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing requested but CUDA is unavailable")
        torch.cuda.synchronize(timing_device)
    start = perf_counter()
    result = function(*args, **kwargs)
    if timing_device.type == "cuda":
        torch.cuda.synchronize(timing_device)
    return result, perf_counter() - start


__all__ = [
    "era5_vorticity",
    "high_correlation_time",
    "kolmogorov_dissipation",
    "pearson_correlation",
    "periodic_total_variation_1d",
    "rmse",
    "synchronized_cuda_time",
    "vorticity_2d_finite_difference",
    "vorticity_2d_spectral",
]
