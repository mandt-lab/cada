"""Paper observation operators with spatial and temporal sampling kept separate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

SpatialDims: TypeAlias = Literal[1, 2]
MaskPhase: TypeAlias = int | Sequence[int]


class Observation(NamedTuple):
    """An observed signal on the model grid and its binary availability mask."""

    value: Tensor
    mask: Tensor


class ObservationOperator(Protocol):
    def __call__(self, state: Tensor) -> Observation: ...


def _validate_factor(factor: int) -> int:
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("spatial factor must be a positive integer")
    return factor


def _validate_state(state: Tensor, spatial_dims: SpatialDims) -> tuple[int, ...]:
    if not isinstance(state, Tensor):
        raise TypeError("state must be a torch.Tensor")
    if state.ndim < spatial_dims + 1:
        layout = "[..., C, X]" if spatial_dims == 1 else "[..., C, H, W]"
        raise ValueError(
            f"state must use layout {layout}, got shape {tuple(state.shape)}"
        )
    spatial_shape = tuple(int(size) for size in state.shape[-spatial_dims:])
    if any(size <= 0 for size in spatial_shape):
        raise ValueError(f"spatial dimensions must be positive, got {spatial_shape}")
    return spatial_shape


def _full_spatial_mask(
    state: Tensor, spatial_mask: Tensor, spatial_dims: int
) -> Tensor:
    view_shape = (1,) * (state.ndim - spatial_dims) + tuple(spatial_mask.shape)
    return spatial_mask.reshape(view_shape).expand_as(state)


def _normalize_phase(
    phase: MaskPhase,
    *,
    spatial_dims: SpatialDims,
    factor: int,
) -> tuple[int, ...]:
    if isinstance(phase, int) and not isinstance(phase, bool):
        phases = (phase,) * spatial_dims
    else:
        phases = tuple(phase)
    if len(phases) != spatial_dims:
        raise ValueError(f"mask phase must contain {spatial_dims} value(s)")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < factor
        for item in phases
    ):
        raise ValueError(f"each mask phase must be an integer in [0, {factor})")
    return phases


@dataclass(frozen=True)
class IdentityObservation:
    """Fully observed identity operator."""

    def __call__(self, state: Tensor) -> Observation:
        if not isinstance(state, Tensor):
            raise TypeError("state must be a torch.Tensor")
        return Observation(state, torch.ones_like(state))


@dataclass(frozen=True)
class AverageDownsample1D:
    """Average non-overlapping 1D blocks and lift by nearest replication."""

    factor: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _validate_factor(self.factor))

    def __call__(self, state: Tensor) -> Observation:
        (length,) = _validate_state(state, 1)
        if not state.is_floating_point():
            raise TypeError("average downsampling requires a floating-point state")
        if length % self.factor:
            raise ValueError(
                f"1D size {length} must be divisible by spatial factor {self.factor}"
            )
        flat = state.reshape(-1, 1, length)
        coarse = F.avg_pool1d(flat, kernel_size=self.factor, stride=self.factor)
        lifted = coarse.repeat_interleave(self.factor, dim=-1).reshape_as(state)
        return Observation(lifted, torch.ones_like(state))


@dataclass(frozen=True)
class AverageDownsample2D:
    """Average non-overlapping 2D blocks and lift by nearest replication."""

    factor: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _validate_factor(self.factor))

    def __call__(self, state: Tensor) -> Observation:
        height, width = _validate_state(state, 2)
        if not state.is_floating_point():
            raise TypeError("average downsampling requires a floating-point state")
        if height % self.factor or width % self.factor:
            raise ValueError(
                f"2D size {(height, width)} must be divisible by spatial factor "
                f"{self.factor}"
            )
        flat = state.reshape(-1, 1, height, width)
        coarse = F.avg_pool2d(flat, kernel_size=self.factor, stride=self.factor)
        lifted = coarse.repeat_interleave(self.factor, dim=-2).repeat_interleave(
            self.factor, dim=-1
        )
        return Observation(lifted.reshape_as(state), torch.ones_like(state))


@dataclass(frozen=True)
class RegularGridMask:
    """MS operator: reveal one regular-grid phase, shared across channels."""

    spatial_dims: SpatialDims
    factor: int
    phase: MaskPhase = 0

    def __post_init__(self) -> None:
        if self.spatial_dims not in (1, 2):
            raise ValueError("spatial_dims must be 1 or 2")
        factor = _validate_factor(self.factor)
        phases = _normalize_phase(
            self.phase, spatial_dims=self.spatial_dims, factor=factor
        )
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "phase", phases)

    def __call__(self, state: Tensor) -> Observation:
        spatial_shape = _validate_state(state, self.spatial_dims)
        spatial_mask = torch.zeros(
            spatial_shape, dtype=state.dtype, device=state.device
        )
        slices = tuple(
            slice(phase, None, self.factor) for phase in self.phase
        )
        spatial_mask[slices] = 1
        mask = _full_spatial_mask(state, spatial_mask, self.spatial_dims)
        return Observation(state * mask, mask)


class BernoulliMask:
    """Irregular spatial mask with reproducible draws from an explicit seed."""

    def __init__(
        self,
        *,
        spatial_dims: SpatialDims,
        keep_probability: float,
        seed: int,
    ) -> None:
        if spatial_dims not in (1, 2):
            raise ValueError("spatial_dims must be 1 or 2")
        if not 0.0 < keep_probability <= 1.0:
            raise ValueError("keep_probability must lie in (0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.spatial_dims = spatial_dims
        self.keep_probability = float(keep_probability)
        self.seed = seed
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def reset(self) -> None:
        """Reset the mask stream to its original seed."""

        self._generator.manual_seed(self.seed)

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self._generator.get_state().clone()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, Tensor):
            raise ValueError("BernoulliMask state requires generator_state")
        self._generator.set_state(generator_state.cpu())

    def __call__(self, state: Tensor) -> Observation:
        spatial_shape = _validate_state(state, self.spatial_dims)
        draw = torch.rand(
            spatial_shape, generator=self._generator, device="cpu"
        )
        spatial_mask = (draw < self.keep_probability).to(
            device=state.device, dtype=state.dtype
        )
        mask = _full_spatial_mask(state, spatial_mask, self.spatial_dims)
        return Observation(state * mask, mask)


@dataclass(frozen=True)
class TemporalCadence:
    """Independent observation-arrival schedule over physical time indices."""

    interval: int = 1
    start: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.interval, bool)
            or not isinstance(self.interval, int)
            or self.interval < 1
        ):
            raise ValueError("interval must be a positive integer")
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
        ):
            raise ValueError("start must be a non-negative integer")

    def is_observation_time(self, time_index: int) -> bool:
        if time_index < self.start:
            return False
        return (time_index - self.start) % self.interval == 0

    def indices(self, length: int) -> tuple[int, ...]:
        """Return scheduled indices in the half-open interval ``[0, length)``."""

        if length < 0:
            raise ValueError("length must be non-negative")
        return tuple(range(self.start, length, self.interval))


def apply_observation(state: Tensor, operator: ObservationOperator) -> Observation:
    """Apply an operator and validate its model-grid observation and mask."""

    result = operator(state)
    if not isinstance(result, Observation):
        raise TypeError("observation operator must return Observation(value, mask)")
    if result.value.shape != state.shape:
        raise ValueError(
            f"observation shape {tuple(result.value.shape)} does not match state "
            f"shape {tuple(state.shape)}"
        )
    if result.mask.shape != state.shape:
        raise ValueError(
            f"mask shape {tuple(result.mask.shape)} does not match state "
            f"shape {tuple(state.shape)}"
        )
    return result


def make_observation_operator(
    kind: str,
    *,
    spatial_dims: SpatialDims,
    factor: int | None = None,
    mask_phase: MaskPhase = 0,
    keep_probability: float | None = None,
    seed: int | None = None,
) -> ObservationOperator:
    """Build a spatial operator without coupling it to temporal cadence."""

    if spatial_dims not in (1, 2):
        raise ValueError("spatial_dims must be 1 or 2")
    name = kind.strip().lower().replace("-", "_")
    if name in {"identity", "full"}:
        if factor not in (None, 1):
            raise ValueError("identity observation does not use a spatial factor")
        return IdentityObservation()
    if name in {"ds", "downsample", "average_downsample"}:
        if factor is None:
            raise ValueError("DS observation requires factor")
        return (
            AverageDownsample1D(factor)
            if spatial_dims == 1
            else AverageDownsample2D(factor)
        )
    if name in {"ms", "regular_mask", "mask"}:
        if factor is None:
            raise ValueError("MS observation requires factor")
        return RegularGridMask(spatial_dims, factor, mask_phase)
    if name in {"bernoulli", "bernoulli_mask", "random_mask"}:
        if keep_probability is None:
            raise ValueError("Bernoulli observation requires keep_probability")
        if seed is None:
            raise ValueError("Bernoulli observation requires an explicit seed")
        return BernoulliMask(
            spatial_dims=spatial_dims,
            keep_probability=keep_probability,
            seed=seed,
        )
    raise ValueError(
        f"unknown observation kind {kind!r}; expected identity, ds, ms, or bernoulli"
    )


__all__ = [
    "AverageDownsample1D",
    "AverageDownsample2D",
    "BernoulliMask",
    "IdentityObservation",
    "MaskPhase",
    "Observation",
    "ObservationOperator",
    "RegularGridMask",
    "SpatialDims",
    "TemporalCadence",
    "apply_observation",
    "make_observation_operator",
]
