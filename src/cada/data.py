"""Lazy, normalized trajectory datasets used by the CADA experiments."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

PathLike: TypeAlias = str | os.PathLike[str]


class ResumableRandomSampler(Sampler[int]):
    """Random permutation sampler with constant-time checkpoint restore."""

    def __init__(self, data_source: Dataset[Any], generator: torch.Generator) -> None:
        self.data_source = data_source
        self.generator = generator
        self.permutation: Tensor | None = None
        self.permutation_generator_state: Tensor | None = None
        self.position = 0

    def __iter__(self):
        if self.permutation is None:
            self.permutation_generator_state = self.generator.get_state().clone()
            self.permutation = torch.randperm(
                len(self.data_source),
                generator=self.generator,
            )
            self.position = 0
        while self.position < len(self.permutation):
            index = int(self.permutation[self.position])
            self.position += 1
            yield index
        self.permutation = None
        self.permutation_generator_state = None
        self.position = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state().clone(),
            "permutation_generator_state": (
                None
                if self.permutation_generator_state is None
                else self.permutation_generator_state.clone()
            ),
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        generator_state = state.get("generator_state")
        permutation_generator_state = state.get("permutation_generator_state")
        position = state.get("position")
        if not isinstance(generator_state, Tensor):
            raise ValueError("sampler state requires generator_state")
        if permutation_generator_state is not None and not isinstance(
            permutation_generator_state,
            Tensor,
        ):
            raise ValueError(
                "sampler permutation_generator_state must be a tensor or None"
            )
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("sampler position must be an integer")
        if permutation_generator_state is None and position != 0:
            raise ValueError("sampler position must be zero without a permutation")
        if not 0 <= position <= len(self.data_source):
            raise ValueError("sampler position is outside the permutation")
        self.generator.set_state(generator_state.cpu())
        if permutation_generator_state is None:
            self.permutation = None
        else:
            replay_generator = torch.Generator(device="cpu")
            replay_generator.set_state(permutation_generator_state.cpu())
            self.permutation = torch.randperm(
                len(self.data_source),
                generator=replay_generator,
            )
        self.permutation_generator_state = (
            None
            if permutation_generator_state is None
            else permutation_generator_state.cpu().clone()
        )
        self.position = position


def _one_dimensional_stats(values: Tensor | Sequence[float] | float) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().flatten().cpu()
    if tensor.numel() == 0:
        raise ValueError("normalization statistics cannot be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("normalization statistics must be finite")
    return tensor.clone()


@dataclass(frozen=True)
class MinMaxNormalizer:
    """Per-channel min/max normalization for tensors with an explicit channel axis."""

    minimum: Tensor | Sequence[float] | float
    maximum: Tensor | Sequence[float] | float
    clip: bool = False

    def __post_init__(self) -> None:
        minimum = _one_dimensional_stats(self.minimum)
        maximum = _one_dimensional_stats(self.maximum)
        if minimum.shape != maximum.shape:
            raise ValueError(
                f"minimum and maximum must have the same shape, got "
                f"{tuple(minimum.shape)} and {tuple(maximum.shape)}"
            )
        if torch.any(maximum <= minimum):
            raise ValueError("each channel maximum must be greater than its minimum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def channels(self) -> int:
        return int(self.minimum.numel())

    def _bounds(self, value: Tensor, channel_dim: int) -> tuple[Tensor, Tensor]:
        if value.ndim == 0:
            raise ValueError("a channel axis is required")
        channel_dim %= value.ndim
        if value.shape[channel_dim] != self.channels:
            raise ValueError(
                f"tensor has {value.shape[channel_dim]} channels on axis "
                f"{channel_dim}, "
                f"but statistics contain {self.channels}"
            )
        shape = [1] * value.ndim
        shape[channel_dim] = self.channels
        minimum = self.minimum.to(device=value.device, dtype=value.dtype).reshape(shape)
        maximum = self.maximum.to(device=value.device, dtype=value.dtype).reshape(shape)
        return minimum, maximum

    def normalize(self, value: Tensor, *, channel_dim: int) -> Tensor:
        minimum, maximum = self._bounds(value, channel_dim)
        if self.clip:
            value = value.clamp(min=minimum, max=maximum)
        return (value - minimum) / (maximum - minimum)

    def denormalize(self, value: Tensor, *, channel_dim: int) -> Tensor:
        minimum, maximum = self._bounds(value, channel_dim)
        return value * (maximum - minimum) + minimum


StatsLike: TypeAlias = MinMaxNormalizer | PathLike


def minmax_normalize(
    value: Tensor,
    minimum: Tensor | Sequence[float] | float,
    maximum: Tensor | Sequence[float] | float,
    *,
    channel_dim: int,
    clip: bool = False,
) -> Tensor:
    """Normalize ``value`` to [0, 1] using scalar or per-channel bounds."""

    return MinMaxNormalizer(minimum, maximum, clip=clip).normalize(
        value, channel_dim=channel_dim
    )


def minmax_denormalize(
    value: Tensor,
    minimum: Tensor | Sequence[float] | float,
    maximum: Tensor | Sequence[float] | float,
    *,
    channel_dim: int,
) -> Tensor:
    """Invert :func:`minmax_normalize`."""

    return MinMaxNormalizer(minimum, maximum, clip=False).denormalize(
        value, channel_dim=channel_dim
    )


def load_channel_minmax(path: PathLike, *, clip: bool = False) -> MinMaxNormalizer:
    """Load the ``{"min", "max"}`` PyTorch statistics used by Kolmogorov/ERA5."""

    stats: Any = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(stats, Mapping) or "min" not in stats or "max" not in stats:
        raise ValueError(f"{path} must contain a mapping with 'min' and 'max'")
    return MinMaxNormalizer(stats["min"], stats["max"], clip=clip)


def load_ks_minmax(path: PathLike, *, clip: bool = False) -> MinMaxNormalizer:
    """Load KS JSON statistics, preferring absolute training-set extrema."""

    with Path(path).open("r", encoding="utf-8") as file:
        stats = json.load(file)
    if not isinstance(stats, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    if "abs_min" in stats and "abs_max" in stats:
        minimum, maximum = stats["abs_min"], stats["abs_max"]
    elif "min" in stats and "max" in stats:
        minimum, maximum = stats["min"], stats["max"]
    else:
        raise ValueError(f"{path} must contain abs_min/abs_max or min/max")
    return MinMaxNormalizer(minimum, maximum, clip=clip)


class TrajectorySource(Protocol):
    """Tensor trajectory source consumed by the temporal adapters below."""

    @property
    def n_trajectories(self) -> int: ...

    @property
    def trajectory_length(self) -> int: ...

    def read_window(self, trajectory: int, start: int, stop: int) -> Tensor: ...

    def close(self) -> None: ...


class _LazyH5TrajectoryDataset(Dataset[Tensor]):
    """HDF5 source with one read-only handle per worker process."""

    def __init__(
        self,
        file: PathLike,
        normalizer: MinMaxNormalizer,
        *,
        swmr: bool = False,
    ) -> None:
        super().__init__()
        self.file = Path(file)
        self.normalizer = normalizer
        self.swmr = bool(swmr)
        self._dataset_key: str | None = None
        self._shape: tuple[int, ...] | None = None
        self._h5: h5py.File | None = None
        self._owner_pid: int | None = None

    def _resolve_key(self, file: h5py.File) -> str:
        raise NotImplementedError

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        raise NotImplementedError

    def _normalize(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def _inspect(self) -> None:
        if self._shape is not None:
            return
        with h5py.File(self.file, mode="r", swmr=self.swmr) as file:
            key = self._resolve_key(file)
            node = file[key]
            if not isinstance(node, h5py.Dataset):
                raise ValueError(f"{key!r} in {self.file} is not a dataset")
            shape = tuple(int(size) for size in node.shape)
        self._validate_shape(shape)
        self._dataset_key = key
        self._shape = shape

    @property
    def dataset_key(self) -> str:
        self._inspect()
        assert self._dataset_key is not None
        return self._dataset_key

    @property
    def shape(self) -> tuple[int, ...]:
        self._inspect()
        assert self._shape is not None
        return self._shape

    @property
    def n_trajectories(self) -> int:
        return self.shape[0]

    @property
    def trajectory_length(self) -> int:
        return self.shape[1]

    def __len__(self) -> int:
        return self.n_trajectories

    def _open_for_process(self) -> h5py.File:
        pid = os.getpid()
        if self._h5 is not None and self._owner_pid != pid:
            self.close()
        if self._h5 is None:
            self._h5 = h5py.File(self.file, mode="r", swmr=self.swmr)
            self._owner_pid = pid
        return self._h5

    def read_window(self, trajectory: int, start: int, stop: int) -> Tensor:
        count = self.n_trajectories
        if trajectory < 0:
            trajectory += count
        if not 0 <= trajectory < count:
            raise IndexError(f"trajectory index {trajectory} is out of range")
        if not 0 <= start < stop <= self.trajectory_length:
            raise IndexError(
                f"invalid state interval [{start}, {stop}) for trajectory length "
                f"{self.trajectory_length}"
            )
        dataset = self._open_for_process()[self.dataset_key]
        value = torch.as_tensor(dataset[trajectory, start:stop], dtype=torch.float32)
        return self._normalize(value)

    def __getitem__(self, index: int) -> Tensor:
        return self.read_window(index, 0, self.trajectory_length)

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None
                self._owner_pid = None

    def __enter__(self) -> _LazyH5TrajectoryDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DenseH5TrajectoryDataset(_LazyH5TrajectoryDataset):
    """Kolmogorov/ERA5 trajectories stored as ``[N, T, C, H, W]`` at key ``x``."""

    def __init__(
        self,
        file: PathLike,
        stats: StatsLike,
        *,
        key: str = "x",
        clip: bool = False,
        swmr: bool = False,
    ) -> None:
        normalizer = (
            stats
            if isinstance(stats, MinMaxNormalizer)
            else load_channel_minmax(stats, clip=clip)
        )
        super().__init__(file, normalizer, swmr=swmr)
        if not key:
            raise ValueError("key cannot be empty")
        self.key = key

    def _resolve_key(self, file: h5py.File) -> str:
        if self.key not in file:
            raise KeyError(f"dataset {self.key!r} not found in {self.file}")
        return self.key

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 5:
            raise ValueError(
                f"dense trajectories must have shape [N, T, C, H, W], got {shape}"
            )
        if any(size <= 0 for size in shape):
            raise ValueError(f"trajectory dimensions must be positive, got {shape}")
        if shape[2] != self.normalizer.channels:
            raise ValueError(
                f"dataset has {shape[2]} channels but statistics contain "
                f"{self.normalizer.channels}"
            )

    def _normalize(self, value: Tensor) -> Tensor:
        return self.normalizer.normalize(value, channel_dim=1)


class KSH5TrajectoryDataset(_LazyH5TrajectoryDataset):
    """KS trajectories stored as ``[N, T, X]`` under ``split/pde_*``."""

    def __init__(
        self,
        file: PathLike,
        stats: StatsLike,
        *,
        split: str,
        dataset_name: str | None = None,
        clip: bool = False,
        swmr: bool = False,
    ) -> None:
        normalizer = (
            stats
            if isinstance(stats, MinMaxNormalizer)
            else load_ks_minmax(stats, clip=clip)
        )
        super().__init__(file, normalizer, swmr=swmr)
        if not split or "/" in split:
            raise ValueError("split must be a non-empty HDF5 group name")
        if dataset_name is not None and (
            not dataset_name.startswith("pde_") or "/" in dataset_name
        ):
            raise ValueError("dataset_name must be an unqualified pde_* name")
        self.split = split
        self.dataset_name = dataset_name

    def _resolve_key(self, file: h5py.File) -> str:
        if self.split not in file or not isinstance(file[self.split], h5py.Group):
            raise KeyError(f"group {self.split!r} not found in {self.file}")
        group = file[self.split]
        if self.dataset_name is not None:
            if self.dataset_name not in group:
                raise KeyError(
                    f"dataset {self.split}/{self.dataset_name} not found in {self.file}"
                )
            return f"{self.split}/{self.dataset_name}"
        candidates = sorted(
            name
            for name, node in group.items()
            if name.startswith("pde_") and isinstance(node, h5py.Dataset)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one pde_* dataset under {self.split!r}, "
                f"found {candidates}; pass dataset_name to disambiguate"
            )
        return f"{self.split}/{candidates[0]}"

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 3:
            raise ValueError(f"KS trajectories must have shape [N, T, X], got {shape}")
        if any(size <= 0 for size in shape):
            raise ValueError(f"trajectory dimensions must be positive, got {shape}")
        if self.normalizer.channels != 1:
            raise ValueError("KS normalization statistics must be scalar")

    def _normalize(self, value: Tensor) -> Tensor:
        value = value.unsqueeze(1)
        return self.normalizer.normalize(value, channel_dim=1)


class PreviewWindowDataset(Dataset[Tensor]):
    """All fixed-horizon windows from a trajectory source.

    ``horizon`` counts transitions, not states: a horizon of ``H`` returns
    ``[x_t, ..., x_(t+H)]`` with shape ``[H + 1, C, ...]``.
    """

    def __init__(
        self,
        source: TrajectorySource,
        *,
        horizon: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be at least one transition")
        if stride < 1:
            raise ValueError("stride must be positive")
        if horizon >= source.trajectory_length:
            raise ValueError(
                f"horizon {horizon} requires {horizon + 1} states, but trajectories "
                f"contain {source.trajectory_length}"
            )
        self.source = source
        self.horizon = int(horizon)
        self.stride = int(stride)
        self._windows_per_trajectory = (
            (source.trajectory_length - self.horizon - 1) // self.stride
        ) + 1

    @property
    def states_per_window(self) -> int:
        return self.horizon + 1

    def __len__(self) -> int:
        return self.source.n_trajectories * self._windows_per_trajectory

    def locate(self, index: int) -> tuple[int, int]:
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(f"window index {index} is out of range")
        trajectory, local = divmod(index, self._windows_per_trajectory)
        return trajectory, local * self.stride

    def __getitem__(self, index: int) -> Tensor:
        trajectory, start = self.locate(index)
        return self.source.read_window(
            trajectory, start, start + self.states_per_window
        )

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> PreviewWindowDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class OneStepTransitionDataset(PreviewWindowDataset):
    """All prior transitions, returned as ``[x_t, x_(t+1)]``."""

    def __init__(self, source: TrajectorySource) -> None:
        super().__init__(source, horizon=1, stride=1)


PriorTransitionDataset = OneStepTransitionDataset


__all__ = [
    "DenseH5TrajectoryDataset",
    "KSH5TrajectoryDataset",
    "MinMaxNormalizer",
    "OneStepTransitionDataset",
    "PreviewWindowDataset",
    "PriorTransitionDataset",
    "ResumableRandomSampler",
    "TrajectorySource",
    "load_channel_minmax",
    "load_ks_minmax",
    "minmax_denormalize",
    "minmax_normalize",
]
