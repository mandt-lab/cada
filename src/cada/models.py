"""Conditional 1D and 2D U-Nets used by the CADA diffusion prior."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F


class _LearnedSinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("sinusoidal_embedding_dim must be a positive even integer")
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, time: Tensor) -> Tensor:
        frequencies = time[:, None] * self.weights[None, :] * (2 * math.pi)
        return torch.cat((time[:, None], frequencies.sin(), frequencies.cos()), dim=-1)


class _RMSNorm(nn.Module):
    def __init__(self, channels: int, spatial_dims: int) -> None:
        super().__init__()
        self.scale = channels**0.5
        self.gain = nn.Parameter(torch.ones(1, channels, *([1] * spatial_dims)))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.gain * self.scale


class _Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int,
        dropout: float,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        self.projection = conv(in_channels, out_channels, 3, padding=1)
        self.norm = _RMSNorm(out_channels, spatial_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        x = self.norm(self.projection(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.dropout(F.silu(x))


class _ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        spatial_dims: int,
        dropout: float,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        self.spatial_dims = spatial_dims
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, out_channels * 2)
        )
        self.block1 = _Block(
            in_channels, out_channels, spatial_dims=spatial_dims, dropout=dropout
        )
        self.block2 = _Block(
            out_channels, out_channels, spatial_dims=spatial_dims, dropout=0.0
        )
        self.residual = (
            conv(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        scale_shift = self.time_mlp(time_embedding)
        scale_shift = scale_shift.reshape(
            scale_shift.shape[0], scale_shift.shape[1], *([1] * self.spatial_dims)
        )
        hidden = self.block1(x, scale_shift.chunk(2, dim=1))
        return self.block2(hidden) + self.residual(x)


class _LinearAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_dims: int,
        heads: int,
        dim_head: int,
        memory_tokens: int,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        hidden = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.spatial_dims = spatial_dims
        self.norm = _RMSNorm(channels, spatial_dims)
        self.to_qkv = conv(channels, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            conv(hidden, channels, 1), _RMSNorm(channels, spatial_dims)
        )
        self.memory = (
            nn.Parameter(torch.randn(2, heads, dim_head, memory_tokens))
            if memory_tokens
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        spatial_shape = x.shape[2:]
        q, k, v = (
            part.reshape(x.shape[0], self.heads, self.dim_head, -1)
            for part in self.to_qkv(self.norm(x)).chunk(3, dim=1)
        )
        if self.memory is not None:
            memory_k, memory_v = (
                repeat(part, "h d n -> b h d n", b=x.shape[0])
                for part in self.memory
            )
            k = torch.cat((memory_k, k), dim=-1)
            v = torch.cat((memory_v, v), dim=-1)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(x.shape[0], self.heads * self.dim_head, *spatial_shape)
        return self.to_out(out)


class _Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_dims: int,
        heads: int,
        dim_head: int,
        memory_tokens: int,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        hidden = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.norm = _RMSNorm(channels, spatial_dims)
        self.to_qkv = conv(channels, hidden * 3, 1, bias=False)
        self.to_out = conv(hidden, channels, 1)
        self.memory = (
            nn.Parameter(torch.randn(2, heads, dim_head, memory_tokens))
            if memory_tokens
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        spatial_shape = x.shape[2:]
        q, k, v = (
            part.reshape(x.shape[0], self.heads, self.dim_head, -1)
            for part in self.to_qkv(self.norm(x)).chunk(3, dim=1)
        )
        if self.memory is not None:
            memory_k, memory_v = (
                repeat(part, "h d n -> b h d n", b=x.shape[0])
                for part in self.memory
            )
            k = torch.cat((memory_k, k), dim=-1)
            v = torch.cat((memory_v, v), dim=-1)
        attention = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attention = attention.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attention, v)
        out = out.reshape(x.shape[0], self.heads * self.dim_head, *spatial_shape)
        return self.to_out(out)


class _Downsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels * 4, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        return self.projection(x)


def _downsample(
    in_channels: int, out_channels: int, spatial_dims: int
) -> nn.Module:
    if spatial_dims == 1:
        return nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1)
    return _Downsample2D(in_channels, out_channels)


def _upsample(in_channels: int, out_channels: int, spatial_dims: int) -> nn.Module:
    conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv(in_channels, out_channels, 3, padding=1),
    )


class _UNet(nn.Module):
    def __init__(
        self,
        *,
        physical_channels: int,
        base_width: int,
        dim_multipliers: Sequence[int],
        sinusoidal_embedding_dim: int,
        dropout: float,
        attention_heads: int,
        attention_dim_head: int,
        spatial_dims: int,
        full_attention: Sequence[bool],
    ) -> None:
        super().__init__()
        if physical_channels < 1 or base_width < 1:
            raise ValueError("physical_channels and base_width must be positive")
        if not dim_multipliers:
            raise ValueError("dim_multipliers cannot be empty")
        if len(full_attention) != len(dim_multipliers):
            raise ValueError("full_attention must match dim_multipliers")

        self.physical_channels = physical_channels
        self.downsample_factor = 2 ** (len(dim_multipliers) - 1)
        conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        memory_tokens = 0 if spatial_dims == 1 else 4

        self.initial = conv(physical_channels * 2, base_width, 7, padding=3)
        widths = [base_width, *(base_width * m for m in dim_multipliers)]
        in_out = tuple(zip(widths[:-1], widths[1:]))
        time_dim = base_width * 4
        self.time_mlp = nn.Sequential(
            _LearnedSinusoidalEmbedding(sinusoidal_embedding_dim),
            nn.Linear(sinusoidal_embedding_dim + 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        def resnet(in_channels: int, out_channels: int) -> _ResnetBlock:
            return _ResnetBlock(
                in_channels,
                out_channels,
                time_dim=time_dim,
                spatial_dims=spatial_dims,
                dropout=dropout,
            )

        def attention(channels: int, use_full: bool) -> nn.Module:
            cls = _Attention if use_full else _LinearAttention
            return cls(
                channels,
                spatial_dims=spatial_dims,
                heads=attention_heads,
                dim_head=attention_dim_head,
                memory_tokens=memory_tokens,
            )

        self.downs = nn.ModuleList()
        for index, ((dim_in, dim_out), use_full) in enumerate(
            zip(in_out, full_attention)
        ):
            last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    (
                        resnet(dim_in, dim_in),
                        resnet(dim_in, dim_in),
                        attention(dim_in, use_full),
                        (
                            conv(dim_in, dim_out, 3, padding=1)
                            if last
                            else _downsample(dim_in, dim_out, spatial_dims)
                        ),
                    )
                )
            )

        middle_width = widths[-1]
        self.middle_block1 = resnet(middle_width, middle_width)
        self.middle_attention = attention(middle_width, True)
        self.middle_block2 = resnet(middle_width, middle_width)

        self.ups = nn.ModuleList()
        reversed_stages = zip(reversed(in_out), reversed(full_attention))
        for index, ((dim_in, dim_out), use_full) in enumerate(reversed_stages):
            last = index == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    (
                        resnet(dim_out + dim_in, dim_out),
                        resnet(dim_out + dim_in, dim_out),
                        attention(dim_out, use_full),
                        (
                            conv(dim_out, dim_in, 3, padding=1)
                            if last
                            else _upsample(dim_out, dim_in, spatial_dims)
                        ),
                    )
                )
            )

        self.final_block = resnet(base_width * 2, base_width)
        self.final = conv(base_width, physical_channels, 1)

    def forward(self, states: Tensor, diffusion_time: Tensor) -> Tensor:
        if states.shape[1] != self.physical_channels * 2:
            raise ValueError(
                f"expected {self.physical_channels * 2} input channels, "
                f"received {states.shape[1]}"
            )
        if diffusion_time.ndim != 1 or diffusion_time.shape[0] != states.shape[0]:
            raise ValueError("diffusion_time must have shape [batch]")
        if any(size % self.downsample_factor for size in states.shape[2:]):
            raise ValueError(
                f"spatial dimensions must be divisible by {self.downsample_factor}"
            )

        x = self.initial(states)
        residual = x
        time_embedding = self.time_mlp(diffusion_time)
        skips: list[Tensor] = []

        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_embedding)
            skips.append(x)
            x = block2(x, time_embedding)
            x = attention(x) + x
            skips.append(x)
            x = downsample(x)

        x = self.middle_block1(x, time_embedding)
        x = self.middle_attention(x) + x
        x = self.middle_block2(x, time_embedding)

        for block1, block2, attention, upsample in self.ups:
            x = block1(torch.cat((x, skips.pop()), dim=1), time_embedding)
            x = block2(torch.cat((x, skips.pop()), dim=1), time_embedding)
            x = attention(x) + x
            x = upsample(x)

        x = self.final_block(torch.cat((x, residual), dim=1), time_embedding)
        return self.final(x)


class UNet1D(_UNet):
    """1D pred-v prior taking ``[previous_state, noisy_next_state]`` channels."""

    def __init__(
        self,
        physical_channels: int,
        *,
        base_width: int = 64,
        dim_multipliers: Sequence[int] = (1, 2, 4, 8),
        sinusoidal_embedding_dim: int = 128,
        dropout: float = 0.0,
        attention_heads: int = 4,
        attention_dim_head: int = 32,
    ) -> None:
        super().__init__(
            physical_channels=physical_channels,
            base_width=base_width,
            dim_multipliers=dim_multipliers,
            sinusoidal_embedding_dim=sinusoidal_embedding_dim,
            dropout=dropout,
            attention_heads=attention_heads,
            attention_dim_head=attention_dim_head,
            spatial_dims=1,
            full_attention=(False,) * len(dim_multipliers),
        )


class UNet2D(_UNet):
    """2D pred-v prior taking ``[previous_state, noisy_next_state]`` channels."""

    def __init__(
        self,
        physical_channels: int,
        *,
        base_width: int = 64,
        dim_multipliers: Sequence[int] = (1, 2, 4, 8),
        sinusoidal_embedding_dim: int = 128,
        dropout: float = 0.0,
        attention_heads: int = 4,
        attention_dim_head: int = 32,
        full_attention: Sequence[bool] | None = None,
    ) -> None:
        if full_attention is None:
            full_attention = (False,) * (len(dim_multipliers) - 1) + (True,)
        super().__init__(
            physical_channels=physical_channels,
            base_width=base_width,
            dim_multipliers=dim_multipliers,
            sinusoidal_embedding_dim=sinusoidal_embedding_dim,
            dropout=dropout,
            attention_heads=attention_heads,
            attention_dim_head=attention_dim_head,
            spatial_dims=2,
            full_attention=full_attention,
        )


__all__ = ["UNet1D", "UNet2D"]
