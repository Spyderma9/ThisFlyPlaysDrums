"""Connectome simulation built on eonsystems fly-brain (cloned to fly/vendor/, imported, never copied).

fly-brain supplies the neuron dynamics (AlphaLIF with its ATan surrogate gradient), the Poisson input
generator and the Shiu et al. parameters. This module supplies what fly-brain hard-codes for FlyWire:
  - any connectome's weights (MaleCNS or FlyWire), stepped at dt = 1 ms
  - cue rates per voice spread over each voice's cue neurons
  - an injected current (the teacher's alpha * I* into motor neurons)
  - a trainable edge subset, removed from the frozen matrix so it isn't counted twice

The frozen recurrent matmul uses its own autograd function so gradients reach the trainable subset
through the fixed wiring, while the fixed weights themselves never get gradients.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fly.connectome import Connectome

FLYBRAIN_CODE = Path(__file__).resolve().parent / "vendor" / "fly-brain" / "code"
DT_MS = 1.0


def _flybrain():
    if str(FLYBRAIN_CODE) not in sys.path:
        sys.path.insert(0, str(FLYBRAIN_CODE))
    import run_pytorch  # fly-brain's PyTorch backend

    return run_pytorch


class _FrozenMatmul(torch.autograd.Function):
    """spikes [B, N] -> spikes @ W.T, with gradient to spikes only (W and W.T are fixed CSR)."""

    @staticmethod
    def forward(ctx, spikes, w, w_t):
        ctx.w_t = w_t
        return torch.sparse.mm(w, spikes.t()).t()

    @staticmethod
    def backward(ctx, grad):
        return torch.sparse.mm(ctx.w_t, grad.t()).t(), None, None


class PlasticEdges(nn.Module):
    """Trainable weights on a fixed set of edges (e.g. Kenyon cell -> MBON). D3 picks which edges."""

    def __init__(self, pre: np.ndarray, post: np.ndarray, weight: np.ndarray, size: int):
        super().__init__()
        self.register_buffer("pre", torch.as_tensor(pre, dtype=torch.long))
        self.register_buffer("post", torch.as_tensor(post, dtype=torch.long))
        self.weight = nn.Parameter(torch.as_tensor(weight, dtype=torch.float32))
        self.size = size

    def forward(self, spikes):
        out = spikes.new_zeros(spikes.shape[0], self.size)
        return out.index_add(1, self.post, spikes[:, self.pre] * self.weight)


class _WideSpike(torch.autograd.Function):
    """fly-brain's spike (Heaviside, identical forward) with its ATan surrogate widened: 1/(1+(pi v/width)^2).

    fly-brain's surrogate has width 1 in mV units, while rest -> threshold is 7 mV, so the gradient is ~0.002 at rest
    and vanishes across the two spiking layers between a cue synapse and a motor neuron. Only the backward pass
    changes: the simulated fly is exactly the same."""

    @staticmethod
    def forward(ctx, v, width):
        ctx.save_for_backward(v)
        ctx.width = width
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (v,) = ctx.saved_tensors
        return grad / (1 + (math.pi * v / ctx.width) ** 2), None


class Brain(nn.Module):
    def __init__(
        self,
        conn: Connectome,
        cue_groups: dict[str, np.ndarray],  # voice name -> neuron indices
        plastic_mask: np.ndarray | None = None,  # bool per edge in conn; these edges become trainable
        batch: int = 1,
        device: str = "cuda",
        surrogate_mv: float | None = None,  # widen the surrogate gradient (training only; forward unchanged)
        tone_idx: np.ndarray | None = None,  # neurons that get a trainable constant current (motor-neuron tone)
    ):
        super().__init__()
        fb = _flybrain()
        self.params = dict(fb.MODEL_PARAMS)
        self.scale = self.params["wScale"]
        self.size, self.batch, self.device = conn.size, batch, device

        frozen = conn if plastic_mask is None else Connectome(
            conn.name, conn.neurons, conn.pre[~plastic_mask], conn.post[~plastic_mask], conn.weight[~plastic_mask]
        )
        self.w = frozen.to_torch(device)
        self.w_t = frozen.to_torch(device, transpose=True)
        self.plastic = None
        if plastic_mask is not None:
            self.plastic = PlasticEdges(
                conn.pre[plastic_mask], conn.post[plastic_mask], conn.weight[plastic_mask], conn.size
            ).to(device)

        self.voices = tuple(cue_groups)
        cue_idx = np.concatenate([cue_groups[v] for v in self.voices])
        # fly-brain gives directly stimulated neurons no refractory period, so Poisson input isn't gated.
        self.neurons = fb.AlphaLIF(batch, conn.size, DT_MS, self.params, exc_indices=torch.as_tensor(cue_idx), device=device)
        self.surrogate_mv = surrogate_mv
        if surrogate_mv is not None:  # an instance attribute, so fly-brain itself is untouched
            self.neurons.neuron.spike_gradient = lambda v: _WideSpike.apply(v, float(surrogate_mv))
        self.tone = None
        if tone_idx is not None:  # resting excitability per neuron, the same at every moment: it can't encode a groove
            self.register_buffer("tone_idx", torch.as_tensor(tone_idx, dtype=torch.long, device=device))
            self.tone = nn.Parameter(torch.zeros(len(tone_idx), device=device))
        self.poisson = fb.PoissonSpikeGenerator(DT_MS, self.params["scalePoisson"], device=device)
        # voice rate [B, V] -> neuron rate [B, N]
        spread = torch.zeros(len(self.voices), conn.size, device=device)
        for i, v in enumerate(self.voices):
            spread[i, torch.as_tensor(cue_groups[v], device=device)] = 1.0
        self.register_buffer("spread", spread)

    def init_state(self):
        return self.neurons.state_init()

    def step(self, state, voice_rates: torch.Tensor, current: torch.Tensor | None = None, generator=None):
        """Advance one 1 ms step. voice_rates [B, V] in Hz; current [B, N] in mV, added to the Poisson drive."""
        conductance, delay_buffer, spikes, v, refrac = state
        stim = self.scale * self.poisson(voice_rates @ self.spread, generator=generator)
        if current is not None:
            stim = stim + current
        if self.tone is not None:
            stim = stim.index_add(1, self.tone_idx, self.tone.expand(stim.shape[0], -1))
        recurrent = _FrozenMatmul.apply(spikes, self.w, self.w_t)
        if self.plastic is not None:
            recurrent = recurrent + self.plastic(spikes)
        return self.neurons(self.scale * recurrent, stim, conductance, delay_buffer, spikes, v, refrac)
