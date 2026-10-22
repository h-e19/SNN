"""
NumPy reference model for the Lava pause experiment.

The neuron update is Lava's floating-point LIF (PyLifModelFloat), step for step:

    u[t] = u[t-1] * (1 - du) + a_in[t]
    v[t] = v[t-1] * (1 - dv) + u[t] + bias
    s[t] = v[t] > vth,   v[s] = 0

so porting to Lava means copying du, dv, vth, bias and the weight matrices.

Parameters are specified in physical units and converted to Lava units:
    tau_m dV/dt = -V + I + mu,   tau_s dI/dt = -I + J * sum(delta(t - t_spike))
    du = dt/tau_s,  dv = dt/tau_m,  w = dv * J / tau_s,  bias = dv * mu
V is in mV above rest (rest = reset = 0), J in mV*ms, so J / tau_m is roughly
the PSP size in mV.

State that a pause has to deal with:
    v  membrane voltage          (Lava LIF Var `v`)
    u  synaptic current          (Lava LIF Var `u`)
    s  spikes in flight: spikes emitted on the last step that the recurrent
       weights deliver on the next step. In Lava these live in channels,
       not in a Var, so get()/set() cannot save them.
External input is indexed by world time and is not network state.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

DT = 0.1  # ms per step

STATE_VARS = ("v", "u", "s")

# Which Vars get restored after the pause, and the situation each case stands for.
# wiped and all are the worst and best bounds; v+u is the realistic digital case
# (v and u sit in core SRAM and are held, spikes on the mesh are lost).
RESTORE = {
    "wiped": (),
    "v": ("v",),
    "u": ("u",),
    "v+u": ("v", "u"),
    "all": ("v", "u", "s"),
}
RESTORE_LABELS = {
    "wiped": "wiped: cores reassigned, no save",
    "v": "v only",
    "u": "u only",
    "v+u": "v+u: stopped digital chip, in-flight spikes lost",
    "all": "all: perfect checkpoint incl. in-flight spikes",
}
# freeze is the digital case. leaky is a rough bound for analog drift (digital SRAM
# does not decay); under leaky, "wiped" means nothing restored, i.e. drifted state.
MODE_LABELS = {"freeze": "digital stop", "leaky": "leaky (rough analog-drift bound)"}


@dataclass
class Network:
    name: str
    pops: dict  # population name -> slice into the neuron vector
    du: np.ndarray
    dv: np.ndarray
    vth: np.ndarray
    bias: np.ndarray
    w_in: np.ndarray  # (n_in, N) external input -> neurons, Lava units
    w_rec: np.ndarray  # (N, N) indexed [src, dst], Lava units
    in_rate: float  # Hz, rate of each external Poisson input

    @property
    def n(self):
        return len(self.du)

    @property
    def n_in(self):
        return self.w_in.shape[0]

    def zero_state(self):
        return dict(v=np.zeros(self.n), u=np.zeros(self.n), s=np.zeros(self.n, bool))


def _block(rng, n_src, n_dst, p, J, tau_m, tau_s):
    """Random sparse block with weight J (mV*ms), converted to Lava units."""
    mask = rng.random((n_src, n_dst)) < p
    return mask * (DT / tau_m) * J / tau_s


def _params(n, tau_m, tau_s, vth, mu):
    return dict(
        du=np.full(n, DT / tau_s),
        dv=np.full(n, DT / tau_m),
        vth=np.full(n, vth),
        bias=np.full(n, DT / tau_m * mu),
    )


def feedforward(seed=1, tau_m=15.0, tau_s=5.0, n_in=1000, n_layer=200, n_layers=2,
                in_rate=10.0, p_in=0.2, J_in=9.5, p_ff=0.2, J_ff=35.0, vth=20.0,
                n_bg=0, J_bg=8.0, p_bg=0.2):
    """input -> L1 -> ... -> Ln, all excitatory. J_ff may be a list, one per layer-to-layer
    projection, to keep rates level in deeper chains.

    n_bg > 0 gives every layer its own pool of independent background inputs (Poisson at
    in_rate): 0.8 * n_bg excitatory with weight J_bg and 0.2 * n_bg inhibitory with
    weight -4 * J_bg, so the mean drive cancels and only fluctuations are added. The
    background inputs are appended after the n_in main inputs."""
    rng = np.random.default_rng(seed)
    n = n_layer * n_layers
    J_ff = np.broadcast_to(J_ff, n_layers - 1)
    w_in = np.zeros((n_in + n_layers * n_bg, n))
    w_in[:n_in, :n_layer] = _block(rng, n_in, n_layer, p_in, J_in, tau_m, tau_s)
    w_rec = np.zeros((n, n))
    for k in range(n_layers - 1):
        src, dst = slice(k * n_layer, (k + 1) * n_layer), slice((k + 1) * n_layer, (k + 2) * n_layer)
        w_rec[src, dst] = _block(rng, n_layer, n_layer, p_ff, J_ff[k], tau_m, tau_s)
    if n_bg:
        bg_rng = np.random.default_rng([seed, 1])  # separate stream: n_bg leaves the main weights unchanged
        n_e = int(0.8 * n_bg)
        for k in range(n_layers):
            rows, cols = n_in + k * n_bg, slice(k * n_layer, (k + 1) * n_layer)
            w_in[rows:rows + n_e, cols] = _block(bg_rng, n_e, n_layer, p_bg, J_bg, tau_m, tau_s)
            w_in[rows + n_e:rows + n_bg, cols] = _block(bg_rng, n_bg - n_e, n_layer, p_bg, -4 * J_bg, tau_m, tau_s)
    pops = {f"L{k + 1}": slice(k * n_layer, (k + 1) * n_layer) for k in range(n_layers)}
    return Network("feedforward", pops, w_in=w_in, w_rec=w_rec, in_rate=in_rate,
                   **_params(n, tau_m, tau_s, vth, 0.0))


def recurrent(seed=1, tau_m=15.0, tau_s=5.0, n_in=1000, n_e=800, n_i=200,
              in_rate=10.0, p=0.2, vth=20.0,
              J_ee=10.0, J_ei=-45.0, J_ie=30.0, J_ii=-60.0, J_ex=15.0, J_ix=12.0,
              n_clusters=0, J_plus=1.0, n_stim=0, J_stim=15.0, p_stim=0.2):
    """Balanced E-I network driven by external Poisson input.
    J_ab is the weight from population b onto population a (Rosenbaum's convention).

    n_clusters > 0 splits E into equal clusters (populations C0, C1, ...): E->E weights
    within a cluster are scaled by J_plus and between clusters by
    J_minus = (n_clusters - J_plus) / (n_clusters - 1), which keeps the mean E->E input
    unchanged (Litwin-Kumar & Doiron 2012, weight clustering).
    n_stim > 0 appends a pool of n_stim stimulus inputs per cluster after the n_in
    background inputs; pool c projects only onto cluster c."""
    rng = np.random.default_rng(seed)
    n = n_e + n_i
    E, I = slice(0, n_e), slice(n_e, n)
    w_in = np.zeros((n_in + n_clusters * n_stim, n))
    w_in[:n_in, E] = _block(rng, n_in, n_e, p, J_ex, tau_m, tau_s)
    w_in[:n_in, I] = _block(rng, n_in, n_i, p, J_ix, tau_m, tau_s)
    w_rec = np.zeros((n, n))
    w_rec[E, E] = _block(rng, n_e, n_e, p, J_ee, tau_m, tau_s)
    w_rec[E, I] = _block(rng, n_e, n_i, p, J_ie, tau_m, tau_s)
    w_rec[I, E] = _block(rng, n_i, n_e, p, J_ei, tau_m, tau_s)
    w_rec[I, I] = _block(rng, n_i, n_i, p, J_ii, tau_m, tau_s)
    pops = dict(E=E, I=I)
    if n_clusters:
        size = n_e // n_clusters
        J_minus = (n_clusters - J_plus) / (n_clusters - 1)
        cluster = np.arange(n_e) // size
        w_rec[E, E] *= np.where(cluster[:, None] == cluster[None, :], J_plus, J_minus)
        stim_rng = np.random.default_rng([seed, 2])  # separate stream: options leave other weights unchanged
        for c in range(n_clusters):
            cols = slice(c * size, (c + 1) * size)
            pops[f"C{c}"] = cols
            rows = slice(n_in + c * n_stim, n_in + (c + 1) * n_stim)
            w_in[rows, cols] = _block(stim_rng, n_stim, size, p_stim, J_stim, tau_m, tau_s)
    return Network("recurrent", pops, w_in=w_in, w_rec=w_rec,
                   in_rate=in_rate, **_params(n, tau_m, tau_s, vth, 0.0))


def poisson_input(n_in, rate_hz, n_steps, seed):
    """External spikes as a list of index arrays, one per world-time step."""
    rng = np.random.default_rng(seed)
    p = rate_hz * DT / 1000.0
    out = []
    for start in range(0, n_steps, 1000):
        chunk = rng.random((min(1000, n_steps - start), n_in), dtype=np.float32) < p
        out.extend(np.flatnonzero(row) for row in chunk)
    return out


def stimulus_input(n_in, base_rate, boost_rate, n_classes, frac, seg_steps, n_steps, seed, pattern_seed=0,
                   stim_steps=None):
    """Stream of class stimuli for the toy task. Every seg_steps the class is redrawn;
    class c raises the rate of its own random subset (frac of the inputs) from
    base_rate to boost_rate. The mean input rate is the same for every class.
    With stim_steps set, only the first stim_steps of each segment show the stimulus
    and the rest is a class-free delay at base_rate (for the memory task).
    pattern_seed fixes the class patterns, seed the label sequence and spikes.
    Returns (spikes as list of index arrays, class label per segment)."""
    prng = np.random.default_rng(pattern_seed)
    rng = np.random.default_rng(seed)
    n_hot = int(frac * n_in)
    rates = np.full((n_classes, n_in), base_rate)
    for c in range(n_classes):
        rates[c, prng.choice(n_in, n_hot, replace=False)] = boost_rate
    rates *= base_rate * n_in / rates.sum(1, keepdims=True)  # equalize total rate
    n_seg = -(-n_steps // seg_steps)
    labels = rng.integers(n_classes, size=n_seg)
    p = rates * DT / 1000.0
    out = []
    p_delay = np.full(n_in, base_rate * DT / 1000.0)
    stim = seg_steps if stim_steps is None else stim_steps
    for seg, c in enumerate(labels):
        n = min(seg_steps, n_steps - seg * seg_steps)
        chunk = rng.random((n, n_in), dtype=np.float32) < np.where(np.arange(n)[:, None] < stim, p[c], p_delay)
        out.extend(np.flatnonzero(row) for row in chunk)
    return out, labels


def simulate(net, x, state, x_offset, n_steps):
    """Advance the network n_steps from `state`. Network step k consumes world input
    x[x_offset + k]. Returns (spikes (n_steps, N) bool, final state)."""
    v, u, s = state["v"].copy(), state["u"].copy(), state["s"].copy()
    keep_u, keep_v = 1.0 - net.du, 1.0 - net.dv
    spikes = np.zeros((n_steps, net.n), bool)
    for k in range(n_steps):
        a = net.w_in[x[x_offset + k]].sum(0)
        src = np.flatnonzero(s)
        if src.size:
            a += net.w_rec[src].sum(0)
        u = u * keep_u + a
        v = v * keep_v + u + net.bias
        s = v > net.vth
        v[s] = 0.0
        spikes[k] = s
    return spikes, dict(v=v, u=u, s=s)


def pause(net, state, mode, restore, n_pause):
    """State after a pause of n_pause steps, followed by restoring the Vars in `restore`.

    freeze: the network is stopped; Vars that are not restored come back at their
            defaults (v = 0, u = 0, no spikes in flight).
    leaky:  the network is stopped but stored state is not perfectly held. Each Var
            decays independently toward its default with its own time constant
            (no u -> v coupling, no spiking); spikes in flight are lost.
            Vars that are not restored keep their decayed value.
    """
    if mode == "freeze":
        held = net.zero_state()
    elif mode == "leaky":
        held = dict(v=state["v"] * (1.0 - net.dv) ** n_pause,
                    u=state["u"] * (1.0 - net.du) ** n_pause,
                    s=state["s"] if n_pause == 0 else np.zeros(net.n, bool))
    else:
        raise ValueError(mode)
    return {k: (state[k] if k in restore else held[k]).copy() for k in STATE_VARS}


def van_rossum_trace(a, b, tau_ms=5.0):
    """Per-step van Rossum divergence between spike rasters a and b, (T, n) bool,
    averaged over neurons. Summing over time and multiplying by dt/tau gives the
    squared van Rossum distance per neuron."""
    decay = np.exp(-DT / tau_ms)
    y = lfilter([1.0], [1.0, -decay], a.astype(float) - b, axis=0)
    return (y ** 2).mean(1)


def rate_hz(spikes):
    return spikes.mean() * 1000.0 / DT


def neuron_rates(spikes):
    """Per-neuron firing rate in Hz, (T, n) bool -> (n,)."""
    return spikes.mean(0) * 1000.0 / DT


def cv_isi(spikes, min_spikes=5):
    """Mean coefficient of variation of inter-spike intervals over neurons with enough spikes."""
    cvs = []
    for j in range(spikes.shape[1]):
        t = np.flatnonzero(spikes[:, j])
        if len(t) >= min_spikes:
            d = np.diff(t)
            cvs.append(d.std() / d.mean())
    return np.mean(cvs) if cvs else np.nan


def count_corr(spikes, bin_ms=20.0):
    """Pairwise spike-count correlation matrix (n, n); silent neurons give NaN rows."""
    b = int(round(bin_ms / DT))
    t = spikes.shape[0] // b * b
    counts = spikes[:t].reshape(-1, b, spikes.shape[1]).sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(counts.T)
