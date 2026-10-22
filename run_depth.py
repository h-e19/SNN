"""
Prediction: post-wipe dead time in a feedforward chain grows with depth, since each layer
has to charge before it can drive the next.

    python3 run_depth.py

One 4-layer chain (weights tuned so every layer fires at about 10-20 Hz). By default
every layer also gets its own balanced background input (zero mean drive, added
fluctuations) and feedforward connections are sparse (p = 0.05), which keeps deep
layers asynchronous instead of bursting. --no-background runs the original chain. Because the
chain is feedforward and weights are drawn layer by layer, layer k here is exactly the
output layer of a k-layer chain, so all depths come from one simulation.

For each layer after resume (10 ms digital stop, input buffered):
    excess dead time  time to the layer's first spike, minus the control's time to its
                      first spike from the same step (deep layers fire in bursts, so even
                      the control can be silent for tens of ms)
    lost output       spikes the layer is short of the control over WINDOW, expressed as
                      ms of the control's output: WINDOW * (1 - N_paused / N_control)
"""

import argparse
import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from pause_sim import DT, RESTORE, RESTORE_LABELS, feedforward, pause, poisson_input, simulate

# Weights tuned for ~15 Hz per layer.
CHAINS = {
    "background": dict(n_bg=500, J_bg=30.0, p_ff=0.05, J_in=5.63, J_ff=[65.82, 61.85, 53.36]),
    "no_background": dict(J_ff=[30.96, 19.17, 14.41]),
}
N_LAYERS = 4
REALIZATIONS, ONSETS_PER, SPACING = 30, 10, 3000  # pauses 300 ms apart
WINDOW = 2000  # 200 ms after resume
PAUSE_MS = 10.0
BIN = 20  # 2 ms
CONDITIONS = ["wiped", "v", "u", "v+u"]


def run_realization(job):
    seed, chain = job
    net = feedforward(n_layers=N_LAYERS, **CHAINS[chain])
    n_pause = int(round(PAUSE_MS / DT))
    onsets = [2000 + k * SPACING for k in range(ONSETS_PER)]
    n_total = onsets[-1] + WINDOW
    x = poisson_input(net.n_in, net.in_rate, n_total, seed)
    chunks, states, state, t = [], [], net.zero_state(), 0
    for t1 in onsets + [n_total]:
        spk, state = simulate(net, x, state, t, t1 - t)
        chunks.append(spk)
        states.append(state)
        t = t1
    control = np.concatenate(chunks)

    def first_spike(spk):
        return [np.argmax(spk[:, sl].any(1)) * DT if spk[:, sl].any() else np.nan for sl in net.pops.values()]

    out = {c: dict(dead=[], lost=[], rate=[], rate_ctrl=[]) for c in CONDITIONS}
    for t1, state1 in zip(onsets, states):
        ref = control[t1:t1 + WINDOW]
        dead0 = first_spike(ref)
        n0 = [ref[:, sl].sum() for sl in net.pops.values()]
        for c in CONDITIONS:
            spk, _ = simulate(net, x, pause(net, state1, "freeze", RESTORE[c], n_pause), t1, WINDOW)
            out[c]["dead"].append(np.subtract(first_spike(spk), dead0))
            out[c]["lost"].append([WINDOW * DT * (1 - spk[:, sl].sum() / n) if n else np.nan
                                   for sl, n in zip(net.pops.values(), n0)])
            out[c]["rate"].append([spk[:, sl].reshape(-1, BIN, sl.stop - sl.start).mean((1, 2))
                                   for sl in net.pops.values()])
            out[c]["rate_ctrl"].append([ref[:, sl].mean() for sl in net.pops.values()])
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-background", action="store_true")
    chain = "no_background" if ap.parse_args().no_background else "background"
    with Pool() as pool:
        out = pool.map(run_realization, [(seed, chain) for seed in range(REALIZATIONS)])
    # res[cond][metric]: (realizations * onsets, layers[, bins])
    res = {c: {k: np.concatenate([o[c][k] for o in out]) for k in out[0][c]} for c in CONDITIONS}

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    with open(os.path.join(outdir, f"depth_{chain}.pkl"), "wb") as f:
        pickle.dump(dict(res=res, chain=CHAINS[chain], pause_ms=PAUSE_MS), f)

    n = len(res["wiped"]["dead"])
    print(f"Feedforward chain ({chain.replace('_', ' ')}), {PAUSE_MS:g} ms digital stop, input buffered, {n} pauses; "
          f"layer k = output of a k-layer chain")
    for metric, desc in [("dead", "excess dead time (ms): median [IQR]"),
                         ("lost", "lost output (ms-equivalent over 200 ms): mean ± 95% CI")]:
        print(f"\n  {desc}")
        print(f"  {'restored':10s}" + "".join(f"{'depth ' + str(k + 1):>22s}" for k in range(N_LAYERS)))
        for c in CONDITIONS:
            v = res[c][metric]
            if metric == "dead":
                cells = [f"{np.nanmedian(v[:, k]):6.1f} [{np.nanpercentile(v[:, k], 25):5.1f},"
                         f"{np.nanpercentile(v[:, k], 75):5.1f}]" for k in range(N_LAYERS)]
            else:
                cells = [f"{np.nanmean(v[:, k]):6.1f} ± {1.96 * np.nanstd(v[:, k]) / np.sqrt(len(v)):4.1f}"
                         for k in range(N_LAYERS)]
            print(f"  {c:10s}" + "".join(f"{s:>22s}" for s in cells))
        center = np.nanmedian if metric == "dead" else np.nanmean
        slope = np.polyfit(np.arange(1, N_LAYERS + 1), center(res["wiped"][metric], 0), 1)[0]
        print(f"  wiped: {slope:.1f} ms per layer (linear fit to the {'medians' if metric == 'dead' else 'means'})")
    plot(res, outdir, chain)


def plot(res, outdir, chain):
    colors = plt.cm.viridis(np.linspace(0, 0.85, N_LAYERS))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    depth = np.arange(1, N_LAYERS + 1)
    for ax, metric in zip(axes[:2], ["dead", "lost"]):
        for c, m in zip(CONDITIONS, ["o", "s", "^", "x"]):
            v = res[c][metric]
            if metric == "dead":
                mid = np.nanmedian(v, 0)
                err = [mid - np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0) - mid]
            else:
                mid = np.nanmean(v, 0)
                err = 1.96 * np.nanstd(v, 0) / np.sqrt(len(v))
            ax.errorbar(depth, mid, yerr=err, fmt=m + "-", capsize=3, label=RESTORE_LABELS[c])
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(depth)
        ax.set_xlabel("depth (output layer of a k-layer chain)")
        ax.set_ylabel("ms")
        ax.set_title("excess dead time vs control (median, IQR)" if metric == "dead"
                     else "lost output, ms-equivalent over 200 ms (mean, 95% CI)", fontsize=9)
    axes[0].legend(fontsize=7)
    ax = axes[2]
    # pooled over pauses: deep layers are bursty, so single-trial control rates can be 0
    r = res["wiped"]["rate"].mean(0) / res["wiped"]["rate_ctrl"].mean(0)[:, None]
    t = (np.arange(r.shape[1]) + 0.5) * BIN * DT
    for k in range(N_LAYERS):
        ax.plot(t, r[k], color=colors[k], label=f"layer {k + 1}")
    ax.axhline(1, color="gray", lw=0.5)
    ax.set_xlim(0, 120)
    ax.set_xlabel("time after resume (ms)")
    ax.set_ylabel("rate / control rate")
    ax.set_title("wiped: layer rates after resume (mean)", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(f"Post-wipe dead time vs depth, feedforward chain ({chain.replace('_', ' ')}), "
                 f"{PAUSE_MS:g} ms digital stop", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"depth_{chain}.png"), dpi=130)


if __name__ == "__main__":
    main()
