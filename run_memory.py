"""
Memory task: does a pause during a delay erase what the network is holding?

    python3 run_memory.py

Network: the recurrent E-I network with E split into N_CLUSTERS clusters
(within-cluster E->E weights x J_PLUS, Litwin-Kumar & Doiron 2012). At J_PLUS = 2.4
one cluster at a time is active (~50 Hz, the others near 0); a stimulus to a cluster's
input pool switches the active cluster, and the switch persists without input. Some
cluster is always active, so each trial's stimulus simply overwrites the last memory.

Trial: STIM_MS stimulus to a random cluster, then DELAY_MS delay with background input
only (about 13 tau_m, so leftover membrane voltage cannot carry the answer). The
readout is "which cluster is most active" over the last WINDOW_MS of the delay, and the
margin is the correct cluster's rate minus the best other cluster's rate (Hz).

Each pause (PAUSE_MS, digital stop) lands PAUSE_AT_MS into the delay, branching off one
uninterrupted control; scoring is paired with the control on the same trial as in
run_task.py. Under "input lost" the network misses PAUSE_MS of background input, which
carries no information about the answer, so it isolates state loss.
"""

import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pause_sim import DT, RESTORE, RESTORE_LABELS, pause, poisson_input, recurrent, simulate

N_CLUSTERS, J_PLUS = 4, 2.4
N_STIM, STIM_RATE = 200, 30.0  # stimulus pool per cluster, Hz while the stimulus is on
N_BG = 1000
STIM_MS, DELAY_MS, WINDOW_MS = 50.0, 200.0, 20.0
PAUSE_MS, PAUSE_AT_MS = 10.0, 100.0
REALIZATIONS, TRIALS = 30, 30
WARMUP_TRIALS = 2  # the first trials are not paused or scored
INPUTS = {"buffered": "input buffered", "dropped": "input lost"}
CONDITIONS = [(r, i) for i in INPUTS for r in RESTORE]

STIM, DELAY, WINDOW = (int(round(ms / DT)) for ms in (STIM_MS, DELAY_MS, WINDOW_MS))
TRIAL = STIM + DELAY
N_PAUSE, PAUSE_AT = int(round(PAUSE_MS / DT)), int(round(PAUSE_AT_MS / DT))


def build():
    return recurrent(n_clusters=N_CLUSTERS, J_plus=J_PLUS, n_stim=N_STIM)


def trial_input(n_trials, seed):
    """Background on inputs 0..N_BG-1 plus, during each trial's stimulus, its cluster's
    pool. Returns (spikes as list of index arrays, cluster per trial)."""
    rng = np.random.default_rng([seed, 3])
    labels = rng.integers(N_CLUSTERS, size=n_trials)
    x = poisson_input(N_BG, 10.0, n_trials * TRIAL, seed)
    p = STIM_RATE * DT / 1000.0
    for k, c in enumerate(labels):
        for t in range(k * TRIAL, k * TRIAL + STIM):
            x[t] = np.concatenate([x[t], N_BG + c * N_STIM + np.flatnonzero(rng.random(N_STIM) < p)])
    return x, labels


def cluster_rates(spk, net):
    """Rate (Hz) of each cluster over the raster spk."""
    return np.array([spk[:, net.pops[f"C{c}"]].mean() * 1000.0 / DT for c in range(N_CLUSTERS)])


def evaluate(rates, label):
    others = np.delete(rates, label)
    return float(rates[label] > others.max()), rates[label] - others.max()


def label(c):
    return f"{c[0]}, {INPUTS[c[1]]}"


def run_realization(seed):
    net = build()
    x, labels = trial_input(TRIALS + WARMUP_TRIALS, seed)
    starts = [(k + WARMUP_TRIALS) * TRIAL for k in range(TRIALS)]
    onsets = [s + STIM + PAUSE_AT for s in starts]
    n_total = (TRIALS + WARMUP_TRIALS) * TRIAL

    chunks, states, state, t = [], [], net.zero_state(), 0
    for t1 in onsets + [n_total]:
        spk, state = simulate(net, x, state, t, t1 - t)
        chunks.append(spk)
        states.append(state)
        t = t1
    control = np.concatenate(chunks)

    out = {c: dict(correct=[], margin=[], identical=[]) for c in ["control"] + CONDITIONS}
    for k, (start, t1, state1) in enumerate(zip(starts, onsets, states)):
        y = labels[k + WARMUP_TRIALS]
        end = start + TRIAL  # input index where the delay ends
        ref = control[end - WINDOW:end]
        a, m = evaluate(cluster_rates(ref, net), y)
        out["control"]["correct"].append(a)
        out["control"]["margin"].append(m)
        out["control"]["identical"].append(1.0)
        for c in CONDITIONS:
            rname, inp = c
            offset = t1 if inp == "buffered" else t1 + N_PAUSE
            spk, _ = simulate(net, x, pause(net, state1, "freeze", RESTORE[rname], N_PAUSE), offset, end - offset)
            win = spk[-WINDOW:]
            a, m = evaluate(cluster_rates(win, net), y)
            out[c]["correct"].append(a)
            out[c]["margin"].append(m)
            out[c]["identical"].append(float(np.array_equal(win, ref)))
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in out.items()}


def paired_stats(res, c):
    a, a0 = res[c]["correct"], res["control"]["correct"]
    m, m0 = res[c]["margin"], res["control"]["margin"]
    lost = int(((a0 == 1) & (a == 0)).sum())
    gained = int(((a0 == 0) & (a == 1)).sum())
    p_acc = stats.binomtest(lost, lost + gained).pvalue if lost + gained else 1.0  # exact McNemar
    diff = (a - a0).ravel()
    ci_acc = 1.96 * diff.std(ddof=1) / np.sqrt(diff.size)
    d = (m - m0).mean(1)  # per realization
    if d.any():
        ci_m = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        p_m = stats.ttest_1samp(d, 0).pvalue
    else:
        ci_m, p_m = 0.0, 1.0
    return dict(acc=a.mean(), d_acc=diff.mean(), ci_acc=ci_acc, lost=lost, gained=gained, p_acc=p_acc,
                d_margin=d.mean(), ci_margin=ci_m, p_margin=p_m, identical=res[c]["identical"].mean())


def main():
    with Pool() as pool:
        out = pool.map(run_realization, range(REALIZATIONS))
    res = {c: {k: np.stack([o[c][k] for o in out]) for k in out[0][c]} for c in ["control"] + CONDITIONS}

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "memory.pkl"), "wb") as f:
        pickle.dump(dict(res=res, J_plus=J_PLUS, delay_ms=DELAY_MS, pause_ms=PAUSE_MS, pause_at_ms=PAUSE_AT_MS), f)

    ctrl = res["control"]
    print(f"Clustered E-I network (J+ {J_PLUS}), {DELAY_MS:g} ms delay, {PAUSE_MS:g} ms digital stop "
          f"{PAUSE_AT_MS:g} ms into the delay; {ctrl['correct'].size} trials per condition "
          f"({REALIZATIONS} streams x {TRIALS}), chance {1 / N_CLUSTERS:.2f}")
    print(f"control: accuracy {ctrl['correct'].mean():.3f}, mean margin {ctrl['margin'].mean():.1f} Hz")
    print(f"{'condition':26s}{'acc':>7s}{'Δacc [95% CI]':>24s}{'lost/gained':>13s}{'p':>8s}"
          f"{'Δmargin Hz [95% CI]':>27s}{'p':>9s}{'spikes = control':>18s}")
    for c in CONDITIONS:
        s = paired_stats(res, c)
        print(f"{label(c):26s}{s['acc']:7.3f}{s['d_acc']:+9.3f} [{s['d_acc'] - s['ci_acc']:+.3f},"
              f"{s['d_acc'] + s['ci_acc']:+.3f}]{s['lost']:>7d}/{s['gained']:<5d}{s['p_acc']:8.1g}"
              f"{s['d_margin']:+10.1f} [{s['d_margin'] - s['ci_margin']:+.1f},{s['d_margin'] + s['ci_margin']:+.1f}]"
              f"{s['p_margin']:9.1g}{s['identical']:18.2f}")
    plot(res, outdir)


def plot(res, outdir):
    colors = dict(zip(RESTORE, plt.cm.viridis(np.linspace(0, 0.9, len(RESTORE)))))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1.1, 1.1]))
    y = np.arange(len(CONDITIONS))[::-1]
    names = [label(c) for c in CONDITIONS]
    for ax, key, xlabel in [(axes[0], "acc", "accuracy"), (axes[1], "margin", "margin (Hz)")]:
        for yi, c in zip(y, CONDITIONS):
            s = paired_stats(res, c)
            ax.errorbar(s[f"d_{key}"], yi, xerr=s[f"ci_{key}"], fmt="o", ms=5, capsize=2, color=colors[c[0]])
        ax.axvline(0, color="black", lw=0.6)
        ax.axhline(len(RESTORE) - 0.5, color="gray", lw=0.4)
        ax.set_yticks(y, names if key == "acc" else [], fontsize=8)
        ax.set_xlabel(f"{xlabel} change vs control, same trials (95% CI)")
    axes[0].set_title(f"control accuracy {res['control']['correct'].mean():.2f} (chance {1 / N_CLUSTERS:.2f})",
                      fontsize=9)
    axes[1].set_title("margin = correct cluster rate − best other", fontsize=9)
    ax = axes[2]
    ident = [paired_stats(res, c)["identical"] for c in CONDITIONS]
    acc = [paired_stats(res, c)["acc"] for c in CONDITIONS]
    for c, i, a in zip(CONDITIONS, ident, acc):
        ax.scatter(i, a, color=colors[c[0]], marker="o" if c[1] == "buffered" else "s", s=40,
                   label=f"{RESTORE_LABELS[c[0]].split(':')[0]}" if c[1] == "buffered" else None)
    ax.axhline(1 / N_CLUSTERS, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("fraction of trials with spikes identical to control (readout window)")
    ax.set_ylabel("memory accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("exact spikes vs memory (● input buffered, ■ input lost)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Memory task: {DELAY_MS:g} ms delay, {PAUSE_MS:g} ms digital stop {PAUSE_AT_MS:g} ms into it "
                 f"(clustered E-I, J+ {J_PLUS}; {res['control']['correct'].size} trials per condition)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "memory.png"), dpi=130)


if __name__ == "__main__":
    main()
