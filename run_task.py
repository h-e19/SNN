"""
Task-level metric: does a pause hurt a toy classification task?

    python3 run_task.py

Every SEG steps (50 ms) the input shows one of N_CLASSES stimuli (a class-specific
subset of the inputs fires faster; total input rate is equal across classes). A ridge
readout on the output population's spike counts per segment names the class. The
readout is trained once per network on a long uninterrupted run, then frozen.

Each realization is a fresh stimulus stream with ONSETS_PER pauses, each landing in the
middle of a stimulus and spaced SPACING segments apart, so each pause is an
independent trial branching off the same uninterrupted control. We score the
interrupted stimulus and the next N_EVAL, on the paused branch and on the control.

Two scores per trial: correct / wrong, and the readout margin (score of the correct
class minus the best other class), which is continuous and shows partial damage.
Statistics are paired against the control on the same trials:
    accuracy: McNemar's exact test on trials that flipped (pooled over all trials)
    margin:   mean change with a 95% t-CI over realizations
Conditions separate two kinds of loss:
    state lost   restore set is not "all"
    input lost   the stimulus shown during the pause never reaches the network
"""

import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pause_sim import DT, RESTORE, RESTORE_LABELS, feedforward, pause, recurrent, simulate, stimulus_input

SEG = 500  # steps per stimulus (50 ms)
N_CLASSES, FRAC, BOOST = 4, 0.25, 25.0
N_TRAIN, LAM = 600, 1000.0
REALIZATIONS = 30
ONSETS_PER, FIRST_SEG, SPACING = 30, 20, 8  # 30 pauses per stream, 400 ms apart
N_EVAL = 3  # stimuli scored after the interrupted one
PAUSE_MS = 10.0
TRAIN_SEED, TEST_SEED = 100, 1000
NETWORKS = {"feedforward": (feedforward, "L2"), "recurrent": (recurrent, "E")}
INPUTS = {"buffered": "input buffered", "dropped": "input lost"}
CONDITIONS = [(r, i) for i in INPUTS for r in RESTORE]  # digital stop only


def stream(net, n_steps, seed):
    return stimulus_input(net.n_in, net.in_rate, BOOST, N_CLASSES, FRAC, SEG, n_steps, seed)


def train(kind):
    """Ridge readout (one-hot targets) on spike counts of an uninterrupted run."""
    build, pop = NETWORKS[kind]
    net = build()
    x, labels = stream(net, N_TRAIN * SEG, TRAIN_SEED)
    spk, _ = simulate(net, x, net.zero_state(), 0, N_TRAIN * SEG)
    X = spk[:, net.pops[pop]].reshape(N_TRAIN, SEG, -1).sum(1).astype(float)[20:]  # skip start-up
    y = labels[20:]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    W = np.linalg.solve(Z.T @ Z + LAM * np.eye(Z.shape[1]), Z.T @ np.eye(N_CLASSES)[y])
    return dict(mu=mu, sd=sd, W=W)


def evaluate(readout, X, y):
    """(correct, margin) per row of spike counts X with true classes y."""
    s = np.c_[(X - readout["mu"]) / readout["sd"], np.ones(len(X))] @ readout["W"]
    rows = np.arange(len(y))
    true = s[rows, y].copy()
    s[rows, y] = -np.inf
    return (true > s.max(1)).astype(float), true - s.max(1)


def label(c):
    return f"{c[0]}, {INPUTS[c[1]]}"


def kind_of_loss(c):
    lost = [w for w, flag in [("state", c[0] != "all"), ("input", c[1] == "dropped")] if flag]
    return " + ".join(lost) + " lost" if lost else "nothing lost"


def run_realization(job):
    """All conditions for one stimulus stream. Returns, per condition and for the control,
    arrays (ONSETS_PER, N_EVAL + 1) of correct and margin; column 0 is the interrupted
    stimulus."""
    kind, seed, readout = job
    build, pop = NETWORKS[kind]
    net = build()
    sl = net.pops[pop]
    n_pause = int(round(PAUSE_MS / DT))
    onsets = [(FIRST_SEG + k * SPACING) * SEG + SEG // 2 for k in range(ONSETS_PER)]
    n_total = onsets[-1] + n_pause + (N_EVAL + 2) * SEG
    x, labels = stream(net, n_total, TEST_SEED + seed)

    # One uninterrupted control, keeping its state at every onset.
    chunks, states, state, t = [], [], net.zero_state(), 0
    for t1 in onsets + [n_total]:
        spk, state = simulate(net, x, state, t, t1 - t)
        chunks.append(spk)
        states.append(state)
        t = t1
    control = np.concatenate(chunks)

    def score(spk, offset, t1):
        """spk[k] consumed input offset + k. The interrupted stimulus counts the spikes
        before t1 (shared with the control) plus those after resume."""
        seg0 = t1 // SEG
        idx = np.arange(seg0, seg0 + N_EVAL + 1)
        X = [control[seg0 * SEG:t1, sl].sum(0) + spk[:(seg0 + 1) * SEG - offset, sl].sum(0)]
        X += [spk[j * SEG - offset:(j + 1) * SEG - offset, sl].sum(0) for j in idx[1:]]
        return evaluate(readout, np.array(X, float), labels[idx])

    out = {c: ([], []) for c in ["control"] + CONDITIONS}
    for t1, state1 in zip(onsets, states):
        runs = {"control": score(control[t1:], t1, t1)}
        for c in CONDITIONS:
            rname, inp = c
            offset = t1 if inp == "buffered" else t1 + n_pause
            spk, _ = simulate(net, x, pause(net, state1, "freeze", RESTORE[rname], n_pause), offset,
                              (N_EVAL + 1) * SEG)
            runs[c] = score(spk, offset, t1)
        for c, (correct, margin) in runs.items():
            out[c][0].append(correct)
            out[c][1].append(margin)
    return kind, {c: dict(correct=np.array(a), margin=np.array(m)) for c, (a, m) in out.items()}


def paired_stats(res, c, col):
    """Accuracy and margin change vs control on one stimulus (column), paired by trial."""
    a, a0 = res[c]["correct"][:, :, col], res["control"]["correct"][:, :, col]
    m, m0 = res[c]["margin"][:, :, col], res["control"]["margin"][:, :, col]
    lost = int(((a0 == 1) & (a == 0)).sum())  # control right, paused wrong
    gained = int(((a0 == 0) & (a == 1)).sum())
    p_acc = stats.binomtest(lost, lost + gained).pvalue if lost + gained else 1.0  # exact McNemar
    diff = (a - a0).ravel()
    ci_acc = 1.96 * diff.std(ddof=1) / np.sqrt(diff.size)
    d = (m - m0).mean(1)  # per realization
    if d.any():
        ci_margin = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        p_margin = stats.ttest_1samp(d, 0).pvalue
    else:
        ci_margin, p_margin = 0.0, 1.0
    return dict(d_acc=diff.mean(), ci_acc=ci_acc, lost=lost, gained=gained, p_acc=p_acc,
                d_margin=d.mean(), ci_margin=ci_margin, p_margin=p_margin)


def main():
    with Pool() as pool:
        readouts = dict(zip(NETWORKS, pool.map(train, NETWORKS)))
        jobs = [(kind, seed, readouts[kind]) for kind in NETWORKS for seed in range(REALIZATIONS)]
        out = pool.map(run_realization, jobs)

    # results[kind][cond][metric]: (realizations, onsets, N_EVAL + 1)
    results = {kind: {c: {m: np.stack([o[c][m] for k, o in out if k == kind]) for m in ["correct", "margin"]}
                      for c in ["control"] + CONDITIONS} for kind in NETWORKS}

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "task.pkl"), "wb") as f:
        pickle.dump(dict(results=results, pause_ms=PAUSE_MS, seg_ms=SEG * DT), f)

    for kind, res in results.items():
        ctrl = res["control"]
        print(f"\n== {kind} ({NETWORKS[kind][1]}): {ctrl['correct'][:, :, 0].size} interrupted stimuli per "
              f"condition ({REALIZATIONS} streams x {ONSETS_PER} pauses), chance {1 / N_CLASSES:.2f}")
        print(f"   control accuracy {ctrl['correct'][:, :, 0].mean():.3f}, "
              f"mean margin {ctrl['margin'][:, :, 0].mean():.3f}")
        for col, name in [(0, "interrupted stimulus"), (1, "next stimulus")]:
            print(f"   -- {name}: change vs control on the same trials")
            print(f"   {'condition':30s}{'loss':22s}{'Δacc [95% CI]':>22s}{'lost/gained':>13s}{'p':>8s}"
                  f"{'Δmargin [95% CI]':>26s}{'p':>9s}")
            for c in CONDITIONS:
                s = paired_stats(res, c, col)
                print(f"   {label(c):30s}{kind_of_loss(c):22s}"
                      f"{s['d_acc']:+8.3f} [{s['d_acc'] - s['ci_acc']:+.3f},{s['d_acc'] + s['ci_acc']:+.3f}]"
                      f"{s['lost']:>7d}/{s['gained']:<5d}{s['p_acc']:8.1g}"
                      f"{s['d_margin']:+10.3f} [{s['d_margin'] - s['ci_margin']:+.3f},"
                      f"{s['d_margin'] + s['ci_margin']:+.3f}]{s['p_margin']:9.1g}")
    plot(results, outdir)


def plot(results, outdir):
    colors = dict(zip(RESTORE, plt.cm.viridis(np.linspace(0, 0.9, len(RESTORE)))))
    fig, axes = plt.subplots(len(results), 3, figsize=(17, 4.4 * len(results)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    for k, (kind, res) in enumerate(results.items()):
        y = np.arange(len(CONDITIONS))[::-1]
        for j, (metric, key) in enumerate([("accuracy", "acc"), ("margin", "margin")]):
            ax = axes[k, j]
            for yi, c in zip(y, CONDITIONS):
                for col, marker, dy in [(0, "s", 0.15), (1, "o", -0.15)]:
                    s = paired_stats(res, c, col)
                    ax.errorbar(s[f"d_{key}"], yi + dy, xerr=s[f"ci_{key}"], fmt=marker, ms=4, capsize=2,
                                color=colors[c[0]], mfc=colors[c[0]] if marker == "o" else "white")
            ax.axvline(0, color="black", lw=0.6)
            ax.axhline(len(RESTORE) - 0.5, color="gray", lw=0.4)
            ax.set_yticks(y, [f"{label(c)}  ({kind_of_loss(c)})" for c in CONDITIONS], fontsize=7)
            ax.set_xlabel(f"{metric} change vs control, same trials (95% CI)")
            ax.set_title(f"{kind} ({NETWORKS[kind][1]}): □ interrupted stimulus, ● next stimulus", fontsize=9)

        ax = axes[k, 2]
        seg = np.arange(N_EVAL + 1)
        ax.plot(seg, res["control"]["margin"].mean((0, 1)), "k-", label="control")
        for c in [("wiped", "buffered"), ("v+u", "buffered"), ("all", "dropped")]:
            ax.plot(seg, res[c]["margin"].mean((0, 1)), "o-" if c[1] == "buffered" else "s--", ms=3,
                    color=colors[c[0]], label=f"{RESTORE_LABELS[c[0]].split(':')[0]}, {INPUTS[c[1]]}")
        ax.set_xticks(seg, ["interrupted"] + [f"+{i}" for i in seg[1:]], fontsize=8)
        ax.set_ylabel("mean readout margin")
        ax.set_title(f"{kind}: margin by stimulus", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Toy {N_CLASSES}-class task, {PAUSE_MS:g} ms digital stop in the middle of a {SEG * DT:g} ms "
                 f"stimulus ({REALIZATIONS} streams × {ONSETS_PER} pauses per condition)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "task.png"), dpi=130)


if __name__ == "__main__":
    main()
