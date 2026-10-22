"""
Do the recurrent network's statistics survive a pause when its exact spikes don't?

    python3 run_stats.py

Each repeat uses its own input realization, paused at step T1. After each pause the
network runs WINDOW steps (5 s), long enough for rate, CV and
pairwise-correlation estimates. Each run is compared with the uninterrupted control on
the steps that consumed the same world input (as in run_grid.py).

Three reference runs give the scale for "statistically the same" and "different":
    independent input   same network and state, a different input realization
                        (the noise floor: two legitimate runs of this network)
    inhibition -10%     same connectivity and input, J_ei and J_ii scaled by 0.9
    other network       different random connectivity, same input
"""

import itertools
import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from pause_sim import (DT, MODE_LABELS, RESTORE, count_corr, cv_isi, neuron_rates, pause, poisson_input,
                       rate_hz, recurrent, simulate, van_rossum_trace)

REPEATS = 8  # independent input realizations, each paused at T1
T1 = 2000
WINDOW = 50000  # 5 s
PAUSE_MS = 10.0  # any nonzero pause fully decorrelates spikes (run_grid.py)
ALT_SEED = 1000  # + repeat index
TRANSIENT_MS, TRANSIENT_BIN_MS = 300, 5

REFERENCES = {
    "independent input": None,
    "inhibition -10%": dict(J_ei=-45.0 * 0.9, J_ii=-60.0 * 0.9),
    "other network": dict(seed=2),
}
CONDITIONS = [(m, r, i) for m, r, i in itertools.product(["freeze", "leaky"], RESTORE, ["buffered", "dropped"])]


def run_repeat(seed):
    """Control, references and every pause condition for one input realization; returns metrics only."""
    t1 = T1
    net = recurrent()
    n_pause = int(round(PAUSE_MS / DT))
    x = poisson_input(net.n_in, net.in_rate, t1 + n_pause + WINDOW, seed)
    _, state1 = simulate(net, x, net.zero_state(), 0, t1)
    control, _ = simulate(net, x, state1, t1, n_pause + WINDOW)
    ref_for = lambda offset: control[offset - t1:offset - t1 + WINDOW]

    res = {"control": measure(ref_for(t1), ref_for(t1), net.pops)}
    for name, changes in REFERENCES.items():
        if changes is None:
            x_alt = poisson_input(net.n_in, net.in_rate, t1 + WINDOW, ALT_SEED + seed)
            spk, _ = simulate(net, x_alt, state1, t1, WINDOW)
        else:
            spk, _ = simulate(recurrent(**changes), x, state1, t1, WINDOW)
        res[name] = measure(spk, ref_for(t1), net.pops)
    for c in CONDITIONS:
        mode, rname, inp = c
        state = pause(net, state1, mode, RESTORE[rname], n_pause)
        offset = t1 if inp == "buffered" else t1 + n_pause
        spk, _ = simulate(net, x, state, offset, WINDOW)
        res[c] = measure(spk, ref_for(offset), net.pops)

    # Spike divergence relative to the independent-input run (1 = unrelated).
    ceiling = res["independent input"]["spike divergence"]
    for r in res.values():
        r["spike divergence"] /= ceiling
    return res


def upper(m):
    return m[np.triu_indices_from(m, 1)]


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def measure(spk, ref, pops):
    """Statistics of one run and its similarity to the control window `ref`."""
    E, I = pops["E"], pops["I"]
    c_run, c_ref = count_corr(spk[:, E]), count_corr(ref[:, E])
    b = int(TRANSIENT_BIN_MS / DT)
    head = spk[:int(TRANSIENT_MS / DT), E]
    return {
        "E rate (Hz)": rate_hz(spk[:, E]),
        "I rate (Hz)": rate_hz(spk[:, I]),
        "CV of ISI (E)": cv_isi(spk[:, E]),
        "mean pair corr (E)": np.nanmean(upper(c_run)),
        "spike divergence": van_rossum_trace(spk[:, E], ref[:, E]).mean(),
        "neuron-rate similarity r": pearson(neuron_rates(spk[:, E]), neuron_rates(ref[:, E])),
        "corr-structure similarity r": pearson(upper(c_run), upper(c_ref)),
        "_transient": head.reshape(-1, b, head.shape[1]).mean((1, 2)) * 1000 / DT,
    }


def main():
    with Pool() as pool:
        out = pool.map(run_repeat, range(REPEATS))
    names = ["control"] + list(REFERENCES) + CONDITIONS
    per = {name: [o[name] for o in out] for name in names}

    metrics = [k for k in per["control"][0] if not k.startswith("_")]
    summary = {name: {m: (np.mean([r[m] for r in rs]), np.std([r[m] for r in rs], ddof=1) / np.sqrt(len(rs)))
                      for m in metrics} for name, rs in per.items()}

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "stats_recurrent.pkl"), "wb") as f:
        pickle.dump(dict(per=per, summary=summary, pause_ms=PAUSE_MS, repeats=REPEATS), f)

    label = lambda n: n if isinstance(n, str) else f"{'digital' if n[0] == 'freeze' else 'leaky'}, {n[1]}, {n[2]}"
    print(f"{'':42s}" + "".join(f"{m[:16]:>18s}" for m in metrics))
    for name in names:
        print(f"{label(name):42s}" + "".join(f"{summary[name][m][0]:12.3f}±{summary[name][m][1]:<5.3f}" for m in metrics))
    plot(per, summary, metrics, names, label, outdir)


def plot(per, summary, metrics, names, label, outdir):
    shown = ["control"] + list(REFERENCES) + [
        ("freeze", r, "buffered") for r in RESTORE] + [
        ("freeze", "wiped", "dropped"), ("freeze", "v+u", "dropped"),
        ("leaky", "wiped", "buffered"), ("leaky", "v+u", "buffered")]
    ref_color, pause_color = "#888888", "#2a6f97"
    y = np.arange(len(shown))[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 0.36 * len(shown) + 1.4), sharey=True)
    for ax, m in zip(axes, metrics):
        for yi, name in zip(y, shown):
            mu, se = summary[name][m]
            color = "black" if name == "control" else ref_color if isinstance(name, str) else pause_color
            ax.errorbar(mu, yi, xerr=se, fmt="o", ms=4, color=color, capsize=2)
        ax.axvline(summary["control"][m][0], color="black", lw=0.6, ls=":")
        lo, hi = summary["independent input"][m]
        ax.axvspan(lo - 2 * hi, lo + 2 * hi, color=ref_color, alpha=0.15, lw=0)
        ax.set_title(m, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.axhline(y[len(REFERENCES)] - 0.5, color="gray", lw=0.4)
    axes[0].set_yticks(y, [label(n) for n in shown], fontsize=8)
    fig.suptitle(f"Recurrent E-I network, {PAUSE_MS:g} ms pause, 5 s after resume "
                 f"(mean ± SEM over {REPEATS} input realizations; band = independent-input run ± 2 SEM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stats_recurrent.png"), dpi=130)

    # Population rate right after resume.
    fig, ax = plt.subplots(figsize=(7, 3.2))
    t = (np.arange(TRANSIENT_MS // TRANSIENT_BIN_MS) + 0.5) * TRANSIENT_BIN_MS
    for name, color in [("control", "black"), (("freeze", "wiped", "buffered"), "#c1121f"),
                        (("freeze", "v+u", "buffered"), "#2a6f97"), (("leaky", "wiped", "buffered"), "#e09f3e")]:
        ax.plot(t, np.mean([r["_transient"] for r in per[name]], 0), color=color, label=label(name))
    ax.set_xlabel("time after resume (ms)")
    ax.set_ylabel("E population rate (Hz)")
    ax.set_title(f"Transient after a {PAUSE_MS:g} ms pause (mean over realizations; leaky = {MODE_LABELS['leaky']})", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stats_transient.png"), dpi=130)


if __name__ == "__main__":
    main()
