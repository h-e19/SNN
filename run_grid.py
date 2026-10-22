"""
Run the pause experiment over the full condition grid and plot the results.

    python3 run_grid.py                      # default time constants
    python3 run_grid.py --tau-m 30 --tau-s 10 --tag slow

For every network the control run is a single uninterrupted run. Every paused run
shares the control's state at t1, gets paused / restored, then runs WINDOW steps.
Paused and control spikes are compared on steps that consumed the same world input:
    buffered: paused step k  vs  control step t1 + k        (network time)
    dropped:  paused step k  vs  control step t1 + dt + k   (world time)

Divergence is the van Rossum trace divided by its value between the control and a
run with independent input (same network), so 0 = identical, ~1 = unrelated.

The whole grid is repeated for several pause onsets t1 and averaged, since whether
e.g. spikes in flight exist at t1 depends on the exact step.
"""

import argparse
import itertools
import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from pause_sim import (DT, MODE_LABELS, RESTORE, RESTORE_LABELS, feedforward, pause, poisson_input, rate_hz,
                       recurrent, simulate, van_rossum_trace)

ONSETS = [2000 + 1000 * i for i in range(8)]  # pause onsets t1 (200 ms ... 900 ms)
WINDOW = 5000  # steps compared after the pause (500 ms)
PAUSES_MS = [0, 1, 2, 5, 10, 20, 50, 100]
MODES = ["freeze", "leaky"]
INPUTS = ["buffered", "dropped"]
NETWORKS = {"feedforward": feedforward, "recurrent": recurrent}
BIN = 50  # steps per bin for stored time courses (5 ms)
INPUT_SEED, ALT_SEED = 0, 1000


def binned(trace):
    return trace.reshape(-1, BIN).mean(1)


def run_network(net, T1):
    max_pause = int(max(PAUSES_MS) / DT)
    n_total = T1 + max_pause + WINDOW
    x = poisson_input(net.n_in, net.in_rate, n_total, INPUT_SEED)

    pre, state1 = simulate(net, x, net.zero_state(), 0, T1)
    post, _ = simulate(net, x, state1, T1, n_total - T1)
    control = np.concatenate([pre, post])

    x_alt = poisson_input(net.n_in, net.in_rate, n_total, ALT_SEED)
    alt, _ = simulate(net, x_alt, net.zero_state(), 0, n_total)
    ceiling = {p: van_rossum_trace(control[T1:, s], alt[T1:, s]).mean() for p, s in net.pops.items()}

    def compare(spikes, offset):
        ref = control[offset:offset + WINDOW]
        out = {}
        for p, s in net.pops.items():
            d = van_rossum_trace(spikes[:, s], ref[:, s]) / ceiling[p]
            out[p] = dict(curve=binned(d), mean=d.mean(),
                          rate=rate_hz(spikes[:, s]), rate_ctrl=rate_hz(ref[:, s]))
        return out

    # Reference perturbation: 1 mV kick to one random neuron, no pause.
    kicked = {k: v.copy() for k, v in state1.items()}
    kicked["v"][np.random.default_rng(5).integers(net.n)] += 1.0
    spk, _ = simulate(net, x, kicked, T1, WINDOW)
    perturb = compare(spk, T1)

    rows = []
    for mode, (rname, restore), inp, pause_ms in itertools.product(MODES, RESTORE.items(), INPUTS, PAUSES_MS):
        n_pause = int(round(pause_ms / DT))
        state = pause(net, state1, mode, restore, n_pause)
        offset = T1 if inp == "buffered" else T1 + n_pause
        spk, _ = simulate(net, x, state, offset, WINDOW)
        res = compare(spk, offset)
        if mode == "freeze" and rname == "all" and inp == "buffered":
            assert not (spk != control[offset:offset + WINDOW]).any(), "full freeze should be exact"
        rows.append(dict(net=net.name, mode=mode, restore=rname, input=inp, pause_ms=pause_ms, res=res))
    return dict(net=net.name, pops=list(net.pops), perturb=perturb, rows=rows,
                in_flight=int(state1["s"].sum()),
                rates={p: rate_hz(control[T1:, s]) for p, s in net.pops.items()})


def _job(args):
    kind, tau_m, tau_s, t1 = args
    net = NETWORKS[kind](tau_m=tau_m, tau_s=tau_s)
    return run_network(net, t1)


def aggregate(per_onset):
    """Average one network's results over pause onsets."""
    first = per_onset[0]
    rows = []
    for i, row in enumerate(first["rows"]):
        res = {}
        for p in first["pops"]:
            means = np.array([r["rows"][i]["res"][p]["mean"] for r in per_onset])
            res[p] = dict(curve=np.mean([r["rows"][i]["res"][p]["curve"] for r in per_onset], 0),
                          mean=means.mean(), sem=means.std(ddof=1) / np.sqrt(len(means)),
                          rate=np.mean([r["rows"][i]["res"][p]["rate"] for r in per_onset]))
        rows.append({**{k: row[k] for k in ("net", "mode", "restore", "input", "pause_ms")}, "res": res})
    perturb = {p: dict(curve=np.mean([r["perturb"][p]["curve"] for r in per_onset], 0),
                       mean=np.mean([r["perturb"][p]["mean"] for r in per_onset])) for p in first["pops"]}
    rates = {p: np.mean([r["rates"][p] for r in per_onset]) for p in first["pops"]}
    return dict(net=first["net"], pops=first["pops"], rows=rows, perturb=perturb, rates=rates,
                in_flight=[r["in_flight"] for r in per_onset])


def plot(results, tag, outdir):
    colors = dict(zip(RESTORE, plt.cm.viridis(np.linspace(0, 0.9, len(RESTORE)))))
    t_ms = (np.arange(WINDOW // BIN) + 0.5) * BIN * DT
    readout = {r["net"]: r["pops"][0] if r["net"] == "recurrent" else r["pops"][-1] for r in results}

    # Time courses after resume, leaky pause of 20 ms.
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 3.2 * len(results)), sharex=True, squeeze=False)
    for i, r in enumerate(results):
        pop = readout[r["net"]]
        for j, inp in enumerate(INPUTS):
            ax = axes[i, j]
            for row in r["rows"]:
                if row["mode"] == "leaky" and row["input"] == inp and row["pause_ms"] == 20:
                    ax.plot(t_ms, row["res"][pop]["curve"], color=colors[row["restore"]], label=RESTORE_LABELS[row["restore"]])
            ax.plot(t_ms, r["perturb"][pop]["curve"], "k--", lw=1, label="1 mV kick, no pause")
            ax.axhline(1, color="gray", lw=0.5)
            ax.set_title(f"{r['net']} ({pop}), {MODE_LABELS['leaky']}, 20 ms, input {inp}", fontsize=9)
            ax.set_ylabel("divergence (1 = unrelated)")
    for ax in axes[-1]:
        ax.set_xlabel("time after resume (ms)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"timecourse_{tag}.png"), dpi=130)

    # Mean divergence over the window against pause length.
    fig, axes = plt.subplots(len(results), 4, figsize=(15, 3.2 * len(results)), sharey=True, squeeze=False)
    for i, r in enumerate(results):
        pop = readout[r["net"]]
        for j, (mode, inp) in enumerate(itertools.product(MODES, INPUTS)):
            ax = axes[i, j]
            for rname in RESTORE:
                pts = [(row["pause_ms"], row["res"][pop]["mean"], row["res"][pop]["sem"]) for row in r["rows"]
                       if (row["mode"], row["input"], row["restore"]) == (mode, inp, rname)]
                ax.errorbar(*zip(*pts), fmt="o-", ms=3, capsize=2, color=colors[rname], label=RESTORE_LABELS[rname])
            ax.set_xscale("symlog", linthresh=1)
            ax.set_title(f"{r['net']} ({pop})\n{MODE_LABELS[mode]}, input {inp}", fontsize=9)
            ax.set_xlabel("pause Δt (ms)")
        axes[i, 0].set_ylabel("mean divergence, 500 ms\n(± SEM over onsets)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"vs_pause_{tag}.png"), dpi=130)


def summarize(results):
    for r in results:
        print(f"\n== {r['net']}  control rates: " + ", ".join(f"{p} {v:.1f} Hz" for p, v in r["rates"].items()))
        pop = r["pops"][0] if r["net"] == "recurrent" else r["pops"][-1]
        print(f"   readout {pop}; 1 mV kick reference: {r['perturb'][pop]['mean']:.3f}; "
              f"spikes in flight at each onset: {r['in_flight']}")
        print(f"   {'mode':7s}{'input':10s}{'restore':8s}" + "".join(f"{p:>7g}" for p in PAUSES_MS) + "   (mean divergence)")
        for mode, inp, rname in itertools.product(MODES, INPUTS, RESTORE):
            vals = [row["res"][pop]["mean"] for row in r["rows"]
                    if (row["mode"], row["input"], row["restore"]) == (mode, inp, rname)]
            print(f"   {mode:7s}{inp:10s}{rname:8s}" + "".join(f"{v:7.3f}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-m", type=float, default=15.0)
    ap.add_argument("--tau-s", type=float, default=5.0)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    jobs = [(kind, args.tau_m, args.tau_s, t1) for kind in NETWORKS for t1 in ONSETS]
    with Pool() as pool:
        out = pool.map(_job, jobs)
    results = [aggregate([o for o in out if o["net"] == kind]) for kind in NETWORKS]
    with open(os.path.join(outdir, f"grid_{args.tag}.pkl"), "wb") as f:
        pickle.dump(dict(args=vars(args), results=results), f)
    summarize(results)
    plot(results, args.tag, outdir)


if __name__ == "__main__":
    main()
