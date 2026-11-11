#!/usr/bin/env python3
"""Regenera las figuras de la tesis desde el panel conjunto baseline/R1.1/R3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DATA = ROOT / "data_vx/baseline_r11_r3_common_20260929"
ROUTES = ROOT / "RUTAS_SAC"
METHODS = ("baseline", "r1_1", "r3")
LABELS = ("Baseline", "R1.1", "R3")
COLORS = ("#1f4e78", "#139b98", "#d97825")
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})


def load_rows():
    payload = json.loads((DATA / "summary.json").read_text())
    if not payload["complete"] or len(payload["episodes"]) != 600:
        raise RuntimeError("La extracción conjunta no está completa")
    rows = {(r["case_id"], r["method"]): r for r in payload["episodes"]}
    cases = sorted({key for key, _ in rows})
    common = [case for case in cases if all(rows[case, m]["success_place"] for m in METHODS)]
    if len(cases) != 200 or len(common) != 188:
        raise RuntimeError("La cohorte de la tesis no coincide")
    return rows, common


def trace(method, case):
    return np.load(DATA / "traces" / method / f"{case}.npz")


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def success(rows, directory):
    fig, ax = plt.subplots(figsize=(8, 3.7))
    x = np.arange(4)
    width = .24
    for i, (m, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        vals = [sum(rows[case, m]["success_place"] for case in
                    {key for key, _ in rows if rows[key, m]["map_id"] == map_id})
                for map_id in range(1, 5)]
        ax.bar(x + (i - 1) * width, vals, width, color=color, label=label)
        for j, value in enumerate(vals):
            ax.text(x[j] + (i - 1) * width, value + .5, str(value),
                    ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=[f"Mapa {i}" for i in range(1, 5)],
           ylabel="Colocaciones / 50 intentos", ylim=(0, 56))
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, directory, "exito")


def phases(rows, common, directory):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.arange(3)
    width = .25
    for i, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        times = [np.mean([rows[case, method]["phase_time_s"][p]
                          for case in common]) for p in range(3)]
        work = [np.mean([rows[case, method]["phase_energy_arm_J"][p] +
                         rows[case, method]["phase_energy_base_J"][p]
                         for case in common]) for p in range(3)]
        axes[0].bar(x + (i - 1) * width, times, width, color=color, label=label)
        axes[1].bar(x + (i - 1) * width, work, width, color=color, label=label)
    for ax, ylabel in zip(axes, ("Tiempo medio (s)", "Trabajo medio, 11 actuadores (J)")):
        ax.set(xticks=x, xticklabels=("P0", "P1", "P2"), ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, directory, "fases")


def tracking(rows, directory, map_id, case):
    if not all(rows[case, method]["success_place"] for method in METHODS):
        raise RuntimeError(f"El caso {case} no es éxito común")
    route = json.loads((ROUTES / f"map_{map_id}.json").read_text())
    path = np.asarray(route["path"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(path[:, 0], path[:, 1], "--", color="#444444", lw=1.8,
            label="Ruta RRT*")
    for method, label, color in zip(METHODS, LABELS, COLORS):
        with trace(method, case) as tr:
            gate = tr["phase"] == 1
            ax.plot(tr["x"][gate], tr["y"][gate], color=color, lw=1.35, label=label)
    ax.set(xlabel="x (m)", ylabel="y (m)",
           title=f"Mapa {map_id}, masa {rows[case, 'r3']['mass_kg']:.2f} kg, "
                 f"semilla {rows[case, 'r3']['case_seed']}")
    ax.axis("equal")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    save(fig, directory, f"seguimiento_mapa{map_id}")


def power(rows, directory, case):
    if not all(rows[case, method]["success_place"] for method in METHODS):
        raise RuntimeError(f"El caso {case} no es éxito común")
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.5), sharex=True)
    for method, label, color in zip(METHODS, LABELS, COLORS):
        with trace(method, case) as tr:
            t = tr["t_s"]
            increment = tr["energy_arm_J"] + tr["energy_base_J"]
            dt = np.diff(np.r_[0., t])
            power_w = increment / dt
            smooth = np.convolve(power_w, np.ones(20) / 20, mode="same")
            axes[0].plot(t, smooth, color=color, lw=1, label=label)
            axes[1].plot(t, np.cumsum(increment), color=color, lw=1.3,
                         label=label)
    axes[0].set_ylabel("Potencia mecánica (W)")
    axes[1].set(xlabel="Tiempo hasta colocación (s)", ylabel="Trabajo acumulado (J)")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3)
    fig.tight_layout()
    save(fig, directory, "potencia")


def lookahead(rows, common, directory):
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    data = [[rows[case, "r3"]["lookahead_delta_rms_m"]
             for case in common if rows[case, "r3"]["map_id"] == map_id]
            for map_id in range(1, 5)]
    if any(any(value is None for value in group) for group in data):
        raise RuntimeError("Faltan mediciones del lookahead")
    box = ax.boxplot(data, tick_labels=[f"Mapa {i}" for i in range(1, 5)],
                     patch_artist=True, showfliers=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#d97825")
        patch.set_alpha(.65)
    ax.axhline(0., color="#444444", lw=.8)
    ax.set(ylabel="Desviación RMS del lookahead nominal (m)",
           title="R3: 188 colocaciones exitosas compartidas")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, directory, "lookahead")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, common = load_rows()
    success(rows, args.output)
    phases(rows, common, args.output)
    tracking(rows, args.output, 1, "m1_mass2_rep01")
    tracking(rows, args.output, 2, "m2_mass2_rep00")
    power(rows, args.output, "m2_mass2_rep00")
    lookahead(rows, common, args.output)
    print(f"FIGURES_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
