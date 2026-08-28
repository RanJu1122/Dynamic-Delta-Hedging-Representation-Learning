"""Optional plots for the pricing SVI/local-vol calibration workflow.

Kept out of the calculation path -- import failures here must never break
`solution.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _to_grid(frame: pd.DataFrame):
    t = np.arange(len(frame.index))
    k = np.asarray(frame.columns, dtype=float)
    return np.meshgrid(k, t), frame.to_numpy()


def plot_surfaces(implied: pd.DataFrame, local: pd.DataFrame,
                  local_repaired: pd.DataFrame | None = None,
                  path: str | Path = "output/surfaces.png",
                  title: str = "SVI implied vs Dupire local volatility"):
    """3D surfaces.  Returns the output path, or None if plotting fails.

    The holes in the raw local volatility panel are the calendar spread
    arbitrage in the quotes, not a plotting artefact.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    except Exception:                                       # noqa: BLE001
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = [(implied, "implied volatility"),
              (local, "local volatility (as quoted)")]
    if local_repaired is not None:
        panels.append((local_repaired, "local volatility (calendar repaired)"))

    fig = plt.figure(figsize=(6.4 * len(panels), 5.2))
    for i, (frame, name) in enumerate(panels, start=1):
        (kk, tt), zz = _to_grid(frame)
        ax = fig.add_subplot(1, len(panels), i, projection="3d")
        ax.plot_surface(kk, tt, zz, cmap="viridis", linewidth=0,
                        antialiased=True, rstride=4, cstride=1)
        ax.set_xlabel("strike level  K / S0")
        ax.set_ylabel("business days from pricing date")
        ax.set_zlabel("volatility")
        ax.set_title(name, fontsize=10)
        ax.view_init(elev=24, azim=-128)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_smile_slices(surface, dates, levels, path="output/smiles.png"):
    """Implied and local volatility smiles for a handful of expiries."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    K = np.asarray(levels, dtype=float) * surface.market.spot

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for T in dates:
        if surface.tau_vol(T) <= 0:
            continue
        axes[0].plot(levels, surface.implied_vol(T, K), label=str(T))
        axes[1].plot(levels, surface.local_vol(T, K), label=str(T))
    for ax, name in zip(axes, ("implied volatility", "local volatility")):
        ax.set_xlabel("strike level  K / S0")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_step4_pricing_errors(frame: pd.DataFrame,
                              path="output/step4_pricing_errors.png"):
    """PV levels and relative errors across the Step 4 strike grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = frame["level"].to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    axes[0].plot(x, frame["bs_pv"], marker="o", label="BS from implied vol")
    axes[0].plot(x, frame["mc_implied_pv"], marker="s", ms=4,
                 label="MC with constant implied vol")
    axes[0].errorbar(x, frame["mc_local_pv"],
                     yerr=frame["mc_local_stderr"], marker="^", ms=4,
                     capsize=3, label="MC with local vol")
    axes[0].set_ylabel("call PV")
    axes[0].set_title("Step 4 vanilla price comparison")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].plot(x, frame["mc_implied_rel_error_pct"], marker="s", ms=4,
                 label="implied-vol MC minus BS")
    axes[1].errorbar(x, frame["mc_local_rel_error_pct"],
                     yerr=100.0 * frame["mc_local_stderr"] / frame["bs_pv"],
                     marker="^", ms=4, capsize=3,
                     label="local-vol MC minus BS")
    axes[1].axhline(1.0, color="grey", ls="--", lw=0.8)
    axes[1].axhline(-1.0, color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("strike level  K / S0")
    axes[1].set_ylabel("relative error (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_terminal_variance_bins(
        frame: pd.DataFrame,
        path="output/step4_terminal_variance_bins.png"):
    """Conditional integrated local variance and implied total variance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = frame["mean_terminal_ratio"].to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 0.75]})
    axes[0].errorbar(
        x, frame["mean_path_integrated_variance"],
        yerr=frame["path_integrated_variance_stderr"], marker="o", capsize=3,
        label=r"mean path $\sum \sigma_{loc}^2\,\Delta\tau_{vol}$")
    axes[0].plot(
        x, frame["mean_implied_total_variance_at_terminal_spot"],
        marker="s", label="mean implied total variance at path terminal spots")
    axes[0].set_ylabel("total variance")
    axes[0].set_title("Path variance conditional on terminal spot")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    widths = np.maximum(
        0.015,
        np.minimum(frame["bin_right_ratio"].replace(np.inf, x.max() + 0.05)
                   - frame["bin_left_ratio"], 0.15).to_numpy(float) * 0.75)
    axes[1].bar(x, 100.0 * frame["path_probability"], width=widths,
                alpha=0.65)
    axes[1].set_xlabel("mean terminal ratio  E[S_T/S0 | bin]")
    axes[1].set_ylabel("path probability (%)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
