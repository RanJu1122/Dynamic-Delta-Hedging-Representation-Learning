"""Continuity-aware, descriptive Step 5 statistics and presentation figures."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def block_sample_indices(segments, *, draws=1000, block=10):
    """Resample consecutive blocks, including short segments as random units.

    Resampling each segment separately would freeze singleton segments and
    understate uncertainty for sparse quality-selected samples.
    """
    segments = np.asarray(segments)
    blocks = []
    for segment in pd.unique(segments):
        positions = np.flatnonzero(segments == segment)
        blocks.extend(positions[start:start+block] for start in range(0, len(positions), block))
    if not blocks:
        return np.empty((draws, 0), dtype=int)
    rng = np.random.default_rng(20260807)
    sampled = np.empty((draws, len(segments)), dtype=int)
    for i in range(draws):
        offset = 0
        while offset < len(segments):
            piece = blocks[rng.integers(len(blocks))][:len(segments)-offset]
            sampled[i, offset:offset+len(piece)] = piece
            offset += len(piece)
    return sampled


def correlation_interval(x, y, segments, *, draws=1000):
    """Paired consecutive blocks of up to 10 rows, retaining missing pairs.

    This is an exploratory interval, not a multiple-testing-adjusted decision.
    """
    a, b = np.asarray(x, float), np.asarray(y, float)
    if np.isfinite(a+b).sum() < 20:
        return np.nan, np.nan
    idx = block_sample_indices(segments, draws=draws)
    aa, bb = a[idx], b[idx]
    good = np.isfinite(aa) & np.isfinite(bb)
    n = good.sum(axis=1)
    aa, bb = np.where(good, aa, 0), np.where(good, bb, 0)
    n = np.maximum(n, 1)
    ax, by = aa.sum(axis=1), bb.sum(axis=1)
    var_a = (aa*aa).sum(axis=1)-ax*ax/n
    var_b = (bb*bb).sum(axis=1)-by*by/n
    denominator = np.sqrt(np.maximum(0, var_a*var_b))
    corr = np.divide((aa*bb).sum(axis=1)-ax*by/n, denominator,
                     out=np.full(draws, np.nan), where=denominator > 1e-15)
    finite = corr[np.isfinite(corr)]
    return tuple(np.quantile(finite, [.025, .975])) if len(finite) else (np.nan, np.nan)


def period_statistics(panel, acf_function, correlation_function, spot_function):
    """Subperiods use chronological Step4 boundary, retaining all calendar rows."""
    boundary = panel.loc[panel["sample"].eq("train"), "observation_date"].max()
    acfs, correlations, regressions = [], [], []
    for name, mask in (("train", panel.observation_date <= boundary),
                       ("test", panel.observation_date > boundary)):
        part = panel.loc[mask].reset_index(drop=True)
        regression, residuals = spot_function(part)
        acfs.append(acf_function(residuals).assign(period=name))
        correlations.append(correlation_function(part).assign(period=name))
        regressions.append(regression.assign(period=name))
    return pd.concat(acfs, ignore_index=True), pd.concat(correlations, ignore_index=True), pd.concat(regressions, ignore_index=True)


def save_diagnostic_plots(result, target):
    """Plots follow the document's Step5 order; predictor results stay in CSV."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/svi-localvol-mpl")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from .factors import schema_for
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP") if f in fonts), "DejaVu Sans")
    schema = schema_for(result.factor_state_panel)
    panel, acf = result.factor_state_panel, result.factor_acf
    titles = ["ATM Beta", "形状因子1", "形状因子2"] if schema.method == "atm_anchored" else ["PC1", "PC2", "PC3"]
    files = []
    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, factor, title in zip(axes, schema.scores, titles):
            data = acf[acf.factor.eq(factor) & acf["transform"].eq("level")]
            ax.plot(data.lag, data.autocorrelation, "o-", color="#087e8b")
            ax.fill_between(data.lag, data.ci_low, data.ci_high, color="#087e8b", alpha=.15)
            ax.axhline(0, color="gray", lw=1)
            ax.set(title=title, xlabel="连续交易日 lag", ylabel="相关系数", xticks=range(1, 11), ylim=(-.45, .6))
            ax.grid(alpha=.2)
        fig.suptitle("自相关：点估计与探索性95%日期块Bootstrap区间")
        fig.tight_layout();fig.savefig(target/"acf_diagnostics_cn.png", dpi=160);plt.close(fig)
        files.append("acf_diagnostics_cn.png")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, factor, title in zip(axes, schema.scores, titles):
            data = panel[["dlogS", factor]].dropna()
            ax.scatter(data.dlogS*100, data[factor], s=9, alpha=.45)
            row = result.spot_regression[result.spot_regression.factor.eq(factor) & result.spot_regression["sample"].eq("all")].iloc[0]
            xx = np.linspace(data.dlogS.min(), data.dlogS.max(), 100)
            ax.plot(xx*100, row.intercept+row.slope*xx, color="#e47723")
            ax.set(title=f"{title}：同日线性R²={row.r_squared:.2%}", xlabel="当日现货对数收益（%）", ylabel="因子值")
            ax.grid(alpha=.2)
        fig.suptitle("同日关系：不能直接当作明日预测证据")
        fig.tight_layout();fig.savefig(target/"spot_relationship_cn.png", dpi=160);plt.close(fig)
        files.append("spot_relationship_cn.png")
        names = {"dlogS":"当日收益", "atm_iv_3m":"3M ATM IV", "atm_iv_change_1d":"IV单日变化",
                 "atm_iv_change_5d":"IV近5日变化", "smile_slope_3m":"3M smile斜率", "term_slope_1y_minus_3m":"1Y−3M IV",
                 "realized_vol_20d":"20日RV", "recent_return_5d":"近5日收益", "recent_return_20d":"近20日收益", "vol_of_vol_20d":"20日vol-of-vol"}
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, relationship, title in zip(axes, ("same_close", "close_t_to_close_t_plus_1"), ("同日：状态t ↔ 因子t", "滞后：状态t ↔ 因子t+1")):
            tab = result.state_correlations[result.state_correlations.relationship.eq(relationship)].pivot(index="feature", columns="factor", values="spearman").reindex(index=list(names), columns=list(schema.scores))
            im = ax.imshow(tab, vmin=-.5, vmax=.5, cmap="RdBu_r", aspect="auto")
            ax.set_yticks(range(len(tab)), [names[n] for n in tab.index]);ax.set_xticks(range(3), titles)
            for i in range(len(tab)):
                for j in range(3):ax.text(j, i, f"{tab.iloc[i,j]:.2f}", ha="center", va="center")
            ax.set_title(title)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, label="Spearman相关")
        fig.suptitle("状态相关：探索线索，尚非独立样本外预测验收")
        fig.subplots_adjust(left=.14, right=.88, wspace=.8, top=.88, bottom=.08)
        fig.savefig(target/"state_correlations_cn.png", dpi=160);plt.close(fig)
        files.append("state_correlations_cn.png")
    return files
