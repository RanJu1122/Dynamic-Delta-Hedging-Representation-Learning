"""Shared precompute must preserve independent-expiry MC and resumable shards."""
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import date_at_tau
from dynamic_alpha_hedging.hedging import MCMapStore
from dynamic_alpha_hedging.mc_library import MCLibrary
from dynamic_alpha_hedging.precompute import run_precompute
from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC
from tests.test_precompute import sample, fake_pricing
from tests.test_pricing_calibration import _surface


def test_shared_snapshots_equal_independent_paths_and_all_estimators():
    surface = _surface().repaired()
    expiries = [date_at_tau(surface, t) for t in (.25, .5, 1.)]
    strikes = np.array([.7, 1., 1.2]) * surface.ref_spot
    bump = .01 * surface.ref_spot
    for antithetic, substeps in ((True, 2), (False, 1)):
        settings = dict(n_paths=400, seed=37, antithetic=antithetic, n_substeps=substeps)
        for alpha in (0., .5, 1., 1.5, 2.):
            def grids(expiry):
                return [LocalVolGrid.build(surface, expiry, n_ratio=81,
                    spot_adj=np.log(1+sign*.01), alpha=alpha) for sign in (0, 1, -1)]

            base, up, down = grids(expiries[-1])
            mc = LocalVolMC(surface, base, **settings)
            shared = mc.bumped_implied_vol_diagnostics_many(strikes, expiries, up, down, bump)
            for expiry in expiries:
                short_base, short_up, short_down = grids(expiry)
                independent = LocalVolMC(surface, short_base, **settings)
                expected = independent.bumped_implied_vol_diagnostics(
                    strikes, expiry, short_up, short_down, bump)
                pd.testing.assert_frame_equal(shared[expiry], expected, check_exact=True)
                prefix = up.prefix(expiry)
                np.testing.assert_array_equal(prefix.sigma, short_up.sigma)
                assert (prefix.n_undefined, prefix.n_clipped, prefix.n_zero) == (
                    short_up.n_undefined, short_up.n_clipped, short_up.n_zero)


def test_grid_prefix_preserves_time_dependent_repairs():
    surface = _surface().repaired()
    short, long = [date_at_tau(surface, t) for t in (.25, .5)]

    def local_vol(T, K):
        values = np.full_like(K, .2)
        values[0] = np.nan
        if T > short:
            values[1:4] = np.nan
            values[-2:] = 7.
        return values

    full = LocalVolGrid.build(surface, long, n_ratio=21, local_vol_fn=local_vol)
    prefix = full.prefix(short)
    expected = LocalVolGrid.build(surface, short, n_ratio=21, local_vol_fn=local_vol)
    np.testing.assert_array_equal(prefix.sigma, expected.sigma)
    assert prefix.n_undefined == expected.n_undefined < full.n_undefined
    assert prefix.n_clipped == expected.n_clipped == 0 < full.n_clipped


def test_mapping_shares_path_advances_and_preserves_all_columns():
    surface = _surface()
    config = replace(DynamicAlphaConfig(), step3_n_paths=400, step3_n_ratio=81)
    tenors = (.25, .5, 1.)
    with TemporaryDirectory() as directory:
        store = MCMapStore(config, tenors, (.8, 1., 1.2), directory)
        original = LocalVolMC.terminal_spots
        with patch.object(LocalVolMC, "terminal_spots", autospec=True, side_effect=original) as calls:
            together = store.measure_many(surface, tenors)
            assert calls.call_count == 2 * len(config.step3_alphas)
        for tenor in tenors:
            pd.testing.assert_frame_equal(together[tenor], store.measure(surface, tenor), check_exact=True)


def test_partial_date_resume_batches_only_missing_tenors():
    with sample() as (root, config, history), fake_pricing() as calls, patch(
            "dynamic_alpha_hedging.precompute._plots", return_value=[]):
        tenors = (.25, .375, .5)
        out = root / "library"
        library = MCLibrary(out, config, tenors, (.9, 1.), writable=True)
        first = history[history.dates[0]]
        library.ensure(first, .5)
        calls.reset_mock()
        report = run_precompute(config, outdir=out, tenors=tenors, levels=(.9, 1.),
                                progress=lambda _: None)
        assert report["reused_jobs"] == 1 and report["computed_jobs"] == 8
        assert calls.call_count == 3
        assert tuple(calls.call_args_list[0].args[2]) == (.25, .375)
        before = {p.name: p.read_bytes() for p in (out / "shards").glob("*.csv")}
        calls.reset_mock()
        again = run_precompute(config, outdir=out, tenors=tenors, levels=(.9, 1.),
                               progress=lambda _: None)
        assert again["reused_jobs"] == 9 and calls.call_count == 0
        assert before == {p.name: p.read_bytes() for p in (out / "shards").glob("*.csv")}
