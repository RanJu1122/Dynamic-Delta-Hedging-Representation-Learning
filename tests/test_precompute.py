"""Precompute integration tests with deterministic pricing doubles; never run MC."""
from contextlib import contextmanager
from dataclasses import replace
import datetime as dt
import io
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from pricing_svi_localvol_calibration.config import TEST_VOL_PARAMS
from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import load_surface_history, EXCLUDED_OBSERVATIONS
from dynamic_alpha_hedging.hedging import MCMapStore
from dynamic_alpha_hedging.mc_library import MCLibrary, engine_signature
from dynamic_alpha_hedging.precompute import run_precompute, middle_dates, _plots, _writer_lock
from dynamic_alpha_hedging.step03 import _anchor_at_sticky_strike, _cell_quality


@contextmanager
def raises(kind, text=""):
    try:
        yield
    except kind as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {kind.__name__}")


@contextmanager
def sample():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data.pkl"
        dates = [dt.date(2026, 8, d) for d in (3, 4, 5)]
        with data.open("wb") as handle:
            pickle.dump({d: TEST_VOL_PARAMS for d in dates}, handle)
        config = replace(DynamicAlphaConfig(), data_path=data)
        history = load_surface_history(data, config.market_conventions,
                                       duplicate_vol_date_policy="first")
        yield root, config, history


def measured(store, surface, tenor):
    rows = []
    for level in store.levels:
        for alpha in store.config.step3_alphas:
            rows.append(dict(calibration_date=surface.market.pricing_date,
                tenor=tenor, level=level, alpha=alpha, strike=level*surface.ref_spot,
                beta_model=.4*(1-alpha)+.02, beta_model_stderr=.001,
                call_delta=.5+.1*(alpha-1), call_delta_stderr=.001,
                grid_undefined_fraction=.2, grid_clipped_fraction=0.,
                price_clipped_for_inversion=False))
    curve = _anchor_at_sticky_strike(pd.DataFrame(rows))
    quality = _cell_quality(curve, store.config)
    return curve.merge(quality[["tenor", "level", "quality_pass", "inverse_available",
                                 "quality_failures"]], on=["tenor", "level"])


@contextmanager
def fake_pricing():
    signature = engine_signature()
    with patch("dynamic_alpha_hedging.mc_library.engine_signature", return_value=signature), \
         patch.object(MCMapStore, "measure", autospec=True, side_effect=measured) as calls:
        yield calls


def run(config, out, **kwargs):
    return run_precompute(config, outdir=out, tenors=(.25,), levels=(.9, 1.),
                          progress=lambda _: None, **kwargs)


def test_library_resume_readonly_subset_and_model_reuse():
    with sample() as (root, config, history), fake_pricing() as calls, patch(
            "dynamic_alpha_hedging.precompute._plots", return_value=[]):
        out = root / "library"
        result = run(config, out)
        assert result["computed_jobs"] == calls.call_count == 3
        assert result["quality_pass_cells"] == 0  # Audit failures do not hide results.
        again = run(config, out)
        assert again["reused_jobs"] == 3 and again["computed_jobs"] == 0
        assert calls.call_count == 3
        reader = MCLibrary(out, replace(config, beta_min_abs_dlogS=.01), (.25,), (1.,))
        curve = reader.get(history[history.dates[0]])
        assert set(curve.level) == {1.} and len(curve) == 5
        assert np.allclose(curve.query("alpha == 1").beta_converter, 0.)
        assert np.allclose(curve.query("alpha == 1").beta_model, .02)
        with raises(RuntimeError, "read-only"):
            reader.ensure(history[history.dates[0]], .25)
        with raises(ValueError, "no extrapolation"):
            MCLibrary(out, config, (.5,), (1.,))
        with raises(ValueError, "numerical mismatch"):
            MCLibrary(out, replace(config, step3_n_paths=2000), (.25,), (1.,))
        stricter = replace(config, step3_alpha_one_abs_tolerance=.001)
        audited = MCLibrary(out, stricter, (.25,), (1.,)).get(history[history.dates[0]])
        assert audited.quality_failures.str.contains("alpha_one_abs_pass").all()
        # A missing day must fail without falling back to the pricer.
        index = json.loads((out / "index.json").read_text())
        del index[next(iter(index))]
        (out / "index.json").write_text(json.dumps(index))
        reader = MCLibrary(out, config, (.25,), (1.,))
        with raises(FileNotFoundError, "missing MC shard"):
            reader.get(history[history.dates[0]])
        assert calls.call_count == 3


def test_library_rejects_stale_quotes_and_corrupted_shards():
    with sample() as (root, config, history), fake_pricing(), patch(
            "dynamic_alpha_hedging.precompute._plots", return_value=[]):
        out = root / "library"
        run(config, out)
        reader = MCLibrary(out, config, (.25,), (1.,))
        surface = history[history.dates[0]]
        with patch.object(surface, "market", replace(surface.market, rate=.08)):
            with raises(ValueError, "changed market/quotes"):
                reader.get(surface)
        shard = sorted((out / "shards").glob("*.csv"))[0]
        shard.write_text(shard.read_text()+"\n")
        with raises(ValueError, "checksum mismatch"):
            reader.get(surface)


def test_plan_only_and_invalid_axes_preserve_existing_reports():
    with sample() as (root, config, history), fake_pricing() as calls, patch(
            "dynamic_alpha_hedging.precompute._plots", return_value=[]):
        planned = root / "planned"
        result = run(config, planned, plan_only=True)
        assert result["status"] == "planned" and not result["mc_run"]
        assert not (planned / "library.json").exists() and calls.call_count == 0
        out = root / "library"
        run(config, out)
        before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
        with raises(ValueError, "different axes"):
            run_precompute(config, outdir=out, tenors=(.25, .5), levels=(.9, 1.))
        assert all((out / name).read_bytes() == value for name, value in before.items())
        with raises(ValueError, "no supported"):
            run_precompute(config, outdir=root / "empty", tenors=(10.,), levels=(1.,))
        assert not (root / "empty" / "manifest.json").exists()


def test_failed_precompute_records_progress_and_resumes():
    with sample() as (root, config, history), fake_pricing() as calls, patch(
            "dynamic_alpha_hedging.precompute._plots", return_value=[]):
        def fail_second(store, surface, tenor):
            if surface.market.pricing_date == history.dates[1]:
                raise RuntimeError("pricing failed for test")
            return measured(store, surface, tenor)
        calls.side_effect = fail_second
        out = root / "library"
        with raises(RuntimeError, "pricing failed"):
            run(config, out)
        manifest = json.loads((out / "manifest.json").read_text())["validation"]
        assert manifest["status"] == "failed" and manifest["completed_jobs"] == 1
        assert manifest["failed_job"]["date"] == str(history.dates[1])
        calls.side_effect = measured
        result = run(config, out)
        assert result["status"] == "complete"
        assert result["reused_jobs"] == 1 and result["computed_jobs"] == 2


def test_writer_lock_rejects_overlapping_precomputes():
    with TemporaryDirectory() as directory:
        with _writer_lock(Path(directory)):
            with raises(RuntimeError, "another precompute"):
                with _writer_lock(Path(directory)):
                    pass


def test_middle_dates_plots_and_raw_anchor_audit():
    assert middle_dates(list(range(9)), 3) == [3, 4, 5]
    assert middle_dates(list(range(9)), 5) == [2, 3, 4, 5, 6]
    with sample() as (root, config, history):
        from types import SimpleNamespace
        surface = history[history.dates[0]]
        store = SimpleNamespace(config=config, levels=(.9, 1.))
        frame = pd.concat([measured(store, surface, t) for t in (.25, .5)])
        paths = _plots(frame, root / "plots", history.dates[0])
        assert len(paths) == 3 and all(Path(p).stat().st_size > 0 for p in paths)
        assert (root / "plots" / f"{history.dates[0]}_beta_alpha.csv").exists()
        quality = _cell_quality(frame[frame.tenor == .25], config)
        assert np.allclose(quality.beta_alpha_1, .02)


def test_cli_precompute_dispatch_and_library_conflicts():
    from contextlib import redirect_stdout, redirect_stderr
    from dynamic_alpha_hedging.cli import cli
    with patch("sys.argv", ["dynamic-alpha", "precompute", "--plan-only", "--plot-dates", "5"]), \
         patch("dynamic_alpha_hedging.precompute.run_precompute", return_value={
             "status": "planned", "excluded_observation_dates": []}) as command, \
         redirect_stdout(io.StringIO()):
        cli()
        config = command.call_args.args[0]
        options = command.call_args.kwargs
        assert config.step3_n_paths == 40000 and config.step3_n_ratio == 801
        assert config.step3_alphas == (0., .5, 1., 1.5, 2.)
        assert options["plot_count"] == 5 and options["plan_only"]
        assert set((.25, .5, 1.)).issubset(options["tenors"])
        assert set((.9, 1.)).issubset(options["levels"])
    for extra in (["--fast"], ["--paths", "500"], ["--mc-cache", "cache"]):
        with patch("sys.argv", ["dynamic-alpha", "step7-legacy", "--mc-library", "lib", *extra]), \
             redirect_stderr(io.StringIO()), raises(SystemExit):
            cli()


def test_restored_equal_spot_dates_are_kept_without_modifying_raw_file():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "data.pkl"
        dates = [dt.date(2026, 3, 30), dt.date(2026, 3, 31), dt.date(2026, 4, 1),
                 dt.date(2026, 4, 2), dt.date(2026, 4, 6)]
        with path.open("wb") as handle:
            pickle.dump({date: TEST_VOL_PARAMS for date in dates}, handle)
        original = path.read_bytes()
        history = load_surface_history(path, DynamicAlphaConfig().market_conventions,
                                       duplicate_vol_date_policy="first")
        assert history.dates == dates
        assert history.skipped.empty and not EXCLUDED_OBSERVATIONS
        assert history.spots.nunique() == 1
        assert path.read_bytes() == original


def test_cli_step7_forwards_library_model_and_uses_frozen_numerics():
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from dynamic_alpha_hedging.cli import cli
    date = __import__("datetime").date(2026, 1, 5)
    frozen = replace(DynamicAlphaConfig(), step3_n_paths=1234, step3_n_ratio=55)
    inputs = SimpleNamespace(config=frozen, forecaster=SimpleNamespace(train_end=date))
    summary = pd.DataFrame(dict(strategy=["fixed_1"],std_error=[0.],
                                std_improvement_vs_best_fixed=[0.]))
    args = ["dynamic-alpha", "step7-legacy", "--mc-library", "library", "--model", "ridge",
            "--model-params", '{"alpha":2}', "--book", "near_atm", "--factors", "2"]
    with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
         patch("dynamic_alpha_hedging.step07.prepare_step7", return_value=inputs) as prepare, \
         patch("dynamic_alpha_hedging.step07._intervals", return_value=([(date,date)],[])), \
         patch("dynamic_alpha_hedging.step07.run_step7",return_value={"summary":summary}) as backtest:
        cli()
        settings = prepare.call_args.args[1]
        assert settings.mc_library == Path("library") and settings.model == "ridge"
        assert settings.model_params == {"alpha":2} and settings.factor_count == 2
        assert backtest.call_args.args[0] is inputs
        assert backtest.call_args.kwargs["cache_dir"] is None
