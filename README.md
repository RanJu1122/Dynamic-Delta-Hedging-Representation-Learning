# Pricing Practice 3 — SVI-based local volatility calibration

Calibrates a Dupire local volatility surface from SVI-JW desk quotes, validates it by
Monte Carlo, and extends into the delta hedging backtest the task leads towards.

Run every command from this directory (`SVI_volatility_surface/`):

```bash
python3 -m pip install -e ".[plots,dev]"
```

Then run:

```bash
python3 solution.py                    # full run: Steps 1-6, writes ./output
python3 solution.py --fast             # 20k Monte Carlo paths instead of 100k
python3 -m svi_localvol                 # equivalent package entry point
python3 -m svi_localvol.tests           # regression checks
```

Before starting the separate dynamic-alpha study, run its read-only preflight:

```bash
python3 -m svi_localvol.research
```

That command audits `svi_data.pkl`, the observation-date calendar, SVI repairs,
the alpha convention and constant-tenor coverage.  It deliberately does not
create an IV panel, estimate beta or run a hedge.  See
[`DYNAMIC_ALPHA_READINESS.md`](DYNAMIC_ALPHA_READINESS.md) for the boundary
between the completed pricing task and the new research.

After installing the project, the console command is also available:

```bash
svi-localvol --fast
```

Requires `numpy`, `scipy`, `pandas`; `matplotlib` only for the optional plots.

---

## Layout

The split is by *what changes together*, not by task step, so the hedging work
reuses the calibration layer untouched.

| File under `svi_localvol/` | Responsibility | Key entry points |
|---|---|---|
| `conventions.py` | the two independent time axes and schedule generation | `dt_vol`, `dt_r`, `nb_biz_days`, `gen_schedule` |
| `params.py` | input containers and the shipped test case | `MarketData`, `SVIJWQuote`, `VolQuoteSet`, `TEST_VOL_PARAMS` |
| `svi.py` | one raw SVI slice: algebra, conversion, per-slice arbitrage | `SVIRaw`, `jw_to_raw`, `raw_to_jw`, `g_function`, `crossedness` |
| `surface.py` | term structure, the two required functions, diagnostics | `VolSurface.implied_vol`, `.local_vol`, `.arbitrage_report`, `.repaired` |
| `dataset.py` | daily quote history and explicit IV-panel axes | `load_surface_history`, `implied_vol_panel` |
| `research.py` | dynamic-alpha conventions and pre-Step-1 audit | `DynamicAlphaConfig`, `run_preflight` |
| `blackscholes.py` | cost-of-carry BS in **total variance** form | `bs_price_w`, `bs_delta_w`, `bs_vega_sqrt_w`, `bs_vanna_w`, `bs_volga_w` |
| `montecarlo.py` | pre-tabulated local vol grid + path engine (Step 4) | `LocalVolGrid.build`, `LocalVolMC.price_european`, `.step4_diagnostics` |
| `deltas.py` | legacy analytic-R delta plus alpha MC diagnostics | `smile_greeks`, `delta_range`, `alpha_mc_delta_curve` |
| `backtest.py` | legacy analytic-R hedge self-check | `MarketState`, `HedgeBacktester`, `roll_surface` |
| `plots.py` | optional surfaces and diagnostic charts | `plot_surfaces`, `plot_step4_pricing_errors`, `plot_terminal_variance_bins` |
| `solution.py` | fixed-signature API + `main()` | `param_convert`, `gen_schedule`, `ImpliedVol`, `localvol` |
| `tests.py` | regression suite | `python3 -m svi_localvol.tests` |

Two design decisions worth knowing before reading the code:

**Black-Scholes is parameterised by total variance `w`, not by `(sigma, T)`.**
The caller supplies the forward (built on `dt_r`), `w` (built on `dt_vol`) and the
discount factor (built on `dt_r`), so the pricing formulas never have to choose a
time convention. This removes the single most common bug in this exercise.

**The local vol grid is pre-tabulated, not called inside the time loop.**
`LocalVolGrid` runs the SVI interpolation once per (date, ratio) and serves
values interpolated in *both* state and time during simulation.

**The Monte Carlo carries two step sizes, not one.** `local_vol` returns
`sqrt((dw/dtau) / D)` with `tau = dt_vol`, so its units are variance per
*business* year. The scheme is consistent only if the accumulated variance
reproduces `w` exactly, which requires integrating the diffusion against
`d(tau_vol)` while the drift and the discount factor stay on Act/365:

```
S_{i+1} = S_i * exp( b*dt_r  -  0.5*sigma^2*dt_v  +  sigma*sqrt(dt_v)*Z )
             \_______/       \____________________________________/
              Act/365                  Business/260
```

The Ito correction belongs with `dt_v`; it is a by-product of the diffusion.
Step 4(2) of the task statement uses a single Act/365 `dt` for both, which
contradicts its own Step 3 definition of `dw/dT = dw/dtau`. See *Known
approximations*.

### Step 4 diagnostic outputs

The full run writes three machine-readable diagnostic tables and two matching figures:

| Output | Contents |
|---|---|
| `output/step4_pricing_errors.csv` | strike-by-strike BS PV, constant-implied-vol MC, local-vol MC, absolute/relative errors, antithetic-pair standard errors, control-variate statistics and paired differences |
| `output/step4_delta_comparison.csv` | strike-by-strike analytic BS delta, BS bump delta, constant-implied-vol MC delta and mentor-scope local-vol MC delta, including standard errors and absolute errors |
| `output/step4_terminal_variance_bins.csv` | terminal `S_T/S0` bins, path counts/probabilities, conditional path-integrated variance distribution and the implied total variance evaluated at each path's terminal spot |
| `output/step4_pricing_errors.png` | PV curves plus relative pricing errors and error bars |
| `output/step4_terminal_variance_bins.png` | conditional path-integrated variance versus implied total variance, plus terminal-bin probabilities |

For each simulated local-vol path, the variance statistic is

```
I_T(path) = sum_i sigma_loc(t_i, S_i)^2 * delta_tau_vol_i.
```

This is the discrete path-integrated variance (the continuous diffusion's
quadratic variation). It is not `sum(daily return^2)`, and it is deliberately
integrated on the Business/260 volatility clock.

The vanilla-price comparison and the terminal-bin comparison answer different
questions. Calibration requires the local-vol terminal distribution to
reproduce vanilla option prices for all strikes. It does **not** require
`E[I_T | S_T in bin]` to equal the implied total variance associated with that
bin: implied variance is a Black-Scholes-equivalent summary of an option price,
not a conditional forecast of path variance. The second CSV is therefore a
joint-distribution diagnostic, rather than another calibration equality.

---

## What the run finds

### The SVI-JW `beta` formula uses the volatility-skew convention

The quoted `Skew` is `d(sigma)/dlogm`, while Gatheral–Jacquier's `psi` is the
corresponding total-variance quantity.  Since `psi = Skew * sqrt(tau)`, the task
formula `jw_beta = rho - 2*Skew*sqrt(omg*tau)/b` is correct.  The internal
variables are named `jw_beta` and `jw_shape` in the code so they cannot be
mistaken for the dynamic study's empirical `beta = -dIV/dlogS` or stickiness
`alpha`.  `test_quoted_skew_is_the_volatility_skew` pins down the convention by
differentiating the calibrated smile.

### The test quotes contain calendar spread arbitrage

Butterfly is clean on every slice (`min g` from 0.0071 to 0.19). Calendar is not:
`Callwing` decays too fast with maturity and then jumps from 0.6038 to 1.2155 at the
last expiry, so total variance *decreases* in `T` on the call wing.

Measured on `y = ln(K/F)` in `[-1.5, +1.5]`:

| adjacent slices | crossedness | affected `y` | note |
|---|---|---|---|
| 2026-09-18 → 2026-10-16 | 6.9e-05 | 0.075 … 0.135 | inside the traded band |
| 2026-10-16 → 2026-11-20 | 5.9e-04 | 0.120 … 1.500 | truncated at the grid edge |
| 2026-11-20 → 2026-12-18 | 1.1e-02 | 0.150 … 1.500 | **unbounded call wing** |
| 2027-01-15 → 2027-02-19 | 2.3e-04 | 0.245 … 0.565 | |
| 2027-02-19 → 2027-03-19 | 2.1e-04 | 0.295 … 0.710 | |

`crossedness` and `y_bad_*` are grid quantities and mean nothing without the
window, so `arbitrage_report` also reports it. Two columns say what happens
*outside* the window: `w` is asymptotically linear with slope `b(1 ± rho)`, so
when the near slice has the steeper wing slope the gap grows without bound and
no finite grid can measure it — that is the `unbounded_call_wing` row above,
whose "crossedness" reads 3.8e-03 on `[-1, 0.6]`, 1.1e-02 on `[-1.5, 1.5]` and
2.2e-02 on `[-1, 3]`. Only the traded band is economically meaningful.

On the requested strike grid this is 205 of 1980 points (10.4%), all at
`level >= 1.10`; on the finer Monte Carlo grid it is 20.5%. `local_vol` returns
`NaN` there on purpose and `step3_arbitrage_flags.csv` labels every point
`ok` / `calendar` / `butterfly` / `both`.

`VolSurface(..., calendar_repair=True)` makes `w` non-decreasing in `tau` by a
running maximum over slices, which removes every `NaN`. Steps 1–3 run on the raw
quotes so the arbitrage gets reported; Steps 4–6 run on the repaired surface,
because a model needs a surface that admits one.

The repair is not free, and Step 4 prints what it costs:

* At the maturity used for the Step 4 check it changes **nothing** — repaired and
  raw implied vols agree to 0.0 bp at every level from 0.80 to 1.30, so the
  Black-Scholes leg is untouched. On intermediate dates it lifts the call wing by
  up to **1.03 vol points** (level 1.30, 2026-12-18) and moves 22.5% of the Step 2
  grid by more than 1 bp.
* Flattening `w` in `tau` means `dw/dtau = 0`, so `sigma_loc` is **exactly zero**
  on 37.5% of the simulation grid (110 of 180 dates at level 1.30). The repaired
  model carries no variance there. It is self-consistent — the Monte Carlo is
  compared against implied vols read off the *same* repaired surface — but it is
  a strong intervention, not a cosmetic one.

### The Dupire denominator *is* the paper's `g`

`D = 1 - (y/w)w' + ¼(-¼ - 1/w + y²/w²)w'² + ½w''` is algebraically identical to
equation (2.1) of Gatheral–Jacquier. So

```
sigma_loc² = (dw/dT) / g(y)
```

numerator ≥ 0 ⟺ no calendar arbitrage, `g > 0` ⟺ no butterfly arbitrage. The local
volatility exists exactly when the implied surface is free of static arbitrage.
Checked to 1e-12 in `test_dupire_denominator_equals_gatheral_g`.

### Step 4: PV and mentor-scope delta checks

The PV check uses the `spot_adj=0` repaired local-vol grid and compares its MC
vanilla prices with prices read from the same implied-volatility surface.

The formal delta check no longer bumps the initial spot against one frozen
local-vol table. For a 1% bump it builds two additional grids through the
delivered function:

```text
grid_up   <- localvol(T, K, log((S0+h)/ref_spot))
grid_down <- localvol(T, K, log((S0-h)/ref_spot))
```

The up/down paths start at `S0+h` and `S0-h` and reuse the same normal draws.
`step4_delta_comparison.csv` reports, for every strike, the analytic BS delta,
the finite-bump BS delta, the constant-implied-vol MC delta and the bumped
local-vol MC delta. The local leg uses the implied leg as a control variate.

At 1e5 paths the ATM standard error is about 0.4% of PV, so the relative-error
column should be read together with its standard error and z-score. Across the
0.80–1.30 strike grid, all local-vol-minus-BS errors are within 1.54 standard
errors and all paired raw-local-minus-implied-MC differences are within 1.73
standard errors. Relative errors become large for far-OTM calls because the PV
denominator is very small; this run does not have enough tail paths to estimate
those relative errors to 1% precision. The deterministic checks live in
`tests.py`:

* `test_scheme_accumulates_exactly_the_quoted_total_variance` — with a constant
  local vol and antithetic draws, `mean(log S_T)` equals
  `b*tau_r − 0.5*sigma²*tau_vol` to machine precision. Any error in either step
  size, in the Ito term's clock, or in the horizon shows up with no noise to hide
  behind.
* `test_mc_clocks_span_exactly_the_option_life` — `sum(dt_r) == tau_r` and
  `sum(dt_v) == tau_vol` to 1e-12.

Independently, a fully implicit PDE under the same local vol surface (validated to
0.005% against Black-Scholes at constant vol) reproduces the input smile to

| K | 0.85 | 0.95 | 1.00 | 1.05 | 1.10 | 1.15 | 1.20 |
|---|---|---|---|---|---|---|---|
| error, vol points | +0.004 | +0.006 | +0.008 | +0.013 | +0.026 | +0.058 | +0.146 |

so the calibration itself is right to well under a basis point ATM. The residual
on the call wing is where the running-max repair lives: `max` of SVI slices has a
convex kink in `y` whose Dirac mass in `d2w/dy2` the analytic derivatives drop, so
the repaired surface is not *exactly* Dupire-invertible. On the raw surface the
same number is +0.21 vol points, so the repair improves it without curing it.

The old frozen-grid bump remains available in the lower-level
`LocalVolMC.price_european(..., bump=...)` API for model diagnostics, but it is no
longer presented as the Step 4 acceptance delta. The mentor-scope comparison is
implemented separately by `LocalVolMC.step4_delta_diagnostics`.

The PV and delta checks use antithetic draws, common random numbers and GBM
control variates driven by the same normals. Their exact expectations use the
same two clocks as the local-vol paths.

### Legacy Steps 5–6: analytic R delta is a range

For the ATM call at the furthest expiry:

| stickiness `R` | | delta |
|---|---|---|
| −1.0 | local-vol-like | 0.3933 |
| 0.0 | sticky strike | 0.5284 |
| +1.0 | sticky moneyness | 0.6635 |

27 delta points of spread on one option. Closed form and bump-and-reprice agree to
1e-9, so this is a choice about smile dynamics, not numerical noise.

`HedgeBacktester` sweeps `R`, minimising the daily hedging P&L dispersion. On a
synthetic history generated with a known true stickiness, the objective recovers it:

| true `R` | recovered `R*` (4 seeds) |
|---|---|
| 0.00 | 0.00 |
| 0.50 | 0.50 |
| 1.00 | 1.00 |

This is a self-check for the original analytic extension only.  It is not the
new research backtest: the dynamic-alpha document defines `alpha=0/1/2`, obtains
delta from local-vol MC, and requires a measured `beta(alpha)` curve.  New work
must not substitute `alpha = R + 1` into this legacy backtester.

---

## Two things to confirm with the desk

1. **`spot_adj` semantics.** The statement applies `y_adj = y - spot_adj` only
   inside the Dupire denominator `D`, leaving `w`, `dw/dy` and `d2w/dy2` at the
   unshifted `y`. The economically consistent reading shifts the whole surface.
   Both are implemented: `local_vol(..., shift_mode='denominator' | 'surface')`,
   defaulting to the literal reading.
2. **Historical observation dates.** `svi_data.pkl` contains many weekend keys.
   Their source timestamp semantics must be confirmed before computing daily
   `dlogS` and `dIV`; `research.py` refuses to repair them silently.

## Where this departs from the task statement

1. **Step 4(2) integrates `sigma_loc²` against an Act/365 `dt`.** Step 3 of the
   same statement defines `dw/dT = dw/dtau` with `tau = dt_vol`, so `sigma_loc` is
   a variance rate per *business* year and integrating it on Act/365 does not
   reproduce `w`. This implementation uses `dt_r` for the drift and the discount
   factor and `dt_v` for the variance and the Ito term, which makes
   `integral sigma² d(measure) == w` exact. On this data the difference is small
   (about +0.008 vol points ATM, ~0.05% of PV) because Business/260 and Act/365
   nearly agree in aggregate here — `sum dt_r = 0.690411` vs
   `sum dt_v = 0.692308` — but the per-step weights differ by 2.1× over a weekend
   and 0.71× on a weekday, and the near-cancellation is an accident of this
   calendar, not a property to rely on.
2. **`spot_adj` semantics** — see below. The Step 4 delta check follows the
   statement's literal denominator convention and rebuilds separate up/down
   grids before comparing with BS.

## Known approximations

- `dw/dT` uses the boundary rule `w/tau` at and beyond the last quoted expiry,
  following the statement, which makes it discontinuous at that node. Harmless for
  a maturity landing there.
- Local volatility on the simulation grid is clipped to `[0.0, 5.0]`; the surface
  peaks at 4.4 in the far wings, so nothing is actually clipped on this data. The
  floor is 0 rather than 0.01 because the calendar repair produces genuine zeros
  and flooring them would inject variance the surface does not have.
- `LocalVolGrid` tabulates on business-day nodes and `sigma_bilinear` interpolates
  linearly between them. `dw/dtau` jumps at each VolDate, so that smooths seven
  jumps across one business day each. Evaluating `local_vol` analytically at every
  sub-step instead removes it, at roughly 50× the cost.
- After the fixes the Monte Carlo sits about +0.4% above BS at 1e5 paths (≈ +1
  stderr) and stops moving once `n_substeps >= 4`; refining the spot-ratio axis
  from 1000 to 8000 nodes moves it by only 0.09%, so the residual is scheme bias
  from the near-non-Lipschitz local vol vertex (the front slice has SVI
  `sigma = 0.017`, i.e. an almost perfect V), not grid resolution.
