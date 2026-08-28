# 修改报告 — SVI local vol 校准

> 这是 2026-08-25 原 pricing task 的历史修改记录，不是动态 Alpha 研究的
> 当前状态页；新研究准备状态见 `DYNAMIC_ALPHA_READINESS.md`。

日期：2026-08-25
范围：Step 3 套利报告、Step 4 蒙特卡洛引擎、交付函数口径、回归测试

验证结论：**Dupire 校准本身是正确的**（单切片纯 SVI 曲面上，PDE 回算隐含波动率误差 ≤ 0.01 个波动率点）。修掉的是 Step 4 蒙特卡洛引擎的三个 bug、Step 3 套利报告的度量截断，以及交付函数与 Step 4 的口径不一致。

---

## 一、修改清单

| # | 位置 | 问题 | 严重度 |
|---|---|---|---|
| 1 | `montecarlo.py` `LocalVolGrid.build` | 模拟少算首个交易日 | **高** |
| 2 | `montecarlo.py` `LocalVolMC` | 方差按 Act/365 积分，与 σ_loc 的 Bus/260 口径不符 | 中 |
| 3 | `montecarlo.py` `terminal_spots` | σ 在整个交易日内冻结，加子步不收敛 | **高** |
| 4 | `surface.py` `arbitrage_report` | y 网格硬编码 `[-1, 0.6]`，crossedness 被截断 | 中 |
| 5 | `solution.py` | 交付的 `localvol` 跑不出 Step 4 的结果 | 中 |
| 6 | `solution.py` | Step 4 打印的百分比写死且互相矛盾 | 低 |
| 7 | `tests.py` | 缺少能抓住 1–3 的确定性回归 | 中 |

---

## 二、逐项说明

### 1. 模拟少算首个交易日（高）

`LocalVolGrid.build` 用 `tau_vol(d) > 0` 过滤日期轴，把定价日本身滤掉了，于是网格从 2026-08-10 而不是 2026-08-07 开始。`LocalVolMC` 的步长直接建在这个截断轴上：

```
修改前：179 步，sum(dt) = 0.682192
        但 tau_r = 0.690411、tau_vol = 0.692308
        -> 少 1 个交易日的方差、3 个日历日的 drift（贴现却按完整 tau_r 算）
修改后：180 步，sum(dt_r) = 0.690411 == tau_r
                sum(dt_v) = 0.692308 == tau_vol
```

**修法**：日期轴保留定价日；τ=0 处局部波动率无定义，用 τ→0⁺ 极限（下一交易日的值）填第一行。

单独这一项就是 −1.19% 的总方差缺口，是三个 bug 里最大的一个。

### 2. 两个时钟混用（中）

`local_vol` 返回 `sqrt((dw/dτ)/D)`，τ 是 `dt_vol`（Business/260），所以单位是**每交易年的方差率**。原代码把它对 Act/365 的步长积分。自洽的充要条件是

```
∫ σ² d(测度) == w(y, T)   精确成立
```

`∫(∂w/∂τ)dτ = w` 成立，`∫(∂w/∂τ)dt_r = w` 不成立。

**修法**：每步携带两个增量 —— drift 和贴现走 `dt_r`（Act/365），方差和 Itô 修正项走 `dt_v`（Bus/260）：

```python
s = s * np.exp(self.b * hr - 0.5 * sig**2 * hv + sig * np.sqrt(hv) * z)
#              \_ Act/365 _/  \__________ Business/260 ____________/
```

注意 `-0.5σ²` 必须跟 `hv` 走，它是扩散项的产物。原代码写成 `(b - 0.5σ²)·hs`，时钟合一时看不出问题，拆开就会错。`terminal_spots_gbm`（控制变量）同步改，否则它的解析期望 `_pv_cv_exact`（用 `σ²·tau_vol`）与模拟值对不上，控制变量本身会引入偏差。

量化（用独立 PDE 隔离，σ 每子步刷新、时间轴完整）：

| 方差积分测度 | K=0.95 | K=1.00 | K=1.10 | K=1.15 |
|---|---|---|---|---|
| Bus/260（修改后） | +0.0065 | **+0.0083** | +0.0278 | +0.0615 |
| Act/365（修改前） | +0.0166 | **+0.0158** | +0.0477 | +0.0930 |

（单位：波动率点。）ATM 只差 0.008 点 ≈ PV 的 0.05%，是三项里最小的，属于原理问题而非结果问题 —— **这一点上 docx 第六节与第七节自相矛盾**，见下文。

### 3. σ 在整个交易日内冻结（高）

`terminal_spots` 用 `grid.sigma_at(i, ...)`，日期索引 `i` 在 `n_substeps` 个子步内不变。所以加子步只细化了对 spot 的依赖，**对时间的依赖永远是一阶误差，加多少子步都不收敛**。`sigma_bilinear` 本来就是为这个写的，但从未被调用（死代码）。

PDE 验证（波动率点，ATM）：

```
nsub= 1 冻结  : -0.0873      nsub= 1 刷新  : +0.0286
nsub= 8 冻结  : -0.0800      nsub= 4 刷新  : +0.0124
              ^ 不收敛       nsub=12 刷新  : +0.0088
                             nsub=32 刷新  : +0.0077   ^ 收敛到 0
```

**修法**：`sigma_bilinear(i + j/m, s/s0_ref)`，在状态和时间两个方向都插值。

### 三项合起来的效果

修改前，MC/BS 相对误差随子步数扫过零点，`n_substeps=2` 的生产设置**恰好落在正的 Euler 偏差与负的时间轴缺口的交叉点上**，报出来的 0.16% 是抵消的结果，不是收敛值：

```
修改前（200k 路径，固定种子）        修改后
  m= 1: +1.733%                        m= 1: +1.737%
  m= 2: +0.416%  <- 生产设置           m= 2: +1.115%
  m= 4: -0.358%                        m= 4: +0.567%   <- 新生产设置
  m= 8: -0.616%                        m= 8: +0.584%
  m=16: -0.496%                        m=16: +0.145%
                                       m=32: +0.506%
```

修改后单调收敛，`m >= 4` 后进入平台。生产运行（100k 路径、m=4）：

```
BS PV               0.05595800
MC PV (control var) 0.05617821  (stderr 2.29e-04 = 0.410 % of PV)
relative error      +0.3935 %   -> PASS (<1%)
error / stderr      +0.96       -> consistent with zero
clocks: sum dt_r = 0.690411 (== tau_r), sum dt_v = 0.692308 (== tau_vol)
```

残余 +0.4% 已排除 spot 网格分辨率（用公共随机数比较，1000 → 8000 个 ratio 节点只移动 0.09%），是前端切片近乎不可 Lipschitz 的 V 型顶点（SVI `sigma = 0.017`）带来的格式偏差。

### 4. 套利报告的度量被截断（中）

`arbitrage_report` 的 y 网格写死成 `linspace(-1.0, 0.6, 321)`，三行的 `y_bad_hi` 正好等于 0.6 —— 那是网格边界，不是套利区间的边界。

更根本的问题：`w(y)` 渐近线性、斜率 `b(1±ρ)`，所以当近端切片的翼斜率更陡时，`w_lo − w_hi` 随 |y| 线性发散，**crossedness 在解析上无界，任何有限网格都测不出来**。2026-11-20 → 2026-12-18 这一对就是如此：

| y 窗口 | `[-1, 0.6]` | `[-1.5, 1.5]` | `[-1, 3]` |
|---|---|---|---|
| crossedness | 3.81e-03 | 1.05e-02 | 2.17e-02 |

**修法**：默认窗口放宽到 `[-1.5, 1.5]`，并新增四列 —— `y_grid_lo` / `y_grid_hi`（度量窗口）、`truncated`（违规区间在边界仍未闭合）、`unbounded_call_wing` / `unbounded_put_wing`（解析判据 `b_lo(1±ρ_lo) > b_hi(1±ρ_hi)`）。Step 3 的打印会在触发时明确说明"只有可交易区间有经济意义"。

### 5. 交付函数与 Step 4 口径不一致（中）

`_SURFACE` 绑的是**未修复**曲面（Step 3 必须如此，题目要求标注套利），但 Step 4 直接用私有的 `surface.repaired()` 建网格，所以交付的 `localvol` 跑不出 Step 4 的结果：

```python
localvol(2026-12-11, [1.20, 1.25, 1.30]) -> [nan nan nan]
```

而题目 Step 4(1) 明确要求"对每个 (date, ratio) 调用 `localvol(date, K=ratio·S0, spot_adj=0)`"。

**修法**：新增 `set_surface()` 和 `using_surface()` 上下文管理器；`LocalVolGrid.build` 增加可选的 `local_vol_fn` 参数。Step 4 现在是：

```python
with using_surface(surface_mc):
    grid = LocalVolGrid.build(..., local_vol_fn=lambda T, K: localvol(T, K, 0.0))
```

网格确实由交付的那个函数产出，同时 Step 3 的默认口径不变（`localvol` 在套利处仍返回 NaN，这是对的）。

### 6. 打印信息（低）

- `"20% of the local vol grid is NaN"` 是写死的，且与 Step 3 打印的 10.4%（不同网格）并列出现，容易误读。现在两个数都实时计算并标明各自的网格。
- `"0 points repaired"` 具有误导性 —— 修复发生在上游曲面，网格层面当然看不到 NaN。现在同时打印原始报价网格的 NaN 数（37,055 / 181,000 = 20.5%）和修复后网格中 **σ_loc 恰好为 0 的点数（67,560 = 37.3%）**，并说明这些零点来自 running-max 把 w 在 τ 方向压平，是一次很强的模型干预。
- 波动率上下限从 `[0.01, 2.0]` 改为 `[0.0, 5.0]`：曲面峰值 4.4，原来的 cap 会削掉真实值；floor 改为 0 是因为修复产生的是真零点，抬到 1% 等于凭空注入方差。改完后本数据上 **0 个点被裁剪**。
- delta 那段现在只输出 MC、`R=-1` 闭式解和 sticky-strike BS 三组数值及误差，不在代码输出中写定性结论；解释放在 README 和本报告中。

### 7. 新增回归测试（中）

原来的验收是"MC PV vs BS PV < 1%"，但 1e5 路径下 stderr 本身就是 PV 的 0.4%，这个判据分辨不出校准正确与否。新增五个测试，其中前三个是**确定性**的：

| 测试 | 抓什么 |
|---|---|
| `test_mc_grid_starts_on_the_pricing_date` | bug 1 |
| `test_mc_clocks_span_exactly_the_option_life` | bug 1 + 2，断言 `sum(dt_r)==tau_r`、`sum(dt_v)==tau_vol` 到 1e-12 |
| `test_scheme_accumulates_exactly_the_quoted_total_variance` | bug 1+2+3。常数波动率 + 对偶变量下布朗增量和恰好为 0，于是 `mean(log S_T) == b·tau_r − 0.5σ²·tau_vol` 成立到机器精度 —— 无噪声 |
| `test_mc_is_a_martingale_under_the_forward_measure` | drift 时钟 |
| `test_substep_refinement_has_converged` | bug 3；按**合并标准误**判定，不用固定百分比（两次不同子步数的运行噪声独立） |

同时修正了 `test_mc_engine_reproduces_black_scholes_when_smile_is_off`：它原来用 `w = σ²·tau_r`，等于把 bug 2 写进了断言；改为 `σ²·tau_vol`。

另外新增两个 Step 4 诊断测试：常数波动率下逐路径积分方差必须精确等于
`sigma²*tau_vol`，以及 strike 表/终值分箱表的字段、概率和 implied-vol MC
误差检查。

**32/32 通过。**

### 8. Step 4 CSV 与可视化（中）

- `step4_pricing_errors.csv`：对 0.80–1.30 的每个 strike 同时输出 BS、常数
  implied-vol MC、local-vol MC、原始/控制变量误差、对偶样本标准误和配对差。
- `step4_terminal_variance_bins.csv`：按 `S_T/S0` 分箱，输出每条路径
  `sum sigma_loc²*delta_tau_vol` 的条件均值、标准误、分位数，以及逐路径终值
  对应 implied total variance 的条件均值。
- 新增两张对应 PNG。代码只负责计算和展示字段，如何解释两个 variance 的差异
  写在 README；没有把某个比例或方向性结论写死在 `solution.py`。
- 修正 antithetic sampling 的标准误：独立样本是 `(Z,-Z)` 的 pair mean，而非
  两条路径各算一个独立观测。控制变量 beta/correlation 也改为在 pair mean 上估计。

---

## 三、docx 本身的矛盾

第六节 Step 3 定义 `dw/dT = ∂w/∂τ`（τ 是 `dt_vol`，Business/260），第七节 Step 4(2) 却用 `dt = 日历天数/365` 同时喂给 drift 和 diffusion。而第三节自己写着"二者日数惯例不同……务必作为独立变量，互不影响"。

按字面实现仍能通过 1% 验收（时钟误差只有 0.05%），所以这是规格瑕疵，不是致命错误 —— 真正让原 Step 4 不可信的是 bug 1 和 bug 3。代码现在按自洽的口径实现，并在 README 的 "Where this departs from the task statement" 里明确记录这处偏离。

---

## 四、独立验证方法

所有结论都用一个**独立编写的全隐式 PDE** 交叉验证，而不是靠蒙特卡洛：

1. 先用常数波动率验证 PDE 本身 —— 复现 BS 到 **0.005%**；
2. 再在单切片纯 SVI 曲面（无插值、无修复、无套利）上跑 —— 隐含波动率回算误差 **≤ 0.01 个波动率点**，证明 Dupire 公式、解析一二阶导、`dw/dτ = w/τ` 边界规则都是对的；
3. 最后在完整曲面上跑，定位残差来源。

完整曲面上收敛后的回算误差（波动率点）：

| K | 0.85 | 0.95 | 1.00 | 1.05 | 1.10 | 1.15 | 1.20 |
|---|---|---|---|---|---|---|---|
| 修复后曲面 | +0.004 | +0.006 | +0.008 | +0.013 | +0.026 | +0.058 | +0.146 |
| 原始曲面 | +0.003 | +0.004 | +0.007 | +0.015 | +0.040 | +0.104 | +0.212 |

ATM 精确到远小于 1bp。call wing 的残差正是 calendar 修复所在的位置：running max 是若干 SVI 切片的逐点极大值，切换点上 y 方向有凸折角，其 `d²w/dy²` 的 Dirac 质量被解析导数丢掉了，所以修复后的曲面**不严格可 Dupire 反演**。原始曲面上同一位置误差更大（+0.212），说明修复在改善但不彻底。

---

## 五、未做的事

- **running-max 修复的折角问题没有根治。** 彻底的做法是按 Gatheral & Jacquier 第 5 节做无套利重新校准（在保持无蝶式套利的前提下，求解使 crossedness 归零的最近 SVI 参数），而不是逐点取极大值。这是重新校准，超出本次 review 的范围；当前做法的代价已在 Step 4 打印和 README 中量化说明。
- **Step 4 ATM 残差本次为 +0.30%（+0.74 标准误）**。strike 网格的远
  OTM 相对误差会因 PV 很小而放大，因此 CSV 同时保留绝对误差、标准误和 z-score；
  本轮所有 local-vol/BS 差异均在 1.54 标准误内。
- **`spot_adj` 语义**仍保留两种实现（`shift_mode='denominator' | 'surface'`），默认按题目字面。需要与交易台确认，README 已列出。

---

## 六、改动文件

| 文件 | 改动 |
|---|---|
| `svi_localvol/montecarlo.py` | 重写：双时钟步长、日期轴含定价日、子步内刷新 σ、网格诊断计数、`_check_horizon` 保护；新增 strike/路径方差诊断和对偶 pair-mean 标准误 |
| `svi_localvol/surface.py` | `arbitrage_report`：默认 y 窗口放宽 + 4 个新列（窗口边界、截断、两个翼的解析无界判据） |
| `svi_localvol/solution.py` | `set_surface` / `using_surface`；Step 4 经由交付的 `localvol` 建网格；Step 3/4 打印重写；`n_substeps` 默认 2 → 4；写出两个新增 CSV/PNG |
| `svi_localvol/tests.py` | 新增 7 个回归；修正 `test_mc_engine_...` 的 `tau_r` → `tau_vol`，修正对偶样本标准误口径 |
| `svi_localvol/backtest.py` | `len(mc.dt)` → `len(mc.dt_r)` |
| `README.md` | 双时钟设计说明、calendar 套利表更新、Step 4 诊断字段与两类 variance 的关系、新增 "Where this departs from the task statement" |
| `output/*` | 全部重新生成，并新增 `step4_pricing_errors.*`、`step4_terminal_variance_bins.*` |

---

## 七、追加修改：mentor 口径的 Step 4 delta

- Step 4 不再把同一张 `spot_adj=0` 网格用于 spot up/down 的正式 delta。
- up 网格通过 `localvol(T, K, log((S0+h)/refSpot))` 重建；down 网格通过
  `localvol(T, K, log((S0-h)/refSpot))` 重建，仍采用 docx 默认的
  `shift_mode='denominator'`。
- up/down local-vol 路径与 constant-implied-vol 路径使用相同随机数。
- 新增 `LocalVolMC.step4_delta_diagnostics`，逐 strike 输出解析 BS delta、
  BS 中心差分 delta、constant-implied-vol MC delta、local-vol MC delta、
  标准误和绝对误差。
- 新增 `output/step4_delta_comparison.csv`。
- Step 4 默认日内子步数由 4 提高到 16，以降低 mentor delta 检查的时间离散误差。
- 新增 `test_step4_mentor_delta_table_is_complete`，检查 up/down 网格确实不同、
  shift 数值正确、表格字段完整且 implied-vol MC delta 与 BS bump delta 一致。
