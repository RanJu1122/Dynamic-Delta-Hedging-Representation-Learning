# Step 7 near-ATM book：实验设置、模型与结果分析

> 历史版本记录：本文的模型特征、rolling Beta 对照及数值属于当时实验。当前代码已改为 [daily-only 流程](DAILY_ONLY_PIPELINE_CN.md)，未据新代码重算本文结果。

本文对应已完成的 `output/dynamic_alpha/step07_near_atm_fast/` 实验，依据该目录的 manifest、CSV 和实际代码核对，不将其他 book 或其他转换器的结果混入本次结论。

核心结论：在六份 near-ATM call 构成的模拟 book 上，未平滑动态策略相对训练期最优固定 alpha 的对冲误差标准差降低 **2.16%**，解析 shadow delta 得到近似结果。改善主要集中在 6M、1Y，但统计区间跨零、换手增加，且 MC 网格质量与部分原始 spot 数据仍需验证。因此这是正向研究迹象，而不是已经确认的稳定对冲优势。

## 1. 实验命令与研究目标

本次命令：

```bash
python -m dynamic_alpha_hedging step7 \
  --book near_atm \
  --factors 2 \
  --converter daily \
  --fast \
  --output output/dynamic_alpha/step07_near_atm_fast
```

对应《动态Alpha对冲研究》7.4：构造不同 strike、不同 maturity 的期权组合，按整个 book 的净 delta 对冲，比较动态 alpha、固定 alpha 和训练期最优固定 alpha，额外报告归因残差与 turnover。

本次不是 56 个合约槽位的 full book，而是规模较小、仍跨期限与 strike 的 near-ATM book。结果只能直接用于评价这个持仓设计，不能推断完整曲面的所有合约都得到改善。

## 2. Setting：究竟跑了什么

### 2.1 Book 设置

| 项目 | 本次取值 | 含义 |
|---|---|---|
| `book` | `near_atm` | 固定采用下表六个期权槽位 |
| tenor | 0.25、0.5、1.0 | 分别对应 3M、6M、1Y 的剩余期限目标 |
| level | 0.9、1.0 | 每日开仓 strike 为 level × 当日 refSpot |
| 期权类型 | 欧式 call | 不是 call/put 混合持仓 |
| `weights` | `contracts` | 每个槽位数量为 1，共六份 |
| `factor_count` | 2 | 重构 beta 时采用 ATM 因子和第一个形状因子 |
| `converter` | `daily` | 按当天曲面计算 beta–alpha–delta 表 |
| `alpha_half_life` | 10 | 仅用于 `dynamic_ema`；未平滑策略同时输出 |
| `hedge_cost_bps` | 0 | 本次没有扣标的交易费用 |
| 期权交易费用 | 未建模 | 未包含期权 bid/ask、滑点 |

六个槽位是：3M/0.9、3M/1.0、6M/0.9、6M/1.0、1Y/0.9、1Y/1.0。

level=0.9 对 call 而言通常是实值，而不是虚值。level=1.0 是以 spot 定义的 ATM，不严格等同于 forward ATM。

每个持有区间开始时：

$$
K_{i,t}=m_iS_t,\qquad T_{i,t}=\operatorname{date\_at\_tau}(t,\tau_i).
$$

在收盘 t 至下一收盘期间，实际 strike、实际 expiry 和数量都冻结。下一天再次按槽位建立新合约。因此这是每日滚动的模拟 book，不是把同六张合约一直持有到期，也不是读取真实 desk 的持仓明细。

代码入口：[Step7Config.axes](../dynamic_alpha_hedging/step07.py)、[contract_interval](../dynamic_alpha_hedging/hedging.py)。

### 2.2 数据、时间与市场约定

| 项目 | 本次设置 |
|---|---|
| 原始数据 | `data/svi_param.pkl` |
| 数据重复项 | 同一重复到期日保留第一个，`duplicate_vol_date_policy=first` |
| 无法构建的曲面 | 允许跳过，并保留审计记录 |
| 波动率时钟 | Business/260 |
| 利率时钟 | Actual/365 |
| 年化利率 r | 0.036 |
| 年化 dividend | 0.03 |
| repo | 0 |
| 日历 | 预设美股节假日，不包含临时一次性休市 |
| 时区配置 | source 为 Asia/Shanghai，market 为 America/New_York |
| daily beta 标签门槛 | `abs(dlogS) >= 0.0025`，即约 0.25% 的对数收益 |
| rolling beta | 60 条观测窗口、至少 20 条有效样本 |

时区配置不表示每个 date-only key 都能恢复原始收盘时间。原始日期和 refSpot 的准确对应仍取决于数据源。

原始 SVI 参数构建每日隐含波动率曲面。给定剩余期限和实际 strike，在曲面上读取 IV，再使用 Black–Scholes 公式估值。历史损益使用原始 SVI 曲面价格，不是交易所真实成交价；MC 则使用现有修复后曲面构建局部波动率网格。

本次实验 manifest 记录的原始数据和 Step 1/2/4/5/6 manifest 的 SHA-256，与撰写本文时的文件一致。哈希说明版本对应，不证明数据本身没有错误。

### 2.3 MC 数值设置

| 参数 | 本次值 | 含义 |
|---|---:|---|
| `step3_n_paths` | 10,000 | fast 模式路径数 |
| `step3_n_ratio` | 201 | 局部波动率网格的 spot-ratio 节点数 |
| `step3_n_substeps` | 2 | 每个基础时间间隔进一步细分 |
| `step3_spot_bump_fraction` | 0.01 | spot 上下各 bump 1% |
| alpha 节点 | 0、0.5、1、1.5、2 | 不是连续 alpha 每个数值都跑一次 MC |
| seed | 20260807 | 固定随机种子 |
| antithetic | true | 使用对偶变量 |
| ratio 范围 | 首节点钳到 0.001，最大 3.0 | 覆盖广于 book 行权价的路径状态空间 |
| local vol floor/cap | 0 / 5 | 按波动率小数值计，5 表示 500% |

同一天、同一期限的两个 strike 共用路径，一组 spot up/down 价格同时提供 beta 与 delta。三个期限、五档 alpha，合计每个日期 15 组配对估值。缓存命中时复用已有结果。

`fast` 是数值诊断设置，不是已获认证的正式精度。增加路径数只针对随机误差，不能自动解决无定义网格、截断或模型偏差。

## 3. 模型：从状态预测到六个对冲量

### 3.1 预测目标仍是下一日 daily surface beta

记 $I_t(\tau,K)$ 为 t 日曲面在固定剩余期限与实际 strike 下的 IV。当前标签使用后一日 strike 为锚：

$$
K_{t+1}=mS_{t+1},\qquad
dIV_{surface,t+1}=I_{t+1}(\tau,K_{t+1})-I_t(\tau,K_{t+1}).
$$

$$
\beta_{t+1}(\tau,m)
=-\frac{dIV_{surface,t+1}}{\log(S_{t+1}/S_t)}.
$$

它是扣除标准化网格移动效应后的固定-strike 曲面变化，而不是原始 grid IV 的直接差分。剩余期限保持可比，实际到期日可以滚动。

使用下一日信息构造事后的监督标签是正常的；预测输入只能用 t 日及之前的信息。Step 7 不会先知道 $S_{t+1}$ 再决定当天仓位。

重要区别：Step 7 实际隔夜持仓固定的是实际 expiry，而不是固定剩余期限。持仓会老化，beta 只描述价格变化中的一部分，不承担解释全部 theta、Gamma 和其他残差的任务。

代码：[step01.py](../dynamic_alpha_hedging/step01.py) 的曲面变化分解、[step02.py](../dynamic_alpha_hedging/step02.py) 的 `_daily_beta`。

### 3.2 Step 4 的 ATM 锚定因子结构

降维是在当前完整的 56 单元建模曲面上进行，不是只对本次六个持仓单元重新做 PCA：

$$
\beta_t(\tau,m)\approx c(\tau,m)+f_1(\tau,m)z_{1,t}
+f_2(\tau,m)z_{2,t}+f_3(\tau,m)z_{3,t}.
$$

- $z_{1,t}$ 严格等于当天 3M ATM daily beta。
- 先在训练期用 ATM 因子回归解释各曲面单元，得到截距 c 和载荷 $f_1$。
- 再对残差做 PCA，得到两个形状因子 $z_2,z_3$。
- ATM 处的截距为 0，第一因子载荷为 1，两个形状因子载荷为 0。

本次 `--factors 2` 重构为：

$$
\widehat\beta_{t+1|t}(\tau,m)
=c(\tau,m)+f_1(\tau,m)\widehat z_{1,t+1|t}
+f_2(\tau,m)\widehat z_{2,t+1|t}.
$$

不同合约通过不同的 c、$f_1$、$f_2$ 得到不同 beta；不是整个 book 共用一个 beta 或一个 alpha。在 3M ATM 锚点，增加第二因子不会改变该单元的重构值，因为它的第二载荷为零。

代码：[step04.py](../dynamic_alpha_hedging/step04.py) 的 `run_step4`。

### 3.3 Step 6 预测器与 16 个输入

采用 `HistGradientBoostingRegressor`，每个因子一个预测器。代码拟合并保存三个预测器，但本次重构只取前两个输出；不是六个合约各自训练一个模型。

| 模型参数 | 值 |
|---|---|
| loss | absolute_error |
| learning_rate | 0.05 |
| max_iter | 200 |
| max_leaf_nodes | 7 |
| min_samples_leaf | 15 |
| l2_regularization | 1.0 |
| early_stopping | false |
| random_state | 20260807 |

绝对误差损失倾向于条件中位数预测，并不直接等价于最小化 book 对冲误差方差。

| 输入组 | 字段 | 含义 |
|---|---|---|
| 当天收益 | `dlogS` | 截至 t 的当日对数收益 |
| IV 水平 | `atm_iv_3m` | 3M spot-ATM IV |
| IV 变化 | `atm_iv_change_1d`、`atm_iv_change_5d` | 最近 1/5 日 ATM IV 变化 |
| 偏斜 | `smile_slope_3m` | 3M 的 level 0.9 和 1.1 IV 差除以 log-level 差 |
| 期限结构 | `term_slope_1y_minus_3m` | 1Y ATM IV 减去 3M ATM IV |
| 已实现波动率 | `realized_vol_20d` | 最近 20 日 dlogS 样本标准差乘以 $\sqrt{260}$ |
| 累计收益 | `recent_return_5d`、`recent_return_20d` | 最近 5/20 日对数收益之和 |
| 波动率的波动 | `vol_of_vol_20d` | 最近 20 日 ATM IV 日变化标准差乘以 $\sqrt{260}$ |
| 最近因子，3 个 | `last_atm_beta_factor`、`last_shape_score_1`、`last_shape_score_2` | 最近一次可观测因子；仅向前填充 |
| rolling 因子，3 个 | `rolling_atm_beta_factor`、`rolling_shape_score_1`、`rolling_shape_score_2` | rolling beta 曲面在相同载荷下的投影 |

前十个是市场状态，后六个是历史因子状态。即使只使用两因子输出，输入仍是全部 16 项；这不是删除第三因子的所有相关输入。

模型在 Step 7 准备阶段按照 Step 6 原训练样本重新拟合，然后冻结。每天更新特征、产生预测，但不每天重新训练。Step 6 的 Ridge 等模型比较结果没有在 Step 7 中自动参与择优或集成。

代码：[step05.py](../dynamic_alpha_hedging/step05.py) 的 `STATE_FEATURES`、[step06.py](../dynamic_alpha_hedging/step06.py) 的 `primary_factor_model`、`fit_factor_forecaster`、`factor_features`。

### 3.4 样本数量为什么有 342、114、492、156？

| 数量 | 含义 |
|---|---|
| 342 | Step 6 的有效训练标签日期 |
| 114 | Step 6 可用于评价真实因子预测误差的测试标签日期 |
| 492 | Step 7 实际用于比较固定 alpha 的训练持有区间 |
| 156 | Step 7 的测试持有区间，2026-01-16 至 2026-08-31 |
| 157 | `factor_forecasts.csv` 的收盘预测日期，包含最后一个尚无下一日持有结果的收盘 |

训练标签截止日为 2026-01-15。Step 7 不以真实下一日 daily beta 是否有效来决定是否对冲，小收益日也必须持有和对冲。

最初 494 个连续业务日训练区间中，两个涉及 2025-04-09 曲面的区间因合约期限超出报价范围被排除，余下 492 个。非连续来源观测的排除另在 manifest 中记录。本次测试区间没有这类合约排除。

所以提高 beta 标签门槛会减少 Step 4–6 的有效标签，却不会自然减少 Step 7 的所有持有日期。

## 4. 每天如何从预测 beta 得到对冲 delta

对当天曲面、每个期限及五档 alpha，固定同一实际 strike 和 expiry，构建 spot up/down 的两张 Local Vol Grid，复用现有 LocalVolMC。

$$
B_t^{raw}(\alpha)=-\frac{IV_t^{up}(\alpha)-IV_t^{down}(\alpha)}
{\log(S_t^{up}/S_t^{down})}.
$$

$$
B_t(\alpha)=B_t^{raw}(\alpha)-B_t^{raw}(1).
$$

$$
\Delta_{MC,t}(\alpha)=\frac{PV_t^{up}(\alpha)-PV_t^{down}(\alpha)}
{S_t^{up}-S_t^{down}}.
$$

若为反解 IV 稳定性选用了 put，其 delta 通过 put-call parity 转换为 call delta。up/down 是同一天的假设场景，不是前一天与后一天的历史价格。

预测 beta 在当天 $B_t(\alpha)$ 上反查得到 alpha，再在当天 delta 节点表插值。不为每个预测出来的小数 alpha 另外跑 MC。

现有减去 $B_t^{raw}(1)$ 的处理保证转换表 alpha=1 对应 beta=0，但不保证原始 MC 的 beta(1) 已经足够接近零，也没有同时把 MC delta(1) 改成 BS delta。

本次没有读取单日 Step 3 inverse，也没有构建多日 pooled 转换器。manifest 中的 `converter_dates=10` 是未使用的默认配置，不能解读为本次用了 10 天转换器。

代码：[hedging.py](../dynamic_alpha_hedging/hedging.py) 的 `MCMapStore.measure`、`invert_beta`、`delta_at_alpha`；原始锚定和检查逻辑复用 [step03.py](../dynamic_alpha_hedging/step03.py)。

## 5. 十三种策略如何比较

| 策略 | 实际做法 |
|---|---|
| `fixed_0`、`fixed_0.5`、`fixed_1`、`fixed_1.5`、`fixed_2` | 所有合约采用同一个固定 alpha，但 delta 随当天市场变化 |
| `best_fixed_train` | 用训练期 book 误差标准差选出最优固定档，测试期冻结 |
| `bs_delta` | 原始历史 SVI IV 对应的 BS delta |
| `rolling_beta` | 截至 t 已知的 rolling beta 反查 alpha |
| `rolling_alpha_mean` | 最近最多 20 个有效历史 beta 反查出的 alpha 均值 |
| `dynamic_raw` | 下一日预测 beta 直接反查当天 alpha |
| `dynamic_ema` | 对原始预测 alpha 做 10 日半衰期平滑，初值为训练期最优固定档 |
| `shadow_bs` | 直接用预测 beta 修正 BS delta，不反查 alpha |
| `shadow_mc_base` | 同样的 beta 调整，但基准换成当天 MC delta(1) |

由于 beta 对的是 log spot 而不是 spot，解析对照必须除以 S：

$$
\Delta_{shadow,BS}=\Delta_{BS}-\frac{\nu}{S}\widehat\beta,
\qquad
\Delta_{shadow,MCbase}=\Delta_{MC}(1)-\frac{\nu}{S}\widehat\beta.
$$

两者使用相同的原始 SVI vega，不截断 beta，也不进行 alpha 平滑。shadow 策略的 alpha 为空是正常现象，不是计算失败。两者的 vega 都是对波动率小数值求导，不是每一个波动率百分点的 vega。

## 6. Book 损益与评价口径

各策略使用相同期权持仓，唯一改变的是标的对冲量：

$$
D_t=\sum_i q_{i,t}\Delta_{i,t},\qquad V_t^{book}=\sum_i q_{i,t}V_{i,t}.
$$

原始对冲误差：

$$
e_t^{raw}=\Delta V_t^{book}-D_t\Delta S_t.
$$

包含融资与标的收益的毛误差：

$$
e_t^{gross}=\Delta V_t^{book}-D_t\Delta S_t
+(D_tS_t-V_t^{book})(e^{r\Delta t_r}-1)
-D_tS_t(e^{q_{eff}\Delta t_r}-1),
$$

其中本次 $q_{eff}=dividend+repo=0.03$。代码进一步用自融资现金账本计算净误差和累计 wealth；交易成本基于净标的再平衡，不把各期权交易量绝对值分别相加。

本次费用为零，所以净误差与含 carry 毛误差在数值精度内一致。Headline 是净误差标准差，不是累计 wealth，也不是单笔误差标准差之和。误差单位是当前定价与数量约定下的价格单位，不是收益率；没有加入真实交易合约乘数，不能直接当作实际账户美元收益。

相对基准的改善定义为：

$$
Improvement_{std}=1-\frac{\operatorname{Std}(e^{strategy})}
{\operatorname{Std}(e^{baseline})}.
$$

正数表示误差下降，负数表示误差增大。标准差使用样本口径；RMSE 同时反映均值偏移和波动。

## 7. 本次输出结果

### 7.1 最优固定 alpha 来自训练期，不是事后选的

训练期 book 误差标准差如下：

| 固定 alpha | 训练期标准差 |
|---|---:|
| 0 | 48.5311 |
| 0.5 | 32.3451 |
| 1 | **30.3203** |
| 1.5 | 44.1372 |
| 2 | 64.4763 |

因此本次 `best_fixed_train = fixed_1`。这与之前单个 3M ATM 实验选择 alpha=0.5 不矛盾，选择目标和持仓范围已经不同。这里的最优也只是五个预设 alpha 档位中的最优，不是连续 alpha 全局最优。

### 7.2 测试期整体表现

| 策略 | 标准差 ↓ | RMSE ↓ | MAE ↓ | 标准差相对 fixed_1 改善 |
|---|---:|---:|---:|---:|
| fixed_0 | 74.3680 | 74.1565 | 55.3792 | −130.55% |
| fixed_0.5 | 47.5291 | 47.3770 | 33.2779 | −47.35% |
| fixed_1 / best_fixed_train | 32.2570 | 32.2041 | 22.1012 | 基准 |
| fixed_1.5 | 43.2548 | 43.2732 | 31.1438 | −34.09% |
| fixed_2 | 68.7942 | 68.8027 | 50.6596 | −113.27% |
| bs_delta | 32.3544 | 32.2998 | 22.2104 | −0.30% |
| rolling_beta | 33.3600 | 33.2878 | 21.8543 | −3.42% |
| rolling_alpha_mean | 35.9870 | 35.8764 | 23.2467 | −11.56% |
| **dynamic_raw** | **31.5589** | **31.4880** | **20.1294** | **+2.16%** |
| dynamic_ema | 32.9339 | 32.8493 | 21.4971 | −2.10% |
| shadow_bs | 31.5326 | 31.4607 | 20.1897 | +2.25% |
| shadow_mc_base | 31.5529 | 31.4819 | 20.1294 | +2.18% |

`dynamic_raw` 的标准差改善为 $1-31.5589/32.2570\approx2.16\%$；RMSE 改善 2.22%，MAE 改善 8.92%。绝对误差的 95% 分位数由 55.2506 降至 50.2626。

MAE 改善比标准差改善大，表明一般日期误差有所改善，但尾部大误差仍然重要。不能把这几个数解释成收益率、分类准确率或 beta 预测 R²。

### 7.3 改善尚不具有充分的统计证据

当前区间用同日期配对、10 个观测长度的循环区块 bootstrap，重复 1,000 次，并取 2.5%/97.5% 分位数。

- dynamic_raw 标准差改善 2.16%，95% 区间为 **−6.24% 至 +15.83%**。
- shadow_bs 标准差改善 2.25%，95% 区间约为 **−5.71% 至 +15.26%**。

两个区间都包含零，不能宣称稳定优于基准。它们是当前测试段上的描述性统计，不覆盖数据源错误、模型筛选、多次试验和 MC 系统偏差的不确定性。

### 7.4 哪些合约改善了

下表是 dynamic_raw 相对 fixed_1 的单合约误差标准差改善：

| 期限 | level 0.9 | level 1.0 |
|---|---:|---:|
| 3M | −0.83% | −1.86% |
| 6M | +3.78% | +2.19% |
| 1Y | +4.48% | +7.39% |

改善集中于 6M、1Y，3M 两个单元仍略差。Book 改善不是这六个百分比的平均，还取决于损益大小及误差相关性。

这个表尚不能证明第二因子有效。需要在同一个六槽位 book、同样日期和费用下比较 `--factors 1` 与 `--factors 2`。不能拿一因子的单个 ATM book 与两因子的六槽位 book 直接归因于因子数。

### 7.5 预测 alpha 的实际位置

| 期限 | level | 预测 beta 均值 | dynamic_raw alpha 均值 | alpha 范围 |
|---|---:|---:|---:|---|
| 3M | 0.9 | 0.0450 | 0.9629 | 0.6718–1.2996 |
| 3M | 1.0 | 0.1257 | 0.8577 | 0.2756–1.7338 |
| 6M | 0.9 | 0.0020 | 1.0000 | 0.7927–1.2663 |
| 6M | 1.0 | 0.0235 | 0.9732 | 0.7161–1.3254 |
| 1Y | 0.9 | −0.0148 | 1.0278 | 0.8511–1.2591 |
| 1Y | 1.0 | −0.0334 | 1.0775 | 0.8657–1.3862 |

大部分调整围绕 alpha=1，而非长期靠近 0 或 2。3M ATM 调整较大，1Y 的预测 beta 平均为负，映射到略高于 1 的 alpha。这是模型预测，不是宣称真实市场的长期 beta 必然为负。

### 7.6 逐日转换与解析对照非常接近

dynamic_raw、shadow_bs、shadow_mc_base 的标准差分别为 31.5589、31.5326、31.5529。
全部动态单元都没有 alpha 截断、inverse fallback 或 delta fallback。当天有效 beta 与预测 beta 的差异约为浮点误差。

这支持：本次逐日转换链路没有相对于直接解析调整表现出明显额外损失。不能把动态改善有限主要归咎于转换器，也不能据此证明 MC 数值已经完全收敛。

`beta_mapping_gap` 接近零只是同一张表正反插值一致，并不是预测误差接近零。

### 7.7 平滑与换手

| 策略 | 标的总换手量 | 误差标准差 |
|---|---:|---:|
| fixed_1 | 9.5630 | 32.2570 |
| dynamic_raw | 22.0378 | 31.5589 |
| dynamic_ema | 10.2014 | 32.9339 |
| shadow_bs | 22.1103 | 31.5326 |

未平滑动态的换手是 fixed_1 的约 2.30 倍。10 日半衰期降低了换手，但没有保留原始动态策略的标准差改善。

按本次相同交易量粗算，单边 1 bp 标的费用约为 fixed_1 的 6.98、dynamic_raw 的 15.96。
这只是累计直接交易费量级，不包含其后融资影响，也不是标准差；不能从 2.16% 中直接相减。净效果应带成本重新计算账本。

### 7.8 归因结果

公共归因基准采用 BS delta，避免 MC/shadow delta 已包含的 spot-vol 效应被重复计入：

$$
\Delta V\approx\Delta_{BS}\Delta S+\tfrac12\Gamma(\Delta S)^2
+time\_pnl+\nu\,term\_roll\_iv-\nu\beta_{effective}\Delta\log S+\varepsilon.
$$

time_pnl 是冻结 IV 后的时间流逝重估；term_roll_iv 是上一日曲面上，期限缩短导致的 IV 变化。book 归因将各合约贡献按数量相加。

fixed_1 的归因残差 RMSE 为 29.7074，dynamic_raw 为 27.7941，下降 **6.44%**；其 95% 改善区间约为 **−0.98% 至 +18.57%**，同样跨零。

这个数表示曲面变化解释能力，不是实际对冲标准差改善。shadow 策略和 dynamic_raw 使用相同有效 beta，本次归因结果相同也正常。

尤其注意：`forecast_attribution_rmse` 对所有策略都是 27.7941，因为该列统一使用同一套原始预测 beta。不要把它当成每个策略各自取得的归因效果；实际策略归因看 `attribution_rmse`。

## 8. 数值质量和数据问题

### 8.1 全项质量通过率只有 14.53%

936 个“日期 × 合约单元”中，136 个通过全部检查。每个单元的检查结果复制到五个 alpha 节点，因此不能把 4,680 个节点当作独立检查样本。

| 审计项目 | 结果 |
|---|---|
| 完整质量检查通过 | 136/936，14.53% |
| 网格检查失败 | 796/936 |
| 原始 beta(1) 超容差 | 31/936，其中 27 个同时存在网格问题 |
| 可逆曲线 | 全部可逆 |
| IV 反解价格截断 | 0 |
| 测试期 inverse/delta fallback | 0 |
| 最大 beta 标准误 | 0.08578，低于当前 0.1 门槛 |
| 全节点平均 delta 标准误 | 0.00450 |
| 最大 delta 标准误 | 0.01185 |

主要门槛是：alpha=1 原始 beta 绝对值不超过 0.03；网格未定义比例不超过 5%；网格截断比例不超过 1%。本次单张 bumped 网格的未定义比例最高约 24.17%，截断比例最高约 7.50%。

代码对无定义局部波动率使用现有填补处理，再应用上下限。质量检查按此前约定只写审计，不删除策略结果。因此“成功出结果”不等于“全部数值质量通过”。

同样，这不等于 85% 的期权价格都错了：远端网格区域也被计数，而 MC 路径未必经常访问。需要补充异常网格位置及路径使用情况诊断，不能只修改门槛把通过率变高。

### 8.2 MC 内部一致性

本次 `mc_consistency.csv` 的统计：

| 字段 | RMSE | 含义 |
|---|---:|---|
| raw_chain_residual | 0.000426 | MC delta 与使用原始 MC beta 的一阶链式近似之差 |
| centered_adjustment_residual | 0.000632 | 相对 MC alpha=1 基准的 delta 调整与归零后 beta 调整之差 |
| mc_one_minus_bs_delta | 0.004043 | 当天 MC alpha=1 delta 与原始 SVI BS delta 的差异 |

中点 IV 被用作无 bump MC IV 的近似，因此这些是有限 bump 下的内部诊断，不是严格的独立统计检验，也不是单靠残差小就能判定最终对冲准确。

### 8.3 重复 spot 与尾部日期

结果中有以下区间：

| 持有区间结束日 | 前一日 refSpot | 当日 refSpot | dlogS |
|---|---:|---:|---:|
| 2026-03-31 | 6368.85 | 6368.85 | 0 |
| 2026-04-01 | 6368.85 | 6368.85 | 0 |
| 2026-04-02 | 6368.85 | 6368.85 | 0 |
| 2026-04-06 | 6368.85 | 6606.93 | 约 0.0367 |

3/31 的 fixed_1 与 dynamic_raw book 误差分别约 −178.47、−178.49；spot 不动时，很难通过调整标的 delta 消除主要的曲面变动损益。

4/6 的 fixed_1 误差约 +150.61，dynamic_raw 约 +203.07，动态在这个大幅变动区间明显更差。以上日期对尾部指标影响很大。

连续相同 spot 不是自动认定错误的充分证据，但值得向数据提供者确认 refSpot 是否 stale、是否与 SVI 参数来自相同观测时点。不能因为这些日期拖累结果就删除，也不能未经核实就归因于预测器失效。

## 9. 输出文件阅读顺序

所有文件位于 `output/dynamic_alpha/step07_near_atm_fast/`。

| 文件 | 内容与读法 |
|---|---|
| `manifest.json` | 首先确认配置、训练截止日、排除区间、输入指纹和运行完成状态 |
| `summary.csv` | 13 个策略的 book headline；先看 std_error、相对改善和置信区间，再看换手、费用、归因 |
| `book_pnl.csv` | 2,028 行＝156 日期×13 策略；净 book delta、现金余额、逐日误差和累计 wealth |
| `cell_summary.csv` | 78 行＝6 单元×13 策略；定位改善集中在哪些期限和 level，不替代 book 指标 |
| `signals.csv` | 12,168 行；日期、预测 beta、实际 alpha、effective_beta、clipping/fallback |
| `option_pnl.csv` | 12,168 行；同一实际合约的前后估值、Greeks、对冲 delta 和损益 |
| `mc_map.csv` | 4,680 行＝156 日期×6 单元×5 alpha；原始 PV/IV、beta、delta、标准误和质量标记 |
| `mc_consistency.csv` | 同样 4,680 行，内部一阶一致性诊断 |
| `factor_forecasts.csv` | 157 个收盘日期的三个因子预测，本次仅前两个用于重构 |
| `forecaster.joblib` | 冻结的模型对象，不是可直接阅读的结果表 |
| `mc_cache/` | 按数据、配置和实现指纹保存的计算缓存 |

本次 daily 模式没有 `converter.csv` 是正常的：每天的转换器都在 mc_map.csv，不存在一张全时期共用的固定表。

补充解释几个容易误读的字段：

- `mean_error`：净对冲误差均值；越接近零越好，但并非文档唯一目标。
- `std_error`：实际样本对冲误差的标准差，不是该指标的统计标准误。
- `rmse` / `mae`：测试区间的实际净对冲误差，不是训练 beta 的误差。
- `std_improvement_vs_best_fixed`：本次和 `std_improvement_vs_alpha_one` 相同，因为训练期选中 alpha=1。
- `final_wealth`：自融资账本终值；不是无需本金的可交易收益保证。
- `mean_alpha_change`：按单元计算相邻 alpha 绝对变化再平均；shadow 没有 alpha，因此为空。
- `total_hedge_turnover`：净标的交易数量的累计，包含首次建立和最终平仓，不是期权数量换手。
- `delta_stderr`：MC 估计误差或相应基准误差，不包含 beta 预测风险；shadow_bs 为零不表示其总风险为零。

CSV 中从浮点运算得到的 level 可能出现 0.8999999999999999 等表示。按 level 做额外统计时应按配置轴归一或使用数值容差，避免将同一 0.9 单元拆成多组；回测选取曲线使用数值容差。

项目忽略 output 目录，因此本报告把核心设置和结果数字写在正文中，便于在 GitHub 阅读；原始 CSV 需在本地查看或另行分享。

## 10. 后续实验建议与可用于 Step 8 的结论

优先级建议：

1. 核实重复 refSpot 区间；确认后再决定是否需要修正数据及重建受影响的整个 pipeline，不按损益好坏删日期。
2. 相同 near-ATM book 比较一因子与两因子，隔离第二因子的增量；同数据同 MC 配置可复用缓存。
3. 预先约定非零交易成本，重算净账本；区分低成本和高成本情景，不把零成本的小改善当成可交易优势。
4. 在预选的代表日期做路径数、网格、时间步、bump 敏感性；核对异常网格位置，而不只看总体填补比例。
5. 然后再考虑 full book 与预测模型优化；不要在同一测试段上反复选因子、半衰期、期限和日期后，仍把最终结果称为独立样本外检验。

下面给出一因子对照命令，本文没有执行它；使用独立输出目录保留本次两因子结果：

```bash
python -m dynamic_alpha_hedging step7 \
  --book near_atm --factors 1 --converter daily --fast \
  --mc-cache output/dynamic_alpha/step07_near_atm_fast/mc_cache \
  --output output/dynamic_alpha/step07_near_atm_1factor_fast
```

可用于 Step 8 的当前结论：

> 在六个 near-ATM call 槽位构成的模拟 book 上，使用 daily surface beta、ATM 锚定两因子重构、冻结梯度提升预测器与逐日 Local Vol MC 转换，未平滑动态对冲相对训练期最优固定 alpha=1 的测试期误差标准差下降约 2.16%，MAE 下降约 8.92%。改善集中于 6M 和 1Y，解析 shadow delta 得到近似结果。与此同时，改善的统计区间跨零，标的换手约为固定策略的 2.30 倍，且 fast MC 网格检查和部分原始 spot 对应关系仍存在待验证事项。因此该实验支持进一步研究动态 beta 的对冲价值，但尚不足以确认稳定的实际对冲优势。

相关说明：[Step 7 框架](STEP7_GUIDE_CN.md)、[解析 Delta 与转换器对照](STEP7_DELTA_COMPARISON_CN.md)、[Step 6–7 代码详解](STEP6_STEP7_CODE_WALKTHROUGH_AND_REVIEW_CN.md)。
