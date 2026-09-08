# Step 7：动态 Alpha 回测框架

新增的解析 delta 对照、训练期多日期转换器、MC 一致性检查和已完成的缓存复算，
见 [Delta 与转换器对照](STEP7_DELTA_COMPARISON_CN.md)。下文开头的“未运行”描述为框架初版状态，
不代表目前没有回测输出。

本版完成可运行框架、真实输入准备检查和小样本测试，**未运行全历史正式 MC 回测，也尚无对冲改善结论**。原始依据为《动态Alpha对冲研究》Step 7.1–7.4；此前审查见 [实施方案](STEP7_AUDIT_AND_PLAN_CN.md)。

## 1. 一套引擎，三种组合

| `--book` | 每天开仓的欧式 call 槽位 | 默认因子数 |
|---|---|---:|
| `atm` | 3M × level 1.0 | 1 |
| `near_atm` | 3M、6M、1Y × level 0.9、1.0 | 2 |
| `full` | 当前 Step 4 的 7 个期限 × 8 个 level，共 56 个 | 2 |

`full` 是当前保留的建模网格，不含已排除的 1M 和 level 1.2。可用 `--factors 3` 比较三因子；不能根据测试表现反复调参再宣称独立样本外优胜。

默认每个槽位持有一份 call。`--weights equal_vega` 将开仓时总 vega 预算设为 100，各槽位等分；单个持有区间内数量冻结。它不是实际 desk 的持仓，也不代表已实现的市场成交。

## 2. 每天实际做什么

1. 收盘 t：用冻结的 Step 6 HGB 模型预测下一日因子。特征只到 t，历史缺失因子只向前填充，不要求下一日 Beta 标签存在。
2. 按 Step 4 已拟合的载荷，在当前槽位计算预测 Beta：

$$
\widehat\beta_{t+1}(\tau,m)=c(\tau,m)+\sum_{k=1}^{p}\widehat z_{k,t+1}f_k(\tau,m).
$$

3. 默认用 t 日曲面测量五档 Alpha 对应的 Beta 和 Delta，反查预测 Beta 得到 Alpha；另生成 EMA 平滑 Alpha。
4. 开仓 strike 为 $K=mS_t$，expiry 由 t 日固定期限换算。持有到下一收盘时，**K、实际到期日和数量都不变**。用下一日原始 SVI 曲面重估同一份合约。
5. 各策略持有相同的期权组合，只改变标的对冲量；下一日按照同样规则换入新合约，现金账户承接换仓与融资。

Step 1 的标签比较固定剩余期限的曲面变化；实际持仓则会老化。因此标签不是持仓价格变化的完整分解，本版单独记录时间流逝和期限曲线滚动项，不把 Beta 拟合改善当作对冲改善。

当前真实输入准备检查：训练标签截止 2026-01-15；494 个连续业务日训练区间，156 个测试区间。训练区间若合约超出报价期限范围，明确排除并记录；测试组合不完整则中止，不按事后收益删除日期。原始相邻观测中的非连续业务日另行记录；测试现金账遇到断档要求显式拆段，不跨越缺失日期假装完成每日对冲。

## 3. Step 3 转换器：不再只依赖一天

默认 `--converter daily`，用当天曲面复用 Step 3 方法，不读取旧单日 inverse：

$$
B_t^{raw}(a)=-\frac{IV_t^{up}(a)-IV_t^{down}(a)}{\log(S_t^{up}/S_t^{down})},
\qquad
\Delta_t(a)=\frac{PV_t^{up}(a)-PV_t^{down}(a)}{S_t^{up}-S_t^{down}}.
$$

同期限的全部 strike 共用路径，一对 up/down 路径同时提供 Beta 和 Delta。即使 IV 反解选用 put，输出也通过 put-call parity 转换为 call Delta。调用只传 `spot_adj=log(S_bump/refSpot)`，Alpha 在核心 Dupire 代码里乘一次。

沿用现有 Step 3 的处理：`beta_converter = beta_model - beta_model(alpha=1)`。**归零后的 Beta=0 是锚定约定，不是原始 MC 精度通过的证明**。原始锚点偏差和标准误仍保留；没有单调投影，也没有用历史 Beta 强行拟合曲线。

预测 Beta 超出曲线范围时截断到端点；曲线不能严格单调反解时 Alpha 回退到 1，明确输出标志。有限 MC Delta 不可用时回退到 BS Delta。其他构建/输入异常显式中止，不悄悄删掉测试日。质量检查不抹掉结果；必须结合 `quality_pass`、回退率、网格异常比例解读。

平滑后的 Alpha 从当天 Delta 节点表线性插值，并非为每个预测 Alpha 再跑 MC。原始/修复 SVI 的 IV 差异记录为 `repair_iv_change`；实际历史估值始终使用原始曲面。

### 转换器精度为什么影响对冲

局部可近似写成：

$$
\delta\alpha\approx\frac{\delta\beta}{\partial B/\partial\alpha},
\qquad
\delta\Delta\approx\frac{\partial\Delta}{\partial\alpha}\delta\alpha,
\qquad
\delta\mathrm{error}\approx-\delta\Delta\,\Delta S.
$$

Beta–Alpha 曲线越平，逆解越敏感；MC 噪声、粗网格、bump 大小、Delta 插值误差和参考曲面状态失配都可能传导到损益。更高路径数只能降低随机误差，不能自动消除网格偏差和模型误差。

正式解释结果前，应在**训练期预选**的普通/高波日期比较路径数、substeps、ratio nodes、spot bump，并用目标 Alpha 直接 MC 检查节点插值误差。逐日转换器解决了单日状态代表性问题，但不等于数值收敛已获认证。

`--converter fixed --reference-inverse ...` 支持单张训练期曲面的 inverse 作为状态敏感性对照。只固定 Beta→Alpha，Delta 仍在当天计算。代码检查参考日期不得晚于训练截止日，但 CSV 无法证明代表日的选择没有用未来信息；选择过程需自行保证只使用训练期。旧 Step 3 自动 medoid 使用全历史，不能直接当作严格无泄漏基准。

## 4. 对照组、平滑与组合账本

包含固定 Alpha=0/0.5/1/1.5/2、`best_fixed_train`、`bs_delta`、`last_observed_factor`、`rolling_alpha_mean`、`dynamic_raw` 和 `dynamic_ema`。

- 最好固定档：按训练期相同 book 的含 carry、未扣交易成本误差标准差选取，测试期冻结。不用测试期挑“最好档”。
- `last_observed_factor`：用 close-t 最近已知 daily 因子预测下一日，和动态策略共用因子数、载荷、转换器及当日 Delta 表。当天因子缺失时的回退规则见 [daily-only 说明](DAILY_ONLY_PIPELINE_CN.md)。
- `rolling_alpha_mean`：过去 20 个有效已实现 Daily Beta 在各自当时曲面反查的 Alpha 均值，不把同一个旧观测重复算多次；无历史时为 1。
- EMA：$\alpha_t^{EMA}=\lambda\alpha_{t-1}^{EMA}+(1-\lambda)\alpha_t^{raw}$，$\lambda=2^{-1/H}$；默认 H=10 个交易日，初值为训练最好固定档。H=0 表示不平滑。5/20 日仅作预设敏感性，不能拿测试集选最优 H。

Book 仅对冲净 Delta：$D_t=\sum_i q_{i,t}\Delta_{i,t}$。标的成本按净交易量 $|D_t-D_{t-1}|S_t$ 计算，不把单笔交易量绝对值相加；包括首次建立和最后平仓。Alpha 变化量与实际标的 turnover 分别报告。

文档原始误差是 $\Delta V-D_t\Delta S$。含融资的毛误差为：

$$
e_t^{gross}=\Delta V-D_t\Delta S+(D_tS_t-V_t)(e^{r\Delta t}-1)
-D_tS_t(e^{q_{eff}\Delta t}-1).
$$

其中 $q_{eff}=r-b$，在当前代码中为 dividend+repo；利息用 Act/365。净误差由现金账户直接计算，含净标的交易成本及其融资影响。每日滚动卖出旧期权、买入新期权的现金流由重设现金余额体现。

当前成本是标的单边交易金额的假设 bps，默认 0；未建模期权 bid/ask、真实滑点、期货基差和实际保证金。输入利率/股息也是现有研究假设。结果不能称为可交易的净收益保证。

## 5. 归因线与指标

为避免重复计算，归因使用 BS Delta，而不是“动态 Local Vol Delta + 完整 vega spot-vol 项”：

$$
\Delta V\approx\Delta_{BS}\Delta S+\tfrac12\Gamma(\Delta S)^2+
\mathrm{time\_pnl}+\nu\,\mathrm{term\_roll\_iv}
-\nu\,\widehat\beta\,\Delta\log S+\varepsilon.
$$

`forecast_attribution_residual` 使用原始预测 Beta，对应文档复用预测模型的归因要求；`attribution_residual` 使用实际实施 Alpha 在转换器上的 Beta，反映平滑/截断的影响。它们与真实对冲 P&L 分开。

主指标为相同日期、相同 book 的 `std_error`。`std_improvement_vs_best_fixed=1-动态净误差std/训练最好固定档净误差std`，正值才是改进。同时报 Alpha=1 对照、RMSE/MAE、绝对误差尾部、成本、净持仓 turnover、Alpha 平滑度及回退比例。

配对区块 bootstrap 使用同一组日期索引重采样策略与基准，10 个观测为块、1000 次，报告 std 改善与归因 RMSE 改善的 95% 区间；少于 20 天不报区间。它只是当前研究样本的不确定性描述，不能修复反复查看测试集造成的选择偏差。

## 6. 命令

在项目根目录、已激活 `.venv` 的终端运行。准备阶段只校验输入链、拟合原 Step 6 模型并保存全日期预测，**不运行 MC**：

```bash
python -m dynamic_alpha_hedging step7 --prepare
```

开始 3M ATM 正式回测：

```bash
python -m dynamic_alpha_hedging step7
```

小型 book / 全保留曲面使用同一引擎，分别输出以保留结果：

```bash
python -m dynamic_alpha_hedging step7 --book near_atm --factors 2 --output output/dynamic_alpha/step07_near_atm
python -m dynamic_alpha_hedging step7 --book full --factors 2 --output output/dynamic_alpha/step07_full
```

正式默认 100000 路径、2 substeps、801 ratio nodes、1% spot bump。`--fast` 是 10000 路径、201 nodes 的运行检查，不作为正式精度。`--paths`、`--substeps`、`--ratio-nodes`、`--spot-bump-fraction` 可做收敛敏感性。`--half-life` 和 `--cost-bps` 分别控制平滑与成本。

首次 ATM 理论上限为 `(494+156) × 1期限 × 5Alpha = 3250 对` up/down MC，另有不支持的训练合约会明确排除；仅测试期为 780 对。全曲面期限数乘 7，不再乘 strike 数或策略数。它明显比原单日 Step 3 更耗时。

每个日期/期限的结果存入输出目录的 `mc_cache/`，key 包含曲面、数值配置和定价实现哈希。中断后同命令重跑复用缓存；仅改变策略平滑/成本也可复用同目录缓存。修改相关定价实现或数值设置后自动失效。终端逐日打印训练/测试进度。

## 7. 输出和代码入口

| 输出 | 先看什么 |
|---|---|
| `summary.csv` | book 级 std 是否胜过最好固定档；区间、成本、回退与归因改善 |
| `book_pnl.csv` | 每日净 Delta、现金、误差、成本和累计财富；逐日排查 |
| `signals.csv` | close-t 因子/Beta、转换器日期、raw/实施 Alpha、截断和回退 |
| `mc_map.csv` | 测试期逐日 Beta/Delta 节点、标准误、原始 Alpha=1 偏差及质量 |
| `option_pnl.csv` | 每个实际 K/expiry 的开平估值、时钟、Greeks、归因项及误差 |
| `cell_summary.csv` | 各期限×level 的单份合约毛误差诊断；不分摊 book 净成本 |
| `factor_forecasts.csv`、`forecaster.joblib` | 全日期预测与冻结预测器；不要加载不可信 joblib |
| `manifest.json` | 实际完成区间、最好固定档、排除、输入哈希和研究假设 |

仅 `--prepare` 时输出 `preparation.json`、预测 CSV 和模型，不产生正式回测 manifest。训练 MC 节点可在缓存中审计。首版以 CSV 为主，不额外生成重复图表。

实现分工：[step06.py](../dynamic_alpha_hedging/step06.py) 提供共享预测器；[hedging.py](../dynamic_alpha_hedging/hedging.py) 负责固定合约估值、联合 MC 表及 inverse；[step07.py](../dynamic_alpha_hedging/step07.py) 负责策略、现金账和统计；[montecarlo.py](../svi_localvol/montecarlo.py) 仍是共用定价核心。旧 Step 1–6 输出未覆盖。

## 2026-09-08 接口更新

独立预计算、只读 MC 库、模型选择、三个快照排除和分段回测的完整参数与命令见 [MC_PRECOMPUTE_CN.md](MC_PRECOMPUTE_CN.md)。历史结果文件尚未按新排除口径重跑。

当前输入特征数为 13，回归 rolling Beta 已完全移除。摘要同时比较最好固定 Alpha 与 `last_observed_factor` 对冲误差；历史报告仍是旧版本结果。
