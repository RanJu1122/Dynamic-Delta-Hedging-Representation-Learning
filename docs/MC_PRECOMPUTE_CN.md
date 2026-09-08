# 历史 MC 预计算与 Step 7 复用接口

更新：2026-09-08。本次完成接口、异常日期与断档处理及模拟测试；没有启动真实历史 MC 预计算或重跑市场回测。已有 Step 7 报告仍对应旧数据口径。

当前已进一步改为 **daily-only**，rolling Beta 的估计、模型输入和对照均已移除。变更与原文要求核对见 [DAILY_ONLY_PIPELINE_CN.md](DAILY_ONLY_PIPELINE_CN.md)。

## 1. 工作流与依赖

```text
原始 svi_param.pkl → 当天 SVI 曲面 → precompute → 独立 MC library
                └→ Step 1 → 2 → 4 → 5 → 6 → Step 7 ← 只读 library
```

预计算不读取 Beta 标签、因子、预测器或未来价格。每个分片是一个 `日期 × tenor`，保存全部预设 `level × alpha` 节点的 MC 结果。每个期限内部的不同 strike 共用路径；不同预测策略不会增加预计算次数。

正式预计算的入口是 `python -m dynamic_alpha_hedging precompute`。该命令会实际运行 MC；本文仅列出命令，没有代为执行。

## 2. 默认参数

| 参数 | 默认值 |
|---|---|
| 数据 | `data/svi_param.pkl` |
| 输出库 | `output/dynamic_alpha/mc_library` |
| 日期 | 全部能够构建曲面且未被排除的快照 |
| tenors，Business/260 年 | `1/6, 0.25, 0.5, 0.75, 1, 1.5, 2`，共 7 档 |
| levels，K/refSpot | `0.4, 0.5, …, 1.1`，共 8 档 |
| Alpha | `0, 0.5, 1, 1.5, 2` |
| 路径数 | `100000` |
| 随机种子 | `20260807` |
| 对偶采样 | 开启；路径数包含全部正负路径 |
| 每业务日子步数 | `2` |
| spot bump | `±1%`，固定 K 和实际到期日 |
| spot-ratio 网格 | `801` 节点，范围 `[0.001, 3]` |
| local vol 下限/上限 | `0 / 5` |
| Beta clamp | `0` |
| 画图日期 | 有效快照序列正中间连续 `3` 个，可选 `5` 个 |
| 利率/dividend/repo | `0.036 / 0.03 / 0`，沿用研究配置 |
| 时间/日历 | 波动率 Business/260、利息 Act/365，沿用配置内 US holiday 列表 |

默认定价轴覆盖当前 full book 的 56 个单元，也覆盖 ATM 和 near-ATM 两种 book。1M 与 level=1.2 不在默认库中，如有需要必须在首次预计算时明确加入。期限超出当天报价范围会记录为 unsupported，不外推。

数值参数可通过 `--paths`、`--seed`、`--substeps`、`--ratio-nodes`、`--ratio-min`、`--ratio-max`、`--vol-floor`、`--vol-cap`、`--spot-bump-fraction`、`--alphas`、`--no-antithetic` 指定。`--fast` 会将路径数/ratio 节点覆盖为 `10000/201`，用于开发检查；应使用独立输出库，不能覆盖正式库。

只写任务计划和覆盖报告，不运行 MC：

```bash
python -m dynamic_alpha_hedging precompute --plan-only --plot-dates 5
```

未来实际启动默认预计算：

```bash
python -m dynamic_alpha_hedging precompute --plot-dates 5
```

同样的命令再次执行会校验并复用已完成分片，只计算缺少的任务。修改数值设置、定价引擎或扩大库轴时应选择新的 `--output`，不能混用已有库。当前接口不提供自动合并旧 `mc_cache` 或自动修复损坏分片。

## 3. 排除哪些日期，如何处理缺口

新增排除仅限下面三个疑似异常快照：

| 日期 | 原因 |
|---|---|
| 2026-03-31 | Spot 仍为 6368.85，但 SVI 参数变化，疑似陈旧参考价格 |
| 2026-04-01 | 同上 |
| 2026-04-02 | 同上 |

保留 2026-03-30 和 2026-04-06。没有修改原始 pickle、没有人工补价，也没有新增“相同 spot 自动删除”的规则。这是事先声明的疑似异常排除，不表示已经核实 Spot 字段应当是每日成交收盘价。

原来无法构建曲面的三天（2023-12-12、2024-08-05、2024-08-06）继续按既有规则跳过；它们不是本次新增排除。因此当前原始数据预期总共跳过 6 个快照。原始导出保留三天记录和原因字段。

- Step 1 保留跨缺口变化作为审计行，并标记不是连续业务日；它不被当作日度标签。
- Step 2 仅计算日度 Beta 并排除跨缺口区间；不再计算 rolling Beta，不存在 20/60 个回归样本的预热要求。
- Step 5 的 5/20 日收益、RV、IV 变化及 vol-of-vol 跨缺口时缺失，待完整窗口恢复；下一日因子标签不跨缺口。
- Step 6 的 last-factor 前向填充在缺口处重置。Ridge 的缺失值填充器仅在训练样本拟合，HGB 使用自身缺失值处理；不从未来补值。
- Step 7 的前一段在最后有效快照平仓，下一段从首个恢复快照建仓。各段分别处理净标的开仓/平仓费用，现金余额和 Alpha 平滑重置，不计算缺失期间的 P&L 或利息。总 wealth 是各段已实现 wealth 的加总，不能解释为缺失期间仍持续交易的实际账户。

因此不会把 3/30→4/6 当作一天；恢复后的第一个有效持有区间是 4/6→4/7。回测分段只由可用日期确定，不由收益或 P&L 选择。统计中的区块抽样也限制在各段内。

旧 Step 1–6 产物需要重新生成。为保留旧实验，应指定独立目录，并保持全部上下游路径一致。例如未来可按顺序运行：

```bash
RESEARCH_OUT=output/dynamic_alpha_excluded
python -m dynamic_alpha_hedging step1 --output "$RESEARCH_OUT/step01"
python -m dynamic_alpha_hedging step2 --input "$RESEARCH_OUT/step01/grid_changes.csv" --output "$RESEARCH_OUT/step02"
python -m dynamic_alpha_hedging step4 --input "$RESEARCH_OUT/step02/beta_daily.csv" --output "$RESEARCH_OUT/step04"
python -m dynamic_alpha_hedging step5 \
  --factors "$RESEARCH_OUT/step04/factor_scores.csv" --loadings "$RESEARCH_OUT/step04/factor_loadings.csv" \
  --iv-state "$RESEARCH_OUT/step01/iv_state.csv" --changes "$RESEARCH_OUT/step01/grid_changes.csv" \
  --daily-beta "$RESEARCH_OUT/step02/beta_daily.csv" \
  --output "$RESEARCH_OUT/step05"
python -m dynamic_alpha_hedging step6 \
  --panel "$RESEARCH_OUT/step05/factor_state_panel.csv" --loadings "$RESEARCH_OUT/step04/factor_loadings.csv" \
  --daily-beta "$RESEARCH_OUT/step02/beta_daily.csv" \
  --output "$RESEARCH_OUT/step06"
```

## 4. Step 7 如何复用同一库

指定 `--mc-library` 后，Step 7 强制使用库内路径数、种子、子步数、网格、bump 和 Alpha 节点；缺分片、内容损坏、当天曲面或引擎改变均报错，不自动启动 MC。不能同时传 `--mc-cache` 或 `--fast/--paths/--substeps/--ratio-nodes/--spot-bump-fraction`。未指定 `--mc-library` 时仍保留旧的“缓存未命中就计算”行为。

以下命令是未来库与上游数据准备好后执行的回测示例，本次未执行：

```bash
python -m dynamic_alpha_hedging step7 \
  --input-root output/dynamic_alpha_excluded \
  --mc-library output/dynamic_alpha/mc_library \
  --book near_atm --factors 2 --model hist_gradient_boosting \
  --model-params '{"max_iter":300,"learning_rate":0.03}' \
  --output output/dynamic_alpha_excluded/step07_hgb

python -m dynamic_alpha_hedging step7 \
  --input-root output/dynamic_alpha_excluded \
  --mc-library output/dynamic_alpha/mc_library \
  --book near_atm --factors 1 --model ridge --model-params '{"alpha":1.0}' \
  --half-life 5 --cost-bps 1 \
  --output output/dynamic_alpha_excluded/step07_ridge
```

支持的预测器：

| 模型 | 默认和允许参数 |
|---|---|
| `hist_gradient_boosting` | 默认 `loss=absolute_error, learning_rate=0.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=15, l2_regularization=1, random_state=20260807`。可覆盖这些字段及 `max_depth`；固定 `early_stopping=False`。 |
| `ridge` | `alpha=1`；训练期中位数填充缺失值并标准化。 |
| `training_mean` | 各因子训练期均值；不接受额外参数。 |

这些模型使用 Step 6 的 13 个 daily-only 特征及对应训练/测试划分，分别预测三个因子。模型选择和显式参数保存在 Step 7 manifest，实际拟合后的模型保存在 `forecaster.joblib`。这里提供复用接口，没有根据测试期表现自动选择最优模型。

| 修改内容 | MC 库能否复用 |
|---|---|
| HGB/Ridge/训练均值模型及超参数 | 能 |
| 因子数 1/2/3、Alpha 平滑、成本、权重 | 能 |
| daily/fixed/pooled 转换器 | 能；fixed 仍需提供参考 inverse，Delta 始终来自当天库 |
| ATM/near-ATM/full book | 能，前提是期限和 level 为库内精确子集且当天报价支持 |
| Daily Beta 门槛、特征、训练设置 | 能复用定价，但需要重新生成对应模型输入产物并通过一致性校验 |
| 质量阈值 | 能；读取时重新计算质量标记，不更改 MC 数值 |
| 新期限/strike、MC 精度、定价引擎或当日曲面/市场参数 | 需要相匹配的定价结果；当前库不自动扩展或覆盖 |

Python 层仍可通过 `prepare_step7` 返回的 `Step7Inputs.forecasts` 对接额外研究预测器，须保留 `prepare_step7` 生成的三个 `last_*` 基准预测及 `naive_uses_close_t_factors` 标记，并提供每个决策日期的三个模型因子预测，自行保证训练/特征仅使用当时可用信息。CLI 目前支持上表三种模型，没有实现任意模型导入。

## 5. Beta 锚点与归因必须同步解释

对每个日期、期限和 level，保留：

\[
B_{raw}(\alpha)=-\frac{IV^{up}-IV^{down}}{\log(S^{up}/S^{down})},\quad
B_{centered}(\alpha)=B_{raw}(\alpha)-B_{raw}(1).
\]

`beta_model` 是原始值，`beta_converter` 是归零值。归零只用于转换器定义，不调整 MC call Delta，也不改实际期权估值、对冲量或 P&L。原始 `B_raw(1)` 继续用于审计，不能拿归零后的 0 证明 MC 无偏。

Step 7 的模型表策略在 Alpha 截断/平滑后，用当天曲线计算实际实施的 `effective_beta`；即使采用 fixed/pooled inverse，也使用当天表做这项实施归因。同时保存原始 `raw_model_beta`。以 BS Delta 为共同基准做归因，不把完整 Beta 项再次叠加到动态 Delta 对冲误差上。

令 `common_pnl` 为 BS Delta、Gamma、时间流逝和期限滚动项之和，则：

\[
R_{centered}=dV-common\_pnl+vega\,B_{centered}\,d\log S,
\]
\[
R_{raw}=dV-common\_pnl+vega\,B_{raw}\,d\log S,
\]
\[
R_{raw}=R_{centered}+anchor\_correction\_pnl.
\]

输出 `attribution_residual`、`raw_model_attribution_residual`、`anchor_correction_pnl` 可直接核对这一恒等式。它们是两个模型解释口径之间的差异，不能把锚点修正项当成实际赚取的 P&L。shadow/解析 BS 策略并非原始 MC 曲线的 Alpha 节点，无法定义的 raw-model 字段保留 NaN。

## 6. 产物、运行状态与图形

- `plan.json`、`coverage.csv`：任务数、报价覆盖和日期计划；`--plan-only` 不创建新定价库。
- `raw_svi_quotes.csv`、`excluded_observations.csv`：原始报价和排除原因。
- `library.json`：定价引擎指纹、数值配置和轴。
- `index.json`、`shards/*.csv`：每个日期/期限分片及其曲面/内容哈希。
- `quality.csv`、`alpha_one_audit.csv`：六项质量审计和原始 beta(1)。失败只标记，不筛掉曲线。
- `plot_dates.csv`、`plots/`：全历史有效日期正中间连续 3/5 个快照。每个日期绘制 ATM 的 Alpha×tenor×Beta 切片、3M 的 Alpha×level×Beta 切片，以及所有预计算 tenor 的 Alpha×level×Beta 分面图。每张图并列 raw/centered，标出 Alpha=1 和 Beta=0；同时保存该日期完整表格。这里有 Alpha、tenor、level、Beta 四个维度，因此采用多张三维切片完整展示。
- `manifest.json`：`running/complete/failed/interrupted`、已完成/复用/新算任务数、失败任务及错误。普通失败后可重跑同一命令续算；进程被强制杀死时状态可能停在 running，但已落盘分片仍可校验复用。

同一目录限制一个预计算写进程；不兼容配置在写报告前拒绝。没有任何有效任务会直接报错，不标记 complete。数值质量通过率、收敛性以及最终对冲优势仍需在实际计算后验证，本次模拟测试不能替代这些验证。
