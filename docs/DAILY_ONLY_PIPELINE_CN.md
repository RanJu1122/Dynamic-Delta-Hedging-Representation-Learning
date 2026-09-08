# Daily-only Beta 流程与朴素预测基准

更新：2026-09-08。此文描述当前代码；旧实验报告中的 rolling Beta 指标属于历史版本。本次仅修改代码、文档与合成测试，没有重新训练历史模型、计算历史 MC 或运行历史对冲回测。

## 原始文档要求与本次研究选择

重新核对了 `docs/source/动态Alpha对冲研究.docx`：

- 开头说明方法不限定，允许根据实验选择实现方法。
- Step 2 建议以回归弹性 (B) 为主、daily realized 比值 (A) 作对照；这是原始建议，不能改写为原文要求 daily-only。
- Step 6 明确写出朴素预测 `z(t+1)=z(t)`，模型应赢过它才能说明有额外预测信息。
- Step 7 要求固定 Alpha、训练内最优固定 Alpha、Alpha 滚动均值、平滑约束及 book 级对冲检验。

用户根据之前的实验选择完全移除 rolling regression Beta。本版采用 daily-only，这是明确记录的研究选择；保留与 `z(t)` 的预测比较及 Step 7 必需的 Alpha 对照。删除一个弱基准不能证明模型更好，结论仍取决于相同样本上是否赢过朴素预测，以及实际对冲是否赢过最好固定 Alpha。

## 当前数据链

```text
当日曲面变化 → daily Beta → Step 4 daily Beta 因子
                         → Step 5 逐单元 daily Beta 预测
                         → Step 6 下一日 daily 因子预测
                         → Step 7 Beta → Alpha → 当日 MC Delta → 对冲与归因
独立 MC precompute ────────────────────────────────┘
```

所有预测特征截止于收盘 t，目标对应下一有效单日区间的 t+1。RV/收益等市场状态仍可用滚动窗口；它们不是 rolling regression Beta，不在本次删除范围内。

| 阶段 | 现在的行为 |
|---|---|
| Step 2 | 只计算 `beta_grid_raw_daily` 与 `beta_surface_daily`，保留阈值、分布、涨跌/波动状态与期限诊断；不再估计回归 Beta 或输出 `beta_rolling*.csv`。 |
| Step 4 | 使用完整 daily Beta 曲面拟合 ATM 锚定因子，保持原来的截面模型与训练期拟合原则。 |
| Step 5 | 11 个输入：10 个市场状态 + 1 个最近已知 daily Beta。比较 sticky-strike、训练均值、daily persistence、Ridge、HGB。 |
| Step 6 | 13 个输入：10 个市场状态 + 3 个最近已知 daily Beta 因子；移除了原来的 3 个 rolling 因子。不再读取 rolling Beta 文件。 |
| Step 7 | 移除 `rolling_beta` 策略，加入 `last_observed_factor` 对冲策略；与模型策略共用因子数、载荷、转换器、当日 MC 表、合约及日期。 |
| MC precompute | 不变；缓存从来不依赖 Beta 估计方法或预测模型，因此有效的独立库可以继续复用。 |

`rolling_alpha_mean` 保留：这是过去有效 daily Beta 在各自当日反查所得 Alpha 的均值，没有使用 rolling Beta 回归。`dynamic_ema` 也保留，它对预测 Alpha 做平滑。每日更新合约的 rolling book 同样与 rolling Beta 无关。

## “前一天预测”与缺失值的明确区别

目标是用 close-t 已知的 `beta_t` 或 `z_t` 预测 t+1。当天值有效时，persistence 就严格等于它。

Daily Beta 在 `abs(dlogS)<0.0025`、零收益或无效变化时缺失；不能用未来数据补值。当前规则：

1. 优先使用 close-t 的有效 daily Beta/因子。
2. 当天无效时，只在同一连续数据区间内使用最近有效值。
3. 缺口后没有任何有效历史值时，朴素预测回退到训练期目标均值；模型特征保留缺失，由训练期拟合的缺失处理或 HGB 处理。

所以基准名称使用 `last_observed_beta` / `last_observed_factor`，不把所有预测都称作“严格前一天”。

- Step 5 的逐笔预测保存 `last_daily_beta` 和 `last_daily_beta_date`。
- Step 6 保存 `naive_uses_close_t_factor`、`naive_training_mean_fallback`，摘要同时报告对应比例。
- Step 7 保存 `naive_beta` 和 `naive_uses_close_t_factors`；朴素策略与动态策略同样覆盖可交易日，不根据 t+1 标签是否有效来选择回测日期。

日期排除和分段回测规则保持：新增排除仅为 2026-03-31、2026-04-01、2026-04-02，不补价，不把 3/30→4/6 当作一天。

## 现在与谁比较，怎么看结果

以前，Step 5/6 既预测 daily Beta，又把 rolling Beta 当作特征和比较基准；Step 5 的可预测性判据也依赖是否超过 rolling Beta。旧代码不会将两者预测再平均，也不会根据测试结果自动切换预测，但 rolling Beta 确实参与了模型输入和评估。

现在改为：

- Step 5：`dIV_rmse_improvement_vs_last_observed_beta`。可预测性判据要求模型相对训练均值 OOS R² 为正，并且 dIV RMSE 小于 daily persistence。该判据是报告字段，不会自动阻止研究者运行 Step 6。
- Step 6 因子级：`oos_r_squared_vs_last_observed_factor`、`rmse_improvement_vs_last_observed_factor`。
- Step 6 曲面级：`dIV_rmse_improvement_vs_last_factor`；一因子模型比较一因子 persistence，两因子比较两因子，三因子比较三因子。不能让因子数量不同造成不公平比较。
- Step 7：除最好固定 Alpha 外，新增 `std_improvement_vs_last_observed_factor` 及其配对置信区间。它衡量实际 hedge error，而不是预测 RMSE。

改善指标一般为 `1 - 模型误差 / 基准误差`。大于零表示更好，小于零表示更差。基准误差为零时，部分指标记为 NaN，不能解释为模型有改善。没有自动从测试结果中挑模型或替换策略。

## 迁移命令及旧结果

之前命令中的以下参数已删除，继续传入会明确报错：

- Step 2：`--window`、`--min-obs`。
- Step 5/6：`--rolling-beta`。

若已有同一日期排除口径的 Step 1，可以直接从 Step 2 更新；Step 4 需要刷新输入来源记录，Step 5/6 必须重新生成并拟合。Step 3 与独立 MC 库不需要因为这次删除 rolling Beta 而重算。旧 Step 5/6 模型含不同特征，不能直接当成新版本模型。

在项目目录中设置 `RUN_DIR` 为已有 Step 1 的新实验目录后，执行：

```bash
PY="$PWD/.venv/bin/python"
RUN_DIR="$PWD/output/rebuild_20260908_excluded3"

"$PY" -m dynamic_alpha_hedging step2 \
  --input "$RUN_DIR/step01/grid_changes.csv" --min-abs-dlogS 0.0025 \
  --output "$RUN_DIR/step02"

"$PY" -m dynamic_alpha_hedging step4 \
  --input "$RUN_DIR/step02/beta_daily.csv" --output "$RUN_DIR/step04"

"$PY" -m dynamic_alpha_hedging step5 \
  --factors "$RUN_DIR/step04/factor_scores.csv" --loadings "$RUN_DIR/step04/factor_loadings.csv" \
  --iv-state "$RUN_DIR/step01/iv_state.csv" --changes "$RUN_DIR/step01/grid_changes.csv" \
  --daily-beta "$RUN_DIR/step02/beta_daily.csv" --output "$RUN_DIR/step05"

"$PY" -m dynamic_alpha_hedging step6 \
  --panel "$RUN_DIR/step05/factor_state_panel.csv" --loadings "$RUN_DIR/step04/factor_loadings.csv" \
  --daily-beta "$RUN_DIR/step02/beta_daily.csv" --output "$RUN_DIR/step06"
```

这些命令会写入指定目录；如需保留该目录内现有 Step 2–6 实验，请先选另一个实验目录并准备对应 Step 1。代码检查 `beta_workflow=daily_only_v1`，防止 Step 7 混用旧的 rolling 特征模型。历史 CSV/报告不因源码修改自动更新；原始数据和已有 MC 库也没有被改写。
