# Dynamic Alpha Step 1–2 运行指南

## 运行

```bash
cd /home/ran/Huatai_intern/SVI_volatility_surface
python -m dynamic_alpha_hedging preflight
python -m dynamic_alpha_hedging step1
python -m dynamic_alpha_hedging step2
```

Step 2 只读取 Step 1 保存的 `grid_changes.csv`，不会隐式重跑 Step 1。

## Step 1：固定矩阵坐标，不固定实际合约

每日标准化曲面定义为：

```text
IV[t, i, j] = IV_t(tau[i], level[j])
K[t, i, j] = level[j] * Spot[t]
T[t, i] = observation_date[t] + tau[i]
```

固定期限网格为 `1M, 2M, 3M, 6M, 9M, 1Y, 1.5Y, 2Y`。2M 在当前数据中
几乎完整覆盖，用来补足变化最快的短端；原始较短 VolDate
先参与每日 SVI 曲面构建，再由该曲面插值得到固定1M格点。

跨日固定的是 `(tau, level)`。实际 strike 和实际 expiry 都允许滚动，因此不追踪
同一张期权。原始格点变化为：

```text
dIV_grid = IV_current(tau, K_current) - IV_previous(tau, K_previous)
```

它包含 `K = level * Spot` 沿前一天 smile 移动的机械效应。Step 1 用前一天
曲面构造无未来信息的有限变动反事实：

```text
smile_crossing_iv = IV_previous(tau, K_current) - IV_previous(tau, K_previous)
dIV_surface = dIV_grid - smile_crossing_iv
```

小变动下，`smile_crossing_iv` 约等于
`smile_slope_logK_previous * dlogS`。有限变动反事实避免一阶近似误差。

Step 1 输出：

- `raw_svi_quotes.csv`：pickle 原始 key、Spot 和逐 VolDate 的 SVI-JW 参数；
- `iv_state.csv`：每日标准化 IV 曲面，包含每天的实际 strike/expiry；
- `grid_changes.csv`：`dIV_grid`、smile crossing 和 `dIV_surface`；
- `skipped_observations.csv`：去重后仍无法构建曲面的日期及错误原因；
- `summary.csv`：三个变化分量的描述统计；
- `tenor_coverage.csv`：期限覆盖情况；
- `manifest.json`：配置、输入哈希和验证结果。

## Step 2：两个日比值，只有一个主标签

诊断用 `beta_grid_raw_daily = -dIV_grid / dlogS`；正式研究使用
`beta_surface_daily = -dIV_surface / dlogS`，其中已扣除前一曲面的 smile crossing。
默认只在 `abs(dlogS)>=0.0025`、分母非零且相邻业务日时生成日度标签。
跨缺口的 Step 1 变化保留为审计记录，不当作日度标签。

当前流程不再计算或输出 rolling regression Beta。输出包括：

- `beta_daily.csv`：两套日比值及可用性标记；
- `summary.csv`：单元统计；
- `beta_threshold_sensitivity.csv`：0.1%/0.25%/0.5%/1% 门槛敏感性；
- `beta_reasonableness.csv`：daily Beta 分布；
- `beta_regime_checks.csv`：涨跌、高低波状态下的 daily 比值统计；
- `beta_term_structure.csv`：ATM daily Beta 期限统计。

```bash
python -m dynamic_alpha_hedging step2 --min-abs-dlogS 0.0025
```

`--window`、`--min-obs` 和 Step 5/6 的 `--rolling-beta` 参数已删除。
完整模型/基准规则见 [DAILY_ONLY_PIPELINE_CN.md](DAILY_ONLY_PIPELINE_CN.md)。

## 日期和输入约定

默认输入是 `data/svi_param.pkl`。key 已经代表美国市场日期，只允许周一至周五；
date-only key 不做统一平移，周末 key 会被拒绝。完整 timestamp 才会从
`Asia/Shanghai` 转换到 `America/New_York`。

当前新文件的 672 个观测日期全部满足星期约定。重复 VolDate 按原始数组顺序
保留第一条，`raw_svi_quotes.csv` 用 `vol_date_occurrence` 和
`kept_by_first_duplicate_policy` 保留审计痕迹。去重后仍无法通过 SVI-JW 转换的
少量日期会写入 `skipped_observations.csv` 并从矩阵中排除。
