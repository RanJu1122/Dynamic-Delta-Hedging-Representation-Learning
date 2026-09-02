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

## Step 2：两个 beta，只有一个主标签

诊断用原始格点 beta：

```text
beta_grid_raw = -dIV_grid / dlogS
```

Step 3–7 使用的正式 beta：

```text
beta_surface = -dIV_surface / dlogS
dIV_surface = intercept - beta_surface_rolling * dlogS + residual
```

`beta_daily.csv` 保存两个日比值；只有 `|dlogS| >= 0.005` 时计算。
`beta_rolling.csv` 保存两个 trailing OLS 结果及 intercept、R²、slope standard
error、nobs。`beta_surface` 是正式研究口径，其 Step 3 sanity check 是
`alpha=1 -> beta_surface ~= 0`。

Step 1 保留并标记相邻有效曲面之间跨越多个交易日的转移；Step 2 的默认日频
估计只使用 `is_next_business_observation=True` 的转移，避免把2至5日累计变化
混入一日 beta。Rolling window 表示最近60个有效曲面变化观测，不是强制60个
日历日；只有当前格点本身有效时才输出 beta。这个约定允许1M使用其真实覆盖
样本，同时绝不对原始期限范围以外的1M曲面做静默外推。

Step 2 还保存文档要求的完整诊断：

- `beta_threshold_sensitivity.csv`：`|dlogS|` 阈值0.25%/0.5%/1%；
- `beta_rolling_sensitivity.csv`：20/40/60/120窗口和等权、`|dlogS|`加权、
  20日半衰期时间衰减的完整时序；
- `beta_rolling_sensitivity_summary.csv`：上述平滑选择的汇总比较；
- `beta_reasonableness.csv`：范围、分位数、正beta比例；
- `beta_regime_checks.csv`：上涨/下跌、高ATM IV/低ATM IV条件回归；
- `beta_term_structure.csv`：ATM beta短端到长端的期限结构。

这些检查以 `beta_surface` 为主；`beta_grid_raw` 仍完整保存在标准daily和rolling
文件中，供后续步骤统一做坐标口径敏感性。

```bash
python -m dynamic_alpha_hedging step2 --window 60 --min-obs 20 \
  --min-abs-dlogS 0.005
```

## 日期和输入约定

默认输入是 `data/svi_param.pkl`。key 已经代表美国市场日期，只允许周一至周五；
date-only key 不做统一平移，周末 key 会被拒绝。完整 timestamp 才会从
`Asia/Shanghai` 转换到 `America/New_York`。

当前新文件的 672 个观测日期全部满足星期约定。重复 VolDate 按原始数组顺序
保留第一条，`raw_svi_quotes.csv` 用 `vol_date_occurrence` 和
`kept_by_first_duplicate_policy` 保留审计痕迹。去重后仍无法通过 SVI-JW 转换的
少量日期会写入 `skipped_observations.csv` 并从矩阵中排除。
