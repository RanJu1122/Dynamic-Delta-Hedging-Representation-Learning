# Dynamic Alpha Step 3

Step 3 用一张固定、calendar-repaired 的代表性 SVI 曲面，正式测量
`Beta → Alpha` 转换器：

```text
给定 Alpha
  → 分别按 S_up / S_down 重建两张 Local Vol Grid
  → 用同一组随机数对固定行权价 K 做 MC 定价
  → 从两边价格反解 IV_up / IV_down
  → beta(alpha) = -(IV_up - IV_down) / log(S_up / S_down)
```

## 与 Step 2 的口径对齐

行权价始终是校准日的 `K = level × refSpot`。Spot bump 后 K 不动，到期日和
定价日也不动。因此它测量的是 Step 2 正式 `beta_surface` 对应的模型响应，
不包含每天重新设置 `level × 当日 refSpot` 产生的 smile crossing。

`beta_model` 是 MC 原始值。理论上 Alpha=1（sticky strike）应有
`beta_model≈0`。为了让 inverse 精确经过 `(beta=0, alpha=1)`，代码只做一个
常数平移：

```text
beta_converter(alpha) = beta_model(alpha) - beta_model(1)
```

这个平移不改变曲线的斜率或单调性。正式版不使用 PAVA、单调投影或任何拟合
去改造原始曲线。全部原始结果都会保存和绘图，质量检查只作为审计标签。
只有原始曲线非单调时，因为不存在唯一的分段线性反函数，该单元不写入
inverse。

## 正式默认配置

- Alpha 节点：`0, 0.5, 1, 1.5, 2`；
- Spot bump：上下各 1%；
- 100,000 条 antithetic 路径；
- 每个 Business/260 时间步再分 2 个子步；
- Local Vol ratio 网格 801 个节点；
- 所有 Alpha 以及 up/down 使用共同随机数；
- 使用常波动 GBM 配对控制变量；
- 期限为 1M、2M、3M、6M、9M、1Y、1.5Y、2Y；
- 未指定日期时，选择完整覆盖期限的 robust IV-grid medoid；也可用
  `--calibration-date YYYY-MM-DD` 指定。

`--fast` 只用于开发检查（10,000 路径、201 ratio 节点），不能作为正式转换器。

## 质量审计

`cell_quality.csv` 对每个 `(tenor, level)` 检查：

- 原始 `beta_model` 随 Alpha 严格递减；
- 原始 `|beta_model(alpha=1)| ≤ 0.03`；
- 最大 Beta 标准误不超过 0.10；
- Alpha 端点的 Beta 跨度至少 0.10，且跨度相对 MC 误差的 z-score 至少为 3；
- IV 反解价格没有触碰无套利边界；
- Local Vol 网格 undefined 比例不超过 5%，clipped 比例不超过 1%。

这些检查共同生成 `quality_pass`，但不会删除曲线或阻止绘图。远翼特别容易因
vega 很小而不通过；失败是可见的研究结果，不会被代码静默平滑。正式 inverse
只以 `inverse_available` 标记原始曲线是否严格单调，因为这是反函数存在的数学
要求，而不是用质量阈值筛选结果。

## 精简后的输出

- `selected_svi_quotes.csv`：所选代表曲面的原始 SVI-JW 参数；
- `beta_alpha_curve.csv`：固定 K 的核心价格、IV、原始 Beta、标准误和归零后 Beta；
- `cell_quality.csv`：每个 `(tenor, level)` 的质量检查和失败原因；
- `alpha_beta_inverse.csv`：原始曲线严格单调格点的分段线性 inverse 节点；
- `beta_alpha_atm.png`：全部 ATM 曲线；
- `beta_alpha_3m_smile.png`：全部 3M strike 切面；
- `manifest.json`：完整配置、输入文件哈希和验收汇总。

调用 `alpha_from_beta()` 时默认禁止区间外外推。只有显式传入 `clip=True` 才会
把超出实测范围的 Beta 截到端点 Alpha。
