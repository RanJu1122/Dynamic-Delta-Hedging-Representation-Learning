# Dynamic Alpha Step 3 结果与 Step 2 Beta 对比

> 本文记录当前正式版 Step 3 的代码逻辑、2025-07-21 代表性曲面的 MC
> 结果、质量审计，以及它与 Step 2 经验 Beta 的对应关系。
>
> 为保证 GitHub 直接渲染，本文只使用原生 Markdown 表格、代码块和纯文本公式，
> 不依赖 LaTeX、MathJax 或额外 Markdown 插件。

## 1. 当前结论摘要

1. Step 3 已经在一张代表性 SVI 曲面上成功测出 360 个原始
   `beta(alpha)` 点：8 个期限 × 9 个 strike level × 5 个 Alpha。
2. ATM 的 Alpha–Beta 关系非常稳定：Beta 随 Alpha 严格下降，Alpha=1 时
   原始 Beta 接近 0，符合 sticky-strike sanity check。
3. 当前 72 个 `(tenor, level)` 单元中有 62 个原始曲线严格单调，可以建立唯一
   的 `Beta -> Alpha` inverse；其余 10 个主要是短期限远翼。
4. 质量检查现在只写入 CSV，不再删除原始结果或隐藏图片。`quality_pass=0/72`
   主要来自“整张 Local Vol 矩形网格”的全局检查，不表示 72 条 MC 曲线都失败。
5. Step 2 的 ATM rolling Beta 全部落在 Step 3 的实测 Beta 范围内。反查后，
   短期限隐含 Alpha 明显低于 1，长期限接近或略高于 1。
6. 当前单日 Step 3 符合研究文档“固定一组 SVI 参数”的要求，但它只给出条件于
   该曲面状态的映射。在正式 OOS 回测前，仍需做多曲面状态敏感性检查。

## 2. Step 2 和 Step 3 分别解决什么问题

### 2.1 Step 2：从历史市场估计经验 Beta

Step 2 使用历史 SVI 曲面的相邻日变化，正式口径是 `beta_surface`：

```text
beta_surface(t, tenor, level)
    = -dIV_surface(t, tenor, level) / dlogS(t)
```

其中 `dIV_surface` 已经从同 level 的总 IV 变化中扣除前一日曲面上的机械 smile
crossing。它要回答的是：

> 历史市场中，Spot 变化时，隐含波动率曲面本身移动了多少？

Step 2 同时保留两类估计：

- `beta_surface_daily`：每日直接比值，噪声较大；
- `beta_surface_rolling`：滚动窗口内带截距 OLS 的斜率，是当前正式主轴。

### 2.2 Step 3：从模型测量 Beta–Alpha 转换器

Step 3 不读取 Step 1/2 的输出，而是在固定 SVI 曲面上人为指定 Alpha，通过
Local Vol MC 测量模型产生的 Beta：

```text
给定 alpha
    -> 分别重建 spot-up / spot-down Local Vol Grid
    -> 对同一个实际 strike 做 MC 定价
    -> 从两边价格反解 IV_up / IV_down
    -> beta_model(alpha) = -(IV_up - IV_down) / log(S_up / S_down)
```

它要回答的是：

> 如果定价模型使用某个 Alpha，这个模型会产生多大的固定-strike Beta？

二者最终通过 inverse 连接：

```text
Step 2 历史或预测 beta_surface
    -> Step 3 alpha_beta_inverse
    -> implied alpha
    -> 用该 alpha 重建 Local Vol Grid 并计算对冲 Delta
```

## 3. Step 3 正式版代码逻辑

主实现位于
[`dynamic_alpha_hedging/step03.py`](../dynamic_alpha_hedging/step03.py)。

### 3.1 选择代表性曲面

如果没有显式指定 `--calibration-date`，代码会：

1. 找出能够无外推覆盖全部研究期限的历史曲面；
2. 在配置的 `(tenor, level)` IV 网格上计算历史中位状态；
3. 选择与中位状态距离最小的 medoid 曲面。

本次选中：

| 项目 | 数值 |
|---|---:|
| Calibration date | 2025-07-21 |
| refSpot | 6325.86 |
| 3M ATM IV | 0.14385 |
| 3M ATM IV 历史百分位 | 约 44% |
| 3M 中央 smile slope | 约 -0.5065 |
| 3M smile slope 历史百分位 | 约 41% |

因此它是一张相对典型的中间状态曲面，不是高波、低波或 skew 的极端样本。

### 3.2 固定实际 strike

每个 strike 只在校准曲面上计算一次：

```text
strike(level) = level * calibration_refSpot
```

当 Spot 上下 bump 时，实际 strike 不变。这与 Step 2 的 `beta_surface` 固定-strike
响应口径一致。

### 3.3 Tenor 与到期日

研究期限为：

```text
1M, 2M, 3M, 6M, 9M, 1Y, 1.5Y, 2Y
```

每个 tenor 都使用 `Business/260` 转成校准曲面上的实际 expiry；MC 漂移仍使用
`Act/365`，方差时间使用 `Business/260`。

### 3.4 Alpha sweep 和 Local Vol Grid

Alpha 节点为：

```text
0.0, 0.5, 1.0, 1.5, 2.0
```

对于每个 `(tenor, alpha)`：

1. Spot 向上 bump 1%，按该 Alpha 重建 `grid_up`；
2. Spot 向下 bump 1%，按该 Alpha 重建 `grid_down`；
3. up/down 使用共同随机数；
4. 使用 antithetic sampling；
5. 使用常波动 GBM 配对控制变量；
6. 同一组终端路径同时为 9 个 strike 定价。

正式参数为：

| 参数 | 数值 |
|---|---:|
| MC paths | 100,000 |
| Time substeps | 2 |
| Local Vol ratio nodes | 801 |
| Spot bump | ±1% |
| Seed | 20260807 |

一共是：

```text
8 tenors * 5 alphas = 40 组模型实验
40 组 * up/down = 80 个 bumped path ensembles
8 tenors * 9 levels * 5 alphas = 360 个 Beta 结果
```

这里不是 360 次独立 MC，因为每个 `(tenor, alpha, bump-side)` 的 9 个 strike 共用
同一组路径。

### 3.5 从 MC 价格反解 Beta

对每个固定 strike：

```text
dIV_model(alpha) = IV_up(alpha) - IV_down(alpha)

beta_model(alpha)
    = -dIV_model(alpha) / log(S_up / S_down)
```

代码会同时计算 call 和 put 估计器，并优先选价格反解有效、Beta 标准误较小的
一边，以减少极端 ITM/OTM 定价不稳定。

### 3.6 Alpha=1 常数归零

`beta_model` 始终保留原始 MC 值。用于 inverse 的 `beta_converter` 只做：

```text
beta_converter(alpha)
    = beta_model(alpha) - beta_model(alpha=1)
```

该操作只消除 Alpha=1 附近的小 MC 截距，不改变曲线斜率、顺序或形状。

正式版不使用：

- PAVA；
- 单调投影；
- 对 Beta–Alpha 曲线的回归拟合；
- 为了可逆而重新排序原始 Alpha 节点。

## 4. 当前 Step 3 输出文件

本地输出目录：

```text
output/dynamic_alpha/step03/
```

| 文件 | 含义 |
|---|---|
| `selected_svi_quotes.csv` | 2025-07-21 代表性曲面的原始 SVI-JW 参数 |
| `beta_alpha_curve.csv` | 360 行价格、IV、原始 Beta、标准误和 converter Beta |
| `cell_quality.csv` | 72 个单元的质量检查、失败原因和可逆标记 |
| `alpha_beta_inverse.csv` | 62 个单调单元的 Beta–Alpha 插值节点，共 310 行 |
| `beta_alpha_atm.png` | 全部期限的 ATM Beta–Alpha 曲线 |
| `beta_alpha_3m_smile.png` | 3M 全部 strike level 的 Beta–Alpha 曲线 |
| `manifest.json` | 配置、输入哈希和验收汇总 |

`output/` 被 `.gitignore` 忽略，因此报告中的输出路径主要用于本地复现，不会随
GitHub 仓库自动上传。

## 5. Step 3 输出结果分析

### 5.1 ATM 原始 Beta–Alpha 曲线

下表使用未经归零的 `beta_model`：

| Tenor | Beta at Alpha=0 | Beta at Alpha=1 | Beta at Alpha=2 |
|---|---:|---:|---:|
| 1M | 0.7247 | 0.0191 | -0.6821 |
| 2M | 0.7092 | 0.0125 | -0.6956 |
| 3M | 0.6860 | 0.0089 | -0.6700 |
| 6M | 0.6256 | 0.0081 | -0.5983 |
| 9M | 0.5532 | 0.0064 | -0.5410 |
| 1Y | 0.5033 | 0.0044 | -0.4949 |
| 1.5Y | 0.4287 | 0.0040 | -0.4218 |
| 2Y | 0.3787 | 0.0019 | -0.3753 |

ATM 可以得到以下结论：

- 8 个期限的 Beta 都随 Alpha 严格下降；
- Alpha=1 的原始 Beta 全部接近 0；
- Alpha=0 产生正 Beta，Alpha=2 产生负 Beta；
- Alpha–Beta 曲线在 ATM 附近接近线性；
- 期限越长，Beta 对 Alpha 的敏感度越低；
- ATM Beta 标准误约为 0.006–0.010，明显小于 Alpha 端点间的 Beta 跨度。

### 5.2 当前质量审计的含义

72 个 `(tenor, level)` 单元的检查通过数：

| 检查 | 通过数 |
|---|---:|
| 原始 Beta 严格递减 | 62 / 72 |
| Alpha=1 绝对误差不超过 0.03 | 53 / 72 |
| 最大 Beta 标准误不超过 0.10 | 51 / 72 |
| Beta 端点跨度和 z-score 合格 | 63 / 72 |
| IV 价格反解没有截断 | 69 / 72 |
| 全矩形 Local Vol Grid 质量 | 0 / 72 |
| 全部质量检查同时通过 | 0 / 72 |

当前逻辑中：

- `quality_pass` 只用于审计，不删除曲线、不阻止绘图；
- `inverse_available` 只要求原始 Beta–Alpha 曲线严格单调；
- 只有非单调曲线因为不存在唯一分段线性反函数，不能进入 inverse。

`grid_quality_pass=0/72` 的主要原因是检查取整个 `ratio=0.001–3.0` 网格、全部
Alpha、up/down 两边的最坏比例。较高 Alpha 的 down-bump 网格在上翼出现 Dupire
分母失效，同一 tenor 下的全部 strike 因共用网格统计而一起失败。

已检查到：

- 静态研究区间 `ratio=0.4–1.2` 内没有 undefined/clipped 点；
- 主要 undefined 区域从 ratio 约 1.226 开始；
- 但是 MC 路径可能运行到该区域，所以不能简单删除这个诊断；
- 更合理的后续指标是“路径实际访问坏网格区域的比例及其价格影响”。

### 5.3 当前 10 个不可逆单元

| Tenor | 不可逆 level |
|---|---|
| 1M | 0.4, 0.5, 0.6, 1.1, 1.2 |
| 2M | 1.1, 1.2 |
| 3M | 1.2 |
| 6M | 1.2 |
| 9M | 1.2 |

1Y、1.5Y、2Y 的 9 个 level 全部具有单调 inverse。不可逆问题主要集中在短期限
远翼，符合普通 MC 在低 vega 期权上反解 IV 较困难的特征。

## 6. Step 2 Beta 与 Step 3 的对比

以下使用 Step 2 正式的 `beta_surface_rolling`，逐个历史时点通过同一
`(tenor, level)` 的 Step 3 inverse 反查 Alpha。

### 6.1 ATM 历史均值

| Tenor | Step 2 ATM rolling Beta 均值 | 逐点反查后的 Alpha 均值 | 解释 |
|---|---:|---:|---|
| 1M | 0.304 | 0.566 | 明显低于 sticky strike |
| 2M | 0.238 | 0.652 | Alpha < 1 |
| 3M | 0.138 | 0.797 | 约为 Alpha=0.8 |
| 6M | 0.052 | 0.915 | 接近 sticky strike |
| 9M | 0.013 | 0.977 | 基本 sticky strike |
| 1Y | -0.012 | 1.025 | 略高于 1 |
| 1.5Y | -0.032 | 1.075 | 略偏向 sticky moneyness |
| 2Y | -0.037 | 1.098 | 略偏向 sticky moneyness |

方向解释：

```text
Step 2 beta > 0  -> Step 3 implied alpha < 1
Step 2 beta = 0  -> Step 3 implied alpha = 1
Step 2 beta < 0  -> Step 3 implied alpha > 1
```

这说明经验 stickiness 有明显期限结构：短期限曲面对 Spot 的反应更强，长期限
逐渐接近 sticky strike，并在 1Y 以后略微越过 Alpha=1。

### 6.2 Beta 范围覆盖

- 全部 8 个 ATM 期限的历史 rolling Beta 都落在 Step 3 的 Alpha `[0,2]`
  实测 Beta 范围内；
- Step 2 一共有 41,184 条非空 rolling Beta；
- 其中 37,198 条位于 Step 3 有单调 inverse 的单元；
- 其中 36,690 条同时位于实测 Beta 范围内；
- 在可逆单元内，Beta 范围覆盖率为 98.6%；
- 按全部 rolling 数据计算，当前可以直接反查 Alpha 的比例约为 89.1%。

这说明 Alpha 节点 `[0,2]` 对 ATM 和大多数曲面单元已经足够宽，不需要普遍做
区间外外推。

### 6.3 Strike 方向的差异

如果一个统一 Alpha 能完全解释整个经验 Beta 曲面，同一天同期限不同 strike
反查出来的 Alpha 应相对接近。当前结果在近 ATM 区域较稳定，但远翼有明显偏离：

- 3M level=1.1 只有约 49.6% 的 rolling Beta 位于模型实测范围；
- 1Y level=1.2 的覆盖率约为 72.3%；
- 多个短期限 level=1.2 单元原始曲线不单调；
- 短期限深度 OTM put 的 Beta 标准误也明显较大。

可能原因包括：

1. 普通 MC 在远翼的低 vega 价格上反解 IV 不稳定；
2. 一张固定 SVI 曲面的模型映射不能覆盖全部历史 smile 状态；
3. 真实市场的 Alpha 本身可能随期限和 moneyness 变化；
4. Step 2 的历史 Beta 在远翼也更加噪声化。

因此，当前最可信的是 ATM 和近 ATM 结果；完整 `0.4–1.2` 网格应继续保留为
诊断，但不应把所有远翼 inverse 视为同等可靠。

### 6.4 Step 2 Beta 的统计不确定性

Step 2 rolling OLS 的 ATM median R-squared 大约为：

| Tenor | Median R-squared |
|---|---:|
| 1M | 0.215 |
| 2M | 0.128 |
| 3M | 0.119 |
| 6M | 0.086 |
| 9M | 0.071 |
| 1Y | 0.084 |
| 1.5Y | 0.096 |
| 2Y | 0.089 |

所以隐含 Alpha 目前是描述性估计，不等于已经证明该 Alpha 的 Delta 最优。正式
结论仍需依靠：

- Beta 的 OOS 预测能力；
- Beta 标准误向 Alpha 的传播；
- Step 7 实际对冲误差；
- 与 Alpha=0/1/2 和样本内最优固定 Alpha 的比较。

## 7. 单日 Step 3 是否足够

### 7.1 为什么当前做法是正确的第一版

研究文档 Step 3 明确要求：

> 固定一组 SVI 参数，对不同 Alpha 做 Spot bump 和 Local Vol MC。

因此，当前在一张固定代表性曲面上建立转换器，符合文档要求。选中的曲面又是
全 IV 网格 medoid，3M ATM IV 和 smile slope 都位于历史中间区域，因此适合作为
baseline。

### 7.2 为什么它还不是已经证明的通用转换器

严格来说，模型关系是：

```text
beta_model
    = beta(alpha; SVI state, tenor, level, numerical conventions)
```

SVI state 包括：

- ATM IV；
- skew；
- put/call wing；
- kurtosis；
- 期限结构；
- calendar repair 后的时间导数。

当前单日转换器隐含假设：

```text
beta(alpha; today's surface state, tenor, level)
    approximately equals
beta(alpha; 2025-07-21 medoid state, tenor, level)
```

如果映射实际随状态变化，那么固定转换器会把两种效应混在一起：

```text
观测到的 implied alpha 变化
    = 真实 stickiness alpha 变化
    + Alpha–Beta 转换器随 SVI 状态变化造成的误差
```

所以当前结果可以证明：

> 在 2025-07-21 这张代表性曲面条件下，Alpha–Beta 关系存在、ATM 近似线性，
> Alpha=1 对应 Beta≈0。

但还不能证明：

> 同一条转换曲线可以无误差地应用于 2023–2026 的所有市场状态。

### 7.3 正式回测前需要的最小敏感性检查

不需要立即对全部历史日期都跑正式 MC。建议：

1. 只从训练期选择低、中、高 ATM IV 状态；
2. 再覆盖平缓、中等、陡峭 skew 状态；
3. 总计选择约 6–12 张代表性曲面；
4. 先在 ATM、level=0.9、level=1.1 上比较 `beta(alpha)`；
5. 比较同一个目标 Beta 在不同曲面上反查出的 Alpha 差异；
6. 如果差异很小，保留单一 medoid 转换器；
7. 如果差异明显，建立按市场状态选择或插值的转换器。

进入 OOS 回测时还要避免未来信息：当前 2025-07-21 medoid 是从完整样本选出的。
正式 Step 7 应只使用训练期数据选择 medoid 或多状态代表曲面，然后冻结到测试期。

## 8. 当前最稳妥的研究判断

目前可以认为：

1. Step 2 与 Step 3 的固定-strike Beta 定义已经对齐；
2. Step 3 的 ATM Alpha–Beta 关系通过了最关键的方向、单调性和 Alpha=1 检查；
3. 3M ATM 历史 Beta 约 0.138，对应代表性曲面下 Alpha 约 0.8；
4. 经验 Alpha 呈现从短期限低于 1、向长期限接近并略高于 1 的期限结构；
5. Alpha `[0,2]` 对全部 ATM 历史 Beta 有充分覆盖；
6. 远翼结果仍受 MC 误差、Local Vol 网格和单曲面状态依赖影响；
7. 当前单曲面转换器适合作为 Step 4 的 baseline，但进入 Step 7 前必须验证
   Alpha–Beta 映射的状态稳定性。

## 9. 复现命令

从项目根目录运行正式 Step 3：

```bash
python -m dynamic_alpha_hedging step3
```

查看命令参数：

```bash
python -m dynamic_alpha_hedging step3 --help
```

注意：`--fast` 仅用于开发检查，不能替代当前 100,000 路径正式结果。

