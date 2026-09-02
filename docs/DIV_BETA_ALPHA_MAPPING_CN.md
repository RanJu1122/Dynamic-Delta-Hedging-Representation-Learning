# Dynamic Alpha 研究中的 dIV、Beta 与 Alpha 映射

本文统一说明动态 Alpha 研究中不同 `dIV`、不同 Beta 的定义，以及历史 Beta 与模型 Alpha 之间的映射关系。

> 注意：本文讨论的是 `dIV`，即隐含波动率（implied volatility）的变化，不是股息率 `div`。

## 1. 每日 IV 矩阵

对每个交易日 $t$，代码从当天 SVI 参数构建完整隐含波动率曲面，然后在固定网格上读取：

$$
IV[t,\tau,m]
$$

其中：

- $\tau$：固定剩余期限，例如 1M、2M、3M、6M；
- $m$：strike level，定义为 $K/S_{\rm ref}$；
- 当天实际 strike 为

$$
K_t=mS_t;
$$

- 当天实际 expiry 根据当天日期和固定 $\tau$ 重新计算。

因此，跨日固定的是

$$
(\tau,m),
$$

但不固定实际 strike、实际到期日和具体期权合约。

例如：

| 日期 | Spot | Level | 实际 strike |
|---|---:|---:|---:|
| $t-1$ | 5000 | 0.9 | 4500 |
| $t$ | 6000 | 0.9 | 5400 |

比较同一个 `level=0.9` 时，比较的是昨天 $K=4500$ 和今天 $K=5400$ 的曲面点。

这与“把每天 refSpot 归一化为 1，再查询 `IV(tau, 0.9)`”完全等价。

完整矩阵保存在 `output/dynamic_alpha/step01/iv_state.csv`。

## 2. 三种 dIV

设：

$$
S_0=S_{t-1},\qquad S_1=S_t,
$$

$$
K_0=mS_0,\qquad K_1=mS_1.
$$

定义：

$$
\sigma_0=IV_{t-1}(\tau,K_0),
$$

$$
\sigma_1=IV_t(\tau,K_1).
$$

还需要一个反事实值：

$$
\sigma_{0\rightarrow1}=IV_{t-1}(\tau,K_1).
$$

它表示不改变昨天的曲面，只在昨天曲面上把查询 strike 从 $K_0$ 移到 $K_1$。

### 2.1 `dIV_grid`

$$
\boxed{
dIV_{\rm grid}
=IV_t(\tau,K_1)-IV_{t-1}(\tau,K_0)
}
$$

即：

$$
dIV_{\rm grid}=\sigma_1-\sigma_0.
$$

它回答的是：每天相同 $(\tau,m)$ 格点上的 IV 总共变化了多少。

它同时包含：

1. 曲面本身的动态变化；
2. strike 从 $K_0$ 移到 $K_1$ 后沿静态 smile 穿行的影响。

因此，它是“把 refSpot 归一化为 1，然后比较相同 tenor 和 level”所对应的原始 dIV。

### 2.2 `smile_crossing_iv`

$$
\boxed{
dIV_{\rm crossing}
=IV_{t-1}(\tau,K_1)-IV_{t-1}(\tau,K_0)
}
$$

即：

$$
dIV_{\rm crossing}=\sigma_{0\rightarrow1}-\sigma_0.
$$

它只使用昨天的曲面，衡量 strike 移动产生的机械 skew 效应。

Spot 变化较小时：

$$
dIV_{\rm crossing}
\approx
\frac{\partial IV}{\partial\log K}d\log S.
$$

因为：

$$
K_t=mS_t
\quad\Rightarrow\quad
d\log K=d\log S.
$$

### 2.3 `dIV_surface`

当前代码定义：

$$
\boxed{
dIV_{\rm surface}
=dIV_{\rm grid}-dIV_{\rm crossing}
}
$$

展开后：

$$
dIV_{\rm surface}
=IV_t(\tau,K_1)-IV_{t-1}(\tau,K_1).
$$

即：

$$
dIV_{\rm surface}=\sigma_1-\sigma_{0\rightarrow1}.
$$

它回答的是：扣除沿昨天 smile 移动的机械影响后，曲面本身在同一个实际 strike 上变化了多少。

需要注意，这里固定了用于比较的实际 strike $K_1$，但仍然固定剩余期限 $\tau$，所以前后实际 expiry 会滚动。它不是同一张真实期权，而是“固定 strike、固定剩余期限”的曲面动态响应。

三者满足精确恒等式：

$$
\boxed{
dIV_{\rm grid}
=dIV_{\rm crossing}+dIV_{\rm surface}
}
$$

具体结果保存在 `output/dynamic_alpha/step01/grid_changes.csv`。

## 3. Spot 变化

所有 Beta 使用的 Spot 变化都是实际 refSpot：

$$
\boxed{
d\log S_t
=\log\left(\frac{S_t}{S_{t-1}}\right)
}
$$

“把 refSpot 归一化为 1”只用于读取相同标准化曲面坐标，不会把真实 Spot 变化也改成 1。

## 4. 两种经济含义不同的 Beta

### 4.1 Grid Beta

$$
\boxed{
\beta_{\rm grid}
=-\frac{dIV_{\rm grid}}{d\log S}
}
$$

它回答的是：Spot 变化 1% 时，相同 $(\tau,m)$ 格点的 IV 总共变化多少。

它同时包含真实曲面动态和静态 smile crossing。这是“refSpot 归一化后固定 tenor 和 level”对应的 Beta。

### 4.2 Surface Beta

$$
\boxed{
\beta_{\rm surface}
=-\frac{dIV_{\rm surface}}{d\log S}
}
$$

它回答的是：扣除固定 level 导致的机械 skew 穿行后，曲面本身对 Spot 的响应是多少。

两者在小变动下近似满足：

$$
\boxed{
\beta_{\rm grid}
\approx
\beta_{\rm surface}
-\frac{\partial IV}{\partial\log K}
}
$$

股指曲面通常为负 skew：

$$
\frac{\partial IV}{\partial\log K}<0.
$$

因此通常有：

$$
\beta_{\rm grid}>\beta_{\rm surface}.
$$

这就是历史结果中 Grid Beta 往往明显更大的原因。

## 5. Daily Beta 和 Rolling Beta

Daily 与 Rolling 不是两套不同的经济定义，而是同一个 Beta 的两种估计方法。

当前输出看似有四个 Beta：

- `beta_grid_raw_daily`；
- `beta_surface_daily`；
- `beta_grid_raw_rolling`；
- `beta_surface_rolling`。

实际结构是：

$$
2\text{种 dIV 口径}\times2\text{种估计方法}.
$$

### 5.1 Daily ratio

逐日直接相除：

$$
\beta_{{\rm grid},t}^{\rm daily}
=-\frac{dIV_{{\rm grid},t}}{d\log S_t},
$$

$$
\beta_{{\rm surface},t}^{\rm daily}
=-\frac{dIV_{{\rm surface},t}}{d\log S_t}.
$$

当

$$
|d\log S|<0.005
$$

即 Spot 变化不足 0.5% 时，当前代码不计算 Daily Beta，避免分母过小导致比值爆炸。

Daily Beta 反应快，但噪声较大，而且不能有效分离每日 IV 漂移。

### 5.2 Rolling OLS Beta

对每一个固定 $(\tau,m)$，使用最近60个有效变化观测回归：

$$
dIV_i=c+b\,d\log S_i+\varepsilon_i.
$$

定义：

$$
\boxed{\beta=-b}.
$$

所以也可以写成：

$$
dIV_i=c-\beta d\log S_i+\varepsilon_i.
$$

其中：

- $c$：截距，吸收与 Spot 无关的平均 IV 漂移；
- $\beta$：Spot 对 IV 的经验影响；
- $R^2$：Spot 变化解释 IV 变化的比例；
- `slope_stderr`：斜率估计误差；
- `nobs`：窗口内有效样本数。

Rolling 默认设置为：

- 最近60个有效观测，不是60个日历日；
- 至少20个样本才输出；
- 只使用相邻有效交易日；
- 基准 OLS 不剔除小幅 Spot 日，0.5%阈值主要用于 Daily ratio。

通常正式研究更适合使用 Rolling OLS，因为回归可以避免单日小分母严重放大，通过截距吸收平均漂移，并输出 $R^2$ 和标准误。

结果分别保存在：

- `output/dynamic_alpha/step02/beta_daily.csv`；
- `output/dynamic_alpha/step02/beta_rolling.csv`。

## 6. Alpha 的含义

当前 Alpha 作用在 Spot bump 后重建 local-vol grid 的 Dupire 坐标中：

| Alpha | 名称 | 经济含义 |
|---:|---|---|
| 0 | frozen/sticky local vol | local-vol 曲面更接近冻结 |
| 1 | sticky strike | 每个实际 strike 上的 IV 不随 Spot 变化 |
| 2 | sticky moneyness | 曲面更接近随 Spot 一起移动 |

Alpha 不是从历史 IV 直接计算出来的。

Step 3 先人为指定 Alpha，再通过 local-vol MC 测量在该 Alpha 下产生的 Beta，从而得到：

$$
\alpha\longrightarrow\beta_{\rm model}(\alpha).
$$

最后才使用历史 Beta 反查 Alpha。

## 7. Fixed-strike Alpha–Beta 映射

Step 3 在一张代表性 SVI 曲面上固定：

$$
K=mS_{\rm ref}.
$$

然后将 Spot 上下 bump 1%，但 strike $K$ 不随 bump 改变。

对每个 Alpha：

1. 重建 up/down local-vol grid；
2. 用 MC 重新定价固定 strike 期权；
3. 把 MC 价格反解成 $IV_{\rm up}$ 和 $IV_{\rm down}$；
4. 计算

$$
\boxed{
\beta_{\rm fixedK}(\alpha)
=-\frac{
IV_{\rm up}(\alpha)-IV_{\rm down}(\alpha)
}{
\log(S_{\rm up}/S_{\rm down})
}
}.
$$

它对应历史数据中的 $\beta_{\rm surface}$，而不是 `beta_grid_raw`。

### Alpha=1 的基准

Sticky strike 意味着固定实际 $K$ 的 IV 不随 Spot 变化：

$$
dIV_{\rm fixedK}\approx0.
$$

因此：

$$
\boxed{
\alpha=1
\quad\Rightarrow\quad
\beta_{\rm surface}\approx0
}.
$$

这就是“Alpha=1 应给出 Beta≈0”的准确适用范围。

## 8. Normalized-grid Alpha–Beta 映射

如果使用固定 $(\tau,m)$ 的 Grid Beta，那么理论模型也必须转换成相同坐标：

$$
\boxed{
\beta_{\rm normalized}(\alpha)
=\beta_{\rm fixedK}(\alpha)
-\frac{\partial IV}{\partial\log K}
}.
$$

它对应历史数据中的 $\beta_{\rm grid}$。

在 Alpha=1 时：

$$
\beta_{\rm fixedK}(1)=0,
$$

但：

$$
\boxed{
\beta_{\rm normalized}(1)
=-\frac{\partial IV}{\partial\log K}
}.
$$

因此，只要 smile 存在 skew：

$$
\beta_{\rm normalized}(1)\neq0.
$$

例如本次代表性曲面的 3M ATM：

$$
\frac{\partial IV}{\partial\log K}\approx-0.585,
$$

所以：

$$
\beta_{\rm normalized}(1)\approx0.585.
$$

这是坐标口径带来的正常结果，不是计算错误。

## 9. Step 3 中 raw model 与 converter 的区别

### 9.1 `beta_model`

`beta_model` 是 MC 直接测量的 Beta。由于 MC 误差，Alpha=1 可能得到 $0.01$、$0.02$ 或 $-0.01$，而不是严格等于零。

它主要用于验证模型和检查数值误差。

### 9.2 `beta_converter`

代码只减去 Alpha=1 处的数值偏差：

$$
\beta_{\rm converter}(\alpha)
=\beta_{\rm model}(\alpha)
-\beta_{\rm model}(1).
$$

因此：

$$
\beta_{\rm converter}(1)=0.
$$

这个常数平移不会改变原始 MC 曲线的形状。正式版不再做单调投影：全部曲线都
会保存，误差、跨度、价格反解和 local-vol 网格检查只写入质量审计。只有原始
曲线非单调时，因为不存在唯一反函数，该格点不写入 fixed-strike inverse。

### 9.3 normalized 口径

$$
\beta_{\rm converter,normalized}(\alpha)
=\beta_{\rm converter}(\alpha)
-\frac{\partial IV}{\partial\log K}.
$$

它可以用于和历史 `beta_grid_raw` 做专项诊断，但不是正式 inverse 的输入，正式
Step 3 因此不再单独保存 normalized 输出文件。

完整结果保存在：

- `output/dynamic_alpha/step03/beta_alpha_curve.csv`；
- `output/dynamic_alpha/step03/cell_quality.csv`；
- `output/dynamic_alpha/step03/alpha_beta_inverse.csv`。

## 10. 历史 Beta 如何映射成 Alpha

历史 Beta 与模型 Beta 必须使用同一种坐标口径。

### 10.1 路径 A：Surface Beta

$$
dIV_{\rm surface}
\longrightarrow
\beta_{\rm surface}^{\rm rolling}
\longrightarrow
\beta_{\rm fixedK}(\alpha)
\longrightarrow
\widehat\alpha.
$$

即：

$$
\boxed{
\widehat\alpha
=f_{\tau,m}^{-1}
\left(\beta_{\rm surface}^{\rm rolling}\right)
}.
$$

这条路径满足：

$$
\alpha=1\longleftrightarrow\beta=0.
$$

当前正式的 `alpha_beta_inverse.csv` 使用的就是这条路径。

### 10.2 路径 B：Grid Beta

$$
dIV_{\rm grid}
\longrightarrow
\beta_{\rm grid}^{\rm rolling}
\longrightarrow
\beta_{\rm normalized}(\alpha)
\longrightarrow
\widehat\alpha.
$$

这条路径保留“refSpot归一化后比较相同 $(\tau,m)$”的定义。

但是它的 Alpha=1 基准是：

$$
\beta_{\rm grid}(1)
=-\frac{\partial IV}{\partial\log K},
$$

而不是零。

### 10.3 不能混用

下面的做法是错误的：

$$
\beta_{\rm grid}^{\rm historical}
\longrightarrow
\beta_{\rm fixedK}^{-1}
\longrightarrow
\alpha.
$$

因为历史 Beta 包含 smile crossing，但模型 inverse 不包含，会导致系统性的 Alpha 偏差。

## 11. 两个研究要求之间的口径差异

“固定 tenor 和 strike level，把每天 refSpot 归一化成1，比较相同矩阵格点”对应：

$$
dIV_{\rm grid},\qquad\beta_{\rm grid}.
$$

“Alpha=1 应给出 Beta≈0”对应：

$$
dIV_{\rm surface},\qquad\beta_{\rm surface}.
$$

这两个要求不能同时对同一个 Beta 成立，因为：

$$
\beta_{\rm grid}
\approx
\beta_{\rm surface}
-\frac{\partial IV}{\partial\log K}.
$$

只要 smile slope 不为零，两者就不相等。

但它们可以分别成立：

$$
\alpha=1
\quad\Rightarrow\quad
\beta_{\rm surface}\approx0,
$$

$$
\alpha=1
\quad\Rightarrow\quad
\beta_{\rm grid}
\approx-\frac{\partial IV}{\partial\log K}.
$$

## 12. 建议的最终研究框架

建议保留两个口径，但必须明确其匹配关系：

| 历史量 | 经济含义 | 对应 Step 3 模型量 | Alpha=1 基准 |
|---|---|---|---|
| `beta_surface_rolling` | 扣除静态 smile crossing 的曲面动态 | `beta_converter` | 约0 |
| `beta_grid_raw_rolling` | 固定 tenor/level 的总 IV 响应 | `beta_converter_normalized` | $-\partial IV/\partial\log K$ |

如果最终严格使用“归一化矩阵”定义，应当：

1. 把 `beta_grid_raw_rolling` 作为历史主标签；
2. 用 `beta_converter_normalized` 构建单独的 normalized inverse；
3. 不再要求该 Beta 在 Alpha=1 时等于零；
4. 同时保留 `beta_surface` 作为去除静态 skew 后的稳健性检验。

如果最终研究和对冲公式必须坚持

$$
\alpha=1\Rightarrow\beta=0,
$$

那么主标签必须是 `beta_surface`，不能是未经调整的 `beta_grid_raw`。

目前代码仍把 `beta_surface` 标记为正式主标签，把 `beta_grid_raw` 和 normalized Step 3 曲线作为诊断。因此，在进入后续步骤前需要最终确认：

> 最终要预测的是“标准化矩阵格点的总变化”，还是“扣除静态 smile crossing 后的曲面动态”？

这个选择决定后续应使用哪一套 Beta–Alpha inverse。
