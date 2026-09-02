# Alpha 参数、Dupire 局部波动率与 Delta 的关系

> 基于《动态Alpha对冲研究》文档与代码实现
> 文件路径：`svi_localvol/surface.py`, `svi_localvol/montecarlo.py`

---

## 一、问题的起点：带波动率曲面时 Delta 是什么

Black-Scholes 的 delta 是在"隐含波动率对 spot 不敏感"这个假设下推导出来的。但市场现实是，当 spot 动时，整张 IV 曲面也在动。所以真实的 delta 是：

```
Δ = ∂V/∂S = Δ_BS(σ_imp) + ν · (∂σ_imp/∂S)
              └── BS delta ──┘   └─ shadow delta ─┘
```

`∂σ_imp/∂S` 就是"stickiness"——spot 动时，某个行权价上的 IV 怎么跟着动。这一项决定了 delta 的值，也是整个 alpha 研究的核心。

---

## 二、从 Implied Vol 到 Local Vol：Dupire 公式

### 2.1 隐含波动率曲面的表示

市场给出的是每个 `(K, T)` 上的隐含波动率 `IV(K, T)`。我们用 total variance 来统一表示：

```
w(y, τ) = IV²(K, T) · τ_vol
```

其中：
- `y = ln(K / F)`，log-forward moneyness
- `F = S_ref · exp(b · τ_r)`，用参考 spot 算的 forward
- `τ_vol`：Business/260 时钟（SVI 报价的时间轴）
- `τ_r`：Act/365 时钟（forward 和 discount 用的时间轴）

代码实现在 [surface.py:183](../svi_localvol/surface.py#L183) 的 `total_variance()`。

### 2.2 Dupire 公式

Dupire（1994）证明：给定一张无套利的隐含波动率曲面，存在唯一的局部波动率函数 `σ_loc(T, K)` 使得这张曲面自洽。它的推导核心是：在风险中性测度下，欧式期权价格满足一个关于到期日和行权价的偏微分方程，从中解出 `σ_loc`。

用 Gatheral 的 total variance 记法，结果是：

```
σ_loc²(T, K) = (∂w/∂τ) / D

D = 1 - (y/w)·(∂w/∂y) + (1/4)·(-1/4 - 1/w + y²/w²)·(∂w/∂y)² + (1/2)·(∂²w/∂y²)
```

**分子 `∂w/∂τ`**：total variance 对时间的斜率，反映"随着到期日延长，这个 strike 上积累了多少方差"。

**分母 `D`**：和 Gatheral g-function 等价，是从 strike 维度的曲率提取出来的修正项。`D ≤ 0` 意味着 butterfly 套利，`∂w/∂τ < 0` 意味着 calendar 套利，两种情况都返回 NaN。

代码实现在 [surface.py:257](../svi_localvol/surface.py#L257) 的 `local_vol()`：

```python
res = self.total_variance(T, K, spot=spot, order=2)   # 算 w, dw/dtau, dw/dy, d2w/dy2
w      = res["w"]
dw_dT  = res["dw_dtau"]
dw_dy  = res["dw_dy"]
d2w_dy2 = res["d2w_dy2"]

D = (1.0
     - (y_adj / w) * dw_dy
     + 0.25 * (-0.25 - 1.0/w + y_adj**2 / w**2) * dw_dy**2
     + 0.5 * d2w_dy2)

var_loc = np.where((D > 0.0) & (dw_dT >= 0.0) & (w > 0.0), dw_dT / D, np.nan)
vol_loc = np.sqrt(var_loc)
```

**关键点**：`local_vol ≠ implied_vol`。Local vol 是隐含曲面对时间和 strike 同时求导后的结果，数值上通常比 ATM 的 IV 更高（因为分子取的是斜率，分母 < 1 时会放大）。两者通过 Dupire 公式联系，但数值永远不相等。

---

## 三、Alpha 参数的引入：控制 Dupire 分母的 y 坐标

### 3.1 问题场景

当我们用 MC 计算 delta 时，需要对 spot 做 bump：

```
Delta ≈ (V(S+ε) - V(S-ε)) / (2ε)
```

对 `V(S+ε)` 做 MC，路径从 `S+ε` 出发。在路径演化的每一步，需要查询当前状态 `(t, S_t)` 处的 local vol。

这里有个隐含问题：**local vol 是从哪张隐含曲面推导出来的，用的是哪个 spot 的 forward？**

当初始 spot 从 `S_ref` 变为 `S`，对于同一个行权价 `K`：

```
原始 moneyness:  y_ref = ln(K / F_ref),  F_ref = S_ref · exp(b·τ)
新的 moneyness:  y_new = ln(K / F_new),  F_new = S   · exp(b·τ)

两者之差：
y_new - y_ref = ln(F_ref / F_new) = ln(S_ref / S) = -spot_adj
其中 spot_adj = ln(S / S_ref)
```

所以当 spot 上涨（`spot_adj > 0`），同一个 K 的 moneyness **往左移**（从 OTM 变成更接近 ATM 甚至 ITM）。

### 3.2 Alpha 的定义

Alpha 控制：在计算 Dupire 分母 D 时，用哪个 y 坐标去查 SVI 曲面：

```
y_adj = y_new - α · spot_adj
```

- `y_new`：用当前 spot 算出的实际 moneyness
- `spot_adj = ln(S / S_ref)`：当前 spot 相对参考 spot 的对数偏移
- `α`：stickiness 参数，取值 `[0, 2]`

**注意**：这个 shift 只进入 Dupire 分母 D，不改变 `w(y, τ)` 本身，也不改变隐含波动率曲面。隐含曲面始终用 `y_new` 查询。

代码位置 [surface.py:289-298](../svi_localvol/surface.py#L289-L298)：

```python
a = (self.alpha_at(T) if alpha is None else validate_stickiness_alpha(alpha))
shift = a * float(spot_adj)
res = self.total_variance(T, K, spot=spot, order=2)
# res["y"] 就是 y_new，用当前 spot 计算的 moneyness
y_adj = res["y"] - shift
```

---

## 四、三个 Alpha 值的含义：代码推导

### 4.1 关键前提：`y` 始终用 `S_ref` 的 forward 计算

先看代码里 `y` 实际是怎么来的。

`local_vol()` 内部调用的是：

```python
res = self.total_variance(T, K, spot=spot, order=2)
# spot 默认 None → forward 用 S_ref
y = ln(K / F_ref),   F_ref = S_ref · exp(b · τ_r)
```

`LocalVolGrid.build()` 里，`K = ratio * S_ref`，调用 `surface.local_vol(T, K, spot_adj=spot_adj, alpha=alpha)` 时不传 `spot`，所以：

```
y     = ln(K / F_ref)       ← 永远用 S_ref 的 forward，与 spot_adj 无关
y_adj = y - alpha · spot_adj
      = ln(K / F_ref) - alpha · ln(S_bump / S_ref)
```

**这是整个推导的基础**：`total_variance` 看到的 `y` 永远是"以 S_ref 为参考的 moneyness"，`spot_adj` 只改变 Dupire 分母里用的坐标偏移量。

### 4.2 三个 Alpha 值的网格行为

设 `K = ratio * S_ref`，对 base / up / down 三个网格，`spot_adj` 分别为 `0, ln(S_up/S_ref), ln(S_dn/S_ref)`。

**Alpha = 0**：

```
y_adj = ln(K / F_ref) - 0 · spot_adj = ln(K / F_ref)
```

对所有三个网格，同一个 ratio 点上 K 相同，`y_adj` 完全一样 → **三个网格数值逐点完全相同**。

这就是"Frozen"的字面含义：local vol 曲面在 `(T, K)` 坐标下数值不变，不管 spot_adj 是什么。MC delta 纯粹来自初始 spot 不同带来的路径差异。

**Alpha = 1**：

```
y_adj = ln(K / F_ref) - 1 · ln(S_up / S_ref)
      = ln(K) - ln(F_ref) - ln(S_up) + ln(S_ref)
      = ln(K) - ln(S_up · exp(b · τ_r))
      = ln(K / F_up)
```

grid_up 在每个 K 点上用的 `y_adj = ln(K / F_up)`，即用 bump 后的 forward 来定位 moneyness。

grid_dn 类似，`y_adj = ln(K / F_dn)`。

三个网格在相同 K 处查的是 SVI 曲面上三个不同的 y 位置，分别对应三个不同的 forward。

**Alpha = 2**：

```
y_adj = ln(K / F_ref) - 2 · ln(S_up / S_ref)
      = ln(K / F_up) - ln(S_up / S_ref)
      = ln(K · S_ref / (S_up · F_up))
```

过度补偿，查询了比 `ln(K/F_up)` 更负的位置（更深 ITM 方向的曲面值）。

### 4.3 为什么 Alpha=1 能复现 BS Delta

BS 假设每个绝对 strike K 上的 IV 不随 spot 变（"Sticky Strike"）：

```
IV(K=5200, S=5000) = IV(K=5200, S=5100)
```

这意味着：当 spot bump 后，我们评估 K=5200 时，应该用 bump 后的 forward `F_up` 来定位 moneyness，然后从 SVI 曲面读取同一个 K 的 IV。这正是 alpha=1 做的：

```
grid_up 里 K=5200 的 local vol
= Dupire(SVI 在 y=ln(5200/F_up) 处的值)
```

而不是（alpha=0 做的）：

```
grid_up 里 K=5200 的 local vol
= Dupire(SVI 在 y=ln(5200/F_ref) 处的值)   ← 冻结了，没有反映 forward 变化
```

**从 shadow delta 的角度**：

```
Δ = Δ_BS + ν · (∂σ_imp/∂S)
```

- Alpha=0：local vol 在 (T, K) 下冻结，隐含"IV 随 forward 移动而移动"，shadow delta ≠ 0
- Alpha=1：local vol 在 (T, K) 下随 forward 一起重新定位，隐含 `∂σ_imp/∂S ≈ 0`，所以 `Δ ≈ Δ_BS`
- Alpha=2：过度补偿，隐含 IV 随 spot 上涨而上涨，shadow delta > 0，`Δ > Δ_BS`

| Alpha | 三个网格是否相同 | SVI 查询坐标 | 隐含曲面动法 | Delta 结果 |
|-------|----------------|-------------|------------|-----------|
| 0 | **完全相同** | `ln(K/F_ref)`（固定） | IV 随 forward 平移 | Frozen local vol delta |
| 1 | 不同 | `ln(K/F_bump)` | Strike K 的 IV 固定 | ≈ BS delta |
| 2 | 不同 | 更负于 `ln(K/F_bump)` | K/S 比例处的 IV 固定 | > BS delta |

---

## 五、Alpha = 0 的 Delta 有什么意义

### 5.1 它是什么

α=0 时，`y_adj = y_new`，不对 Dupire 分母做任何调整。这意味着：

Local vol 网格在 `(t, S/S_ref)` 坐标系中被当成**固定函数**来使用——不管初始 spot 是多少，路径演化时都从同一张预先计算好的网格查询 `σ_loc(t, S/S_ref)`。

"Frozen"指的不是某个数值冻结，而是**不对 local vol 曲面做任何因 spot 变化的再调整**。

### 5.2 它和 BS delta 的差异

BS delta 假设每个 K 上的 IV 不随 spot 变（shadow delta = 0）。Alpha=0 时，local vol 网格在 `(t, ratio)` 空间里是固定的，但从 `V(S+ε)` 和 `V(S-ε)` 两个 MC 路径看，两条路径在"同一个 ratio 点"查到的 local vol 完全相同，但起始的 spot 不同——这意味着路径最终到达的 `S_T` 分布不同，但波动率的"形状"是一样的。

**Shadow delta 的模型值 ≠ 0**：α=0 的 MC delta 计算了 local vol 模型**内在的** delta，即让 local vol 曲面固定在 `(t, S/S_ref)` 坐标系时，期权价值关于 spot 的偏导。这和 BS delta 不同，因为 BS 是在 IV 坐标系里求导，local vol 是在物理空间求导。

### 5.3 金融意义

文档第 0 节给出了三个 delta 的对应关系：

- **α=0**：等价于让 local vol 曲面静止不动（"Frozen local vol"），delta 来自纯粹的概率论——spot 动了之后，在已知 local vol 曲面下，到期收益的期望值变化多少。这是**最小的 delta**，因为没有考虑 IV 曲面会随 spot 移动带来的额外对冲需求。

- **α=1**：等价于 BS delta，即假设每个 strike 上的 IV 固定（Sticky Strike）。市场通常用这个作为基准对冲。

- **α=2**：等价于 Sticky Moneyness delta，假设相对 moneyness `K/S` 相同的期权 IV 固定。Spot 上涨后，OTM call 的 IV"跟着"上涨，产生比 BS 更大的 delta。

### 5.4 Alpha=0 delta 的实际用途

在研究文档的 Step 3 里，需要建立 `β(α)` 曲线：

```
β = -dIV/d(logS)   ← 经验 stickiness 度量
```

α=0 给出 β 的一个端点：local vol 曲面完全冻结时，模型隐含的 beta 是多少。
α=1 应该给出 β ≈ 0（Sticky Strike，作为 MC 实现正确性的 sanity check）。
α=2 给出另一个端点。

这条 `β(α)` 曲线就是"换算器"：Step 7 中，预测出市场当天的经验 beta，再通过这条曲线反查到对应的 alpha，用来算对冲 delta。

---

## 六、代码的完整执行路径

### 6.1 LocalVolGrid 的构建

[montecarlo.py:120](../svi_localvol/montecarlo.py#L120)，`LocalVolGrid.build()` 预先计算网格：

```python
# 对每个 (日期, spot_ratio) 网格点
for i, T in enumerate(dates):
    for ratio in ratios:
        K = ratio * surface.market.spot   # K = ratio × S_ref
        # 调用 surface.local_vol()，传入 spot_adj 和 alpha
        raw[i] = surface.local_vol(T, K, spot_adj=spot_adj, alpha=alpha)
```

**计算 delta 时构建三个网格**（[step04:35-39](../pricing_svi_localvol_calibration/step04_mc_validation.py#L35-L39)）：

```python
grid      = LocalVolGrid.build(surface, maturity, spot_adj=0.0)
grid_up   = LocalVolGrid.build(surface, maturity, spot_adj=ln(S+ε)/S_ref)
grid_down = LocalVolGrid.build(surface, maturity, spot_adj=ln(S-ε)/S_ref)
```

三个网格在相同的 `(date, ratio)` 点上，因为 `spot_adj` 不同（进而影响 `y_adj`），查到的 local vol 值不同。

### 6.2 MC 路径演化

[montecarlo.py:306-315](../svi_localvol/montecarlo.py#L306-L315)，双时钟 SDE：

```python
for i in range(len(self.dt_r)):
    hr = self.dt_r[i] / m        # Act/365 漂移步长
    hv = self.dt_v[i] / m        # Business/260 方差步长
    for j in range(m):
        ratio = s / s0_ref       # 当前 spot 除以 S_ref
        sig = self.grid.sigma_bilinear(i + j/m, ratio)  # 查网格
        s = s * exp(b*hr - 0.5*sig²*hv + sig*sqrt(hv)*z[i*m+j])
```

关键：漂移用 Act/365，方差用 Business/260，两个时钟独立积分，保证 `∫ σ_loc² d(τ_vol) = w`。

### 6.3 Delta 的有限差分

[montecarlo.py:401-410](../svi_localvol/montecarlo.py#L401-L410)，**复用同一组随机数**（Common Random Numbers）：

```python
z = self._draw()   # 一次抽取，三次复用

base  = pv_lv(s0,      grid,      z)
up    = pv_lv(s0 + ε,  grid_up,   z)  # 复用 z！
down  = pv_lv(s0 - ε,  grid_down, z)  # 复用 z！

delta = (up - down) / (2*ε)
```

复用 z 是关键：否则 1% 的 bump 信号会被 MC 噪声淹没。相同的随机数让两条路径高度相关，差分后噪声相消。

---

## 八、数值示例：三种 Delta 的完整计算

### 8.1 参数设定

为使计算干净，令 b=0（无 carry，F=S），r=0.05：

| 参数 | 值 |
|------|----|
| S_ref | 100 |
| K | 100（平值 call） |
| T | 1 年，τ_vol = τ_r = 1.0 |
| b | 0，r = 0.05 |
| F_ref | 100，df = e^(-0.05) ≈ 0.9512 |
| σ_imp | 0.20（8.2–8.4 用平坦微笑；8.5 引入偏度） |
| ε（bump 大小）| 1（1%）→ S_up=101，S_dn=99 |
| spot_adj_up | ln(101/100) ≈ 0.00995 |
| spot_adj_dn | ln(99/100) ≈ -0.01005 |

---

### 8.2 BS Delta（解析公式）

代码：[blackscholes.py:53](../svi_localvol/blackscholes.py#L53) `bs_delta_w()`。

```
w    = σ² × τ_vol = 0.04
√w   = 0.20

d1   = ln(F/K)/√w + 0.5·√w = 0/0.20 + 0.10 = 0.10
d2   = d1 - √w = -0.10

N(0.10) = 0.5398
carry_factor = e^(b·τ_r) = 1.0

Δ_BS = df × carry × N(d1) = 0.9512 × 1.0 × 0.5398 = 0.5135
```

---

### 8.3 MC Implied Vol Delta（等价于 BS 有限差分）

用固定 σ_imp=0.20 在三个 spot 下分别定价，相当于用常数波动率跑 MC：

```
V(S=100):
  F=100, d1=0.10, d2=-0.10
  N(0.10)=0.5398, N(-0.10)=0.4602
  V = 0.9512 × (100×0.5398 - 100×0.4602) = 0.9512 × 7.96 = 7.572

V(S=101):
  F_up=101, d1_up = ln(101/100)/0.20 + 0.10 = 0.0498 + 0.10 = 0.1498
  d2_up = -0.0502
  N(0.1498)≈0.5595, N(-0.0502)≈0.4800
  V = 0.9512 × (101×0.5595 - 100×0.4800) = 0.9512 × 8.51 = 8.094

V(S=99):
  F_dn=99, d1_dn = ln(99/100)/0.20 + 0.10 = -0.0503 + 0.10 = 0.0498
  d2_dn = -0.1502
  N(0.0498)≈0.5199, N(-0.1502)≈0.4403
  V = 0.9512 × (99×0.5199 - 100×0.4403) = 0.9512 × 7.43 = 7.068

Δ_imp_MC = (8.094 - 7.068) / (2×1) = 0.513 ≈ Δ_BS = 0.5135  ✓
```

微差来自 1% 的有限步长。本质上，MC 路径用 dS = σ_imp·S·dW 演化；当 σ_imp 对所有 spot 一样，上下两条路径只差初始点，有限差分就是 BS 公式的数值微分。

---

### 8.4 Local Vol MC Delta（alpha=1，平坦微笑）

#### 第一阶段：Dupire 计算 σ_loc

代码：[surface.py:289](../svi_localvol/surface.py#L289) `local_vol()`。

平坦微笑 `w(y,τ) = 0.04·τ`，所有导数：

```
∂w/∂τ    = 0.04
∂w/∂y    = 0          ← 无偏度
∂²w/∂y²  = 0          ← 无曲率

D = 1 - (y_adj/w)·0 + 0.25·(...)·0² + 0.5·0 = 1（对任意 y_adj）

σ_loc = √(0.04 / 1) = 0.20 = σ_imp
```

平坦微笑是 Dupire 的退化情形：D=1 对所有 y_adj 恒成立，所以无论 alpha 取何值，σ_loc 都等于 0.20。平坦微笑下三种方法在数值上必然相等；alpha 的差异要在有偏度的曲面上才会显现（见 8.5 节）。

#### 第二阶段：三个 LocalVolGrid 的构建

代码：[step04_mc_validation.py:36-40](../pricing_svi_localvol_calibration/step04_mc_validation.py#L36-L40)，[montecarlo.py:151-162](../svi_localvol/montecarlo.py#L151-L162)。

网格里每个节点 `K = ratio × S_ref = ratio × 100`，调用 `surface.local_vol(T, K, spot_adj=spot_adj_up, alpha=1)`：

```python
# total_variance() 里：
y = ln(K / F_ref) = ln(ratio × 100 / 100) = ln(ratio)
# ← y 永远用 S_ref 的 forward，与 spot_adj 无关

# local_vol() 里（alpha=1）：
shift  = 1 × spot_adj_up = 1 × 0.00995
y_adj  = y - shift
       = ln(ratio) - ln(101/100)
       = ln(ratio × 100 / 101)
       = ln(K / F_up)              ← 关键等式
```

对 ratio=1.0 (K=100)：

```
Alpha=0：y_adj = 0 - 0×0.00995 = 0     → D=1 → σ_loc=0.20（与 base 相同，Frozen）
Alpha=1：y_adj = 0 - 1×0.00995 = -0.00995 = ln(100/101) = ln(K/F_up) → D=1 → σ_loc=0.20
```

平坦曲面下数值上一样，但 **alpha=1 把 Dupire 的视角从"相对 F_ref"换成了"相对 F_up"**。这个换视角在有偏度时会产生不同的 σ_loc 值（见 8.5 节）。

#### 第三阶段：MC 路径演化

代码：[montecarlo.py:306-315](../svi_localvol/montecarlo.py#L306-L315)。

三组路径从 S_0 = 100/101/99 出发，复用同一组随机数 z（CRN）：

```python
z = self._draw()          # 一次抽取，三组复用消除 MC 噪声

# 每步：ratio = s / s0_ref（s0_ref=100）
sig = grid.sigma_bilinear(t, ratio)   # 查网格
s   = s × exp(b·hr - 0.5·sig²·hv + sig·√hv·z)
```

平坦曲面下三个网格在同一 ratio 处给出相同的 σ_loc=0.20，三组路径等效于常数波动率 SDE，有限差分结果和 BS 一致。

#### 第四阶段：有限差分 delta

代码：[montecarlo.py:401-413](../svi_localvol/montecarlo.py#L401-L413)。

```
Δ_local_α1 = (V(101, grid_up) - V(99, grid_dn)) / (2×1)
           = (8.094 - 7.068) / 2 = 0.513 ≈ Δ_BS  ✓
```

---

### 8.5 有偏度曲面：alpha 的差异如何显现

引入负偏度曲面：`w(y, τ) = (0.20 - 0.10·y)² · τ`（OTM put vol 高于 ATM，OTM call vol 低于 ATM）。

在 K=100，y=0，τ=1：

```
w        = 0.04
∂w/∂τ    = 0.04
∂w/∂y    = 2×(0.20)×(-0.10)×1 = -0.04     ← 偏度项，非零
∂²w/∂y²  = 2×(0.10)²×1 = 0.02
```

**base 网格（spot_adj=0）**，y_adj=0：

```
D = 1 - (0/0.04)×(-0.04) + 0.25×(-0.25 - 25 + 0)×(-0.04)² + 0.5×0.02
  = 1 - 0 - 0.01010 + 0.010 = 0.99990

σ_loc_base = √(0.04/0.99990) = 0.2001
```

**grid_up（spot_adj_up=0.00995），alpha=0**：

```
y_adj = 0 - 0×0.00995 = 0        ← 与 base 相同
D = 0.99990                       ← 与 base 相同
σ_loc_up_α0 = 0.2001              ← 完全冻结（Frozen）
```

**grid_up（spot_adj_up=0.00995），alpha=1**：

```
y_adj = 0 - 1×0.00995 = -0.00995

y_adj/w        = -0.00995/0.04 = -0.2488
D_term1        = -(-0.2488)×(-0.04) = -0.009950
inner          = -0.25 - 25 + (0.00995)²/(0.04)² = -25.25 + 0.0619 = -25.188
D_term2        = 0.25×(-25.188)×(-0.04)² = -0.010075
D_term3        = 0.5×0.02 = 0.010

D_up_α1 = 1 - 0.009950 - 0.010075 + 0.010 = 0.98998

σ_loc_up_α1 = √(0.04/0.98998) = 0.2010     ← 与 base 不同
```

对比汇总：

| 网格 | K=100 处 y_adj | σ_loc | 说明 |
|------|---------------|-------|------|
| base（spot=100） | 0 | 0.2001 | F=100，ATM |
| grid_up alpha=0 | 0 | 0.2001 | 冻结，与 base 完全相同 |
| grid_up alpha=1 | -0.00995 = ln(100/101) | 0.2010 | 用 F_up=101 重新定位，与 base 不同 |

**经济含义**：在负偏度曲面里，K=100 从"ATM"变成"OTM put"（相对于 F_up=101），偏度把 σ_loc 推高到 0.2010。这正是 sticky strike 假设的预期：当 spot 上涨，K=100 成为 put 侧，其波动率应该略高。Alpha=1 通过 y_adj = ln(K/F_up) 精确捕捉了这一效应。

Alpha=0 则让网格停在 y_adj=0，仿佛 F 没有移动，σ_loc 冻结在 0.2001 不变，MC delta 偏低（没有计入偏度随 forward 移动带来的额外 hedge 需求）。

---

### 8.6 三种 Delta 的联系（总结）

| 方法 | 波动率输入 | 等价假设 | Alpha |
|------|-----------|---------|-------|
| BS 解析 | σ_imp(K) 不随 S 变 | Sticky Strike | — |
| MC implied vol | 三组路径用相同 σ_imp(K) | Sticky Strike | — |
| MC local vol α=1 | grid_up 用 y_adj=ln(K/F_up) | Sticky Strike | 1 |
| MC local vol α=0 | 三个网格完全相同 | Frozen local vol | 0 |

前三者在数学上等价，因为：
1. 用固定 σ_imp(K) 跑 MC → 期权价格 = BS 价格（定义）
2. alpha=1 的 local vol 用 y_adj=ln(K/F_bump) → Dupire 分母的视角等同于"从 F_bump 重新 calibrate"
3. 从同一张 sticky-strike IV 曲面重新 calibrate → 得到和"用 sigma_imp 直接 MC"相同的价格
4. 因此 alpha=1 local vol MC delta = BS delta

Alpha=0 是另一个极端：local vol 曲面冻结在 (T, K) 坐标下，delta 是 local vol 模型"内在"的对冲比率，与 BS delta 不同。

---

## 七、总结

| 问题 | 答案 |
|------|------|
| Implied → Local 怎么来的？ | Dupire 公式：`σ_loc² = (∂w/∂τ) / D`，分母 D 从 IV 曲面的 strike 曲率提取 |
| Local vol ≠ Implied vol？ | 是的，数值不同。Local vol 是对 IV 曲面求导的结果，一般比 IV 更高 |
| Alpha 的作用？ | 控制 Dupire 分母 D 里 `y_adj` 的坐标：`y_adj = y_new - α·spot_adj` |
| Alpha=1 为什么能复现 BS delta？ | α=1 隐含"每个 strike 上的 IV 不随 spot 变"（Sticky Strike），shadow delta ≈ 0，Δ ≈ Δ_BS |
| Alpha=0 的 delta 有什么意义？ | Local vol 曲面在物理坐标 `(t, S/S_ref)` 下冻结时，期权的自然对冲比率。是 `β(α)` 曲线的一个端点 |
| 为什么都需要三个网格？ | Forward 依赖 spot（`F = S·exp(b·τ)`），三个 spot 对应三个不同的 moneyness，进而三个不同的 local vol 曲面 |
