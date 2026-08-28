# `spot_adj`、mentor 的 Delta 检查与 Stickiness Ratio \(R\)

> **历史/legacy 文档。** 本文记录原 pricing task 的解析 smile-shift
> `R=-1/0/1` 约定，不是 `动态Alpha对冲研究.docx` 的研究约定。新研究统一使用
> `alpha=0/1/2`，并通过 local-vol MC 得到 delta；两者不能只改变量名后互换。
> 当前有效的衔接约定见 `DYNAMIC_ALPHA_READINESS.md`。

本文只解释三个容易混淆的概念：

1. `spot_adj` 本身是什么；
2. mentor 所说的 up/down bump 应该怎样做；
3. 原扩展代码中定义的 legacy Stickiness Ratio \(R\) 是什么。

全文坚持以下记号：

- `spot_adj`：一次具体 spot 变化对应的 log-spot 变动量；
- \(R\)：Stickiness Ratio，是一个无量纲响应系数；
- 真正进入调整后 moneyness 的量是二者的乘积 \(R\times spot\_adj\)。

这三个量不能互相替代。当前输入数据中所有期限的 \(R=1\)，因此数值上

\[
R\times spot\_adj=spot\_adj,
\]

但这只是当前数据的特殊情况，不代表 `spot_adj` 和 \(R\) 是同一个概念。

---

## 1. 先完全不谈 \(R\)：`spot_adj` 是什么

当前参考 spot 是：

\[
S_{\mathrm{ref}}=1.
\]

如果我们考察一个新的 spot \(S'\)，定义：

\[
spot\_adj=\ln\frac{S'}{S_{\mathrm{ref}}}.
\]

所以 `spot_adj` 描述的是：

> 新 spot 相对于校准曲面时参考 spot 的实际 log 变动。

它是一个已经发生的、具体的位移量，不是关于未来 smile 如何移动的规则。

### 1.1 Spot 上涨 1%

\[
S_{\mathrm{up}}=1.01,
\]

\[
spot\_adj_{\mathrm{up}}
=\ln\frac{1.01}{1}
\approx 0.00995033.
\]

### 1.2 Spot 下跌 1%

\[
S_{\mathrm{down}}=0.99,
\]

\[
spot\_adj_{\mathrm{down}}
=\ln\frac{0.99}{1}
\approx -0.01005034.
\]

上下两个数不是完全对称的，因为 log 函数不是线性的：

```text
up   spot_adj = +0.00995033
down spot_adj = -0.01005034
```

### 1.3 `spot_adj` 不等于 spot

需要区分：

| 量 | up 情景 | down 情景 | 用途 |
|---|---:|---:|---|
| 新 spot \(S'\) | 1.01 | 0.99 | 改变 forward、路径初值和 payoff 分布 |
| `spot_adj` | 0.00995033 | -0.01005034 | 告诉 local-vol 公式参考 spot 改变了多少 |

计算 delta 时，这两个量都需要：

```text
up:
    模拟初始 spot = 1.01
    localvol 的 spot_adj = log(1.01 / 1)

down:
    模拟初始 spot = 0.99
    localvol 的 spot_adj = log(0.99 / 1)
```

只改变路径初值而不改变 `spot_adj`，和 mentor 提议的做法不是同一个计算。

---

## 2. Docx 对 `spot_adj` 的原始定义

Docx Step 3 给出的定义是：

\[
y_{\mathrm{adj}}
=y-spot\_adj\times StickinessRatio.
\]

其中原始 moneyness 是：

\[
y=\ln\frac{K}{F_{\mathrm{ref}}(T)}.
\]

Dupire 分母写成：

\[
D
=1-\frac{y_{\mathrm{adj}}}{w}w_y
+\frac12w_{yy}
+\frac14
\left(
-\frac14-\frac1w+\frac{y_{\mathrm{adj}}^2}{w^2}
\right)w_y^2.
\]

local variance 为：

\[
\sigma_{\mathrm{loc}}^2
=\frac{w_\tau}{D}.
\]

这里必须注意 docx 的字面顺序：

1. \(w,w_\tau,w_y,w_{yy}\) 仍由 Step 2 的同一套 \(w(y,T)\) 构造得到；
2. 计算 Dupire 分母时，把 \(y\) 换成 \(y_{\mathrm{adj}}\)；
3. `spot_adj` 先表示 spot 的实际 log 变动；
4. Stickiness Ratio 再决定这个变动有多少进入 \(y_{\mathrm{adj}}\)。

因此，按照 docx 的字面定义：

> `spot_adj` 并不是直接把 Step 2 的 implied-vol smile 整体平移；它首先用于调整 Dupire 分母中的 moneyness 坐标。

---

## 3. 当前项目的默认 `localvol()` 正在做什么

题目要求的公开函数是：

```python
localvol(T, K, spot_adj)
```

它最终调用：

```python
surface.local_vol(
    T,
    K,
    spot_adj=spot_adj,
    shift_mode="denominator",
)
```

`shift_mode="denominator"` 是默认值，对应 docx 的字面实现。

代码内部先执行：

```python
surface_shift = 0.0
res = total_variance(
    T,
    K,
    spot_adj=surface_shift,
    order=2,
)
```

因此下列量都在原始 \(y\) 上求值：

\[
w(y,T),\qquad
w_\tau(y,T),\qquad
w_y(y,T),\qquad
w_{yy}(y,T).
\]

随后只调整：

\[
y_{\mathrm{adj}}=y-spot\_adj.
\]

当前代码没有在这里再显式乘每个 slice 的 `StickinessRatio`。不过当前输入中所有期限都是：

```python
StickinessRatio = 1
```

所以当前数据下：

\[
y-spot\_adj\times 1=y-spot\_adj,
\]

数值结果与 docx 相同。

最终 local vol 是：

\[
\sigma_{\mathrm{loc}}^2(T,K;spot\_adj)
=
\frac{w_\tau(y,T)}
{D\big(y-spot\_adj;\,w(y,T),w_y(y,T),w_{yy}(y,T)\big)}.
\]

所以当前公开函数的真实含义是：

```text
spot_adj 不改变 w 和它的导数
spot_adj 改变 Dupire 分母使用的 y 坐标
分母改变以后，local vol 数值随之改变
```

这不是简单地把最终 local-vol 曲线左右平移。

---

## 4. 为什么改变 spot 后需要给 `localvol()` 传 `spot_adj`

基准 spot 下：

\[
y_{\mathrm{ref}}
=\ln\frac{K}{F_{\mathrm{ref}}}.
\]

如果 spot 从 \(S_{\mathrm{ref}}\) 变成 \(S'\)，新的 forward 为：

\[
F'=F_{\mathrm{ref}}\frac{S'}{S_{\mathrm{ref}}},
\]

所以新 moneyness 是：

\[
\begin{aligned}
y'
&=\ln\frac{K}{F'}\\
&=\ln\frac{K}{F_{\mathrm{ref}}}
-\ln\frac{S'}{S_{\mathrm{ref}}}\\
&=y_{\mathrm{ref}}-spot\_adj.
\end{aligned}
\]

这就是 mentor 要传入：

```python
log(shifted_spot / ref_spot)
```

的直接原因。

对于当前 `StickinessRatio=1`：

\[
y_{\mathrm{adj}}=y_{\mathrm{ref}}-spot\_adj=y'.
\]

所以 `spot_adj` 在这里的作用是：

> 当 spot/forward 改变后，让 Dupire 分母使用新的 forward moneyness。

---

## 5. Mentor 的 up/down Delta 检查：一步一步做

Mentor 的建议是：

> 算 delta 时，localvol 参数 `spot_adj` 分别传入  
> `log(1.01/1)` 和 `log(0.99/1)`，检查 local-vol MC delta 能否和 BS delta 对上。

这项检查先不需要讨论任意 \(R\)，因为当前数据已经给定所有 `StickinessRatio=1`。

### 5.1 基准情景

基准 spot：

\[
S_0=1.
\]

基准调整量：

\[
spot\_adj_0=\ln(1/1)=0.
\]

基准 local-vol 网格：

```text
每个 (date, ratio)
    K = ratio × ref_spot
    sigma_base = localvol(date, K, spot_adj=0)
```

### 5.2 Up 情景

第一步，设置新的路径初值：

\[
S_{\mathrm{up}}=1.01.
\]

第二步，计算本次实际 log-spot 变动：

\[
spot\_adj_{\mathrm{up}}=\ln(1.01/1).
\]

第三步，重新生成 up local-vol 网格：

```text
每个 (date, ratio)
    K = ratio × ref_spot
    sigma_up = localvol(
        date,
        K,
        spot_adj=log(1.01/1),
    )
```

第四步，用这张 `grid_up` 从 1.01 开始模拟：

```text
terminal_spots(
    initial_spot=1.01,
    grid=grid_up,
    random_numbers=z,
)
```

得到：

\[
PV_{\mathrm{LV,up}}.
\]

### 5.3 Down 情景

路径初值：

\[
S_{\mathrm{down}}=0.99.
\]

实际 log-spot 变动：

\[
spot\_adj_{\mathrm{down}}=\ln(0.99/1).
\]

重新生成 down local-vol 网格：

```text
每个 (date, ratio)
    K = ratio × ref_spot
    sigma_down = localvol(
        date,
        K,
        spot_adj=log(0.99/1),
    )
```

再用相同随机数 \(z\) 从 0.99 开始模拟，得到：

\[
PV_{\mathrm{LV,down}}.
\]

### 5.4 Local-vol MC delta

\[
\Delta_{\mathrm{LV}}
=
\frac{
PV_{\mathrm{LV,up}}-PV_{\mathrm{LV,down}}
}{
1.01-0.99
}.
\]

也就是：

\[
\Delta_{\mathrm{LV}}
=
\frac{
PV_{\mathrm{LV,up}}-PV_{\mathrm{LV,down}}
}{0.02}.
\]

up/down 必须使用相同随机数，这样两边的随机误差大部分相消。

---

## 6. BS 一侧怎样比较

Docx Step 4 的 BS 一侧先取基准 implied vol：

\[
\sigma_{\mathrm{imp}}
=ImpliedVol(T,K).
\]

然后使用同一个 \(\sigma_{\mathrm{imp}}\) 分别计算：

\[
PV_{\mathrm{BS,up}}
=BS(S=1.01,K,\sigma_{\mathrm{imp}}),
\]

\[
PV_{\mathrm{BS,down}}
=BS(S=0.99,K,\sigma_{\mathrm{imp}}).
\]

于是：

\[
\Delta_{\mathrm{BS,FD}}
=
\frac{
PV_{\mathrm{BS,up}}-PV_{\mathrm{BS,down}}
}{0.02}.
\]

小 bump 下，它应接近 cost-of-carry BS 解析 delta。

Mentor 要比较的是：

\[
\Delta_{\mathrm{LV}}
\quad\text{vs}\quad
\Delta_{\mathrm{BS}}.
\]

这项检查的目的不是证明所有 smile dynamics 都只有一个正确答案，而是检查：

> 当 spot 变动后，local-vol Dupire 分母使用了正确的新 moneyness，并且 up/down 分别重建了对应的 local-vol 网格时，local-vol 对这个 vanilla 的 bump delta 能否复现同一 implied-vol 定价口径下的 BS delta。

有限 bump、MC 噪声、时间离散和网格插值都会产生小误差，因此验收使用约 `0.001` 的绝对误差容忍度，而不是要求机器精度完全相等。

---

## 7. 当前 Step 4 已如何接入 mentor 的这项检查

修改前的 Step 4 只建立一张 `spot_adj=0` 的网格，并让 up/down 复用它；
那是 frozen-grid delta，不是 mentor 要求的口径。

现在 Step 4 会建立三张网格：

```python
grid_base = LocalVolGrid.build(..., spot_adj=0.0)
grid_up = LocalVolGrid.build(..., spot_adj=log(1.01/1))
grid_down = LocalVolGrid.build(..., spot_adj=log(0.99/1))
```

| 情景 | 路径初始 spot | 使用的 local-vol 网格 |
|---|---:|---|
| base | 1.00 | `grid(spot_adj=0)` |
| up | 1.01 | `grid(spot_adj=log(1.01/1))` |
| down | 0.99 | `grid(spot_adj=log(0.99/1))` |

up/down local-vol 路径和 constant-implied-vol 路径使用同一组随机数。
结果写入：

```text
output/step4_delta_comparison.csv
```

因此这次修改的关键不在于 bump 是不是 1%，而在于：

> up/down 是否分别重新调用 `localvol(..., spot_adj)` 并重建网格。

---

## 8. 到这里才引入 \(R\)

现在再讨论后面代码定义的 \(R\)。

\(R\) 是 Stickiness Ratio。它回答的是：

> 给定一次已经发生的 `spot_adj`，有多少比例、以什么方向进入 smile/moneyness 调整？

一般公式是：

\[
y_{\mathrm{adj}}
=y-R\times spot\_adj.
\]

这里两者分工非常明确：

| 量 | 回答的问题 |
|---|---|
| `spot_adj` | 这次 spot 实际变了多少？ |
| \(R\) | 曲面/moneyness 对这次变化响应多少？ |

例如 spot 从 1 变到 1.01，始终有：

\[
spot\_adj=\ln(1.01).
\]

然后不同 \(R\) 给出不同的有效调整：

| \(R\) | \(R\times spot\_adj\) | \(y_{\mathrm{adj}}\) |
|---:|---:|---|
| 0 | 0 | \(y\) |
| 0.5 | \(0.5\ln(1.01)\) | \(y-0.5\ln(1.01)\) |
| 1 | \(\ln(1.01)\) | \(y-\ln(1.01)\) |
| -1 | \(-\ln(1.01)\) | \(y+\ln(1.01)\) |

`spot_adj` 没有随着 \(R\) 改变；改变的是 \(R\) 乘在它前面的响应比例。

当前输入数据所有期限：

```python
StickinessRatio = 1
```

所以 docx/mentor 的计算直接表现为：

\[
y_{\mathrm{adj}}=y-spot\_adj.
\]

这并不是省略了 \(R\) 的概念，而是因为当前 \(R=1\)。

---

## 9. \(R\) 如何描述 implied smile 的动态

到 Step 5 的扩展代码里，\(R\) 被用来定义完整 implied smile 的动态：

\[
y_{\mathrm{eval}}(S)
=
\ln\frac{K}{F_{\mathrm{ref}}}
-R\ln\frac{S}{S_{\mathrm{ref}}}.
\]

因为：

\[
spot\_adj=\ln\frac{S}{S_{\mathrm{ref}}},
\]

所以它仍然可以写成：

\[
y_{\mathrm{eval}}=y-R\times spot\_adj.
\]

### 9.1 \(R=0\)：sticky strike

\[
y_{\mathrm{eval}}=y.
\]

spot 变化时，固定绝对 strike \(K\) 的 implied vol 不变。于是固定-vol 的 BS delta 就是这一口径下的 delta。

### 9.2 \(R=1\)：sticky moneyness

\[
y_{\mathrm{eval}}=y-spot\_adj.
\]

spot 上涨 1% 时，smile 的 strike 位置也大致上涨 1%，因此 smile 相对于 \(K/S\) 保持不变。

### 9.3 \(R=-1\)：local-vol-like

\[
y_{\mathrm{eval}}=y+spot\_adj.
\]

spot 上涨时，曲面读取位置朝相反方向移动。扩展代码把这一端称为 local-vol-like。

在负 skew 下，不同 \(R\) 会产生不同 delta；今天的 PV 可以一样，但 spot bump 后读取到的 implied vol 不同，所以 delta 不同。

---

## 10. Docx 默认与 Step 5 扩展是两种不同的计算位置

这是当前项目最需要明确标注的地方。

### 10.1 Docx / 公开 `localvol()` 默认口径

默认：

```python
shift_mode="denominator"
```

执行：

```text
w、w_tau、w_y、w_yy：仍在原始 y 上计算
y_adj：使用 y - spot_adj × StickinessRatio
只有 Dupire 分母中的 moneyness 被调整
```

这正是 mentor 所说“把 `log(shifted_spot/refspot)` 传给 localvol 参数”的直接上下文。

### 10.2 Step 5 `deltas.py` 的完整 smile-dynamics 口径

Step 5 使用：

```text
整个 total-variance smile 在 y - R × spot_adj 上求值
w 及其导数一起改变
```

它回答的是更广泛的问题：

> 市场 implied smile 在 spot 变化以后整体如何移动？

两者使用了相似的 \(y-R\times spot\_adj\) 记号，但这个调整进入计算的位置不同：

| 口径 | 调整进入哪里 |
|---|---|
| Docx 默认 localvol | 只进入 Dupire 分母的 \(y_{\mathrm{adj}}\) |
| Step 5 完整 smile dynamics | 进入 \(w,w_y,w_{yy}\) 的整个曲面求值位置 |

因此不能直接把两者当成完全相同的实现。

当前 `surface.py` 同时保留了：

```python
shift_mode="denominator"
shift_mode="surface"
```

- mentor/docx 的这项 delta 检查应先按默认 `denominator` 口径理解；
- 如果要研究 Step 5 的完整 smile movement，才使用 `surface` 口径。

---

## 11. 最简调用链

### Mentor 要检查的 local-vol delta

```text
ref_spot = 1

up:
    S_up = 1.01
    spot_adj_up = log(1.01 / 1)
    grid_up = LocalVolGrid.build(
        spot_adj=spot_adj_up,
        shift_mode="denominator",
    )
    PV_up = MC(initial_spot=1.01, grid=grid_up, z=same_z)

down:
    S_down = 0.99
    spot_adj_down = log(0.99 / 1)
    grid_down = LocalVolGrid.build(
        spot_adj=spot_adj_down,
        shift_mode="denominator",
    )
    PV_down = MC(initial_spot=0.99, grid=grid_down, z=same_z)

delta_LV = (PV_up - PV_down) / 0.02
```

### BS 对照

```text
sigma_imp = ImpliedVol(maturity, strike)

PV_BS_up   = BS(spot=1.01, sigma=sigma_imp)
PV_BS_down = BS(spot=0.99, sigma=sigma_imp)

delta_BS_FD = (PV_BS_up - PV_BS_down) / 0.02
```

最后比较：

\[
\left|\Delta_{\mathrm{LV}}-\Delta_{\mathrm{BS}}\right|.
\]

---

## 12. 一句话记忆

不要记成：

```text
spot_adj 就是 R
```

应该记成：

```text
spot_adj = 这一次 spot 实际发生的 log 变动
R        = 曲面对这个变动响应多少的系数

effective adjustment = R × spot_adj
```

在当前数据中：

```text
R = 1
```

所以 mentor 给出的 up/down 参数直接是：

```text
spot_adj_up   = log(1.01/1)
spot_adj_down = log(0.99/1)
```

而 mentor 的核心要求是：

> 不要只把 MC 初始 spot 改成 1.01/0.99；还要把对应 `spot_adj` 传进 `localvol()`，分别重建 up/down local-vol 网格，再和固定 implied-vol 口径的 BS delta 比较。
