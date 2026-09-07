# Step 6–7 代码、模型、回测与归因详解

本文依据当前代码、各步骤 manifest、Step 6 输出，以及 2026-09-07 完成的 Step 7 运行结果编写。分析对象是以下命令产生的结果：

```bash
python -m dynamic_alpha_hedging step7 --book atm --factors 1 \
  --converter fixed \
  --reference-inverse output/dynamic_alpha/step03/alpha_beta_inverse.csv \
  --fast
```

这是一份 **3M ATM、单因子、固定 Beta–Alpha 反查曲线、快速 MC、零交易成本** 的研究性回测。它已经完整执行，但不是正式数值精度版本，也不能据此认定动态 Alpha 已经优于固定 Alpha。

本文重点解释 Step 6 和 Step 7；为了让数据链闭合，也会说明它们如何接收 Step 1、2、4、5 的结果。主要代码入口如下：

- [step06.py](../dynamic_alpha_hedging/step06.py)：因子预测、Beta 曲面还原和样本外评价；
- [step07.py](../dynamic_alpha_hedging/step07.py)：回测编排、策略、现金账户、归因和汇总；
- [hedging.py](../dynamic_alpha_hedging/hedging.py)：固定合约估值、每日 MC 映射、Beta 反查 Alpha、Delta 插值；
- [surface.py](../svi_localvol/surface.py)：SVI 隐含波动率和带 Alpha 的 Dupire local vol；
- [montecarlo.py](../svi_localvol/montecarlo.py)：LocalVolGrid、路径模拟、控制变量、IV 反解与 bump Delta。

---

## 1. 整条数据链在研究什么

当前 pipeline 的主线是：

```text
svi_param.pkl
  -> 每日 SVI 曲面和 Spot
  -> Step 1: dIV_surface(t, tenor, level)
  -> Step 2: Daily / Rolling Beta
  -> Step 4: 56 维 Daily-Beta 曲面分解为 3 个因子
  -> Step 5: 构造 close-t 状态特征
  -> Step 6: 预测下一日 Beta 因子
  -> Step 7: Beta -> Alpha -> Local-vol Delta -> 对冲回测与归因
```

最核心的时间关系是：在日期 (t) 收盘，只允许使用截至 (t) 已知的数据，预测 (t+1) 的 Beta，再据此计算从 (t) 持有到 (t+1) 的 Delta。实际的 (S_{t+1})、曲面和期权价格只能用于事后评价。

### 1.1 原始数据

源文件是 `data/svi_param.pkl`。每个日期包含：

- 当天 `Spot`，同时作为该日曲面的 `refSpot`；
- 一组滚动的 `VolDate`；
- 每个 VolDate 的 `ATMVol`、`Skew`、`Putwing`、`Callwing`、`Kurt` 和 `StickinessRatio`。

[data_loader.py 的 `build_surface()`](../dynamic_alpha_hedging/data_loader.py#L283) 将每个日期的数据转成 `VolSurface`。当前约定是：

- date-only key 原样作为市场日期，不做无法验证的跨时区平移；
- 同一日重复 VolDate 保留第一条；
- rate=3.6%、dividend=3%、repo=0 是配置假设，不是原始数据字段；
- Business/260 用于波动率时钟，Act/365 用于利率和现金账户；
- 无法构建 SVI 曲面的日期被记录并跳过。

当前原始文件有 672 个日期，成功构建 669 张曲面；3 张失败。Step 1 有 668 个相邻曲面转移，其中 18 个不是相邻业务日。

### 1.2 Step 1 和 Step 2 给 Step 6 的 Beta 标签

对每个固定的剩余期限 τ 和 level (m)，Step 1 每天先读：

$$
IV_t(\tau,m)=\sigma_t\bigl(\tau,K_t=mS_t\bigr).
$$

直接比较两天的同 level 会混入“沿前一天 smile 换 strike”的机械变化，所以 [step01.py 的 `_grid_changes()`](../dynamic_alpha_hedging/step01.py#L38) 计算：

$$
\begin{aligned}
dIV_{grid,t} &= \sigma_t(\tau,mS_t)-\sigma_{t-1}(\tau,mS_{t-1}),\\
smile\_crossing_t &= \sigma_{t-1}(\tau,mS_t)-\sigma_{t-1}(\tau,mS_{t-1}),\\
dIV_{surface,t} &= dIV_{grid,t}-smile\_crossing_t\\
&=\sigma_t(\tau,mS_t)-\sigma_{t-1}(\tau,mS_t).
\end{aligned}
$$

因此 `dIV_surface` 比较的是两天曲面在同一个实际 strike (mS_t) 上的变化，同时两边都取相同剩余期限 τ。它是曲面运动标签，不是在跟踪一张固定到期日的真实期权。

[step02.py 的 `_daily_beta()`](../dynamic_alpha_hedging/step02.py#L72) 定义：

$$
\beta_{t,\tau,m}^{daily}
=-\frac{dIV_{surface,t,\tau,m}}{\log(S_t/S_{t-1})}.
$$

正式 Daily Beta 只在相邻业务日且

$$
|d\log S_t|\ge 0.0025
$$

时保留，避免很小的分母放大噪声。

Rolling Beta 不是 Daily Beta 的移动平均。[step02.py 的 `_rolling_beta()`](../dynamic_alpha_hedging/step02.py#L98) 对每个 $\left(\tau,m\right)$ 使用最近最多 60 个有效相邻转移做带截距回归：

$$
dIV_{surface,s}=a-\beta^{rolling}_{t,\tau,m}d\log S_s+\varepsilon_s,
$$

至少需要 20 个观测。Rolling 回归不应用 Daily Beta 的 0.0025 门槛，而是直接使用有限、相邻的 `dIV` 与 `dlogS` 样本。它利用多天样本降噪，主要作为 Step 6 的状态特征和 benchmark；Step 6 的预测标签仍然是下一日 Daily Beta。

---

## 2. Step 4 如何把 Beta 曲面变成三个因子

Step 6 并不直接预测 56 个互不相关的 Beta。它使用 Step 4 建立的截面结构。

当前保留网格是：

- 7 个 tenor：2M、3M、6M、9M、1Y、1.5Y、2Y；
- 8 个 level：0.4、0.5、0.6、0.7、0.8、0.9、1.0、1.1；
- 合计 (7\times8=56) 个 Beta 单元。

1M 和 level=1.2 保留在上游诊断中，但因覆盖或信噪比较差，不进入 Step 4–6 的主模型。Step 4 只使用 Daily Beta 56 个单元都存在的日期，共 456 天；前 342 天训练载荷，后 114 天只投影，不重新拟合载荷。

### 2.1 第一因子：3M ATM 的真实 Beta

[step04.py 的 `run_step4()`](../dynamic_alpha_hedging/step04.py#L168) 将

$$
z_{1,t}=\beta_{t,3M,ATM}
$$

直接定义为第一因子，然后在训练期对每个曲面单元 $j=(\tau,m)$ 回归：

$$
\beta_{t,j}\approx c_j+\lambda_{1,j}z_{1,t}.
$$

在 3M ATM 锚点强制：

$$
c_{ATM}=0,\qquad \lambda_{1,ATM}=1.
$$

所以 3M ATM 并不是 PCA 算出来的抽象因子；它就是当天观测到的 3M ATM Daily Beta，有直接金融含义。

### 2.2 第二和第三因子：残差 PCA

先得到单因子残差：

$$
r_{t,j}=\beta_{t,j}-c_j-\lambda_{1,j}z_{1,t}.
$$

只在训练期残差矩阵上做 SVD，取前两个方向作为 `shape_loading_1/2`，并将所有日期的残差投影成 `shape_score_1/2`。最终：

$$
\widehat\beta_{t,j}
=c_j+\lambda_{1,j}z_{1,t}
+\lambda_{2,j}z_{2,t}
+\lambda_{3,j}z_{3,t}.
$$

当前 PCA 只做中心化，没有按单元方差标准化，也没有按 vega 或对冲风险加权。三因子训练解释率为 86.46%，测试重构 R² 为 77.08%；两因子测试重构 R² 为 71.40%。这里是使用当日真实因子重构当日真实 Beta 的能力，不是预测能力。

---

## 3. Step 6：从 close-t 状态预测下一日因子

### 3.1 Step 6 读取哪些文件

[step06.py 的 `load_step6_inputs()`](../dynamic_alpha_hedging/step06.py#L62) 读取：

| 输入 | 来源 | 用途 |
|---|---|---|
| `step05/factor_state_panel.csv` | Step 5 | 日期、状态特征、当日和下一日因子标签 |
| `step04/factor_loadings.csv` | Step 4 | (c_j,\lambda_{1,j},\lambda_{2,j},\lambda_{3,j}) |
| `step02/beta_daily.csv` | Step 2 | 下一日真实 Beta 曲面，用于样本外评分 |
| `step02/beta_rolling.csv` | Step 2 | close-t Rolling Beta 曲面和滚动因子特征 |

`_canonical_axes()` 把 CSV 浮点数映射回配置中的标准 tenor/level，防止 `1/12` 写入 CSV 后出现尾数差异。

### 3.2 16 个预测特征

Step 5 构造的 10 个市场状态特征定义在 [step05.py](../dynamic_alpha_hedging/step05.py#L30)：

1. `dlogS`：从前一市场观测到 close t 的收益；
2. `atm_iv_3m`：t 日 3M ATM IV；
3. `atm_iv_change_1d`：3M ATM IV 一日变化；
4. `atm_iv_change_5d`：3M ATM IV 最近 5 日累计变化；
5. `smile_slope_3m`：3M level 0.9 到 1.1 的 IV 对 log-level 斜率；
6. `term_slope_1y_minus_3m`：1Y ATM IV 减 3M ATM IV；
7. `realized_vol_20d`：20 日收益标准差乘 √260；
8. `recent_return_5d`；
9. `recent_return_20d`；
10. `vol_of_vol_20d`：3M ATM IV 一日变化的 20 日年化标准差。

[step06.py](../dynamic_alpha_hedging/step06.py#L31) 再加入：

- 3 个 `last_*`：截至 t 最近可见的三个 Daily-Beta 因子，缺失时只向前填充；
- 3 个 `rolling_*`：将 Step 2 Rolling Beta 曲面按 Step 4 载荷投影出的三个滚动因子。

合计 16 个输入特征。这里的 `dlogS` 是已经发生、在 close t 已知的 $t-1\to t$ 收益，不是未来 $t\to t+1$ 收益。

### 3.3 标签如何对齐并防止泄漏

[step06.py 的 `_prediction_panel()`](../dynamic_alpha_hedging/step06.py#L213) 验证下一行的 `previous_date` 必须等于当前 `observation_date`，然后定义：

$$
X_t=state_t,\qquad y_t=(z_{1,t+1},z_{2,t+1},z_{3,t+1}).
$$

只有确实相邻的市场观测才形成标签。`label_dlogS` 只用于把 Beta 误差换算为 dIV 误差，不进入特征。下一日 Spot、IV、收益和 Beta 都没有放进 (X_t)。

训练/测试不 shuffle，沿用 Step 4 的时间切分：训练标签 342 个，截止 2026-01-15；测试标签 114 个，2026-01-16 至 2026-08-31。

### 3.4 比较了哪些模型

[step06.py 的 `_fit_models()`](../dynamic_alpha_hedging/step06.py#L252) 对三个因子分别比较：

- `training_mean`：始终预测训练均值；
- `last_observed_factor`：用最近可见因子预测下一日；
- `rolling_factor`：使用 Rolling Beta 投影因子；
- `ridge_state`：16 特征的 Ridge；
- `hist_gradient_boosting`：主非线性模型。

Ridge 先做中位数填补、缺失指示和标准化，用三折 `TimeSeriesSplit` 在 7 个正则强度中按 MAE 选择。HGB 的参数在 [step06.py 的 `primary_factor_model()`](../dynamic_alpha_hedging/step06.py#L172) 固定为：

```text
loss = absolute_error
learning_rate = 0.05
max_iter = 200
max_leaf_nodes = 7
min_samples_leaf = 15
l2_regularization = 1.0
early_stopping = False
random_state = 20260807
```

HGB 是一组顺序拟合的浅树。每一轮新树主要修正前面模型的残差，学习率控制每棵树的贡献，叶子数和 L2 控制复杂度。当前使用绝对误差，因此模型更接近条件中位数预测；它并不是直接为最小化对冲误差方差而训练。HGB 原生处理 NaN，缺失值在每个分裂处学习进入哪一侧。实现行为可参考 [scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)。

### 3.5 Step 6 如何还原 Beta 曲面

对每个模型分别得到三个预测因子，然后 [step06.py 的 `_surface_summary()`](../dynamic_alpha_hedging/step06.py#L367) 构造嵌套模型：

$$
\widehat\beta_{t+1,j}^{(p)}
=c_j+\sum_{k=1}^{p}\lambda_{k,j}\widehat z_{k,t+1},
\qquad p\in\{1,2,3\}.
$$

在 3M ATM：(c=0,\lambda_1=1,\lambda_2=\lambda_3=0)，所以无论请求 1、2、3 个因子，3M ATM 预测都完全等于 ŷ₁。第二、第三因子只影响其他 tenor/level。

Beta 预测还被换算为曲面变化预测：

$$
\widehat{dIV}_{t+1,j}
=-\widehat\beta_{t+1,j}\,d\log S_{t+1}.
$$

未来收益只在测试结束后代入评分，不作为事前特征。

### 3.6 Step 6 当前结果

主 HGB 在 114 个有完整 Daily-Beta 曲面的测试日期上：

| 目标 | OOS R²（相对训练均值） | 相关系数 | 判断 |
|---|---:|---:|---|
| 3M ATM 因子 | 21.37% | 0.488 | 有可预测信号，但误差仍大 |
| shape 1 | 3.01% | 0.144 | 很弱 |
| shape 2 | -5.11% | 0.068 | 不如训练均值 |

全曲面 HGB 的结果：

| 因子数 | Beta OOS R² | dIV RMSE 相对 Rolling 改善 |
|---:|---:|---:|
| 1 | 3.96% | 1.16% |
| 2 | 7.00% | 3.05% |
| 3 | 6.98% | 3.23% |

三因子相对两因子的 dIV 改善只有约 0.18 个百分点，而第三因子自身 OOS R² 为负，因此当前没有强证据在对冲中引入第三因子。3M ATM 的 dIV RMSE 相对 Rolling 改善为 8.90%，明显好于全曲面平均，这也是首个 Step 7 诊断选择 3M ATM 单因子的原因。

置换重要性显示，ATM 因子的主要特征是 3M smile slope、3M ATM IV、1Y–3M 期限斜率和近期收益。但重要性是在已被反复查看的测试集上测得，只能解释当前模型，不能用于无限次筛特征后继续把同一测试期称为独立样本外。

### 3.7 Step 6 输出文件

| 文件 | 内容与读法 |
|---|---|
| `factor_predictions.csv` | 114 个测试标签上，三个因子、五种模型的真实值和预测值；先筛 `model=hist_gradient_boosting` |
| `factor_model_summary.csv` | 每个因子、模型的 RMSE、MAE、相关系数和 OOS R² |
| `surface_model_summary.csv` | 将预测因子还原成 56 单元 Beta 后，按 overall、3M ATM 和各 tenor 评价 Beta/dIV |
| `feature_importance.csv` | HGB 在测试期的 permutation importance；负值表示打乱该特征未使评分恶化 |
| `factor_predictions.png` | 三个因子的真实值与 HGB 预测时序 |
| `forecast_skill.png` | 因子 OOS R² 和 1/2/3 因子 dIV 改善 |
| `manifest.json` | 输入哈希、特征、模型、切分、主结果和软件运行约定 |

Step 6 当前没有保存可直接部署的模型对象；Step 7 按相同代码重新拟合并保存自己的 `forecaster.joblib`。

---

## 4. Step 7 准备阶段做了什么

[step07.py 的 `prepare_step7()`](../dynamic_alpha_hedging/step07.py#L74) 先做以下工作：

1. 读取 Step 1、2、4、5、6 的 manifest；
2. 对每个上游输入重新计算 SHA-256，拒绝陈旧或被替换的输入；
3. 检查 Beta 门槛、训练比例、利率、假日、tenor 和 level 配置必须与 Step 6 一致；
4. 重新加载 Step 5 状态面板、Step 4 载荷和 Step 2 Beta；
5. 使用与 Step 6 完全相同的 `primary_factor_model()`，只在原 342 个训练标签上重新拟合三个 HGB；
6. 对训练截止日及以后所有可用 close 日期生成预测，不要求下一日 Daily Beta 标签存在；
7. 加载原始 SVI 历史；
8. 若使用固定转换器，则读取参考 inverse，并验证参考日期不晚于训练截止日。

Step 6 原输出只包含 114 个能够事后评分的测试标签。Step 7 必须在所有实际可对冲日期产生信号，因此得到从 2026-01-15 到 2026-08-31 的 157 个 close 预测，对应 156 个持有区间。新增的 42 个小收益日不能因为事后 Daily Beta 不可计算而从回测删除。原 114 日预测与 Step 6 CSV 完全一致，最大差异为零。

当前固定 inverse 来自 2025-07-21 的 Step 3 曲线。虽然日期在训练截止日前，但旧 Step 3 自动选择“代表日”时看过全历史，因此选择过程有轻微前视问题。代码只能检查参考日期，无法从 CSV 证明代表日选择没有使用未来样本。

该参考日的 3M ATM 曲线本身严格单调、inverse 可用，Alpha=1 原始 Beta 约 0.00886，Beta 标准误不超过约 0.01043；但它因为全 ratio 网格质量没有通过而 `quality_pass=False`。因此“可以反查”和“数值质量已正式验收”也要分开。

---

## 5. Step 7 每个持有区间的合约是什么

当前 `--book atm --factors 1` 表示每天只持有一份 3M ATM 欧式 call。

同一引擎也支持：

| book | 合约槽位 | 默认预测因子 |
|---|---|---:|
| `atm` | 3M × level 1.0，共 1 份 | 1 |
| `near_atm` | 3M/6M/1Y × level 0.9/1.0，共 6 份 | 2 |
| `full` | Step 4 的 7 tenor × 8 level，共 56 份 | 2 |

`weights=contracts` 时每个槽位一份；`weights=equal_vega` 时每天将总 vega 预算设为 100 并在槽位间等分。一个持有区间内数量不再变化。

[hedging.py 的 `contract_interval()`](../dynamic_alpha_hedging/hedging.py#L39) 在 close t 定义：

$$
K_t=S_t,\qquad
T_t=\text{从 }t\text{ 向后约 }0.25\times260=65\text{ 个业务日的日期}.
$$

从 $t\to t+1$ 持有时，实际 strike $K_t$、实际 expiry $T_t$ 和数量都固定。下一日用 $t+1$ 的原始 SVI 曲面重新估值同一个 $(K_t,T_t)$ 合约：

$$
dV_t=V_{t+1}(K_t,T_t)-V_t(K_t,T_t).
$$

完成该区间后，再按新一天的 $S_{t+1}$ 和固定 3M tenor 建立下一份合约。因此它是“每日滚动建立的标准期限 book”，不是将一张期权从 3M 一直持有到到期。

历史 mark 使用 SVI implied vol 和 Black–Scholes 计算 `pv`、`bs_delta`、`vega`、`gamma`。Alpha 只用于构造候选对冲 Delta，不会篡改实际历史期权价格。

---

## 6. 每日 Beta–Alpha–Delta 映射如何计算

### 6.1 Alpha 在核心 local vol 中的位置

[surface.py 的 `local_vol()`](../svi_localvol/surface.py#L282) 使用 Dupire–Gatheral 公式：

$$
\sigma_{loc}^2=\frac{\partial w/\partial\tau}{D},
$$

其中分母中的 log-moneyness 被改为：

$$
y_{adj}=y-\alpha\cdot spot\_adj.
$$

`spot_adj` 是 ±spot bump 相对基准 Spot 的 log 变化。调用方只传一次 `alpha` 和一次 `spot_adj`；核心代码内部完成乘法。

### 6.2 当天 MC 节点

[hedging.py 的 `MCMapStore.measure()`](../dynamic_alpha_hedging/hedging.py#L116) 对每个 close t、tenor 和

$$
\alpha\in\{0,0.5,1,1.5,2\}
$$

执行：

1. 将当天原始 SVI 曲面做 calendar repair，保证用于 MC 的总方差沿期限不下降；
2. 建立基准 `LocalVolGrid`；
3. 令 (h=1\%\times S_t)，分别用 (S_t+h) 和 (S_t-h) 重建两个 Alpha local-vol grid；
4. 使用相同随机数和 antithetic 路径模拟 up/down 终值；
5. 同一 tenor 下所有 strike 共用路径；
6. 同一对路径同时得到 up/down PV、反解 IV、Beta 和 call Delta。

路径更新在 [montecarlo.py](../svi_localvol/montecarlo.py#L288) 中近似为：

$$
S_{u+du}=S_u\exp\left[
b\,dt_r-\frac12\sigma_{loc}^2dt_v
+\sigma_{loc}\sqrt{dt_v}Z
\right],
$$

其中漂移使用 Act/365，方差使用 Business/260。local vol 同时对时间和 spot ratio 插值。

MC 定义：

$$
\begin{aligned}
\beta^{raw}_t(\alpha)
&=-\frac{IV_t^{up}(\alpha)-IV_t^{down}(\alpha)}
{\log[(S_t+h)/(S_t-h)]},\\
\Delta_t(\alpha)
&=\frac{PV_t^{up}(\alpha)-PV_t^{down}(\alpha)}{2h}.
\end{aligned}
$$

代码同时计算 call 和 put，优先选择 IV 反解没有 clipping 且 Beta 标准误更小的一边；如果选中 put，Delta 通过 put-call parity 转成 call Delta。Black–Scholes 路径控制变量用来降低 MC 方差。

### 6.3 为什么还要将 Alpha=1 锚定为 Beta=0

沿用 Step 3 的命名约定：

$$
\beta^{converter}_t(\alpha)
=\beta^{raw}_t(\alpha)-\beta^{raw}_t(1).
$$

这样 Alpha=1 精确对应 converter Beta=0。必须注意：这是对有限 MC 和数值离散误差的锚定，不等于原始 `beta_model(alpha=1)` 天然严格为零。原始偏差仍保存在 `beta_alpha_one_raw` 和质量列中。

### 6.4 这次 `fixed converter` 到底固定了什么

当前命令的固定参考曲线是 2025-07-21 的 3M ATM inverse：

| Alpha | 参考 Beta |
|---:|---:|
| 0.0 | 0.6772 |
| 0.5 | 0.3395 |
| 1.0 | 0 |
| 1.5 | -0.3385 |
| 2.0 | -0.6789 |

`fixed` 只用于将预测 Beta 反查成 Alpha。[hedging.py 的 `invert_beta()`](../dynamic_alpha_hedging/hedging.py#L156) 对严格递减节点做分段线性反插值；超出范围时截到 0 或 2，不单调或非有限时回退 Alpha=1。

但是得到 Alpha 后，真正使用的 Delta 仍然在 **当天** MC 曲线中插值：

$$
\widehat\Delta_t
=\operatorname{Interp}_{\alpha}
\{\Delta_t(0),\Delta_t(0.5),\ldots,\Delta_t(2)\}.
$$

因此此次运行不是“把 2025-07-21 的 Delta 用到 2026 年”，而是：

```text
预测 Beta
  -> 用 2025-07-21 曲线反查 Alpha
  -> 用每个当前日期的 MC 曲线把 Alpha 转成当天 Delta
```

默认 `--converter daily` 则连 Beta→Alpha 反查也使用当天曲线。它更能适应市场状态，但计算量没有减少，因为 fixed 模式也仍需每天 MC 计算 Delta。

本次 `--fast` 让当天 Delta 表使用 10,000 路径和 201 个 ratio nodes；参考 inverse 文件本身来自之前正式 Step 3 的 100,000 路径、801 nodes。两部分精度不同。

---

## 7. 从 Step 6 预测到每个策略的 Alpha

在 3M ATM 单因子下，由于锚点载荷为 1、截距为 0：

$$
\widehat\beta_{t+1,3M,ATM}=\widehat z_{1,t+1}.
$$

[step07.py](../dynamic_alpha_hedging/step07.py#L273) 对每个区间生成以下策略：

| 策略 | Alpha 来源 |
|---|---|
| `fixed_0/0.5/1/1.5/2` | 固定五档 |
| `best_fixed_train` | 在训练期按相同 book 对冲误差 std 从五档中选择 |
| `bs_delta` | 直接使用解析 BS Delta，不使用 MC Alpha Delta |
| `rolling_beta` | 截至 t 已知的 Step 2 Rolling Beta 反查 Alpha |
| `rolling_alpha_mean` | 最近 20 个有效已实现 Daily Beta 逐日反查 Alpha 后取均值 |
| `dynamic_raw` | Step 6 预测 Beta 直接反查 Alpha |
| `dynamic_ema` | 对 `dynamic_raw` Alpha 做 EMA |

训练期 494 个可用区间中，各固定档对冲误差 std 为：

| Alpha | 训练期 std |
|---:|---:|
| 0 | 7.428 |
| 0.5 | **6.099** |
| 1 | 6.614 |
| 1.5 | 8.661 |
| 2 | 11.464 |

所以 `best_fixed_train=0.5`。该选择在测试期开始前冻结。

当前选择指标是训练期 `carry_hedge_error` 的 std，不含标的交易成本，也不是整段现金账户的净财富目标。因此未来启用非零成本时，最好固定档和平滑参数都应在训练期按含成本的统一目标重新选择。

EMA 半衰期 (H=10) 天：

$$
\lambda=2^{-1/H},\qquad
\alpha_t^{EMA}=\lambda\alpha_{t-1}^{EMA}+(1-\lambda)\alpha_t^{raw}.
$$

初值是训练期最好固定 Alpha=0.5。`H=0` 才等于不平滑。

当前测试期：

| 信号 | Alpha 均值 | 标准差 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| dynamic raw | 0.815 | 0.364 | 0.136 | 1.512 |
| dynamic EMA | 0.761 | 0.234 | 0.358 | 1.112 |
| rolling Beta | 0.835 | 0.183 | 0.473 | 1.135 |
| rolling Alpha mean | 0.931 | 0.306 | 0.414 | 1.558 |

156 天均没有 Alpha 截断、inverse 回退或 Delta 回退。

---

## 8. 对冲误差与自融资现金账户

### 8.1 原始对冲误差

对一份 long call、short Delta 股标的，文档最直接的误差是：

$$
e_t^{raw}=dV_t-\Delta_t dS_t.
$$

它保存在 `option_pnl.csv` 的 `raw_hedge_error`。

### 8.2 Carry 调整

[step07.py 的 `hedge_error()`](../dynamic_alpha_hedging/step07.py#L164) 加入现金融资和标的收益率：

$$
e_t^{gross}
=dV_t-\Delta_t dS_t
+(\Delta_tS_t-V_t)(e^{r\Delta t}-1)
-\Delta_tS_t(e^{q_{eff}\Delta t}-1),
$$

其中

$$
q_{eff}=r-b=dividend+repo.
$$

### 8.3 Book 现金账

对多合约 book：

$$
V_t^{book}=\sum_i q_iV_{i,t},\qquad
D_t^{book}=\sum_i q_i\Delta_{i,t}.
$$

只交易净标的 Delta，不将每个合约的绝对 Delta turnover 相加。现金账户开仓时：

$$
C_t=W_t-V_t^{book}+D_t^{book}S_t-cost_t.
$$

这里 (D>0) 表示 long option 对应 short (D) 股，所以卖空股票带来 (DS) 现金。下一日现金计息、支付 short stock 的 dividend/repo，再与期权和股票头寸合并得到 `wealth`。`net_error` 是相对原财富无风险增长后的增量。

当前 `cost_bps=0`，所以 `net_error` 与 `gross_error` 只相差浮点误差。代码目前仅支持净标的单边交易金额成本，不含期权 bid/ask、期货基差、滑点、保证金或真实资金曲线。

表中的 P&L 和 `final_wealth` 是一份模型 call 的价格点数，未乘真实合约乘数，也没有用初始资本归一化。因此它们不是收益率；用于比较同一 book、同一日期上的策略误差才有意义。

---

## 9. 归因的原理和两种残差

归因回答的不是“策略实际赚了多少钱”，而是：“当天真实期权价格变化能够由哪些风险因子解释？”

对一日变化做近似：

$$
dV
\approx \Delta_{BS}dS
+\frac12\Gamma(dS)^2
+time\_pnl
+\nu\,term\_roll\_iv
+\nu\,d\sigma_{spot}
+\varepsilon.
$$

Beta 定义意味着：

$$
d\sigma_{spot}\approx-\beta d\log S.
$$

因此：

$$
\varepsilon
=dV-\left[
\Delta_{BS}dS+\frac12\Gamma(dS)^2+time\_pnl
+\nu\,term\_roll\_iv
-\nu\beta d\log S
\right].
$$

代码写成等价形式：

$$
\varepsilon=dV-common\_pnl+\nu\beta d\log S.
$$

具体组成在 [step07.py](../dynamic_alpha_hedging/step07.py#L307)：

- `bs_delta*dS`：解析 BS Delta 的 spot 一阶项；
- `0.5*gamma*dS^2`：spot 二阶项；
- `time_pnl`：在前一日 IV 和利率参数冻结时，将同一合约剩余期限缩短到下一日的精确 BS 价格差；
- `term_roll_iv`：在前一日曲面上，沿同一 strike 从原期限移动到下一日剩余期限时的 IV 变化；
- `-vega*beta*dlogS`：预测的 spot-vol 曲面变化。

使用 BS Delta 是刻意的。动态 local-vol Delta 已经隐含一部分 spot-vol 响应；如果再给动态 Delta 完整加上 `vega*(-beta*dlogS)`，会重复计算同一个效应。

Step 7 输出两种归因残差：

1. `forecast_attribution_residual`：使用 Step 6 **原始预测 Beta**。这是文档 Step 7.2 最直接的事前模型归因；同一天所有策略的该列相同，因此 `summary.csv` 中每个策略的 `forecast_attribution_rmse` 都相同。
2. `attribution_residual`：使用实际实施 Alpha 在当天 MC 曲线中对应的 `effective_beta`。它会反映 Alpha 截断、固定档和 EMA 平滑后的实际弹性。

Alpha=1 的 converter Beta 为 0，所以它是“不用 spot-vol Beta 解释”的归因基准。归因残差仍会包含：SVI 曲面与模型不一致、真实期限结构变化、volga/vanna 和更高阶项、离散大幅跳动、利率误差、数据噪声以及 Beta 预测误差。因此残差不可能被期望为零。

---

## 10. 当前 Step 7 输出逐个怎么看

### 10.1 `signals.csv`

每个日期、cell、策略一行。重点列：

- `feature_date`：做预测和建仓的 close t；
- `label_date`：持有结束的 t+1；
- `predicted_beta`：Step 6 的原始预测；
- `raw_predicted_alpha`：原始预测 Beta 反查的 Alpha；
- `alpha`：该策略实际采用的 Alpha；
- 三个因子列：Step 6 对下一日的预测；
- `converter_date`：本次固定为 2025-07-21；
- `effective_beta`：实际实施 Alpha 在当天 MC 曲线对应的 Beta；
- 三个 fallback/clip flag：inverse 截断、inverse 回退、Delta 回退。

固定策略行也会重复记录同一个 `predicted_beta` 和 `raw_predicted_alpha`，便于同日对齐；这不代表固定策略使用了预测 Beta。

### 10.2 `mc_map.csv`

本次 156 个测试日期 × 5 个 Alpha = 780 行。包括：

- up/down spot、PV、IV 和标准误；
- `beta_model` 与 `beta_converter`；
- `call_delta` 与 `call_delta_stderr`；
- 网格 undefined/clipped 比例；
- Alpha=1 原始偏差；
- `quality_pass`、`inverse_available` 和失败原因。

训练期节点单独保存在 `mc_cache/`，没有重复拼入 `mc_map.csv`。

### 10.3 `option_pnl.csv`

每个日期、合约槽位、策略一行。它将真实合约身份、两日 mark、Greeks、Alpha/Delta、原始误差、carry 误差和两种归因残差放在一起，是排查某一天损益的主表。

### 10.4 `book_pnl.csv`

按策略和日期汇总。当前 ATM book 只有一个合约，所以每策略 156 行；扩展为多个 tenor/level 后仍是每策略每天一行。重点看 `book_delta`、`gross_error`、`net_error`、`cost`、`wealth` 和净标的 turnover。

### 10.5 `summary.csv`

这是 headline 结果。`std_error` 越小越好；改善定义为：

$$
improvement=1-
\frac{std(e^{strategy})}{std(e^{baseline})}.
$$

置信区间使用同一组日期配对的循环 block bootstrap，块长 10 个观测，重复 1000 次。它描述当前样本的不确定性，但不能消除多次查看同一测试集造成的研究选择偏差。

### 10.6 其余输出

| 文件或目录 | 内容 |
|---|---|
| `cell_summary.csv` | 每个策略、tenor、level 的单合约 gross error 与归因诊断；ATM 本次只有一个 cell |
| `factor_forecasts.csv` | 从训练截止日开始的 157 个 close-t 因子预测，包含无法事后形成 Daily Beta 标签的日期 |
| `forecaster.joblib` | Step 7 重新拟合的三个 HGB；只能加载可信本地文件，joblib 不是安全交换格式 |
| `mc_cache/` | 每个日期、tenor 的 Alpha 节点缓存；key 含曲面、数值参数和相关代码 hash |
| `manifest.json` | 本次正式回测的唯一权威配置与完成记录，包括训练/测试区间、排除、固定档和输入 hash |
| `preparation.json` | 此目录先前 `--prepare` 的记录；正式回测完成后应以较新的 `manifest.json` 为准 |

本次没有生成 Step 7 图。图表属于后续报告增强，不影响 CSV 中的核心结果。

---

## 11. 当前 Step 7 结果分析

### 11.1 对冲误差

| 策略 | std | RMSE | MAE | q95 | q99 | 相对训练最好固定档 |
|---|---:|---:|---:|---:|---:|---:|
| fixed 0 | 11.127 | 11.092 | 7.687 | 18.996 | 37.648 | -34.90% |
| fixed 0.5 / train best | 8.248 | 8.223 | 5.198 | 14.836 | 33.312 | 基准 |
| fixed 1 | **7.268** | **7.254** | 5.048 | 13.101 | 27.653 | +11.88% |
| fixed 1.5 | 8.871 | 8.863 | 6.373 | 19.273 | 23.930 | -7.55% |
| fixed 2 | 12.061 | 12.053 | 8.639 | 25.547 | 36.201 | -46.21% |
| rolling Beta | 7.695 | 7.674 | 4.839 | 13.259 | 28.565 | +6.71% |
| dynamic raw | 7.608 | 7.586 | **4.624** | **11.434** | 25.562 | +7.77% |
| dynamic EMA | 7.883 | 7.859 | 4.809 | 13.609 | 28.551 | +4.43% |

客观结论：

- `dynamic_raw` 相对训练期选出的 fixed 0.5，std 下降 7.77%；但 95% 区间为 -2.76% 到 28.22%，跨过零。
- 测试期真正最好的固定档是 Alpha=1，std=7.268。动态 raw 的 std 比它高 4.67%，EMA 高 8.46%。
- 动态 raw 的 MAE 和 q95 更好，说明大部分日期误差较小；但少数极端日拉高 std。
- 训练期最好 Alpha=0.5，测试期却是 Alpha=1，说明最优固定 Alpha 存在样本不稳定或制度变化。
- 解析 BS Delta 与 Alpha=1 MC Delta 几乎同水平，符合 Alpha=1 应接近 sticky-strike BS Delta 的预期。

动态 raw 最大两个绝对误差日是 2026-03-31 和 2026-04-06，两天合计占其平方误差约 52%。4 月 6 日动态 raw 误差约 57.67，而 fixed Alpha=1 约 34.34。结论对极少数日期非常敏感。

### 11.2 平滑与 turnover

动态 raw 的平均日 Alpha 变化为 0.194，累计净标的 turnover 为 5.475；EMA 分别降至 0.0147 和 1.479，turnover 下降约 73%。

但在零成本假设下，EMA 的 std 从 7.608 上升到 7.883。这正是文档所说的平滑权衡：平滑显著降低交易量，却损失动态响应。当前不能仅凭零成本结果选半衰期；应在训练期预定义 0/5/10/20 日方案，再在相同成本情景下比较。

### 11.3 归因

- Alpha=1 的归因 RMSE：6.411；
- dynamic raw 按实施 Alpha 的归因 RMSE：6.265，改善 2.28%；95% 区间跨零；
- 原始预测 Beta 的 `forecast_attribution_rmse`：6.080，相对 Alpha=1 约改善 5.17%，对应的配对区间也跨零；
- EMA 实施 Beta 的归因 RMSE 为 6.638，反而差于 Alpha=1。

这说明原始 Beta 预测对解释 spot-vol P&L 有弱正向迹象，但平滑后的实际 Alpha 不再完全对应该预测，且证据不足以排除偶然性。

### 11.4 MC 数值质量

- 156 天均能严格单调反解；
- 所有 Alpha=1 原始 Beta 偏差绝对值都小于 0.03，平均绝对值约 0.00615，最大约 0.02535；
- Beta 标准误均值约 0.012–0.015，最大约 0.0407；
- Delta 标准误均值约 0.0023–0.0030，最大约 0.0080；
- 只有 26/156 天通过全部质量检查；130 天因全网格 undefined/clipped 比例超过阈值而失败；
- 3M ATM 位置的 calendar repair 前后 IV 差异为零，但这不能证明完整路径访问区域没有被数值填补。

此外，固定参考曲线反查出的 `dynamic_raw` Alpha，放到每天不同的 MC 曲线后，其 `effective_beta` 与原始 `predicted_beta` 的 RMSE 约为 0.066。这正是 fixed converter 的状态失配；它是下一轮 daily converter 对照实验需要测量的量，不是模型 Beta 预测误差本身。

所以当前结果适合验证代码链和观察量级，不适合作为正式精度结论。`quality_pass=False` 没有导致回退，因为“数学上可逆”和“全部数值质量阈值通过”是两个不同条件。

### 11.5 原始 Spot 数据疑点

原始 `svi_param.pkl` 中：

| 日期 | Spot |
|---|---:|
| 2026-03-30 | 6368.85 |
| 2026-03-31 | 6368.85 |
| 2026-04-01 | 6368.85 |
| 2026-04-02 | 6368.85 |

四天 Spot 完全相同，但 SVI 曲面继续变化。3/30→3/31 的 3M ATM 固定合约 IV 从 23.491% 降到 20.741%，期权 PV 变化约 -36.63，而 (dS=0)。任何 Delta 都无法对冲 (dS=0) 时的曲面跳变。

这可能是真实源数据的 stale/forward-filled Spot，也可能是数据生产约定，不能未经确认直接删除。它还会影响 Daily Beta 标签，因为分母为零。应向 mentor 或数据提供方确认这几天的 `Spot` 是否真实收盘值、是否经过前向填充、SVI 与 Spot 是否同一时间快照。

---

## 12. 改进建议：按优先级推进

### P0：先解决数据和评价有效性

1. **核实重复 Spot。** 输出连续重复 Spot、零收益但 IV 大变、Spot 跳跃和 SVI 更新时间不一致的日报；先查源，不按回测表现删日期。
2. **保存原始时间戳和数据血缘。** 当前 date-only key 无法验证美股收盘时点；应保留 source timestamp、timezone、surface snapshot time、spot source 和 underlying 标识。
3. **确认 refSpot 的含义。** 它必须是与 SVI 拟合同一时点、同一标的定义的 Spot/forward anchor，而不是缓存值。
4. **使用真实 rate/dividend/repo term structure。** 当前常数只适合研究原型，会影响 forward、PV、Delta 和 carry。
5. **建立冻结的最终 holdout。** Step 4–7 的现有 2026 测试期已经被多次查看；后续调特征、转换器和平滑后，需要一段未参与决策的新数据，或严格 walk-forward 评价。
6. **报告两套样本。** 主结果保留所有合法日期；另做预先定义的数据质量敏感性样本，例如“确认 stale Spot 的日期”，但必须同时报告、不能只留表现更好的版本。

### P1：改进 Beta 预测目标

当前 Daily Beta 是比值，天然有异方差：

$$
\beta_t=-dIV_t/d\log S_t.
$$

即使设置 0.0025 门槛，靠近门槛的标签仍明显更噪。更适合对冲目的的方案是直接拟合：

$$
dIV_{t+1,j}
=a_j(state_t)-\beta_j(state_t)d\log S_{t+1}+\varepsilon_{t+1,j}.
$$

训练损失中可以使用已经实现的 $d\log S_{t+1}$ 和 $dIV_{t+1}$，因为它们是标签的一部分；但预测特征仍只能使用 close t 信息。这样避免先除以小收益再预测一个高噪比值。

若仍预测 Daily Beta，至少应比较：

- 按 $|d\log S|$、$(d\log S)^2$、vega 或预估 Delta 敏感度加权；
- winsorization/robust loss，但阈值必须训练内确定；
- 预测分布或不确定度，再将高不确定度预测向 Rolling Beta/Alpha=1 收缩；
- 直接优化下一日 dIV RMSE，而不是 Beta MAE；
- 面向对冲时，进一步用训练期 `vega*dlogS` 或实际合约 P&L 敏感度加权。

当前 HGB 使用 `absolute_error`，更偏向条件中位数；回测 headline 是误差标准差。应在训练期比较 `squared_error`、Huber/稳健目标和风险加权目标，再由 walk-forward 对冲 std 决定，而不是默认 MAE 最优就等于对冲最优。

### P1：改进模型验证，而非先盲目加复杂度

样本只有 342 个训练标签，直接换神经网络很容易过拟合。更有价值的顺序是：

1. 使用 expanding-window walk-forward，分别记录每个预测日的训练截止点；
2. 给 HGB 的叶子数、最小叶样本、L2、学习率做嵌套时序验证；
3. 保留训练均值、Rolling Beta、Ridge 和 Alpha=1 基准；
4. 对特征做稳定性和消融分析，而不是只看一次测试集 permutation importance；
5. 增加只在 close t 可见的 SVI raw/JW 参数、skew curvature、下跌状态、波动率分位数和 regime interaction；
6. 将模型输出限制或收缩到训练期经济合理范围，并报告触边频率。

需要特别关注非对称性：股指下跌日、上涨日以及高波/低波状态下 Beta 可能不同。但 (t+1) 的涨跌方向不能作为事前模型特征，只能使用 t 日以前的 regime 概率或状态。

### P1：重新评估曲面因子结构

shape 2 OOS R² 为负，暂时不适合主回测。对全曲面可以比较：

- 一因子、两因子和逐单元模型；
- 对期限/level 做平滑的多任务回归；
- 按 Daily Beta 的估计方差、vega 或 book 风险做加权 PCA；
- 使用能处理缺失单元的低秩模型，而不是只保留 56 单元全部存在的 456 天；
- 对不同单元标准化后再 PCA，并同时报告未标准化结果。

但 3M ATM 是显式锚点，当前首个 ATM 回测不需要第二、第三因子。只有扩展 near-ATM/full book 时，曲面因子改进才直接影响 Delta。

### P0/P1：验证 Beta–Alpha–Delta 转换器

1. **比较 fixed 与 daily converter。** 固定曲线测试状态迁移误差；daily 曲线是主候选。两次运行应使用不同输出目录，避免覆盖。
2. **正式 MC 收敛。** 在训练期预选普通、高波、陡 skew 日期，比较 paths、substeps、ratio nodes 和 bump 0.5%/1%/2%；指标包括 Beta、Alpha 反解和 Delta。
3. **局部路径质量。** 当前质量检查覆盖整个 ratio grid，可能被远离 ATM 的区域支配；应同时输出路径访问分位数内的 undefined/clipped 比例。
4. **插值误差。** 随机抽取节点之间 Alpha，直接重跑 MC，与五点线性插值 Delta 比较。
5. **不确定度传播。** 估计

$$
\delta\alpha\approx
\frac{\delta\beta}{\partial\beta/\partial\alpha},
\qquad
\delta\Delta\approx
\frac{\partial\Delta}{\partial\alpha}\delta\alpha,
$$

将预测误差和 MC 标准误传递到对冲误差区间。Beta–Alpha 曲线越平，inverse 越不稳定。
6. **严格无泄漏参考日。** 若保留 fixed converter，对代表日的选择只能使用训练期；旧全历史 medoid 只能作为诊断。

### P1：改进 Step 7 策略选择和成本

- 训练期 fixed Alpha=0.5、测试期 fixed Alpha=1，说明单次训练选择不稳定。应在训练期做 rolling/expanding 验证，并报告最优 Alpha 随时间变化，而不是将测试期最优档反向当参数。
- EMA 半衰期应在训练期预设比较 0/5/10/20，联合交易成本选取；不要看测试结果后只保留最好值。
- 增加 0.1、0.5、1、2 bps 等净标的成本敏感性；若实际用期货，应重写 carry、合约乘数、换月和成本模型。
- 增加 option bid/ask 或至少说明期权每日滚动的换仓成本未计。当前 `final_wealth` 不是可交易策略收益。
- 采用 near-ATM 六合约 book 后，应按固定合约数和等 vega 两种权重都报告；主指标仍是 net book Delta 的误差，而不是单期权平均。
- 对 3/31、4/6 等极端日做预定义事件归因：Spot、IV、Gamma、Beta、期限滚动和数据质量逐项解释，不能事后删除。

### P2：工程与可复现性

- Step 7 当前重新拟合模型，而不是读取 Step 6 保存的可部署模型。虽然已验证预测完全一致，长期应让 Step 6 正式保存 frozen model、训练截止日、特征 schema 和软件版本，Step 7 只加载。
- `joblib` 被代码直接 import，但 `pyproject.toml` 没有直接声明；虽然 scikit-learn 通常依赖它，仍建议显式加入依赖，避免再次出现 `ModuleNotFoundError`。
- 每种 book、factor 数、converter、成本和 MC 精度使用独立输出目录；manifest 记录 git commit、Python/NumPy/pandas/scikit-learn 版本。
- 增加 Step 7 图表：累计误差、滚动 std、Alpha 与 Beta 时序、turnover、误差散点、极端日贡献和 MC 质量覆盖。
- 缓存应继续包含曲面、数值配置和代码 hash；正式报告只引用完整运行的 manifest，不能混用不同参数缓存。

---

## 13. 下一轮最合理的实验顺序

1. 向数据提供方确认 2026-03-30 至 2026-04-02 的重复 Spot，以及 Spot/SVI 同步规则；
2. 保持 Step 6 模型和所有回测参数不变，用独立目录正式比较 fixed converter 与 daily converter；
3. 在训练期代表状态上完成 MC paths/substeps/grid/bump 收敛，确定正式数值参数；
4. 用现有 3M ATM 做全日、数据质量敏感性和预设成本/EMA 情景；
5. 改为直接 dIV/风险加权的 Beta 模型，用 walk-forward 训练验证，不触碰最终 holdout；
6. ATM 结果稳定后再运行 3M/6M/1Y × 0.9/1.0 的六合约 book；
7. 只有 shape 因子预测和数值质量都过关后，再扩展 56 单元全曲面。

当前最稳妥的研究结论是：Step 6 的 3M ATM Beta 含有一定事前预测信号；Step 7 的动态 Alpha 在 MAE 和多数日期上有改善迹象，但尚未在对冲误差标准差上超过 Alpha=1，也没有统计显著性。数据同步、极端日期和 MC 网格质量是先于“换更复杂模型”的主要问题。

---

## 14. 后续运行时的目录约定

不要让不同实验覆盖同一个 `step07/`。例如：

```bash
# 当前 fixed converter 的快速诊断应保留在独立目录
python -m dynamic_alpha_hedging step7 --book atm --factors 1 \
  --converter fixed \
  --reference-inverse output/dynamic_alpha/step03/alpha_beta_inverse.csv \
  --fast --output output/dynamic_alpha/step07_atm_fixed_fast

# daily converter 快速对照
python -m dynamic_alpha_hedging step7 --book atm --factors 1 \
  --converter daily --fast \
  --output output/dynamic_alpha/step07_atm_daily_fast

# 数值参数确定后的正式 daily converter
python -m dynamic_alpha_hedging step7 --book atm --factors 1 \
  --converter daily \
  --output output/dynamic_alpha/step07_atm_daily_formal
```

`--fast` 只验证流程和量级。正式运行前应先完成本文 P0/P1 中的数据确认与数值收敛，不应只是去掉 `--fast` 就直接把结果写进最终结论。
