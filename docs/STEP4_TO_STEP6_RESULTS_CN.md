# 动态 Alpha 研究：Step 4–6 模型与结果总结

## 研究链条

Step 2 已得到每日 Beta 曲面：

$$
\beta_{t,\tau,m}=-\frac{dIV_{\mathrm{surface},t,\tau,m}}{d\log S_t}.
$$

Step 4–6 的目标是把每天 56 个 Beta 单元压缩为少数因子，并用日期 $t$ 已知的信息预测下一观测日 $t+1$ 的 Beta：

```text
Daily Beta surface → 因子分解 → 下一日因子预测 → 下一日 Beta surface
```

当前范围包括 7 个期限（2M、3M、6M、9M、1Y、1.5Y、2Y）和 8 个 level（0.4–1.1），共 56 个曲面单元。正式 Daily Beta 门槛为 $|d\log S|\geq0.0025$。

## Step 4：Beta 曲面因子化

### 做了什么

Step 4 将每天的 56 维 Beta 曲面表示为：

$$
\beta_{t,c}\approx a_c+z_{1,t}f_{1,c}+z_{2,t}f_{2,c}+z_{3,t}f_{3,c},
$$

其中 $c=(\tau,m)$。

- 第一因子直接锚定为 3M ATM Beta：$z_{1,t}=\beta_{t,3M,ATM}$。
- 第二、第三因子是扣除第一因子后，对训练集残差曲面做 PCA 得到的两个形状因子。
- 在 3M ATM 锚点，$a=0$、$f_1=1$、$f_2=f_3=0$，因此该点的重构值严格等于 $z_1$。
- 载荷只使用按时间排序的 342 个训练日拟合；114 个测试日仅使用冻结载荷投影，没有重新拟合。

### 结果

| 因子数 | 训练集累计解释率 | 测试集累计重构 $R^2$ |
|---:|---:|---:|
| 1：3M ATM 因子 | 14.08% | 26.08% |
| 2：加入第一形状因子 | 76.17% | 71.40% |
| 3：加入第二形状因子 | 86.46% | 77.08% |

结论：3M ATM 对自身是精确锚点，但单独不足以表示完整曲面；第二因子贡献最大，两个因子已捕捉主要结构；第三因子只有较小的边际增益。

主要输出：

- [explained_variance.csv](../output/dynamic_alpha/step04/explained_variance.csv)：1/2/3 因子的整体解释与重构能力。
- [factor_loadings.csv](../output/dynamic_alpha/step04/factor_loadings.csv)：各因子在不同 tenor、level 上的载荷。
- [factor_scores.csv](../output/dynamic_alpha/step04/factor_scores.csv)：每天的 $z_1,z_2,z_3$。
- [reconstruction_by_cell.csv](../output/dynamic_alpha/step04/reconstruction_by_cell.csv)：每个 $(tenor,level)$ 的重构质量。

## Step 5：直接预测 Daily Beta 的可行性检验

### 做了什么

Step 5 不使用因子压缩，而是为 56 个 Beta 单元分别建立下一日预测模型：

$$
\widehat\beta_{t+1,c}=g_c(state_t).
$$

输入是日期 $t$ 收盘时可获得的 spot、IV 水平与变化、smile slope、term slope、实现波动率和历史收益等状态变量。主要比较：训练均值、Rolling Beta、Ridge 和 `HistGradientBoostingRegressor`。标签是下一观测日真实的 Daily Beta，不是 Rolling Beta。

### 结果

逐单元梯度提升模型的整体测试集结果为：

- Beta OOS $R^2$：8.66%；
- 预测与真实 Beta 的相关系数：0.302；
- dIV RMSE 相对 Rolling Beta 改善：3.13%。

ATM 单元通常比完整 smile 更容易预测。例如 3M ATM 的 Beta OOS $R^2$ 为 22.57%，相关系数为 0.504，dIV RMSE 相对 Rolling Beta 改善 6.72%。多数 2M–1.5Y ATM 单元有正向改善，2Y ATM 在该指标上为 -1.70%。

结论：Daily Beta 存在一定样本外预测能力，尤其集中在 ATM；但 56 个独立模型较分散，因此 Step 5 更适合作为可行性验证和 Step 6 的直接预测基准。

主要输出：

- [daily_beta_model_summary.csv](../output/dynamic_alpha/step05/daily_beta_model_summary.csv)：整体、期限和逐单元的模型指标。
- [daily_beta_predictions.csv](../output/dynamic_alpha/step05/daily_beta_predictions.csv)：逐日、逐单元预测值。
- [factor_state_panel.csv](../output/dynamic_alpha/step05/factor_state_panel.csv)：Step 6 使用的状态与下一日因子标签面板。

## Step 6：预测因子并重构下一日 Beta 曲面

### 模型设计

Step 6 用三个独立模型预测下一日因子：

$$
\widehat z_{k,t+1}=g_k(state_t),\qquad k=1,2,3.
$$

主模型为 `HistGradientBoostingRegressor`。它使用 16 个日期 $t$ 可知特征，能够学习阈值效应、非线性和特征交互；同时保留训练均值、昨日因子、Rolling 因子和 Ridge 作为基准。模型按时间切分为 342 个训练标签日和 114 个测试标签日，不随机打乱。

预测因子通过 Step 4 的冻结载荷重构下一日 Beta 曲面：

$$
\widehat\beta_{t+1,c}=a_c+\sum_{k=1}^{p}\widehat z_{k,t+1}f_{k,c},\qquad p=1,2,3.
$$

### 因子预测结果

| 因子 | OOS $R^2$ | 相关系数 | 判断 |
|---|---:|---:|---|
| $z_1$：3M ATM Beta | 21.37% | 0.488 | 有明确预测能力 |
| $z_2$：第一形状因子 | 3.01% | 0.144 | 很弱 |
| $z_3$：第二形状因子 | -5.11% | 0.068 | 暂无可靠预测能力 |

3M ATM 被锚定为 $z_1$，因此没有曲面重构误差；但 21.37% 的预测 $R^2$ 不是锚定自动产生的，它来自日期 $t$ 状态对日期 $t+1$ 真实 Beta 的样本外预测。

### 曲面预测结果

| 重构版本 | Beta OOS $R^2$ | dIV RMSE 相对 Rolling Beta 改善 |
|---:|---:|---:|
| 1 因子 | 3.96% | 1.16% |
| 2 因子 | 7.00% | 3.05% |
| 3 因子 | 6.98% | 3.23% |

在 3M ATM 单点，梯度提升模型的 Beta OOS $R^2$ 为 21.37%，相关系数为 0.488，dIV RMSE 相对 Rolling Beta 改善 8.90%。完整曲面上，从一个因子增加到两个因子有明确收益；第三因子仅增加约 0.18 个百分点的 dIV 改善，且自身预测 $R^2$ 为负。

结论：若先研究 3M ATM，对 $z_1$ 做单因子预测即可；若覆盖完整曲面，两因子模型是当前更简洁、稳健的主版本，三因子作为敏感性检验。

主要输出：

- [factor_model_summary.csv](../output/dynamic_alpha/step06/factor_model_summary.csv)：三个因子的样本外预测指标。
- [surface_model_summary.csv](../output/dynamic_alpha/step06/surface_model_summary.csv)：重构后 Beta/dIV 曲面的整体、3M ATM 和分期限指标。
- [factor_predictions.csv](../output/dynamic_alpha/step06/factor_predictions.csv)：各测试日的真实与预测因子。
- [feature_importance.csv](../output/dynamic_alpha/step06/feature_importance.csv)：状态特征的置换重要性。

## 综合结论与下一步

Step 4 证明两个因子足以表示 Beta 曲面的主要结构；Step 5 证明下一日 Daily Beta 并非完全不可预测；Step 6 用三个因子模型代替 56 个逐单元模型，确认可预测性主要来自 3M ATM 因子，第二形状因子贡献有限，第三因子暂不稳定。

当前结果只完成了 Beta 预测，尚未调用 Step 3 的 Beta–Alpha 转换器，也没有计算动态 Alpha Delta 或对冲 P&L。后续 Step 7 应先以 3M ATM 为主实验，将预测 Beta 通过 Step 3 反查 Alpha，再比较动态 Alpha、固定 Alpha 和 BS Delta 的样本外对冲误差；完整曲面扩展以两因子版本为主。
