# Step 7：解析 Delta 与转换器对照

> 历史版本记录：本文的模型特征、rolling Beta 对照及数值属于当时实验。当前代码已改为 [daily-only 流程](DAILY_ONLY_PIPELINE_CN.md)，未据新代码重算本文结果。

本次不改变 Step 1–6、原始 MC 定价方法或原有回测策略。只在同一 Step 7
账本中增加解析对照、MC 一致性诊断，以及训练期多日期转换器。

## 1. 代码与计算

- `dynamic_alpha_hedging/hedge_comparison.py`：三个小函数，分别计算 shadow delta、MC 一致性和多日正向曲线平均。
- `dynamic_alpha_hedging/step07.py`：复用现有预测、合约、费用、归因和现金账本。
- `dynamic_alpha_hedging/hedging.py`、`step03.py` 和核心 MC 引擎未因本次对照而修改；旧 MC 缓存仍按原来的内容指纹验证。

当前 beta 定义为 $\beta=-dIV_{surface}/d\log S$，所以必须除以当天 spot：

$$
\Delta_{shadow,BS}=\Delta_{BS}-\frac{\nu}{S}\widehat\beta_{t+1|t}.
$$

`shadow_bs` 使用原始历史 SVI 的 BS delta 和 vega。它不反查 alpha、不截断 beta，
也不做 alpha 平滑。alpha 和 converter_date 输出为空，绝不填一个虚假的 alpha。
解析 delta 没有强制限制在 [0,1]。

第二个诊断对照 `shadow_mc_base` 使用：

$$
\Delta_{shadow,MCbase}=\Delta_{MC}(1)-\frac{\nu}{S}\widehat\beta_{t+1|t}.
$$

两者使用相同的原始 SVI vega，仅替换基准 delta，用于观察 MC 基准误差的影响。
`shadow_mc_base` 不反查 alpha，但依赖当天 MC 的 alpha=1 delta。
其中 delta_stderr 仅表示基准 MC 的数值误差，不包含预测 beta 的不确定性。

所有策略持有完全相同的隔夜合约和数量，费用、融资、再平衡及最后平仓沿用同一套代码。
归因仍以 BS delta 为公共基准；归因残差不是实际对冲误差。
特别是两个 shadow 策略的 beta 相同，归因残差会相同，但其交易 delta 和 hedge error 可以不同。

## 2. 三种转换模式

| 模式 | beta → alpha | alpha → delta |
|---|---|---|
| `fixed` | 指定的单日期参考曲线，也可读取本版导出的 pooled converter.csv | 当天 MC |
| `pooled` | 训练期多个日期的正向曲线等权平均，然后反查 | 当天 MC |
| `daily` | 当天已知曲面的曲线 | 当天 MC |

`pooled` 默认在支持目标 book 的训练期日期中，按时间顺序均匀选择 10 天。
不是根据测试误差挑日期，也不是声称这 10 天覆盖了所有市场状态。
可用日期不足 10 天时使用全部可用日期，少于 2 天报错。

对每个期限和 level，先在相同 alpha 节点计算：

$$
\overline B(\alpha;\tau,m)=\frac{1}{N}\sum_{d=1}^N B_d(\alpha;\tau,m),
\qquad B_d(\alpha)=\beta_{raw,d}(\alpha)-\beta_{raw,d}(1).
$$

然后对平均曲线求逆，而不是平均各日期反查得到的 alpha。各日期必须具有相同节点；
不丢掉非有限值再悄悄平均，也不做单调投影。无法反查时沿用已有显式 fallback。
`beta_date_std` 是各日期转换关系的离散程度，**不是 MC 标准误**。
`source_dates` 记录所有来源日期；`calibration_date` 表示最后一个来源日，并非某张合成市场曲面。
所有来源日不得晚于训练截止日。旧单日 Step 3 自动 medoid 的全历史选日问题仍需单独审查，
这次保留它只为复现旧实验，不把它包装成严格训练期选日基准。

## 3. 运行命令

在项目根目录、已安装依赖的环境运行。以下命令保留旧 step07 输出，
共用 `step07/mc_cache`；完全匹配的数据、MC 参数和代码才会命中缓存。
缓存缺失时正常 CLI 会计算 MC；本次验证另外禁止了 cache miss 后计算，因此确认没有新增历史 MC。

```bash
# 单日参考：与旧实验对应，额外输出两个解析策略
python -m dynamic_alpha_hedging step7 --book atm --factors 1 --fast \
  --converter fixed \
  --reference-inverse output/dynamic_alpha/step03/alpha_beta_inverse.csv \
  --mc-cache output/dynamic_alpha/step07/mc_cache \
  --output output/dynamic_alpha/step07_comparison/fixed

# 10 个训练日期共同构建转换器
python -m dynamic_alpha_hedging step7 --book atm --factors 1 --fast \
  --converter pooled --converter-dates 10 \
  --mc-cache output/dynamic_alpha/step07/mc_cache \
  --output output/dynamic_alpha/step07_comparison/pooled

# 当天曲面转换器
python -m dynamic_alpha_hedging step7 --book atm --factors 1 --fast \
  --converter daily \
  --mc-cache output/dynamic_alpha/step07/mc_cache \
  --output output/dynamic_alpha/step07_comparison/daily
```

本次已经生成以上三个目录。重跑同一 output 会更新该目录；要保留本次结果请换 output。
`--prepare` 仍然只做输入准备，不构建 pooled 曲线、不启动 MC。
多日期汇总放在 Step 7 是为了直接使用其训练截止日、book 和缓存，不再复制一条 Step 3 工作流。
全曲面使用 `--book full --factors 2`；ATM 缓存不能替代不同网格的全曲面 MC。
去掉 `--fast` 后是不同精度的实验，旧 fast 缓存不会被冒充为正式缓存。

## 4. 重点输出

| 文件 | 看什么 |
|---|---|
| `summary.csv` | 同一账本的 std_error、rmse、与 fixed_1 的改善；重点比较 dynamic_raw、shadow_bs、shadow_mc_base |
| `signals.csv` | predicted_beta、effective_beta、beta_mapping_gap，以及 clipping/fallback |
| `option_pnl.csv` | 每日每合约实际 delta 和损益，检查是否真用了同一个信号 |
| `mc_consistency.csv` | 每个日期/单元/alpha 的一阶一致性检查，不增加 MC |
| `converter.csv` | fixed 使用的参考表，或 pooled 平均曲线、跨日期标准差和来源日期 |
| `manifest.json` | 本次配置、训练截止日和转换器来源日期 |

`beta_mapping_gap = effective_beta - predicted_beta`。
对 MC 策略，effective_beta 从**当天**曲线按实际使用的 alpha 读取。
因此固定转换器即使反查成功，也不保证当天实际 beta 等于预测 beta。
对 shadow 策略，effective_beta 就是预测 beta；这只是信号使用约定，不表示已验证真实市场响应。

MC 一致性使用 up/down 反解 IV 的中点近似无 bump 的 MC IV，在同一基准下计算 Greeks：

$$
e_{raw}=\Delta_{MC}(\alpha)-\left(\Delta_{BS,mid}(\alpha)-\frac{\nu_{mid}(\alpha)}{S}\beta_{raw}(\alpha)\right).
$$

$$
e_{centered}=\Delta_{MC}(\alpha)-\left(\Delta_{MC}(1)-\frac{\nu_{mid}(1)}{S}\beta_{converter}(\alpha)\right).
$$

分别对应 `raw_chain_residual`、`centered_adjustment_residual`。
另有 `mc_one_minus_bs_delta` 检查 MC 基准与历史原始 SVI BS 基准的差异。
价格反解截断单独保留标记。中点近似、有限 bump 和模型修复都可能影响检查；
这些残差不是独立 Monte Carlo z 检验，不能根据它们宣称数值收敛。

## 5. 本次缓存复算结果

设置：3M ATM、1 因子、相同 Step 6 冻结预测、156 个测试区间、零交易成本、
10k 路径、201 ratio nodes、1% spot bump。训练截止日 2026-01-15。
旧输出没有删除或覆盖，旧策略净误差复现的最大绝对差约为 $9.1\times10^{-13}$。

| 对冲策略 | 净误差标准差（越小越好） | RMSE |
|---|---:|---:|
| fixed_1 | 7.2684 | 7.2540 |
| BS delta | 7.2714 | 7.2570 |
| 解析 shadow_bs | 7.4145 | 7.3924 |
| 解析 shadow_mc_base | 7.4091 | 7.3871 |
| dynamic_raw：单日固定 | 7.6079 | 7.5858 |
| dynamic_raw：10 日汇总 | 7.6509 | 7.6287 |
| dynamic_raw：当天转换 | 7.4035 | 7.3815 |

两个 shadow 策略在三种转换模式下结果完全相同，这是它们确实绕过转换器的检查之一。
三种模式的 dynamic_raw 均无 alpha 截断、无 inverse fallback。
其 beta_mapping_gap RMSE 分别约为 0.06605、0.07311、$3.65\times10^{-17}$。
逐日模式最后这个接近零是同一张表正反插值的一致性，不是预测误差接近零。

10 日平均曲线在 alpha=0/0.5/1/1.5/2 的 beta 为
0.6624/0.3306/0/-0.3335/-0.6664；alpha=0 的跨日期 beta 标准差约 0.1276。
汇总关系有明显状态离散程度，多加日期并没有保证当天映射更准确。

780 个 MC 节点的 raw_chain_residual RMSE 约 0.000852，centered_adjustment_residual
RMSE 约 0.000847，MC alpha=1 与 BS delta 差异的 RMSE 约 0.001324。
所有节点的 IV 反解都没有价格截断。这里使用旧 fast 数值设置，不是高精度认证。

**结论：** 解析版比单日固定转换版好，但当天转换版与解析版更接近且略好。
这支持固定参考曲面失配是额外误差来源之一，不能据此否定整条 MC 方法。
所有动态版本仍未胜过 fixed_1，因此也不能说预测信号本身已经足够好。
多日平均在这次预设对照中没有改善，不应根据这段测试集重新挑一组“获胜日期”。
后续优先检查数据异常、预测目标与对冲损失的一致性，并在预定日期做 MC 精度敏感性；
本表只是同一测试段上的描述性对照，不是统计显著性或未来表现保证。
