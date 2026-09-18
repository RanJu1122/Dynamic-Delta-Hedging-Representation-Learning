# Step 7 当前指南：固定续开book、共享SR与评价边界

更新：2026-09-15。依据当前 `cli.py`、`step07_fixed_book.py`、`fixed_book.py`、`step07_shared.py`、`sr_pricing.py`。本页合并旧固定book计划、续开说明、共享SR介绍、旧Delta对照及旧迁移命令。历史报告结果没有被当作新代码验收结果。

当前研究停在Step4解读；本轮没有运行Step5–7。当前输入是否足够运行Step7，必须先通过所选产物链的prepare检查。

## 入口必须分清

| 命令 | 合约生命周期 | 定价 |
|---|---|---|
| `step7` / `step7-fixed` | 当前默认：63槽位，各合约持有到期，再按原槽位期限/level续开 | 全book共享SR规则，新MC |
| `step7-sr` | 每日重建标准期限book，单持有区间内固定K/expiry | 全book共享SR规则，新MC |
| `step7-legacy` | 旧逐合约标量Alpha／每日滚动book实验 | 用标量Alpha表的Delta插值 |

默认网格为2/3/6/9/12/18/24月 × K/S=0.4:0.1:1.2，共63节点。`legacy56`仅是显式旧实验；默认不再漏掉1.2。

## 数据链与预测

1. Step4确定固定截距、载荷和因子定义。当前默认3M ATM加两个残差PC，也保留普通PCA接口。
2. Step5产生状态面板和下一日因子标签；逐节点直接预测仅作诊断基准。
3. Step6训练和评价因子预测。Step7调用共享的`fit_factor_forecaster()`，按训练标签日期重新拟合相同形式的预测器，再生成各决策日期预测；不是直接读取Step5逐节点预测CSV。
4. 用固定截距和载荷把预测因子重构为节点Beta，再经当日转换库反查Alpha。
5. 三种SR规则对同一book产生不同Delta，按净book Delta对冲。

Step5/6/7从上游manifest推断factor_method；显式模式必须一致。scores/loadings的basis ID必须匹配，不能跨模式或跨拟合版本混用。

特征截止close-t，不得使用t+1的收益、RV、IV变化或真实Beta。Step4测试投影分数不是t−1日已知数据。已用于反复挑模型的区间不能继续宣称最终未见测试。

## 三种动态SR规则与对照组

`constant_sr`、`term_sr`、`term_spot_sr`共享同一组预测因子，区别在于Alpha规则在常数、期限或期限×spot维度上的表达与插值，不是三套不同的预测器。细节以`AlphaSurface`和节点转换实现为准。

`step7`当前默认`--strategy-set dynamic`，只跑三种raw动态策略。不能用只包含动态策略的输出声称胜过最好固定Alpha。

- `--strategy-set raw_controls`：三种raw动态策略与对照。
- `--strategy-set full`：增加平滑、历史因子等完整对照变体。
- `--raw-only`是`raw_controls`兼容别名。
- `--strategy-set full --fixed-baseline alpha_one`：预先固定Alpha=1为基准，跳过训练期固定Alpha回测与选优；测试期保留固定Alpha=0/0.5/1/1.5/2、三种raw动态、三种历史因子、三种EMA（half-life>0）、滚动Alpha均值和BS Delta，共16种策略。不会生成`best_fixed_train`策略。预测器仍只用原Step6训练样本重新拟合，滚动Alpha可以用测试开始前的已知历史初始化；EMA从Alpha=1开始。

需要评价时，应预先固定对照与策略集合。主判断是相同持仓、相同日期下，相对预定基准的净book对冲误差是否改善；并报告成本、turnover、平滑和回退率。默认基准是训练期选定的最好固定Alpha；`--fixed-baseline alpha_one`改为固定Alpha=1，不从测试期重新选基准。

## 固定续开book规则

- 初始63槽位；同槽位数量、原始tenor和原始level保存。
- 持有期间实际K和日历expiry固定；不能每天把它当回原始剩余期限。
- 到期结算旧合约，再以“原始tenor、原始level×续开当日spot、原数量”开新合约。
- `--no-renew-expired`才运行到期不补新的旧实验。
- 缺口期间保留同一book与上次标的持仓，恢复后累计跨缺口损益；这与legacy分段重建账户不是同一规则。
- 样本末日未到期头寸继续盯市，不人为强制卖出。
- 老化合约超出报价期限边界时，默认`--mark-extrapolation flat_iv`并记录；可选`reject`。这不意味着Step1/2的缺失网格也做flat填补。
- 当前默认近到期≤10业务日使用最多0.5%的较小bump，不大于基本bump。
- `--flat-spot-alpha-one`使用当前与上次已经观察到的相同spot判断，不读取未来是否零收益；可显式关闭。

2026-03-31、04-01、04-02已恢复，原始spot不修改；零收益标签缺失不等于删除实际持有区间。

## 转换器与新SR定价的关系

转换库只读：`beta_converter(alpha)=beta_model(alpha)−beta_model(1)`。居中锚点0是定义约定，不能证明原始Alpha=1数值精度通过。

2026-09-15修复后，价格截断、Beta或标准误非有限的估计不进入反查，无效Alpha=1锚点使整条曲线不可用；失败沿用Alpha=1回退。其他质量阈值仍需诊断。旧库必须按匹配引擎重建，详见[MC修复](../MC_CONTROL_FIX_CN.md)。

当前默认Step7使用转换库获取Alpha，再由`SharedSRPricer`对实际共享SR规则和持仓运行新MC。`--mc-library`和`--mc-cache`可以同时提供，分别表示只读转换库和新SR缓存。旧`step7-legacy`两个选项互斥，不要套用旧文档限制。

## 准备与运行

上游必须具有匹配的Step1/2/4/5/6与新转换库。仅用当前Step4输出文件夹不能直接跑完整Step7。

```bash
# 先自行准备完整input-root；这里的命令不会被文档整理自动执行。
.venv/bin/python -m dynamic_alpha_hedging step7 \
  --input-root output/step04_dual_20260915/atm_anchored \
  --mc-library output/rebuild_20260914_restored_40k_full63/mc_library_40k_shared_cv_fix \
  --mc-cache output/rebuild_20260914_restored_40k_full63/sr_cache_40k_cv_fix \
  --output output/step04_dual_20260915/atm_anchored/step07_cv_fix \
  --factor-method atm_anchored --factors 3 --paths 40000 \
  --strategy-set full --prepare
```

prepare会拟合预测器并验证计划/覆盖，不运行新的SR MC。完成准备且确认所选实验设计后，同命令去掉`--prepare`才运行回测。`--strategy-set full`是为了评估基准而显式选择，区别于程序默认只跑三种动态策略。

共享SR默认40k路径，其他数值设置按CLI继承转换库或显式覆盖；不要把旧报告100k默认套用到当前入口。

## 结果应该怎么看

- 先核对manifest、实际策略集合、样本与定价指纹。
- 重点看`summary.csv`的book级对冲误差及对固定档改善；仅在对应对照实际存在时解释改善。
- Alpha=1基准模式看`raw_std_improvement_vs_alpha_one`、`net_std_improvement_vs_alpha_one`和`net_rmse_improvement_vs_alpha_one`；正数表示改善。`improvement_ci_low/high`是相对`comparison_baseline`的raw标准差改善区间，此模式为`fixed_1`。`raw_std_improvement_vs_best_fixed`留空，不能解释成对最好固定Alpha的改进。
- `book_pnl.csv`用于现金、持仓、carry、成本、误差和财富核对。
- 节点Alpha、预测Beta、实施规则、边界截断和回退审计定位转换问题。
- 实际对冲损益与Beta驱动的事后归因残差分别报告，不能重复加入同一spot-vol项。
- 同时看净标的turnover、Alpha平滑与尾部误差；不能只报某个单节点改善。

完整账本推导保留在[账本与归因参考](STEP7_FULL_BOOK_BACKTEST_ACCOUNTING_CN.md)，该文历史数值来自旧每日滚动book；其结果不能直接移用于当前固定续开book。

旧`step7-legacy`仍支持daily/fixed/pooled转换器及解析shadow Delta对照；这些是研究对照接口，不意味着当前默认共享SR策略集合自动包含所有legacy策略。
