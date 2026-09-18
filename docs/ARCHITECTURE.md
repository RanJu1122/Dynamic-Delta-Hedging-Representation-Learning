# Architecture / 当前代码入口

更新：2026-09-15。本页替代早期中文架构快照；具体数学定义和运行参数链接到单一维护入口。

## Dependency rule

```text
svi_localvol (core) ─→ pricing_svi_localvol_calibration
                 └─→ dynamic_alpha_hedging
```

The core imports neither workflow. The workflows do not import each other.

| Core module | Responsibility |
|---|---|
| `conventions.py` | Business/260、Act/365、日历 |
| `params.py` / `svi.py` | 参数容器、SVI-JW/raw转换、导数与静态套利 |
| `surface.py` | 总方差、IV、Dupire与修补 |
| `blackscholes.py` | 价格、Greeks、IV反解 |
| `montecarlo.py` | LocalVol网格、路径、联合Beta/Delta估计 |

## Dynamic study

```text
原始SVI ─→ Step1 → Step2 → Step4 → Step5 → Step6 ─→ Step7
     └──→ precompute → 独立转换库 ───────────────────┘
     └──→ Step3（单日映射与诊断，不是precompute前置条件）
```

| Module | 当前职责 | 文档 |
|---|---|---|
| `step01.py` | constant_tau、K/S网格；拆分smile crossing与surface change | [Step1–2](step1_guide.md) |
| `step02.py` | daily-only实测Beta；小收益/缺失标记 | [Daily-only](DAILY_ONLY_PIPELINE_CN.md) |
| `step03.py` | 固定K的单日Alpha–Beta映射及质量 | [Step3](step3_guide.md) |
| `step04.py` / `factors.py` | ATM锚定或普通PCA；训练基底与配套标识 | [Step4词典](STEP4_METRICS_GUIDE_CN.md) |
| `step04_report.py` | 中文解读、截面/残差图、指标汇总 | 同上 |
| `step05.py` | 因子状态面板、可预测性诊断、逐节点预测基准 | [Daily-only](DAILY_ONLY_PIPELINE_CN.md) |
| `step06.py` | close-t状态→t+1因子；重构Beta，比较预测基准 | 同上 |
| `precompute.py` / `mc_library.py` | 跨期限共享MC、日期×期限分片、指纹校验 | [Precompute](MC_PRECOMPUTE_CN.md) |
| `step07_fixed_book.py` / `fixed_book.py` | 当前默认固定续开63槽位book | [Step7](STEP7_GUIDE_CN.md) |
| `step07_shared.py` / `sr_pricing.py` | 共享SR规则、实际持仓新MC；同时服务每日滚动实验 | 同上 |
| `step07.py` / `hedging.py` | 上游准备、共享预测器接入；legacy逐合约回测和公共估值逻辑 | 同上 |

默认全链full63：2–24月7期限，K/S=0.4–1.2共9档。Step4完整日期前75%拟合、后25%冻结基底投影，不填缺失，不做节点方差标准化。默认首因子是实测3M ATM，其他两项是残差PC；普通PCA以配置显式选择。两模式均支持下游。

Step4的测试投影是重构，不是未来预测。Step7重新调用共享预测器拟合入口，只用训练标签日期；不是直接加载Step5逐节点预测结果。

## Pricing and provenance

当前`step7`是`step7-fixed`的别名，复用转换库但为共享SR规则运行新MC；旧入口为`step7-legacy`。固定book跨数据缺口持有，不能套用legacy的分段现金账说明。

2026-09-15已修复价差控制系数用于单腿IV价格的问题；相关旧MC库与新引擎指纹不匹配，需新库。见[修复说明](../MC_CONTROL_FIX_CN.md)。该问题不改变Step2历史标签或Step4数值。

各stage的manifest记录配置、输入SHA256、日期范围和validation；factor_basis_id防止因子载荷混用。输出在`output/`，原始输入在`data/`。指标解释应绑定具体产物版本，历史报告不自动更新为新结果。
