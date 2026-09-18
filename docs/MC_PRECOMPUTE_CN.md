# MC 预计算：当前参数、共享路径与使用边界

更新：2026-09-15。当前入口和参数以 `dynamic_alpha_hedging/cli.py`、`precompute.py`、`mc_library.py` 为依据；控制变量修复详见 [修复与重跑](../MC_CONTROL_FIX_CN.md)。本页替代早期40k迁移说明、逐文件修改记录和过时调用耗时报告。

## 依赖与两套定价入口

```text
原始SVI ─→ precompute ─→ Beta–Alpha转换库 ──────────────┐
    └──→ Step1 → Step2 → Step4 → Step5 → Step6 ─→ Step7
```

`precompute`不读取历史Beta标签、PCA或预测器，不依赖先运行Step1/2/3。每个任务按日期×期限保存分片；默认一个分片包含9个level×5个Alpha的结果。

| 项目 | precompute | 当前默认step7 / step7-fixed |
|---|---|---|
| 目的 | 固定标量Alpha对应的Beta、Delta与质量诊断 | 预测Beta形成整张SR规则后，对实际持仓定价 |
| 主要入口 | `LocalVolMC`与批量期限定价 | `sr_pricing.SharedSRPricer` / `simulate_snapshots` |
| 跨期限共享 | 同日、同Alpha、同bump推进到最远期限，各到期点保存快照 | 同一SR规则与bump下，不同实际到期合约共享路径 |
| strike共享 | 同期限所有level使用同组终点股价 | 同到期不同strike使用同组终点股价 |
| 转换库 | 写入分片、索引和质量诊断 | 只读Beta–Alpha转换数据；另跑新的共享SR MC |

因此“Step7只读转换库”不等于“Step7不再运行MC”。旧`step7-legacy`读取标量Alpha表Delta，属于另一个入口。两条链不能混报性能或数值结果。

## 默认参数

| 项目 | 值 |
|---|---|
| 数据 | `data/svi_param.pkl` |
| 网格 | 2/3/6/9/12/18/24月 × K/refSpot=0.4:0.1:1.2，共63 |
| Alpha | 0、0.5、1、1.5、2 |
| MC | 40,000路径；seed=20260807；antithetic开启 |
| 子步 | 每业务日2个 |
| ratio网格 | 801节点，范围0.001–3 |
| local vol floor / cap | 0 / 5 |
| spot bump | 上下各1%，固定K与到期日 |
| 市场假设 | rate=0.036，dividend=0.03，repo=0 |
| 时钟 | 波动率Business/260；利息与贴现Act/365 |

40,000包括对偶正负路径；标准误按对偶配对观测计算。`--fast`为10,000路径和201网格的开发检查，使用独立目录。期限超出当日报价范围记录unsupported，不外推。

## 日期与完整性

2026-03-31、04-01、04-02已经恢复，原Spot与SVI保持原值；零收益只使Step2日比值不可用，不删除整张IV曲面。原始文件问题见[Spot平台审计](SPOT_PLATEAU_TWO_FILES_20260914_CN.md)。无法构建的2023-12-12、2024-08-05、2024-08-06仍由数据加载规则记录跳过。

本次既有原始数据计划包含669个可构建日期、4,679个支持的日期×期限任务。这个数量依赖当前输入数据与覆盖范围，并非写死的通用验收值。

## 2026-09-15控制变量修复后的规则

Delta继续用价差控制系数；up/down用于IV反解的价格分别用单腿价格系数。仍需价格截断或Beta/标准误非有限的估计标记无效，生产Beta写NaN，原始诊断值另存unchecked列。无效Alpha=1锚点会使整条居中曲线不可用。

这修复了估计目标错配，不保证全部翼部节点精度达标。其他标准误、LocalVol质量阈值仍须审计；`beta_inversion_valid`不等于`quality_pass`。

旧`mc_library_40k_shared`的4,679分片完整、13.16%全质量通过率等统计属于修复前库。其引擎指纹与修复后代码不同，不能靠修改指纹复用。

## 重跑与续跑

下面只列命令，本次文档整理未启动MC。保留旧库，给修复后结果单独目录：

```bash
cd /home/ran/Huatai_intern/SVI_volatility_surface
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m dynamic_alpha_hedging precompute \
  --output output/rebuild_20260914_restored_40k_full63/mc_library_40k_shared_cv_fix \
  --paths 40000 --seed 20260807 --substeps 2 \
  --ratio-nodes 801 --ratio-min 0.001 --ratio-max 3 \
  --vol-floor 0 --vol-cap 5 --spot-bump-fraction 0.01 --antithetic
```

先加`--plan-only`可以只写计划。相同命令可复用已完成且校验通过的分片。一个日期批次尚未落盘就中断时，需重算该批次未保存期限。改变引擎、数值参数、数据或网格时使用新的输出库。

完成后先看manifest的status与计划/完成任务数，再看quality.csv和分片中的价格反解、Beta标准误、跨度、原始Alpha=1偏差及LocalVol网格问题。运行完整和数值精度是两种验收。

## 哪些变动需要重算

- 只改Step4降维方式、预测器、因子数、成本或平滑：不改变转换库的定价输入；但需要匹配的模型产物。
- 改Step2标签阈值：重做相关模型链，定价库在数据/引擎/参数一致时可共用。
- 改MC控制变量、网格、paths、bump、市场参数或原始曲面：需要匹配的新定价结果。
- 当前共享SR回测另有`--mc-cache`，须与转换库分开；定价代码/参数改变也会影响缓存指纹。

下游运行与三种策略见[Step7当前指南](STEP7_GUIDE_CN.md)。
