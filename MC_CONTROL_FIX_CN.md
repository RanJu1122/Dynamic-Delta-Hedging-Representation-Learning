# MC 控制变量修复与重跑（2026-09-15）

本次修复同时覆盖 `svi_localvol/montecarlo.py` 和 `dynamic_alpha_hedging/sr_pricing.py`。

- Delta 继续用针对上下腿价差拟合的系数、共同随机数和 antithetic 配对标准误。
- IV 反解使用两条腿各自拟合的价格系数。Beta 标准误由新的两腿价格样本联合传播，保留协方差。
- Call/put 仍按价格反解是否有效、Beta 标准误择优。修复后候选可能变化，因此不能承诺所有最终输出 Delta 与旧库逐项相等。
- 若选中的候选仍需截断价格，或 Beta/标准误非有限，`beta_inversion_valid=False`，生产字段 `beta_model`、`beta_model_stderr` 写为 NaN。未经有效性处理的数值保存在 `beta_model_unchecked`、`beta_model_stderr_unchecked`，价格及反解 IV 也仍可审计。
- 无效 Beta 不参与反查；无效 Alpha=1 锚点会使整条中心化换算曲线不可用。沿用反查失败回退 Alpha=1 的既有行为。SR 的有效 Delta 仍可用于对冲，无效 Beta 的模型归因保持缺失。
- 其他现有标准误、LocalVol 质量阈值仍用于诊断，没有新增任意数值过滤阈值。修复不保证所有翼部节点都收敛。

## 字段解释

| 字段 | 含义 |
|---|---|
| `pv_up/down`；SR 的 `mc_pv_up/down` | 单腿价格系数调整后的价格，用于 IV 反解 |
| `delta_pv_up/down`；SR 的 `mc_delta_pv_up/down` | 价差系数调整后的价格，仅用于核对 Delta |
| `delta_control_beta` | 价差控制系数 |
| `price_control_beta_up/down` | 两条腿各自的价格控制系数 |
| 旧名称 `price_control_beta` | 为兼容保留，仍指价差控制系数 |
| `beta_inversion_valid` | 无价格截断且 Beta、线性化标准误有限；不等同于精度达标 |

`(pv_up-pv_down)/(Spot_up-Spot_down)` 不再等于输出的 Delta。
核对 Delta 应改用 `delta_pv_*`；旧引擎的 Put 候选还需加平价项。
SR 输出的两组价格已经都转换为 Call 价格。

## 已复现案例

2023-11-24、3M、level=0.5、Alpha=0；40k 路径、801 网格、2 子步、1% bump。

| seed | 原实现 Beta / SE | 修复后 Beta / SE | 修复后价格截断 |
|---|---|---|---|
| 20260807 | 14.85845 / 18,856,892 | 3.51873 / 0.38815 | 否 |
| 20260808 | 2.84002 / 1.18703 | 4.07956 / 0.33085 | 否 |
| 20260809 | 2.19495 / 0.65178 | 3.75589 / 0.33463 | 否 |

第一个种子的 up 价格由 -0.049401 修正为 0.204088，价格标准误由 0.339686 降至 0.045080。
这三个种子的候选均为 Put，Delta 标准误分别保持 0.00106985、0.00120530、0.00107639。
这些是代表节点的回归验证，不是全库精度证明，也不是 Beta 真值。

## 1. 重建全历史 precompute

在项目目录运行。使用新目录，原库保留用于对照；不能只改旧库的 engine 字段绕过版本校验。
代码指纹已经变化，旧库会报告 `MC library engine mismatch`。

```bash
cd /home/ran/Huatai_intern/SVI_volatility_surface
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

.venv/bin/python -m dynamic_alpha_hedging precompute \
  --output output/rebuild_20260915_cv_fix/mc_library_40k_shared \
  --paths 40000 --seed 20260807 --substeps 2 \
  --ratio-nodes 801 --ratio-min 0.001 --ratio-max 3 \
  --vol-floor 0 --vol-cap 5 --spot-bump-fraction 0.01 --antithetic
```

期限、level 和 Alpha 沿用默认 full63 与五个 Alpha 节点；其余数值设置与旧库一致。
预计 669 个日期、4,679 个有效日期×期限分片。缺失期限仍记录在 `coverage.csv`。
上述命令中断后，可原样执行以复用新库内已完成的分片。先加 `--plan-only` 可只检查计划。

完成后检查该目录 `manifest.json` 的 `validation.status == "complete"`、
`completed_jobs == date_tenor_jobs == 4679`，并查看 `quality.csv` 与分片中的有效性字段。
完成预计算不等于所有节点均通过数值质量检查。

Step3 是单日期展示报告，precompute 不依赖已有 Step3 CSV；不需要先执行 Step3。
如果需要更新 Step3 报告，可单独运行：

```bash
.venv/bin/python -m dynamic_alpha_hedging step3 \
  --output output/rebuild_20260915_cv_fix/step03 --paths 40000
```

## 2. 准备 Step7 的训练产物

本次 MC 修复不改变 Step1/2 实测 Beta，也不改变 Step4/5/6 的训练数据和算法。
已经完成且相互匹配的 Step1/2/4/5/6 可以直接复用。

目前 `output/step04_dual_20260915/{pca,atm_anchored}` 已有 Step4，尚需各自完成 Step5/6
才能做两模式回测。下面以 PCA 为例；换成 `cv_method=atm_anchored` 可建立另一条研究链。
同一个新 MC 换算库可以供两个模式共用。

```bash
cv_method=pca
cv_source=output/rebuild_20260914_restored_40k_full63
cv_study="output/step04_dual_20260915/$cv_method"

# 复用已有 Step1/2，满足 Step7 要求的完整目录结构。
for cv_stage in step01 step02; do
  if [ ! -e "$cv_study/$cv_stage" ]; then
    ln -s "$(realpath "$cv_source/$cv_stage")" "$cv_study/$cv_stage"
  fi
done

.venv/bin/python -m dynamic_alpha_hedging step5 \
  --factors "$cv_study/step04/factor_scores.csv" \
  --loadings "$cv_study/step04/factor_loadings.csv" \
  --iv-state "$cv_study/step01/iv_state.csv" \
  --changes "$cv_study/step01/grid_changes.csv" \
  --daily-beta "$cv_study/step02/beta_daily.csv" \
  --output "$cv_study/step05" --factor-method "$cv_method"

.venv/bin/python -m dynamic_alpha_hedging step6 \
  --panel "$cv_study/step05/factor_state_panel.csv" \
  --loadings "$cv_study/step04/factor_loadings.csv" \
  --daily-beta "$cv_study/step02/beta_daily.csv" \
  --output "$cv_study/step06" --factor-method "$cv_method"
```

## 3. 用新库运行 Step7

先完成全历史 precompute 和对应模式 Step5/6；沿用上一段设置的 `cv_study`、`cv_method`。
首先检查输入与覆盖范围，不启动新 SR MC：

```bash
.venv/bin/python -m dynamic_alpha_hedging step7 \
  --input-root "$cv_study" \
  --mc-library output/rebuild_20260915_cv_fix/mc_library_40k_shared \
  --mc-cache output/rebuild_20260915_cv_fix/sr_cache_40k \
  --output "$cv_study/step07_cv_fix" \
  --factor-method "$cv_method" --paths 40000 --prepare
```

检查通过后，运行完全相同的命令并去掉 `--prepare`。
默认运行固定到期续持组合的三种动态 SR 策略。新 SR 缓存也会按代码指纹隔离；这里显式指定新目录便于对照。

## 验证命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

`tests/test_control_variates.py` 包含真实异常格点、两定价器一致性、配对误差传播、
仍然越界时的无效 Beta 隔离，以及无效 Beta 不破坏有效 Delta/实际对冲损益的测试。
