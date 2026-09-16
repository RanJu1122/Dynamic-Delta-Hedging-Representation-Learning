"""Human-readable Step 4 diagnostics, separate from the fitted factor basis."""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import schema_for

# Display only: omit divisions by small observed anchor beta, never filter fitting.
SHAPE_DISPLAY_MIN_ABS_ANCHOR = 0.05


def diagnostic_tables(result):
    """Summarize existing errors without averaging node R² into global R²."""
    rows = []
    for _, row in result.explained_variance.iterrows():
        k = int(row.factor_number)
        cell = result.reconstruction_by_cell[f"test_reconstruction_r_squared_{k}factor"]
        item = dict(
            factor_count=k,
            train_reconstruction_r2=row.cumulative_train_explained_variance_ratio,
            test_reconstruction_r2=row.cumulative_test_reconstruction_r_squared,
            test_node_r2_median=cell.median(),
            test_negative_node_count=int((cell < 0).sum()),
            test_defined_node_count=int(cell.notna().sum()),
        )
        for sample in ("train", "test"):
            errors = result.scores.loc[result.scores["sample"].eq(sample),
                                       f"reconstruction_rmse_{k}factor"]
            item[f"{sample}_beta_rmse"] = float(np.sqrt(np.mean(errors**2)))
        rows.append(item)
    summary = pd.DataFrame(rows)
    variance = result.reconstruction_by_cell[["tenor", "level", "actual_beta_std_train"]].copy()
    variance["train_variance"] = variance.actual_beta_std_train**2
    total = variance.train_variance.sum()
    variance["train_variance_share"] = variance.train_variance / total if total > 0 else np.nan
    return summary, variance


def _plots(result, summary, variance, target):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/svi-localvol-mpl")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return []
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei") if f in names), "DejaVu Sans")
    files = []
    with plt.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1.6, 1]})
        ax = axes[0]
        for col, label, color in (("train_reconstruction_r2", "训练：拟合时用过的日期", "#8496ad"),
                                   ("test_reconstruction_r2", "测试：冻结形状，已知当天因子", "#087e8b")):
            ax.plot(summary.factor_count, summary[col] * 100, "o-", label=label, color=color)
            for k, value in zip(summary.factor_count, summary[col]):
                ax.annotate(f"{value:.1%}", (k, value * 100), xytext=(0, 8 if col.startswith("train") else -17), textcoords="offset points", ha="center", fontsize=10)
        ax.set(xticks=[1, 2, 3], xlabel="保留因子数", ylabel="重构 R²（%）", title="① 少数因子能还原多少变化？")
        ax.legend(fontsize=8, loc="lower right")
        scores = summary[["train_reconstruction_r2", "test_reconstruction_r2"]].to_numpy() * 100
        ax.set_ylim(float(np.nanmin(scores))-10, float(np.nanmax(scores))+10)
        ax.grid(alpha=.2)
        table = result.reconstruction_by_cell.pivot(index="tenor", columns="level", values="test_reconstruction_r_squared_3factor")
        image = axes[1].imshow(table, origin="lower", aspect="auto", cmap="RdYlGn", norm=TwoSlopeNorm(vmin=min(-.25, float(table.min().min())), vcenter=0, vmax=max(1., float(table.max().max()))))
        for i in range(len(table)):
            for j in range(len(table.columns)):
                axes[1].text(j, i, f"{table.iloc[i,j]:.0%}", ha="center", va="center", fontsize=8)
        axes[1].set_xticks(range(len(table.columns)), [f"{m:g}" for m in table.columns])
        axes[1].set_yticks(range(len(table)), [f"{t*12:g}M" for t in table.index])
        axes[1].set(xlabel="K / S（1 = ATM）", title="② 三因子在哪些节点表现差？\n负数 = 比训练均值重构更差")
        fig.colorbar(image, ax=axes[1], fraction=.04)
        shares = variance.groupby("tenor").train_variance_share.sum(min_count=1)
        axes[2].barh([f"{12*t:g}M" for t in shares.index], shares*100, color="#8496ad")
        for i, v in enumerate(shares):
            axes[2].text(v*100+.6, i, f"{v:.1%}", va="center", fontsize=9)
        axes[2].set(xlabel="训练总节点方差占比（%）", title="③ 总分主要被哪些期限主导？")
        axes[2].set_xlim(0, max(1, float(shares.max()) * 125))
        fig.suptitle("Step 4 阅读入口：曲面压缩能力，不是明日预测准确率", fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, .92))
        name = "step4_dashboard_cn.png"
        fig.savefig(target/name, dpi=150)
        plt.close(fig)
        files.append(name)

        # Predefined display maturities; no selection by reconstruction performance.
        tenors = sorted(result.surface_examples.tenor.unique())
        selected = list(dict.fromkeys(min(tenors, key=lambda t: abs(t-ref)) for ref in (2/12, result.config.step4_anchor_tenor, 1.)))
        anchored = result.config.step4_factor_method == "atm_anchored"
        fig, axes = plt.subplots(3, len(selected), figsize=(5.2*len(selected), 11), squeeze=False)
        examples = result.surface_examples
        dates = list(examples.observation_date.unique())
        colors = dict(zip(dates, plt.get_cmap("tab10").colors))
        loadings = result.loadings
        skipped = []
        for j, tenor in enumerate(selected):
            sub = examples[examples.tenor.eq(tenor)]
            for date, group in sub.groupby("observation_date", sort=False):
                group = group.sort_values("level")
                color = colors[date]
                axes[0,j].plot(group.level, group.actual_beta, "o-", ms=3, color=color, label=str(date))
                if anchored:
                    amplitude = float(group.atm_beta_observed.iloc[0])
                    if abs(amplitude) >= SHAPE_DISPLAY_MIN_ABS_ANCHOR:
                        axes[1,j].plot(group.level, (group.actual_beta-group.factor_intercept)/amplitude, "o-", ms=3, color=color)
                    else:
                        skipped.append(str(date))
                else:
                    # Ordinary PCA has no privileged observed-ATM amplitude.
                    axes[1,j].plot(group.level, group.actual_beta-group.factor_intercept, "o-", ms=3, color=color)
                axes[2,j].plot(group.level, group.actual_beta-group.reconstructed_beta_1factor, "o-", ms=3, color=color)
            if anchored:
                f = loadings[loadings.tenor.eq(tenor)].sort_values("level")
                axes[1,j].plot(f.level, f.atm_beta_loading, "k--", lw=2, label="训练拟合固定形状 f_ATM")
            axes[0,j].set_title(f"{12*tenor:g}M：同样的五个训练日期")
            for i in range(3):
                axes[i,j].axhline(0, lw=.7, color="gray")
                axes[i,j].grid(alpha=.2)
                axes[i,j].set_xlabel("K / S")
            raw_limits, residual_limits = axes[0,j].get_ylim(), axes[2,j].get_ylim()
            shared_limits = (min(raw_limits[0], residual_limits[0]),
                             max(raw_limits[1], residual_limits[1]))
            axes[0,j].set_ylim(*shared_limits)
            axes[2,j].set_ylim(*shared_limits)
        axes[0,0].set_ylabel("① 原始 Beta\n曲线高低与弯曲一起看")
        axes[1,0].set_ylabel("② (Beta − 截距) / 当天 ATM Beta\n单因子足够时，应接近黑色虚线" if anchored else "② Beta − 训练均值\n普通 PCA 不要求与 ATM 成比例")
        axes[2,0].set_ylabel("③ Beta − 单因子重构\n与第一行同刻度；接近 0 才说明足够")
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.94), ncol=min(5,len(dates)), fontsize=9)
        if anchored:
            axes[1,0].legend(fontsize=8)
        note = (f"归一化只作展示：|ATM Beta| < {SHAPE_DISPLAY_MIN_ABS_ANCHOR:g} 时省略；本次省略 {len(set(skipped))} 天。" if anchored else "普通 PCA 的第二行只中心化，不解释为 ATM 归一形状。")
        fig.suptitle("多日截面：幅度之外，形状是否也在变化？", fontsize=18)
        fig.text(.5,.015,note + " 示例不能证明统计稳定，也不能区分真实变化与观测噪声。", ha="center", fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.9))
        name = "shape_stability_cn.png"
        fig.savefig(target/name, dpi=150)
        plt.close(fig)
        files.append(name)
    return files


def save_readable_report(result, target: Path, source: Path) -> list[str]:
    """Write a portable HTML report, CSV summaries and explanatory PNGs."""
    summary, variance = diagnostic_tables(result)
    summary.to_csv(target/"decision_summary.csv", index=False)
    variance.to_csv(target/"variance_contribution.csv", index=False)
    plots = _plots(result, summary, variance, target)
    v = result.validation
    anchored = result.config.step4_factor_method == "atm_anchored"
    schema = schema_for(result.config.step4_factor_method)
    method = "3M ATM 锚定" if anchored and np.isclose(result.config.step4_anchor_tenor,.25) and np.isclose(result.config.step4_anchor_level,1.) else schema.method
    table = summary.rename(columns={
        "factor_count":"因子数", "train_reconstruction_r2":"训练重构 R²", "test_reconstruction_r2":"测试重构 R²",
        "test_node_r2_median":"测试节点 R² 中位数", "test_negative_node_count":"测试负 R² 节点数",
        "test_defined_node_count":"R² 有定义的节点数", "train_beta_rmse":"训练 Beta RMSE", "test_beta_rmse":"测试 Beta RMSE"})
    for col in ("训练重构 R²", "测试重构 R²", "测试节点 R² 中位数"):
        table[col] = table[col].map(lambda x:f"{x:.2%}")
    for col in ("训练 Beta RMSE", "测试 Beta RMSE"):
        table[col] = table[col].map(lambda x:f"{x:.4f}")
    last = summary.iloc[-1]
    worst = result.reconstruction_by_cell.nsmallest(5, "test_reconstruction_r_squared_3factor")[["tenor","level","test_reconstruction_r_squared_3factor","test_residual_rmse_3factor"]].copy()
    worst.tenor = worst.tenor.map(lambda t:f"{12*t:g}M")
    worst.columns = ["期限","K/S","测试重构 R²","测试 Beta RMSE"]
    worst["测试重构 R²"] = worst["测试重构 R²"].map(lambda x:f"{x:.2%}")
    params = [
        ("分解方法", schema.method, "锚定模式：实测 ATM + 两个残差 PC；普通 PCA：三个主成分"),
        ("节点数", v["surface_cell_count"], "每个日期的一张曲面；不是独立日期样本数"),
        ("期限（月）", ", ".join(f"{12*t:g}" for t in result.config.step4_tenors), "constant_tau；每天保持研究期限，非同一固定到期合约"),
        ("K/S", ", ".join(f"{m:g}" for m in result.config.step4_strike_levels), "1 是现货 ATM；0.4 表示行权价为现货的 40%"),
        ("参考节点", f"{12*result.config.step4_anchor_tenor:g}M / {result.config.step4_anchor_level:g}", "锚定模式在此固定截距0、首载荷1、残差载荷0"),
        ("训练比例", result.config.step4_train_fraction, "完整日期按时间排序，前75%训练、后25%投影；不打乱"),
        ("中心化／标准化", "有截距／不按节点方差标准化", "同等绝对误差权重；高方差节点更影响总分"),
        ("因子数", 3, "固定输出1、2、3因子诊断；没有自动因子数选择"),
        ("Step 2 收益阈值", "以输入 Step 2 manifest 为准", "Step4不重新计算日比值；只读取已保存的 usable 标记与数值"),
        ("缺失", "完整保留网格才入样本", "一个节点缺失也整天剔除；不插值、不填零"),
        ("归一化绘图阈值", SHAPE_DISPLAY_MIN_ABS_ANCHOR, "只影响新形状图，防止小分母放大；不改变PCA或样本"),
    ]
    blocks = []
    def p(text): blocks.append(f"<p>{text}</p>")
    def heading(text): blocks.append(f"<h2>{text}</h2>")
    def image(name, caption):
        path = target/name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            blocks.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    blocks.append(f"<h1>Step 4：{html.escape(method)}，先看懂再往下走</h1>")
    p("<b>结论边界：可以进入 Step 5 检验可预测性；不能把当前重构结果当作预测成功或回测验收。</b>三个因子是待验证的研究表示。")
    p(f"完整曲面 {v['complete_surface_date_count']} / {v['input_date_count']} 天；排除 {v['excluded_incomplete_date_count']} 天。训练 {v['train_start_date']}—{v['train_end_date']}（{v['train_date_count']}天），测试 {v['test_start_date']}—{v['test_end_date']}（{v['test_date_count']}天）。")
    heading("先看这三个问题")
    blocks.append(table.to_html(index=False, escape=True, border=0))
    p(f"三个因子测试总体重构 R² = <b>{last.test_reconstruction_r2:.2%}</b>，节点中位数 = <b>{last.test_node_r2_median:.2%}</b>，有 <b>{int(last.test_negative_node_count)}</b> 个节点比训练均值更差。总体分数不是节点分数的简单平均。")
    p("R² 的参照物是各节点训练期均值：100% = 重构无误差，0 = 与训练均值的平方误差一样，负数 = 更差。77%意为平方误差相对参照减少77%，不是77%的日期预测正确。")
    p("RMSE 是重构 Beta 的典型误差量级，越小越好。Beta RMSE=0.5 时，乘以假设的1%现货对数变化，对应约0.005 IV（0.5个波动率百分点）的误差尺度；这不是已测得的实际 IV RMSE。")
    image("step4_dashboard_cn.png", "从左到右：压缩效果、逐节点薄弱点、总分的期限权重。3M ATM 的100%来自锚定恒等式。" if anchored else "从左到右：压缩效果、逐节点薄弱点、总分的期限权重。")
    heading("叠截面已经实现，应该怎样看？")
    p("原有 surface_cross_sections.png 已经画了固定等距五个训练日期，上排原始 Beta，下排 Beta/实测参考ATM。原图不扣截距，只跳过 |ATM|≤1e−8 的日期。它适合检查文档的纯乘法假设，但与当前有截距模型不是同一口径。")
    p("新图第一行看原始曲线，第二行在锚定模式下看 (Beta−截距)/ATM：如果只有一个ATM幅度在变，各条曲线应贴近黑色虚线。第三行不作除法，直接看 Beta−单因子重构，且与原图同刻度；明显偏离0说明ATM幅度还原不够。普通PCA第二行只减训练均值。")
    p("黑色虚线是训练拟合形状，不是当天数据。归一化曲线交叉、弯曲和斜率发生明显变化时，一个固定形状不足以描述这些观测；仍不能据此证明存在可预测的状态变化，也可能混有日比值噪声。五天只是示例，整体判断要结合所有日期的重构误差。")
    image("shape_stability_cn.png", "固定展示2M、参考期限、12M中最接近的可用期限；所有期限见下一张图。新图不使用测试表现挑日期。")
    image("surface_cross_sections.png", "原有全期限多日截面图；与新图使用相同的固定训练日期。")
    heading("具体模型怎样算？")
    if anchored:
        blocks.append("<pre>β̂[t,c] = a[c] + fATM[c] × A[t] + f1[c] × z1[t] + f2[c] × z2[t]\nA[t] = 当天参考ATM实测Beta\n训练各节点OLS：β[t,c] ~ 常数 + A[t]\nr[t,c] = β[t,c] − a[c] − fATM[c] × A[t]\n训练残差矩阵做SVD，前两个右奇异向量是 f1、f2\nz1[t] = Σc r[t,c] f1[c]；z2[t] = Σc r[t,c] f2[c]</pre>")
        p("a 是固定截距；f 是固定空间权重；A/z 才是每天变化的数字。首因子是实测ATM，<b>不是普通PCA的PC1</b>。形状因子只能称残差PC1/PC2，不能未经分析就命名为斜率/曲率。")
        p("测试日也先拿当天真实曲面算A和残差，再投影出z。因此这里回答已知当天数据能否压缩；Step6才用t日信息预测t+1日的A/z。锚点重构误差0来自a=0、fATM=1、f1=f2=0，不是独立验证成绩。")
    else:
        blocks.append("<pre>β̂[t,c] = 训练均值[c] + Σk PC分数[t,k] × PC载荷[c,k]\n训练期中心化矩阵做SVD；测试期只减训练均值并投影到冻结载荷。</pre>")
    heading("参数与样本")
    blocks.append(pd.DataFrame(params, columns=["项目","本次设置","含义"]).to_html(index=False, escape=True, border=0))
    p("学习率、树深度、迭代轮数、MC路径数等不参与Step4。PCA这里用确定性的全矩阵SVD；没有预测器训练、滚动窗口或随机划分。正负号只为展示固定为最大绝对载荷为正，同时翻转分数不改变重构。")
    heading("最薄弱的五个节点")
    blocks.append(worst.to_html(index=False, escape=True, border=0, float_format=lambda x:f"{x:.4f}"))
    heading("如何决定下一步")
    p("原文PC1>70%、前两PC>85%是方法参考，针对普通PCA。当前首因子受ATM约束，不能把15%的ATM解释率叫作普通PC1解释率。代码 three_factor_train_85pct_pass 是另加的三因子诊断标记，不是原文规定的三因子验收线，false也不阻止保存或Step5。")
    p("可保留锚定三因子作为Step5研究入口：先检验ATM和两个形状因子的事前可预测性，再决定Step6真正要预测哪些因子。额外形状因子重构贡献大，不等于它们容易预测。单独预测ATM并摊到full63，现有低单因子重构分数不足以支持。")
    p("尚未闭合：低维表示仍有遗漏、部分节点弱、观察到的形状变化可能含噪声。后续优先比较训练均值与最近有效值基准，做按时间的预测验证，并按节点检查；Step7才验证对冲。已反复查看的测试区间不能再称完全未见测试集。")
    heading("每个输出文件看什么")
    output_rows = [
        ("decision_summary.csv", "先看：因子数、整体R²/RMSE、节点中位数与负R²数量"),
        ("explained_variance.csv", "各因子增量/累计解释率和SVD奇异值；详见完整词典"),
        ("factor_loadings.csv", "每节点训练均值、截距和固定载荷（不是每日信号）"),
        ("factor_scores.csv", "每日ATM、残差PC分数、样本划分和当天重构RMSE"),
        ("reconstruction_by_cell.csv", "逐节点训练波动率，以及1/2/3因子训练/测试R²和RMSE"),
        ("date_coverage.csv", "哪些日期完整，哪些被排除"),
        ("surface_examples.csv", "五个固定训练日期的原始、1/2/3因子重构及截距"),
        ("variance_contribution.csv", "训练节点方差与占比；衡量整体分数偏重哪些区域"),
        ("manifest.json", "配置、输入哈希、日期范围、数学一致性检查；不是预测成绩"),
    ]
    blocks.append(pd.DataFrame(output_rows,columns=["文件","用途"]).to_html(index=False,escape=True,border=0))
    guide = Path(__file__).resolve().parent.parent/"docs"/"STEP4_METRICS_GUIDE_CN.md"
    if guide.exists():
        text = guide.read_text()
        (target/"METRICS_GUIDE_CN.md").write_text(text)
        blocks.append("<details><summary>展开完整指标词典、定义和公式（也另存为 METRICS_GUIDE_CN.md）</summary><pre class='guide'>"+html.escape(text)+"</pre></details>")
    heading("复现与来源")
    p("输入：<code>"+html.escape(str(source))+"</code>。输入哈希和拟合basis标识见同目录manifest。Step4读取Step2历史Beta，不读取precompute；这次形状诊断不需要重跑MC。")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step4 中文解读</title><style>body{font:16px/1.8 system-ui,sans-serif;color:#18313c;background:#f6f8fa;margin:0}main{max-width:1200px;margin:auto;background:white;padding:32px}h1{font-size:30px}h2{margin-top:40px;border-bottom:2px solid #dce8eb;padding-bottom:6px}table{width:100%;border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:8px 12px;border-bottom:1px solid #dce8eb;text-align:left}th{background:#eaf3f5}img{width:100%;height:auto}figure{margin:25px 0}figcaption{color:#55717d;font-size:14px}pre{white-space:pre-wrap;background:#f3f7f8;padding:18px;overflow-wrap:anywhere}.guide{font:14px/1.8 system-ui,sans-serif}summary{cursor:pointer;font-weight:bold;padding:20px;background:#eaf3f5}code{overflow-wrap:anywhere}</style><main>'+"\n".join(blocks)+"</main></html>"
    (target/"READ_ME_FIRST_CN.html").write_text(page)
    return plots
