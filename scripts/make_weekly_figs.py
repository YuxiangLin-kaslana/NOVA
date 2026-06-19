#!/usr/bin/env python
"""生成周报图表:6 张 PNG + 合并成一份多页 PDF。
中文+拉丁用 assets/NotoSansCJKsc-Regular.otf(单字体含两者)。轻量 CPU,登录节点即可跑。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrow

import matplotlib.font_manager as fm
_FONT = os.path.join(os.path.dirname(__file__), "assets", "NotoSansCJKsc-Regular.otf")
fm.fontManager.addfont(_FONT)
_NAME = fm.FontProperties(fname=_FONT).get_name()
plt.rcParams["font.family"] = _NAME          # 含中文+拉丁,单字体搞定
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "log", "figs")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

C_FROZEN = "#9aa0a6"
C_BOOT   = "#1a73e8"
C_ORACLE = "#34a853"
C_LLM    = "#ea4335"

figs = []


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    figs.append(fig)
    print("wrote", p)


# 图1:Task A 四臂整体准确率
def fig1():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    arms = ["llm_only\n(全靠LLM)", "frozen\n(闭集下界)", "bootstrap★\n(本方法)", "oracle\n(人工上界)"]
    vals = [21.2, 48.9, 74.7, 97.0]
    errs = [2.5, 2.4, 7.7, 1.7]
    cols = [C_LLM, C_FROZEN, C_BOOT, C_ORACLE]
    bars = ax.bar(arms, vals, yerr=errs, capsize=5, color=cols, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("整体准确率 (%)")
    ax.set_ylim(0, 110)
    ax.set_title("图1  Task A:四个对照臂的整体准确率(5 seed,后段)", fontweight="bold")
    ax.annotate("差距≈22pt\n纯 LLM 命名噪声", xy=(2.4, 86), xytext=(1.3, 100),
                fontsize=9, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig1_taskA_arms.png")


# 图2:Task A 新类召回
def fig2():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    arms = ["frozen", "bootstrap★", "oracle"]
    vals = [0.0, 57.2, 95.3]
    errs = [0.0, 16.8, 3.7]
    cols = [C_FROZEN, C_BOOT, C_ORACLE]
    bars = ax.barh(arms[::-1], vals[::-1], xerr=errs[::-1], capsize=5,
                   color=cols[::-1], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontweight="bold")
    ax.set_xlabel("新类 correlation_break 召回 (%)")
    ax.set_xlim(0, 110)
    ax.set_title("图2  新类召回:闭集是真的 0%", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig2_taskA_newclass.png")


# 图3:四段递进 frozen vs bootstrap
def fig3():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    stages = ["A 单新类\n分类", "B 多新类\n分类", "检测桥\n(新类检测召回)", "D 类型化\n早预警"]
    frozen = [0, 0, 15, 0]
    boot   = [57, 88, 88, 100]
    x = range(len(stages))
    w = 0.38
    ax.bar([i - w / 2 for i in x], frozen, w, label="frozen(闭集)", color=C_FROZEN, edgecolor="black", linewidth=0.6)
    ax.bar([i + w / 2 for i in x], boot,   w, label="bootstrap(本方法)", color=C_BOOT, edgecolor="black", linewidth=0.6)
    for i, (f, b) in enumerate(zip(frozen, boot)):
        ax.text(i - w / 2, f + 2, f"{f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 2, f"{b}%", ha="center", fontsize=9, fontweight="bold")
        ax.annotate("", xy=(i + w / 2, b - 4), xytext=(i - w / 2, f + 4),
                    arrowprops=dict(arrowstyle="->", color="#d93025", lw=1.4))
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("召回 / 准确率 (%)")
    ax.set_ylim(0, 115)
    ax.set_title("图3  主线四段:闭集做不到 → 本方法救回(越右越接近论文问题)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3_progression.png")


# 图4:检测桥整体召回拆解
def fig4():
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.barh([1], [0.78], color=C_FROZEN, edgecolor="black", height=0.5, label="frozen 0.78")
    ax.barh([0], [0.78], color=C_BOOT, edgecolor="black", height=0.5)
    ax.barh([0], [0.19], left=[0.78], color="#fbbc04", edgecolor="black", height=0.5,
            label="+0.19 全是捞回的新类")
    ax.text(0.78 / 2, 1, "0.78", va="center", ha="center", color="white", fontweight="bold")
    ax.text(0.78 / 2, 0, "0.78", va="center", ha="center", color="white", fontweight="bold")
    ax.text(0.78 + 0.19 / 2, 0, "+0.19", va="center", ha="center", fontweight="bold")
    ax.text(0.99, 0, "0.97", va="center", ha="left", fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["bootstrap", "frozen"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("整体检测召回")
    ax.set_title("图4  检测桥:召回 0.78→0.97 的提升几乎全来自救回漏检新类", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_detection_bridge.png")


# 图5:早预警时间轴
def fig5():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    onset = 8
    # 前兆窗背景
    ax.axvspan(0, onset, color="#fff3cd", alpha=0.8)
    ax.axvspan(onset, onset + 3, color="#f8d7da", alpha=0.7)
    ax.text(onset / 2, 2.7, "前兆窗(弱信号 strength=0.6)", ha="center", fontsize=9)
    ax.text(onset + 1.5, 2.7, "事件爆发", ha="center", fontsize=9, color="#a11")
    ax.axvline(onset, color="#a11", ls="--", lw=1.2)
    # frozen 行
    for t in range(onset):
        ax.text(t + 0.5, 2.0, "×", ha="center", color=C_FROZEN, fontsize=13)
    ax.text(-0.3, 2.0, "frozen", ha="right", va="center", fontweight="bold")
    ax.text(onset / 2, 1.55, "全程报不出(词表里没有该类型)", ha="center", fontsize=8.5, color=C_FROZEN)
    # bootstrap 行
    ax.text(-0.3, 1.0, "bootstrap", ha="right", va="center", fontweight="bold", color=C_BOOT)
    ax.text(0.5, 1.0, "✓", ha="center", color=C_BOOT, fontsize=15, fontweight="bold")
    ax.annotate("", xy=(onset, 1.0), xytext=(0.6, 1.0),
                arrowprops=dict(arrowstyle="->", color=C_BOOT, lw=2))
    ax.text(onset / 2 + 0.5, 0.6, "lead-time = 8 窗,在前兆最早端就报出正确新类型 T(type-FAR 仅 4.8%)",
            ha="center", fontsize=8.5, color=C_BOOT)
    ax.set_xlim(-1.5, onset + 3.2)
    ax.set_ylim(0.2, 3.1)
    ax.set_xticks(range(0, onset + 1, 2))
    ax.set_xlabel("时间(窗)→  0 = onset-8,虚线 = onset")
    ax.set_yticks([])
    ax.set_title("图5  类型化早预警:bootstrap 提前 8 窗报对类型,frozen 全程为 0", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_early_warning.png")


# 图6:LLM 成本衰减 + 新类准确率爬升
def fig6():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    seg = [1, 2, 3, 4, 5]
    llm = [43, 41, 31, 26, 17]
    acc = [32, 43, 62, 68, 82]
    l1, = ax.plot(seg, llm, "o-", color=C_LLM, lw=2, label="LLM 调用率 (%)")
    l2, = ax.plot(seg, acc, "s-", color=C_BOOT, lw=2, label="新类准确率 (%)")
    for x, y in zip(seg, llm):
        ax.text(x, y + 2.5, f"{y}", ha="center", fontsize=9, color=C_LLM)
    for x, y in zip(seg, acc):
        ax.text(x, y - 5, f"{y}", ha="center", fontsize=9, color=C_BOOT)
    ax.set_xlabel("时间段")
    ax.set_ylabel("百分比 (%)")
    ax.set_xticks(seg)
    ax.set_ylim(0, 100)
    ax.set_title("图6  Task A:越学越准(↑),越来越不靠 LLM(↓)", fontweight="bold")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig6_llm_decay.png")


fig1(); fig2(); fig3(); fig4(); fig5(); fig6()

pdf_path = os.path.join(OUT, "weekly_report_figs.pdf")
with PdfPages(pdf_path) as pdf:
    for f in figs:
        pdf.savefig(f)
print("wrote", pdf_path)
print("OK", len(figs), "figures")
