"""前兆窗口感知预警评测(论文第一核心组成)。

普通 AD 评测(win_label = 窗内有无异常点)会把**事件中/事件后**的检测也算成功 →
虚高预警性能。本模块按论文区分:

  有效早预警(valid)   : 报警落在有效前兆窗 [onset-l_max, onset-l_min] 内 —— 有足够干预时间
  过早报警(premature) : 报警比有效窗还早(lead_time > l_max)且无邻近事件 → 误报
  迟滞预警(late)       : 报警落在 (onset-l_min, onset) —— 太晚,来不及干预
  事件中/后(post)      : 报警落在 [onset, end] —— 只是检测,不算预警

事件级口径(关键):一个事件只有在其有效前兆窗内被报警,才算**成功早预警**;否则即便事后
检测到也算 missed(早预警角度)。同时报 lead-time。这正是避免"靠临近 onset 观测刷虚高"的设计。
"""
from __future__ import annotations

import numpy as np

from .actions import event_regions, find_events


def precursor_metrics(
    point_labels: np.ndarray,
    window_ends: np.ndarray,
    window_pred: np.ndarray,
    l_min: int,
    l_max: int,
) -> dict:
    """前兆感知评测。

    point_labels : [T] 逐点真值(仅评测用)
    window_ends  : [W] 每个窗的结束时刻 index(决策落点)
    window_pred  : [W] 每个窗是否报警 0/1
    """
    T = len(point_labels)
    events = find_events(point_labels)
    n_events = len(events)
    ends = np.asarray(window_ends, dtype=np.int64)
    pos = np.asarray(window_pred, dtype=np.int64) == 1

    # ---- 窗级:每个报警落在哪个区(用逐点 region 在窗结束点取值) ---- #
    regions = event_regions(T, point_labels, l_min, l_max)        # 0 稳定/1 有效前兆/2 迟滞/3 事件中后
    win_region = regions[np.clip(ends, 0, T - 1)]
    REG = {0: "stable", 1: "valid", 2: "late", 3: "post"}
    alarms_by_region = {REG[r]: int((pos & (win_region == r)).sum()) for r in REG}
    n_alarms = int(pos.sum())

    # ---- 事件级:每个事件按"最早的报警落点"归类 ---- #
    valid = late = post = missed = 0
    lead_times = []
    for ev in events:
        v_lo, v_hi = ev.onset - l_max, ev.onset - l_min          # 有效前兆窗(点坐标)
        in_valid = pos & (ends >= v_lo) & (ends <= v_hi)
        in_late = pos & (ends > ev.onset - l_min) & (ends < ev.onset)
        in_post = pos & (ends >= ev.onset) & (ends <= ev.end)
        if in_valid.any():
            valid += 1
            lead_times.append(int(ev.onset - ends[in_valid].min()))  # 越大=越早报
        elif in_late.any():
            late += 1
        elif in_post.any():
            post += 1
        else:
            missed += 1

    # 早预警 recall:只有"有效窗内报警"才算成功
    ew_recall = valid / max(1, n_events)
    # 任意检测 recall(含迟滞/事后)——用来暴露"虚高":普通口径会把这些都算成功
    any_recall = (valid + late + post) / max(1, n_events)
    # 早预警 precision:报警里多少是"有效前兆窗"内的(stable 内的=误报/过早)
    ew_precision = alarms_by_region["valid"] / max(1, n_alarms)
    ew_f1 = 2 * ew_precision * ew_recall / max(1e-12, ew_precision + ew_recall)

    return {
        "l_min": int(l_min), "l_max": int(l_max),
        "n_events": int(n_events),
        "event_outcomes": {"valid": valid, "late": late, "post": post, "missed": missed},
        "ew_recall": float(ew_recall),         # 有效早预警事件占比
        "any_detect_recall": float(any_recall),  # 含迟滞/事后(普通口径的"虚高"版)
        "inflation": float(any_recall - ew_recall),  # 普通口径相对早预警虚高了多少
        "ew_precision": float(ew_precision),
        "ew_f1": float(ew_f1),
        "lead_time_mean": float(np.mean(lead_times)) if lead_times else 0.0,
        "lead_time_median": float(np.median(lead_times)) if lead_times else 0.0,
        "alarms_by_region": alarms_by_region,
        "n_alarms": n_alarms,
    }
