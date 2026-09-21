"""稳健滑动窗口异常检测核心算法。

思路：
1. 对每个点，取其之前 ``window_size`` 个点（严格 trailing window）计算
   稳健基线：中位数 median 与 MAD（median absolute deviation）。
2. 以 MAD * 1.4826 作为稳健标准差估计，计算稳健 z 分数：
       z = (x - median) / (1.4826 * MAD)
   当 |z| >= threshold 时标记为异常点（突变 / 水平偏移）。
3. 允许把相邻异常点之间至多 ``merge_gap`` 个正常点桥接，合并成异常区间。
4. 每个异常区间给出基线、分数与触发点的人类可读解释。

基线策略 ``baseline_policy``：
- ``rolling``（默认，保持原始行为）：所有历史点一律滑入基线窗口；
- ``guarded``：只有判为正常的点才允许进入后续基线，孤立尖刺不会污染
  其后的中位数与离散度；当连续异常点数达到 ``rebase_after`` 时，用该异常
  段最近 ``min_window`` 个点重建基线并递增代次（generation），触发点本身
  仍保持异常，且区间不因此被拆分。

采样断档感知 ``gap_reset_factor``（可选，>1）：
启用后以排序后相邻点正时间差的中位数估算常规采样间隔，相邻点间隔超过
``gap_reset_factor`` 倍估算间隔即切分为新的采样分段；每个分段独立执行
``min_window`` 预热，rolling/guarded 的基线、连续异常计数与重建代次全部
从头开始，异常区间也不允许跨越断档合并。
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from statistics import median
from typing import Any

# 正态分布下 MAD 与标准差的换算系数
MAD_SCALE = 1.4826
# 基线恒定时（MAD=0 且标准差=0）又与基线不一致，使用的有限大 z 分数哨兵值
ZERO_MAD_ZSCORE = 999.0


def _sample_std(values: list[float], mean: float) -> float:
    """样本标准差（n-1 修正）；点数不足 2 时返回 0。"""
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _window_stats(history: list[float]) -> tuple[float, float, float, str]:
    """返回 (中位数, 原始MAD, 稳健离散度, 离散度方法)。

    优先使用 1.4826*MAD；当 MAD=0（小窗口落在多峰 / 离散信号上会误判）时，
    回退到样本标准差；两者都为 0 才视为真正退化的恒定基线。
    """
    med = median(history)
    mad = median([abs(x - med) for x in history])
    robust_sd = mad * MAD_SCALE
    if robust_sd > 0:
        return med, mad, robust_sd, "mad"
    mean = sum(history) / len(history)
    std = _sample_std(history, mean)
    if std > 0:
        return med, mad, std, "std_fallback"
    return med, mad, 0.0, "constant"


def _zscore(value: float, med: float, robust_sd: float) -> tuple[float, bool]:
    """计算稳健 z 分数；返回 (zscore, 基线是否退化)。"""
    degenerate = robust_sd == 0.0
    if degenerate:
        return (0.0 if value == med else ZERO_MAD_ZSCORE), True
    return (value - med) / robust_sd, False


def _compute_sampling_segments(
    timestamps: list[str], gap_reset_factor: float | None
) -> tuple[list[tuple[int, int]], list[dict[str, Any]], float | None]:
    """按采样断档切分排序后的序列。

    返回 ``(segments, gap_boundaries, median_interval_seconds)``：
    - ``segments``：``[start, end)`` 形式的分段索引范围，覆盖全部点；
    - ``gap_boundaries``：每个断档一条记录，给出断档前后的点、间隔秒数、
      倍数及两侧分段编号；
    - ``median_interval_seconds``：相邻点正时间差的中位数（秒）；未启用
      断档感知，或没有正时间差可估算（所有时间戳相同）时为 ``None``。

    时间戳已在请求层去重，但这里仍只统计正时间差（非正差值不参与中位数
    估算，也不会被当作断档）。
    """
    n = len(timestamps)
    if gap_reset_factor is None:
        return [(0, n)], [], None

    dts = [datetime.fromisoformat(ts) for ts in timestamps]
    deltas = [
        (dts[i] - dts[i - 1]).total_seconds()
        for i in range(1, n)
        if (dts[i] - dts[i - 1]).total_seconds() > 0
    ]
    if not deltas:
        return [(0, n)], [], None

    median_interval = median(deltas)
    boundaries: list[dict[str, Any]] = []
    segments: list[tuple[int, int]] = []
    start = 0
    segment_id = 0
    for i in range(1, n):
        delta = (dts[i] - dts[i - 1]).total_seconds()
        if delta > 0 and median_interval > 0 and delta > median_interval * gap_reset_factor:
            segments.append((start, i))
            boundaries.append(
                {
                    "before_index": i - 1,
                    "after_index": i,
                    "before_time": timestamps[i - 1],
                    "after_time": timestamps[i],
                    "gap_seconds": delta,
                    "gap_ratio": delta / median_interval,
                    "left_segment": segment_id,
                    "right_segment": segment_id + 1,
                }
            )
            start = i
            segment_id += 1
    segments.append((start, n))
    return segments, boundaries, median_interval


def analyze_series(
    timestamps: list[str],
    values: list[float],
    window_size: int = 20,
    min_window: int = 5,
    threshold: float = 3.0,
    merge_gap: int = 2,
    baseline_policy: str = "rolling",
    rebase_after: int | None = None,
    gap_reset_factor: float | None = None,
) -> dict[str, Any]:
    """执行检测并返回可 JSON 序列化的完整结果。

    :param timestamps: 已排序、已去重的 ISO-8601 时间戳字符串
    :param values: 与 timestamps 对齐的数值
    :param baseline_policy: ``rolling``（默认，所有点滑入基线）或
        ``guarded``（仅正常点进入基线，连续异常达到 ``rebase_after`` 时重建）
    :param rebase_after: guarded 模式下触发基线重建的连续异常点数，
        默认取 ``min_window``，且必须不小于 ``min_window``；
        rolling 模式忽略该参数
    :param gap_reset_factor: 大于 1 时启用采样断档感知：以相邻点正时间差
        的中位数为常规采样间隔，间隔超过该倍数即切分新分段，每段独立预热
        与维护基线状态；为 None（默认）时保持原有整体检测行为
    """
    n = len(values)
    # 每个被评估点的注解；预热点（未启用断档时为前 min_window 个点）没有
    # 足够基线，不参与评估
    annotations: list[dict[str, Any] | None] = [None] * n
    is_anomaly = [False] * n
    zscores: list[float | None] = [None] * n
    # guarded 模式下各异常点评分时实际使用的基线成员（rolling 不需要）
    baseline_windows: list[tuple[int, ...] | None] = [None] * n
    baseline_rebuild_count = 0

    evaluated_start = min_window
    if baseline_policy == "guarded" and rebase_after is None:
        rebase_after = min_window

    sampling_segments, gap_boundaries, median_interval = _compute_sampling_segments(
        timestamps, gap_reset_factor
    )

    for seg_id, (seg_start, seg_end) in enumerate(sampling_segments):
        if seg_end - seg_start <= min_window:
            # 分段内点数不足以完成预热：该段没有任何被评估点
            continue
        if baseline_policy == "guarded":
            baseline_rebuild_count += _run_guarded(
                timestamps=timestamps,
                values=values,
                window_size=window_size,
                min_window=min_window,
                threshold=threshold,
                rebase_after=rebase_after,
                range_start=seg_start,
                range_end=seg_end,
                annotations=annotations,
                is_anomaly=is_anomaly,
                zscores=zscores,
                baseline_windows=baseline_windows,
                segment_id=seg_id,
            )
        else:
            _run_rolling(
                timestamps=timestamps,
                values=values,
                window_size=window_size,
                min_window=min_window,
                threshold=threshold,
                range_start=seg_start,
                range_end=seg_end,
                annotations=annotations,
                is_anomaly=is_anomaly,
                zscores=zscores,
                segment_id=seg_id,
            )

    # 每个采样分段独立扫描异常点，区间不可能跨越断档合并
    segments: list[dict[str, Any]] = []
    for seg_id, (seg_start, seg_end) in enumerate(sampling_segments):
        if seg_end - seg_start <= min_window:
            continue
        segments.extend(
            _build_segments(
                timestamps=timestamps,
                values=values,
                is_anomaly=is_anomaly,
                zscores=zscores,
                annotations=annotations,
                baseline_windows=baseline_windows,
                window_size=window_size,
                threshold=threshold,
                merge_gap=merge_gap,
                evaluated_start=seg_start + min_window,
                range_end=seg_end,
                range_lo=seg_start,
                segment_id=seg_id if gap_reset_factor is not None else None,
            )
        )

    evaluated_indices = [i for i, ann in enumerate(annotations) if ann is not None]
    if gap_reset_factor is None:
        # 未启用断档感知时保持历史输出：前 min_window 个点不评分的连续区间
        evaluated_point_count = max(0, n - evaluated_start)
        anomaly_point_count = sum(1 for i in range(evaluated_start, n) if is_anomaly[i])
        output_points = [annotations[i] for i in range(evaluated_start, n)]
    else:
        evaluated_point_count = len(evaluated_indices)
        anomaly_point_count = sum(1 for i in evaluated_indices if is_anomaly[i])
        output_points = [annotations[i] for i in evaluated_indices]

    return {
        "point_count": n,
        "window_size": window_size,
        "min_window": min_window,
        "threshold": threshold,
        "merge_gap": merge_gap,
        "baseline_policy": baseline_policy,
        # rolling 不使用 rebase_after，统一回显 null；guarded 回显生效值
        "rebase_after": rebase_after if baseline_policy == "guarded" else None,
        "baseline_rebuild_count": baseline_rebuild_count,
        "gap_reset_factor": gap_reset_factor,
        "estimated_interval_seconds": median_interval,
        "sampling_segment_count": len(sampling_segments),
        "gap_boundaries": gap_boundaries,
        "evaluated_start_index": evaluated_start,
        "evaluated_point_count": evaluated_point_count,
        "anomaly_point_count": anomaly_point_count,
        "anomaly_segment_count": len(segments),
        "segments": segments,
        "points": output_points,
    }


def _run_rolling(
    *,
    timestamps: list[str],
    values: list[float],
    window_size: int,
    min_window: int,
    threshold: float,
    range_start: int,
    range_end: int,
    annotations: list[dict[str, Any] | None],
    is_anomaly: list[bool],
    zscores: list[float | None],
    segment_id: int = 0,
) -> None:
    """rolling 策略：所有历史点一律滑入 trailing window。

    仅评估 ``[range_start + min_window, range_end)`` 内的点，基线也只取
    本采样分段（``[range_start, i)``）内的点，分段从头预热。
    """
    for i in range(range_start + min_window, range_end):
        lo = max(range_start, i - window_size)
        history = values[lo:i]
        med, mad, robust_sd, dispersion_method = _window_stats(history)
        z, degenerate = _zscore(values[i], med, robust_sd)
        zscores[i] = z
        is_anomaly[i] = abs(z) >= threshold

        annotations[i] = {
            "index": i,
            "timestamp": timestamps[i],
            "value": values[i],
            "baseline_median": med,
            "baseline_mad": mad,
            "baseline_robust_sd": robust_sd,
            "dispersion_method": dispersion_method,
            "zscore": z,
            "degenerate_baseline": degenerate,
            "is_anomaly": is_anomaly[i],
            # rolling 不区分代次，也不发生重建
            "baseline_generation": 0,
            "baseline_reset": False,
            "segment": segment_id,
        }


def _run_guarded(
    *,
    timestamps: list[str],
    values: list[float],
    window_size: int,
    min_window: int,
    threshold: float,
    rebase_after: int,
    range_start: int,
    range_end: int,
    annotations: list[dict[str, Any] | None],
    is_anomaly: list[bool],
    zscores: list[float | None],
    baseline_windows: list[tuple[int, ...] | None],
    segment_id: int = 0,
) -> int:
    """guarded 策略：只有正常点进入基线，长异常段触发重建。

    仅评估 ``[range_start + min_window, range_end)`` 内的点；基线种子、
    连续异常计数与重建代次都以本采样分段为界从头开始。
    返回基线重建（代次递增）次数。
    """
    # 基线只容纳已接纳点的索引（重建后为异常段末尾的 min_window 个点）；
    # 正常点滑入、异常点排除，因此成员索引在原序列上可能不连续。
    accepted: deque[int] = deque(range(range_start, range_start + min_window))
    generation = 0
    consecutive_anomalies = 0
    rebuild_count = 0
    # 连续异常期间 accepted 不变，多个异常点可共享同一份窗口成员快照
    window_snapshot: tuple[int, ...] | None = tuple(accepted)

    for i in range(range_start + min_window, range_end):
        members = tuple(accepted)
        history = [values[k] for k in members]
        med, mad, robust_sd, dispersion_method = _window_stats(history)
        z, degenerate = _zscore(values[i], med, robust_sd)
        zscores[i] = z
        anomaly = abs(z) >= threshold
        is_anomaly[i] = anomaly
        # 该点评分所使用的代次（重建发生在评分之后，不影响本点评分）
        scoring_generation = generation

        reset = False
        if anomaly:
            # 记录该点评分时实际使用的基线成员（供异常区间展示，正常点无需记录）
            if window_snapshot is None:
                window_snapshot = members
            baseline_windows[i] = window_snapshot
            consecutive_anomalies += 1
            if consecutive_anomalies >= rebase_after:
                # 用该异常段最近 min_window 个点重建基线并递增代次；
                # 当前点仍是异常，且原异常区间不被拆分。
                accepted = deque(range(i - min_window + 1, i + 1))
                generation += 1
                rebuild_count += 1
                consecutive_anomalies = 0
                reset = True
                window_snapshot = None
        else:
            # 只有判为正常的点允许进入后续基线，孤立尖刺被排除在外
            consecutive_anomalies = 0
            accepted.append(i)
            if len(accepted) > window_size:
                accepted.popleft()
            window_snapshot = None

        annotations[i] = {
            "index": i,
            "timestamp": timestamps[i],
            "value": values[i],
            "baseline_median": med,
            "baseline_mad": mad,
            "baseline_robust_sd": robust_sd,
            "dispersion_method": dispersion_method,
            "zscore": z,
            "degenerate_baseline": degenerate,
            "is_anomaly": anomaly,
            "baseline_generation": scoring_generation,
            "baseline_reset": reset,
            "segment": segment_id,
        }

    return rebuild_count


def _build_segments(
    *,
    timestamps: list[str],
    values: list[float],
    is_anomaly: list[bool],
    zscores: list[float | None],
    annotations: list[dict[str, Any] | None],
    window_size: int,
    threshold: float,
    merge_gap: int,
    evaluated_start: int,
    range_end: int | None = None,
    range_lo: int = 0,
    segment_id: int | None = None,
    baseline_windows: list[tuple[int, ...] | None] | None = None,
) -> list[dict[str, Any]]:
    """扫描异常布尔序列，按 merge_gap 合并相邻异常点为区间。

    只扫描 ``[evaluated_start, range_end)``，因此区间不会跨越采样断档。
    """
    n = len(values) if range_end is None else range_end
    segments: list[dict[str, Any]] = []
    i = evaluated_start

    while i < n:
        if not is_anomaly[i]:
            i += 1
            continue

        seg_start = i
        last_anomaly = i
        j = i + 1
        # 相邻两个异常点索引差 <= merge_gap + 1 即允许桥接
        while j < n and (j - last_anomaly) <= merge_gap + 1:
            if is_anomaly[j]:
                last_anomaly = j
            j += 1
        seg_end = last_anomaly

        segments.append(
            _describe_segment(
                seg_start=seg_start,
                seg_end=seg_end,
                timestamps=timestamps,
                values=values,
                is_anomaly=is_anomaly,
                zscores=zscores,
                annotations=annotations,
                window_size=window_size,
                threshold=threshold,
                range_lo=range_lo,
                segment_id=segment_id,
                baseline_windows=baseline_windows,
            )
        )
        i = seg_end + 1

    return segments


def _describe_segment(
    *,
    seg_start: int,
    seg_end: int,
    timestamps: list[str],
    values: list[float],
    is_anomaly: list[bool],
    zscores: list[float | None],
    annotations: list[dict[str, Any] | None],
    window_size: int,
    threshold: float,
    range_lo: int = 0,
    segment_id: int | None = None,
    baseline_windows: list[tuple[int, ...] | None] | None = None,
) -> dict[str, Any]:
    """构造单个异常区间：基线、分数、触发点与解释。"""
    # 区间基线取第一个触发点评分时实际使用的基线窗口（突变发生前的基线）。
    # rolling 下即 seg_start 之前的连续 trailing window；guarded 下只含已
    # 接纳点，成员索引在原序列上可能不连续（被排除的尖刺不会出现在这里）。
    if baseline_windows is not None and baseline_windows[seg_start] is not None:
        member_indices = baseline_windows[seg_start]  # type: ignore[assignment]
        base_lo, base_hi = member_indices[0], member_indices[-1] + 1
        baseline = [values[k] for k in member_indices]
    else:
        base_lo = max(range_lo, seg_start - window_size)
        base_hi = seg_start
        baseline = [values[k] for k in range(base_lo, base_hi)]
    med, mad, robust_sd, dispersion_method = _window_stats(baseline)

    trigger_indices = [k for k in range(seg_start, seg_end + 1) if is_anomaly[k]]
    triggers: list[dict[str, Any]] = []
    max_idx = trigger_indices[0]
    for k in trigger_indices:
        z = zscores[k]
        assert z is not None
        ann = annotations[k]
        assert ann is not None
        point_med = ann["baseline_median"]
        point_sd = ann["baseline_robust_sd"]
        point_method = ann["dispersion_method"]
        deviation = values[k] - point_med
        direction = "上升" if deviation > 0 else "下降"
        if ann["degenerate_baseline"]:
            reason = (
                f"基线窗口内 MAD=0 且标准差=0（基线恒定为 {point_med:g}），观测值 {values[k]:g} "
                f"与基线不一致，按最强突变处理（z={z:g}，阈值 {threshold:g}）"
            )
        elif point_method == "std_fallback":
            reason = (
                f"基线窗口 MAD=0，回退为样本标准差尺度：z 分数 {z:+.2f} 的绝对值"
                f"超过阈值 {threshold:g}：观测值 {values[k]:g} 相对滑动基线中位数 {point_med:g} "
                f"{direction}偏离 {abs(deviation):g}（样本标准差 {point_sd:.4g}）"
            )
        else:
            reason = (
                f"稳健 z 分数 {z:+.2f} 的绝对值超过阈值 {threshold:g}："
                f"观测值 {values[k]:g} 相对滑动基线中位数 {point_med:g} "
                f"{direction}偏离 {abs(deviation):g}（1.4826·MAD 稳健标准差 {point_sd:.4g}）"
            )
        triggers.append(
            {
                "index": k,
                "timestamp": timestamps[k],
                "value": values[k],
                "zscore": z,
                "baseline_median": point_med,
                "baseline_mad": ann["baseline_mad"],
                "dispersion_method": point_method,
                "deviation": deviation,
                "direction": direction,
                "reason": reason,
            }
        )
        if abs(z) > abs(zscores[max_idx] or 0.0):
            max_idx = k

    seg_z = [zscores[k] for k in range(seg_start, seg_end + 1) if zscores[k] is not None]
    mean_abs_z = sum(abs(z) for z in seg_z) / len(seg_z)
    max_z = zscores[max_idx]
    length = seg_end - seg_start + 1
    span_direction = triggers[0]["direction"]

    explanation = (
        f"区间 {timestamps[seg_start]} ~ {timestamps[seg_end]}（{length} 个点，"
        f"其中 {len(triggers)} 个点触发）：首个触发点为 {timestamps[seg_start]}，"
        f"值 {values[seg_start]:g} 相对基线中位数 {med:g} {span_direction}突变；"
        f"区间最大 |z|={abs(max_z):.2f}（{timestamps[max_idx]}，值 {values[max_idx]:g}），"
        f"平均 |z|={mean_abs_z:.2f}，判定阈值 {threshold:g}。"
    )

    result = {
        "start_index": seg_start,
        "end_index": seg_end,
        "start_time": timestamps[seg_start],
        "end_time": timestamps[seg_end],
        "point_count": length,
        "anomaly_point_count": len(triggers),
        "baseline": {
            "median": med,
            "mad": mad,
            "robust_sd": robust_sd,
            "dispersion_method": dispersion_method,
            "scale": MAD_SCALE,
            "window_size": len(baseline),
            "window_start_time": timestamps[base_lo],
            "window_end_time": timestamps[base_hi - 1],
        },
        "score": {
            "max_abs_zscore": abs(max_z),
            "max_signed_zscore": max_z,
            "max_zscore_index": max_idx,
            "max_zscore_time": timestamps[max_idx],
            "mean_abs_zscore": mean_abs_z,
        },
        "triggers": triggers,
        "explanation": explanation,
    }
    if segment_id is not None:
        result["segment"] = segment_id
    return result
