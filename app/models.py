"""请求 / 响应数据模型。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def parse_timestamp(value: object) -> datetime:
    """把时间戳解析为 UTC aware datetime。

    支持：
    - 数字：按秒或毫秒 epoch 自动识别；
    - 字符串：ISO-8601（如 ``2026-09-20T10:00:00+08:00``，
      ``2026-09-20 10:00:00`` 也可）；无时区按 UTC 处理。
    """
    if isinstance(value, bool):
        raise ValueError("时间戳不能是布尔值")
    if isinstance(value, (int, float)):
        ts = float(value)
        if not math.isfinite(ts):
            raise ValueError("数字时间戳必须有限")
        if abs(ts) > 1e11:  # 大于约 5138 年的秒数，按毫秒处理
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"无法解析时间戳字符串: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise ValueError(f"不支持的时间戳类型: {type(value).__name__}")


class PointIn(BaseModel):
    """单个输入观测点。"""

    timestamp: object = Field(description="时间戳：ISO-8601 字符串，或 epoch 秒/毫秒数字")
    value: float = Field(description="观测数值")

    @field_validator("value")
    @classmethod
    def _finite_value(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value 必须是有限数值（不接受 NaN/Infinity）")
        return v


class DetectRequest(BaseModel):
    """检测请求。"""

    series_id: str | None = Field(
        default=None,
        max_length=128,
        description="可选的业务序列标识，便于检索与去重",
    )
    points: list[PointIn] = Field(min_length=2, description="带时间戳的数值序列")
    window_size: int = Field(default=20, ge=2, le=10_000, description="滑动窗口大小")
    min_window: int = Field(
        default=5, ge=1, le=10_000, description="开始评估所需的最小基线点数"
    )
    threshold: float = Field(
        default=3.0, gt=0, le=100, description="稳健 z 分数阈值，|z| 超过即判为异常"
    )
    merge_gap: int = Field(
        default=2,
        ge=0,
        le=10_000,
        description="合并异常区间时允许桥接的连续正常点数",
    )
    baseline_policy: Literal["rolling", "guarded"] = Field(
        default="rolling",
        description=(
            "基线策略：rolling=所有历史点滑入基线（默认）；"
            "guarded=仅正常点进入基线，连续异常达到 rebase_after 时重建基线"
        ),
    )
    rebase_after: int | None = Field(
        default=None,
        ge=1,
        le=10_000,
        description=(
            "guarded 模式下连续异常达到该点数即重建基线（默认取 min_window，"
            "且不得小于 min_window）；rolling 模式忽略"
        ),
    )
    include_points: bool = Field(
        default=False, description="响应是否包含逐点注解（存储中始终保留）"
    )

    @model_validator(mode="after")
    def _check_windows(self) -> "DetectRequest":
        if self.min_window > self.window_size:
            raise ValueError("min_window 不能大于 window_size")
        if len(self.points) <= self.min_window:
            raise ValueError(
                f"至少需要 {self.min_window + 1} 个点（min_window={self.min_window}），"
                f"当前只有 {len(self.points)} 个"
            )
        if self.baseline_policy == "guarded" and self.rebase_after is not None:
            if self.rebase_after < self.min_window:
                raise ValueError(
                    f"rebase_after 不能小于 min_window"
                    f"（rebase_after={self.rebase_after} < min_window={self.min_window}）"
                )
        return self


class Baseline(BaseModel):
    median: float
    mad: float
    robust_sd: float
    dispersion_method: Literal["mad", "std_fallback", "constant"] = Field(
        description="离散度来源：mad=1.4826·MAD；std_fallback=MAD 为 0 时回退样本标准差；constant=基线恒定"
    )
    scale: float
    window_size: int
    window_start_time: str
    window_end_time: str


class SegmentScore(BaseModel):
    max_abs_zscore: float
    max_signed_zscore: float
    max_zscore_index: int
    max_zscore_time: str
    mean_abs_zscore: float


class Trigger(BaseModel):
    index: int
    timestamp: str
    value: float
    zscore: float
    baseline_median: float
    baseline_mad: float
    dispersion_method: Literal["mad", "std_fallback", "constant"]
    deviation: float
    direction: Literal["上升", "下降"]
    reason: str


class AnomalySegment(BaseModel):
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    point_count: int
    anomaly_point_count: int
    baseline: Baseline
    score: SegmentScore
    triggers: list[Trigger]
    explanation: str


class AnnotatedPoint(BaseModel):
    index: int
    timestamp: str
    value: float
    baseline_median: float
    baseline_mad: float
    baseline_robust_sd: float
    dispersion_method: Literal["mad", "std_fallback", "constant"]
    zscore: float
    degenerate_baseline: bool
    is_anomaly: bool
    baseline_generation: int = Field(
        default=0,
        description="评分该点时使用的基线代次；guarded 模式每次重建递增，rolling 恒为 0",
    )
    baseline_reset: bool = Field(
        default=False,
        description="该点是否触发了基线重建（用其所在异常段末尾 min_window 个点重建）",
    )


class DetectSummary(BaseModel):
    point_count: int
    window_size: int
    min_window: int
    threshold: float
    merge_gap: int
    baseline_policy: Literal["rolling", "guarded"] = "rolling"
    rebase_after: int | None = None
    baseline_rebuild_count: int = Field(default=0, description="guarded 模式基线重建（代次递增）次数")
    evaluated_start_index: int
    evaluated_point_count: int
    anomaly_point_count: int
    anomaly_segment_count: int


class DetectResponse(DetectSummary):
    id: str
    series_id: str | None = None
    created_at: str
    segments: list[AnomalySegment]
    points: list[AnnotatedPoint] | None = None

    model_config = ConfigDict(extra="ignore")


class DetectionListItem(BaseModel):
    id: str
    series_id: str | None
    created_at: str
    point_count: int
    anomaly_segment_count: int
    anomaly_point_count: int
    threshold: float
    window_size: int


class DetectionListResponse(BaseModel):
    items: list[DetectionListItem]
    total: int


class HealthResponse(BaseModel):
    status: str
    database: str
