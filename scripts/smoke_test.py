"""端到端冒烟测试：算法正确性 + FastAPI 接口 + SQLite 持久化。

运行（需安装 requirements）：
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.detection import analyze_series  # noqa: E402
from app.main import app  # noqa: E402


def _gap_series(seg0: int, seg1: int, gap_minutes: int = 30, step_minutes: int = 1):
    """构造两段等间隔采样、中间严格隔 gap_minutes 的序列时间戳。"""
    start = datetime(2026, 9, 20, 8, 0, 0)
    minutes = list(range(0, seg0)) + list(
        range(seg0 - 1 + gap_minutes, seg0 - 1 + gap_minutes + seg1)
    )
    return [(start + timedelta(minutes=m)).isoformat() for m in minutes]



def test_algorithm_spikes_and_shift() -> None:
    n = 40
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(n)]
    # 单点突增
    values[25] = 20.0
    # 水平偏移：连续 5 个点明显抬升
    values[30:35] = [16.0, 16.1, 15.9, 16.2, 15.8]
    ts = [f"2026-09-20T08:{i:02d}:00Z" for i in range(n)]

    result = analyze_series(ts, values, window_size=20, min_window=5, threshold=3.0, merge_gap=2)

    assert result["anomaly_segment_count"] == 2, result["segments"]
    seg1, seg2 = result["segments"]

    # 单点突增区间
    assert seg1["start_index"] == seg1["end_index"] == 25
    assert seg1["score"]["max_abs_zscore"] >= 3.0
    assert seg1["triggers"][0]["direction"] == "上升"

    # 水平偏移区间，5 个点应被合并为一个区间
    assert seg2["start_index"] == 30 and seg2["end_index"] == 34
    assert seg2["point_count"] == 5
    assert "基线" in seg2["explanation"]

    # 逐点注解
    annotated = result["points"]
    assert len(annotated) == n - 5
    assert all("zscore" in p and "baseline_median" in p for p in annotated)
    print("✓ 算法测试通过：单点突变 + 水平偏移识别与合并正确")


def test_constant_baseline() -> None:
    values = [5.0] * 15 + [8.0, 5.0]
    ts = [f"2026-09-20T09:{i:02d}:00Z" for i in range(len(values))]
    result = analyze_series(ts, values, window_size=10, min_window=5, threshold=3.0, merge_gap=1)
    assert result["anomaly_segment_count"] == 1
    trigger = result["segments"][0]["triggers"][0]
    assert trigger["zscore"] == 999.0
    assert "基线恒定" in trigger["reason"]
    print("✓ 恒定基线（MAD=0）退化为强突变哨兵处理正确")


def test_no_anomaly() -> None:
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.07, 0.93]
    ts = [f"2026-09-20T10:{i:02d}:00Z" for i in range(len(values))]
    result = analyze_series(ts, values, window_size=5, min_window=3, threshold=5.0, merge_gap=1)
    assert result["anomaly_segment_count"] == 0
    assert result["anomaly_point_count"] == 0
    print("✓ 无异常数据正确返回空区间")


def test_rolling_defaults_unchanged() -> None:
    """rolling（默认）结果保持原有行为，新增字段取默认值。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    values[10] = 20.0
    ts = [f"2026-09-20T11:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(ts, values, window_size=20, min_window=5, threshold=3.0)
    assert result["baseline_policy"] == "rolling"
    assert result["rebase_after"] is None
    assert result["baseline_rebuild_count"] == 0
    assert all(
        p["baseline_generation"] == 0 and p["baseline_reset"] is False
        for p in result["points"]
    )
    # 断档感知默认关闭：不产生新的有效状态，旧响应完全兼容
    assert result["gap_reset_factor"] is None
    assert result["estimated_interval_seconds"] is None
    assert result["sampling_segment_count"] == 1
    assert result["gap_boundaries"] == []
    assert all(p["segment"] == 0 for p in result["points"])
    assert all("segment" not in s for s in result["segments"])
    print("✓ rolling 默认模式：结果与新增默认字段正确")


def test_guarded_isolated_spike_excluded() -> None:
    """guarded 模式：孤立尖刺不得进入后续基线，不触发重建。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
    values[10] = 20.0  # 孤立尖刺
    ts = [f"2026-09-20T12:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0,
        baseline_policy="guarded",
    )
    assert result["baseline_policy"] == "guarded"
    assert result["rebase_after"] == 5  # 默认取 min_window
    assert result["baseline_rebuild_count"] == 0
    assert [(s["start_index"], s["end_index"]) for s in result["segments"]] == [(10, 10)]

    by_index = {p["index"]: p for p in result["points"]}
    # 尖刺被排除：其后正常点的基线中位数仍贴近 10，离散度未被 20 放大
    assert abs(by_index[11]["baseline_median"] - 10.0) < 0.2
    assert by_index[11]["is_anomaly"] is False
    assert by_index[10]["baseline_reset"] is False
    assert all(p["baseline_generation"] == 0 for p in result["points"])

    # 区间基线窗口不得包含被排除的尖刺：成员来自 0..9（5 个种子 + 5..9 正常点），
    # 时间上界停在尖刺前一个点
    seg = result["segments"][0]
    assert seg["baseline"]["window_size"] == 10
    assert seg["baseline"]["window_end_time"] == "2026-09-20T12:09:00Z"
    print("✓ guarded 孤立尖刺：不污染后续中位数/离散度，且不触发重建")


def test_guarded_rebuild_on_consecutive_anomalies() -> None:
    """guarded 模式：连续异常达到 rebase_after 时重建基线、递增代次但不拆区间。"""
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(40)]
    values[20:30] = [16.0] * 10  # 持续水平偏移
    ts = [f"2026-09-20T13:{i:02d}:00Z" for i in range(len(values))]

    result = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0, merge_gap=2,
        baseline_policy="guarded",
    )
    by_index = {p["index"]: p for p in result["points"]}

    # 第 5 个连续异常点 i=24 触发重建；该点仍保持异常、区间不拆分
    assert by_index[24]["baseline_reset"] is True
    assert by_index[24]["is_anomaly"] is True
    assert by_index[24]["baseline_generation"] == 0  # 评分用旧代次
    assert result["baseline_rebuild_count"] >= 1
    assert [(s["start_index"], s["end_index"]) for s in result["segments"]][0] == (20, 24)

    # 重建后 i=25..29 对新基线（全 16）恢复正常，代次为 1
    for i in range(25, 30):
        assert by_index[i]["is_anomaly"] is False
        assert by_index[i]["baseline_generation"] == 1
        assert by_index[i]["baseline_reset"] is False

    # 自定义 rebase_after（>= min_window）：首次重建延后到 i=26
    custom = analyze_series(
        ts, values, window_size=20, min_window=5, threshold=3.0,
        baseline_policy="guarded", rebase_after=7,
    )
    assert custom["rebase_after"] == 7
    resets = [p["index"] for p in custom["points"] if p["baseline_reset"]]
    assert resets[0] == 26
    print("✓ guarded 连续异常：达 rebase_after 重建基线并递增代次，触发点仍异常、区间不拆分")


def test_gap_detection_boundaries_and_segment_ids() -> None:
    """断档感知：中位数间隔估算、断档边界与逐点分段编号正确。"""
    n0, n1 = 10, 10
    ts = _gap_series(n0, n1, gap_minutes=30, step_minutes=1)
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(n0 + n1)]

    result = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0,
        merge_gap=1, gap_reset_factor=3.0,
    )
    # 常规间隔 60 秒；断档 30 分钟 = 1800 秒，30 倍，超出 3 倍阈值
    assert result["gap_reset_factor"] == 3.0
    assert result["estimated_interval_seconds"] == 60.0
    assert result["sampling_segment_count"] == 2
    assert len(result["gap_boundaries"]) == 1
    boundary = result["gap_boundaries"][0]
    assert boundary["before_index"] == 9 and boundary["after_index"] == 10
    assert boundary["gap_seconds"] == 1800.0
    assert boundary["gap_ratio"] == 30.0
    assert boundary["left_segment"] == 0 and boundary["right_segment"] == 1

    # 每段独立预热：各跳过前 3 个点，共评估 (10-3)*2 = 14 个点
    assert result["evaluated_point_count"] == 14
    by_index = {p["index"]: p for p in result["points"]}
    assert all(by_index[i]["segment"] == 0 for i in range(3, 10))
    assert all(by_index[i]["segment"] == 1 for i in range(13, 20))
    # 预热点（段内前 3 个，含 i=10,11,12）不参与评估、不输出注解
    assert 10 not in by_index and 11 not in by_index and 12 not in by_index
    print("✓ 断档感知：估算间隔 / 断档边界 / 逐点分段编号 / 分段预热正确")


def test_gap_no_merge_across_boundary() -> None:
    """断档两侧的异常即使索引距离满足 merge_gap 也不得跨断档合并。"""
    n0, n1 = 10, 8
    ts = _gap_series(n0, n1, gap_minutes=60, step_minutes=1)
    values = [10.0, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.0, 10.1, 10.0,
              10.0, 9.9, 10.1, 10.0, 9.9, 10.0, 10.1, 10.0]
    values[9] = 20.0   # 第一段最后一个点
    values[13] = 20.0  # 第二段首个被评估点（段内索引 3），与 i=9 差 4

    # 关闭断档感知：merge_gap=3 允许桥接，两点被并入同一区间
    off = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0, merge_gap=3,
    )
    assert [(s["start_index"], s["end_index"]) for s in off["segments"]] == [(9, 13)]

    # 开启断档感知：区间在断档处强制切开
    on = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0, merge_gap=3,
        gap_reset_factor=3.0,
    )
    assert [(s["start_index"], s["end_index"]) for s in on["segments"]] == [(9, 9), (13, 13)]
    assert [s["segment"] for s in on["segments"]] == [0, 1]
    assert on["anomaly_point_count"] == 2
    print("✓ 断档感知：异常区间不跨越断档合并（即使满足 merge_gap 桥接距离）")


def test_gap_independent_warmup_rolling() -> None:
    """rolling 下分段预热：第二段前 min_window 个点不评分，基线不跨断档取点。"""
    n0, n1 = 8, 8
    ts = _gap_series(n0, n1, gap_minutes=120, step_minutes=1)
    # 第一段水位 10，第二段水位 16——若共用基线，第二段会全部判为异常
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(8)]
    values += [16.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(8)]

    result = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0,
        merge_gap=1, gap_reset_factor=3.0,
    )
    by_index = {p["index"]: p for p in result["points"]}
    assert result["anomaly_segment_count"] == 0
    # 第二段首个被评估点 i=11 的基线只来自第二段（16 水位），不回看到断档前
    assert abs(by_index[11]["baseline_median"] - 16.0) < 0.2
    assert by_index[11]["segment"] == 1
    print("✓ 断档感知：分段独立预热与基线（水位迁移在断档后不再误报）")


def test_gap_guarded_state_resets() -> None:
    """guarded 下连续异常计数与重建代次在断档后从头开始。"""
    n0, n1 = 10, 10
    ts = _gap_series(n0, n1, gap_minutes=30, step_minutes=1)
    values = [10.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(20)]
    for i in range(5, 10):
        values[i] = 16.0
    for i in range(13, 18):
        values[i] = 22.0

    result = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0,
        baseline_policy="guarded", rebase_after=3, gap_reset_factor=3.0,
    )
    by_index = {p["index"]: p for p in result["points"]}

    # 第一段：i=7 触发一次重建（段内第 3 个连续异常），其后评分为代次 1
    assert by_index[7]["baseline_reset"] is True
    assert by_index[7]["baseline_generation"] == 0  # 重建评分后生效，本点仍用旧代次
    assert by_index[8]["baseline_generation"] == 1
    # 第二段：代次从头计，i=15 再次触发重建，重建计数为两段之和
    assert by_index[13]["baseline_generation"] == 0
    assert by_index[15]["baseline_reset"] is True
    assert by_index[16]["baseline_generation"] == 1
    assert result["baseline_rebuild_count"] == 2
    assert by_index[7]["segment"] == 0 and by_index[15]["segment"] == 1

    # 对照：不启用断档感知时，第二段沿用代次 1 的基线状态继续累计
    off = analyze_series(
        ts, values, window_size=10, min_window=3, threshold=3.0,
        baseline_policy="guarded", rebase_after=3,
    )
    off_by = {p["index"]: p for p in off["points"]}
    assert off_by[13]["baseline_generation"] >= 1
    print("✓ 断档感知：guarded 基线 / 连续计数 / 重建代次在各分段从头开始")


def test_gap_short_segment_not_evaluated() -> None:
    """点数不足的分段不评估；存在多个断档时逐段处理。"""
    start = datetime(2026, 9, 20, 8, 0, 0)
    minutes = list(range(0, 8)) + list(range(60, 62)) + list(range(120, 128))
    ts = [(start + timedelta(minutes=m)).isoformat() for m in minutes]
    values = [10.0] * 8 + [50.0, 50.0] + [10.0] * 8

    result = analyze_series(
        ts, values, window_size=5, min_window=3, threshold=3.0, gap_reset_factor=3.0
    )
    assert result["sampling_segment_count"] == 3
    assert len(result["gap_boundaries"]) == 2
    # 中间段只有 2 个点（<= min_window），整段预热，50.0 的点不评分、不产生异常
    assert result["evaluated_point_count"] == (8 - 3) * 2
    assert result["anomaly_segment_count"] == 0

    # 无断档（所有间隔都不超过倍数）时退化为单段，结果与不启用结构一致
    flat_ts = [(start + timedelta(minutes=i)).isoformat() for i in range(10)]
    nogap = analyze_series(
        flat_ts, [10.0] * 10, window_size=5, min_window=3, gap_reset_factor=5.0
    )
    assert nogap["sampling_segment_count"] == 1 and nogap["gap_boundaries"] == []
    assert nogap["estimated_interval_seconds"] == 60.0
    print("✓ 断档感知：小分段不评估 / 多断档 / 无断档退化为单段")


def test_gap_validation_under_three_points() -> None:
    """启用 gap_reset_factor 少于 3 个点：请求模型返回明确校验错误。"""
    from app.models import DetectRequest

    point = {"timestamp": "2026-09-20T10:00:00Z", "value": 1.0}
    try:
        DetectRequest(
            points=[point, point.copy()],
            window_size=5,
            min_window=1,
            gap_reset_factor=3.0,
        )
    except Exception as exc:
        message = str(exc)
        assert "gap_reset_factor" in message and "3" in message, message
    else:
        raise AssertionError("少于 3 个点启用断档感知应校验失败")

    # 3 个点可以启用（配合较小的 min_window）；factor 必须 > 1
    DetectRequest(
        points=[
            point,
            dict(point, timestamp="2026-09-20T10:01:00Z"),
            dict(point, timestamp="2026-09-20T10:02:00Z"),
        ],
        window_size=5,
        min_window=2,
        gap_reset_factor=1.5,
    )
    print("✓ 断档感知：<3 点明确校验错误，3 点可启用，因子必须 >1")


def test_api_end_to_end(tmp_db: str = "/tmp/tsad_smoke.db") -> None:
    import os

    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DB_PATH"] = tmp_db

    # 重新初始化数据库（app 模块已导入，直接调用 init_db）
    from app import database

    database.DB_PATH = tmp_db
    database.init_db(tmp_db)

    with TestClient(app) as client:
        _exercise_api(client)


def _exercise_api(client: "TestClient") -> None:
    # 健康检查
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    # 提交检测
    payload = {
        "series_id": "test-series",
        "window_size": 10,
        "min_window": 3,
        "threshold": 3.0,
        "merge_gap": 1,
        "include_points": True,
        "points": [
            {"timestamp": f"2026-09-20T08:{i:02d}:00", "value": 1.0 + (i % 2) * 0.05}
            for i in range(15)
        ]
        + [{"timestamp": "2026-09-20T08:15:00", "value": 9.0}],
    }
    r = client.post("/api/v1/detections", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    detection_id = body["id"]
    assert body["anomaly_segment_count"] == 1
    assert body["segments"][0]["triggers"][0]["timestamp"] == "2026-09-20T08:15:00+00:00"
    assert len(body["points"]) > 0

    # 按 ID 查询
    r = client.get(f"/api/v1/detections/{detection_id}")
    assert r.status_code == 200
    assert r.json()["id"] == detection_id
    assert r.json()["segments"][0]["baseline"]["window_size"] == 10

    # include_points=false 不返回逐点
    r = client.get(f"/api/v1/detections/{detection_id}", params={"include_points": False})
    assert r.json()["points"] is None

    # 列表 + series_id 过滤
    r = client.get("/api/v1/detections", params={"series_id": "test-series"})
    assert r.status_code == 200 and r.json()["total"] == 1

    # 404
    r = client.get("/api/v1/detections/nonexistent")
    assert r.status_code == 404

    # 参数校验：min_window > window_size
    bad = dict(payload)
    bad["min_window"], bad["window_size"] = 20, 10
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # guarded 模式：回显策略与重建统计，逐点注解带代次/重建标记
    guarded_payload = dict(payload)
    guarded_payload["series_id"] = "test-series-guarded"
    guarded_payload["baseline_policy"] = "guarded"
    guarded_payload["rebase_after"] = 3
    r = client.post("/api/v1/detections", json=guarded_payload)
    assert r.status_code == 201, r.text
    gbody = r.json()
    assert gbody["baseline_policy"] == "guarded"
    assert gbody["rebase_after"] == 3
    assert "baseline_rebuild_count" in gbody
    assert all(
        "baseline_generation" in p and "baseline_reset" in p
        for p in gbody["points"]
    )
    guarded_id = gbody["id"]

    # 历史结果取回时策略字段持久化保留
    r = client.get(f"/api/v1/detections/{guarded_id}")
    assert r.status_code == 200
    assert r.json()["baseline_policy"] == "guarded"
    assert r.json()["baseline_rebuild_count"] == gbody["baseline_rebuild_count"]

    # 参数校验：rebase_after < min_window
    bad = dict(payload)
    bad["baseline_policy"] = "guarded"
    bad["min_window"], bad["rebase_after"] = 3, 2
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 参数校验：非法 baseline_policy
    bad = dict(payload)
    bad["baseline_policy"] = "strict"
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    # 采样断档感知：两段每分钟采样中间夹 30 分钟断档（09:00~09:09 -> 09:39~09:48）
    gap_minutes = list(range(0, 10)) + list(range(39, 49))
    gap_points = [
        {
            "timestamp": (datetime(2026, 9, 20, 9, 0) + timedelta(minutes=m)).isoformat(),
            "value": 10.0 + (i % 2) * 0.05,
        }
        for i, m in enumerate(gap_minutes)
    ]
    gap_payload = {
        "series_id": "test-series-gap",
        "window_size": 10,
        "min_window": 3,
        "threshold": 3.0,
        "merge_gap": 1,
        "gap_reset_factor": 3.0,
        "include_points": True,
        "points": gap_points,
    }
    r = client.post("/api/v1/detections", json=gap_payload)
    assert r.status_code == 201, r.text
    gp = r.json()
    assert gp["gap_reset_factor"] == 3.0
    assert gp["estimated_interval_seconds"] == 60.0
    assert gp["sampling_segment_count"] == 2
    assert len(gp["gap_boundaries"]) == 1
    bound = gp["gap_boundaries"][0]
    assert bound["before_index"] == 9 and bound["after_index"] == 10
    assert bound["gap_seconds"] == 1800.0
    assert {p["segment"] for p in gp["points"]} == {0, 1}
    gap_id = gp["id"]

    # 历史结果取回：断档字段持久化保留
    r = client.get(f"/api/v1/detections/{gap_id}")
    assert r.status_code == 200
    fetched = r.json()
    assert fetched["gap_reset_factor"] == 3.0
    assert fetched["sampling_segment_count"] == 2
    assert len(fetched["gap_boundaries"]) == 1

    # 默认不启用断档感知时回显兼容字段
    r = client.post("/api/v1/detections", json=payload)
    assert r.status_code == 201, r.text
    default_body = r.json()
    assert default_body["gap_reset_factor"] is None
    assert default_body["estimated_interval_seconds"] is None
    assert default_body["sampling_segment_count"] == 1
    assert default_body["gap_boundaries"] == []

    # 参数校验：启用 gap_reset_factor 但只有 2 个点
    bad = dict(gap_payload)
    bad["points"] = gap_points[:2]
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422
    assert "gap_reset_factor" in r.text

    # 参数校验：gap_reset_factor 必须 > 1
    bad = dict(gap_payload)
    bad["gap_reset_factor"] = 1.0
    r = client.post("/api/v1/detections", json=bad)
    assert r.status_code == 422

    print("✓ API 端到端测试通过：提交 / 查询 / 列表 / 404 / 参数校验 / guarded 策略 / 断档感知")


if __name__ == "__main__":
    test_algorithm_spikes_and_shift()
    test_constant_baseline()
    test_no_anomaly()
    test_rolling_defaults_unchanged()
    test_guarded_isolated_spike_excluded()
    test_guarded_rebuild_on_consecutive_anomalies()
    test_gap_detection_boundaries_and_segment_ids()
    test_gap_no_merge_across_boundary()
    test_gap_independent_warmup_rolling()
    test_gap_guarded_state_resets()
    test_gap_short_segment_not_evaluated()
    test_gap_validation_under_three_points()
    test_api_end_to_end()
    print("\n全部冒烟测试通过 ✅")
