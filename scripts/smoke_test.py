"""端到端冒烟测试：算法正确性 + FastAPI 接口 + SQLite 持久化。

运行（需安装 requirements）：
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.detection import analyze_series  # noqa: E402
from app.main import app  # noqa: E402


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

    print("✓ API 端到端测试通过：提交 / 查询 / 列表 / 404 / 参数校验 / guarded 策略")


if __name__ == "__main__":
    test_algorithm_spikes_and_shift()
    test_constant_baseline()
    test_no_anomaly()
    test_rolling_defaults_unchanged()
    test_guarded_isolated_spike_excluded()
    test_guarded_rebuild_on_consecutive_anomalies()
    test_api_end_to_end()
    print("\n全部冒烟测试通过 ✅")
