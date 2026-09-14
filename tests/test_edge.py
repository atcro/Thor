"""Tests for edge/ (Agent A): feature parity, ONNX inference, and the FastAPI app.

No broker or network: EDGE_MQTT_ENABLED=0 and CONTROL_PLANE_URL points at a closed port.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.simulator.generate import generate_fleet
from edge import features as ef
from edge.inference import ModelStore, extract_probability, find_newest_model, predict_one

N_FEATURES = len(ef.FEATURE_NAMES)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _train_rf(seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, N_FEATURES)).astype(np.float32)
    y = (X[:, 0] + X[:, 10] + 0.3 * rng.normal(size=300) > 0).astype(int)
    return RandomForestClassifier(n_estimators=12, max_depth=4, random_state=seed).fit(X, y), X


def _export(rf, path: Path, version: str, zipmap: bool, baseline: dict | None = None) -> None:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    opts = None if zipmap else {id(rf): {"zipmap": False}}
    onx = to_onnx(
        rf,
        initial_types=[("input", FloatTensorType([None, N_FEATURES]))],
        options=opts,
        target_opset=15,
    )
    path.write_bytes(onx.SerializeToString())
    sidecar = {
        "model_name": "thor-bearing-classifier",
        "version": version,
        "features": ef.FEATURE_NAMES,
        "window_rows": 12,
        "calibration": "none",
    }
    if baseline is not None:
        sidecar["baseline"] = baseline
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")


def _pandas_reference(df: pd.DataFrame, window: int, baseline: dict) -> pd.DataFrame:
    """Independent pandas implementation of the documented formulas (one asset)."""
    x = np.arange(window)

    def slope(y: np.ndarray) -> float:
        return float(np.polyfit(x, y, 1)[0])

    r = df.rolling(window)
    out = pd.DataFrame(
        {
            "vib_rms_mean": r["vibration_rms"].mean(),
            "vib_rms_slope": r["vibration_rms"].apply(slope, raw=True),
            "vib_kurt_mean": r["vibration_kurtosis"].mean(),
            "vib_crest_mean": r["vibration_crest"].mean(),
            "bearing_temp_mean": r["bearing_temp_c"].mean(),
            "bearing_temp_slope": r["bearing_temp_c"].apply(slope, raw=True),
            "temp_delta": r["bearing_temp_c"].mean() - r["motor_temp_c"].mean(),
            "current_mean": r["current_a"].mean(),
            "rpm_mean": r["rpm"].mean(),
            "load_mean": r["load_pct"].mean(),
        }
    )
    vb, bb = baseline["vibration_rms"], baseline["bearing_temp_c"]
    out["vib_rms_z"] = (out["vib_rms_mean"] - vb["mean"]) / vb["std"]
    out["bearing_temp_z"] = (out["bearing_temp_mean"] - bb["mean"]) / bb["std"]
    out["vib_rms_resid"] = out["vib_rms_mean"] - vb["mean"]
    return out.dropna().reset_index(drop=True)[ef.FEATURE_NAMES]


@pytest.fixture(scope="module")
def asset_df() -> pd.DataFrame:
    df, _ = generate_fleet(days=2, seed=5)
    return df[df["asset_id"] == "MTR-042"].reset_index(drop=True)


@pytest.fixture(scope="module")
def baseline(asset_df: pd.DataFrame) -> dict:
    head = asset_df.iloc[: len(asset_df) // 5]
    return {
        "vibration_rms": {
            "mean": float(head["vibration_rms"].mean()),
            "std": float(head["vibration_rms"].std()),
        },
        "bearing_temp_c": {
            "mean": float(head["bearing_temp_c"].mean()),
            "std": float(head["bearing_temp_c"].std()),
        },
    }


# --------------------------------------------------------------------------------------
# edge/features.py
# --------------------------------------------------------------------------------------


def test_feature_names_match_interface_contract() -> None:
    assert ef.FEATURE_NAMES == [
        "vib_rms_mean",
        "vib_rms_slope",
        "vib_kurt_mean",
        "vib_crest_mean",
        "bearing_temp_mean",
        "bearing_temp_slope",
        "temp_delta",
        "current_mean",
        "rpm_mean",
        "load_mean",
        "vib_rms_z",
        "bearing_temp_z",
        "vib_rms_resid",
    ]
    assert ef.RAW_COLUMNS == [
        "vibration_rms",
        "vibration_kurtosis",
        "vibration_crest",
        "bearing_temp_c",
        "motor_temp_c",
        "current_a",
        "rpm",
        "load_pct",
    ]


def test_single_window_formulas() -> None:
    w = 5
    buf = np.zeros((w, 8))
    buf[:, 0] = [1, 2, 3, 4, 5]  # vibration_rms: slope 1, mean 3
    buf[:, 3] = [50, 52, 54, 56, 58]  # bearing: slope 2, mean 54
    buf[:, 4] = 40  # motor temp -> temp_delta 14
    buf[:, 1], buf[:, 2], buf[:, 5], buf[:, 6], buf[:, 7] = 3.0, 3.2, 100, 1480, 55
    base = {
        "vibration_rms": {"mean": 2.0, "std": 0.5},
        "bearing_temp_c": {"mean": 50.0, "std": 2.0},
    }
    f = dict(zip(ef.FEATURE_NAMES, ef.compute_features_np(buf, base), strict=True))
    assert f["vib_rms_mean"] == pytest.approx(3.0)
    assert f["vib_rms_slope"] == pytest.approx(1.0)
    assert f["bearing_temp_mean"] == pytest.approx(54.0)
    assert f["bearing_temp_slope"] == pytest.approx(2.0)
    assert f["temp_delta"] == pytest.approx(14.0)
    assert f["vib_rms_z"] == pytest.approx((3.0 - 2.0) / 0.5)
    assert f["bearing_temp_z"] == pytest.approx((54.0 - 50.0) / 2.0)
    assert f["vib_rms_resid"] == pytest.approx(1.0)
    assert (
        f["vib_kurt_mean"],
        f["vib_crest_mean"],
        f["current_mean"],
        f["rpm_mean"],
        f["load_mean"],
    ) == (
        pytest.approx(3.0),
        pytest.approx(3.2),
        pytest.approx(100),
        pytest.approx(1480),
        pytest.approx(55),
    )
    # no baseline -> mean 0 / std 1
    g = dict(zip(ef.FEATURE_NAMES, ef.compute_features_np(buf), strict=True))
    assert g["vib_rms_z"] == pytest.approx(3.0) and g["vib_rms_resid"] == pytest.approx(3.0)
    assert ef.slope(np.array([7.0])) == 0.0
    with pytest.raises(ValueError):
        ef.compute_features_np(np.zeros((3, 7)))


def test_per_regime_baseline_selection() -> None:
    per = {
        "R1": {"vibration_rms": {"mean": 1.0, "std": 0.1}},
        "R3": {"vibration_rms": {"mean": 3.0, "std": 0.3}},
        "bearing_temp_c": {"mean": 60.0, "std": 5.0},  # flat entry shared by every regime
    }
    b1 = ef.select_baseline(per, "R1")
    assert b1["vibration_rms"]["mean"] == 1.0 and b1["bearing_temp_c"]["mean"] == 60.0
    b3 = ef.select_baseline(per, "R3")
    assert b3["vibration_rms"]["mean"] == 3.0
    b2 = ef.select_baseline(per, "R2")  # unknown regime -> only the flat entries
    assert "vibration_rms" not in b2 and b2["bearing_temp_c"]["std"] == 5.0
    assert ef.select_baseline(None, "R1") == {}
    rows = [{c: 1.0 for c in ef.RAW_COLUMNS} | {"regime": "R3"} for _ in range(4)]
    f = ef.compute_features_dict(rows, per)
    assert f["vib_rms_resid"] == pytest.approx(1.0 - 3.0)


def test_rolling_matches_pandas_reference(asset_df: pd.DataFrame, baseline: dict) -> None:
    window = 12
    ref = _pandas_reference(asset_df, window, baseline)
    got = ef.rolling_features(asset_df[ef.RAW_COLUMNS].to_numpy(), window, baseline)
    assert got.shape == (len(asset_df) - window + 1, N_FEATURES)
    np.testing.assert_allclose(got, ref.to_numpy(), rtol=1e-9, atol=1e-9)
    # single-window path agrees with the vectorized path, row by row (spot check)
    for i in (0, 17, len(got) - 1):
        one = ef.compute_features_np(asset_df[ef.RAW_COLUMNS].to_numpy()[i : i + window], baseline)
        np.testing.assert_allclose(one, got[i], rtol=1e-9, atol=1e-9)
    assert ef.rolling_features(np.zeros((5, 8)), 12).shape == (0, N_FEATURES)


def test_parity_with_ml_architect_reference(asset_df: pd.DataFrame, baseline: dict) -> None:
    """Shared test with Agent B: edge.features must equal agents.ml_architect.features."""
    try:
        mod = importlib.import_module("agents.ml_architect.features")
    except ModuleNotFoundError:
        pytest.skip("agents/ml_architect/features.py not built yet")
    names = getattr(mod, "FEATURE_NAMES", None)
    ref_fn = getattr(mod, "compute_features_np", None)
    if names is None or ref_fn is None:
        pytest.skip("agents.ml_architect.features lacks FEATURE_NAMES/compute_features_np")
    assert list(names) == ef.FEATURE_NAMES
    window = 12
    buf = asset_df[ef.RAW_COLUMNS].to_numpy()[:window]
    ours = ef.compute_features_np(buf, baseline)
    try:
        theirs = np.asarray(ref_fn(buf, baseline), dtype=float).reshape(-1)
    except TypeError:
        theirs = np.asarray(ref_fn(buf), dtype=float).reshape(-1)
        ours = ef.compute_features_np(buf)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------------
# edge/inference.py
# --------------------------------------------------------------------------------------


def test_extract_probability_shapes() -> None:
    zip_out = [np.array([0, 1]), [{0: 0.7, 1: 0.3}, {0: 0.1, 1: 0.9}]]
    np.testing.assert_allclose(
        extract_probability(zip_out, ["output_label", "output_probability"]), [0.3, 0.9]
    )
    tensor = [
        np.array([0, 1], dtype=np.int64),
        np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
    ]
    np.testing.assert_allclose(
        extract_probability(tensor, ["label", "probabilities"]), [0.3, 0.9], rtol=1e-6
    )
    np.testing.assert_allclose(extract_probability([np.array([[0.25]])], ["p"]), [0.25])
    with pytest.raises(ValueError):
        extract_probability([np.array([1, 0])], ["label"])


@pytest.mark.parametrize("zipmap", [True, False])
def test_onnx_inference_matches_sklearn(tmp_path: Path, zipmap: bool) -> None:
    rf, X = _train_rf()
    _export(rf, tmp_path / "thor-bearing-classifier_v1.onnx", "v1", zipmap)
    store = ModelStore(tmp_path)
    assert store.refresh() is True
    assert store.refresh() is False  # unchanged
    model = store.current
    assert model is not None and model.version == "v1" and model.window_rows == 12
    expected = rf.predict_proba(X[:20])[:, 1]
    got = [
        predict_one(model, dict(zip(ef.FEATURE_NAMES, X[i].tolist(), strict=True)))
        for i in range(20)
    ]
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in got)
    with pytest.raises(KeyError):
        predict_one(model, {"vib_rms_mean": 1.0})


def test_newest_model_wins_and_broken_sidecar_is_skipped(tmp_path: Path) -> None:
    assert find_newest_model(tmp_path / "nope") is None
    rf, _ = _train_rf()
    _export(rf, tmp_path / "m_v1.onnx", "v1", True)
    (tmp_path / "orphan.onnx").write_bytes(b"\x00")  # no sidecar -> ignored
    store = ModelStore(tmp_path)
    store.refresh()
    assert store.current is not None and store.current.version == "v1"
    _export(rf, tmp_path / "m_v2.onnx", "v2", False)
    os.utime(tmp_path / "m_v2.onnx", (2_000_000_000, 2_000_000_000))
    assert store.refresh() is True and store.current.version == "v2"
    (tmp_path / "m_v3.onnx").write_bytes(b"garbage")
    (tmp_path / "m_v3.json").write_text(
        json.dumps({"version": "v3", "features": ["not_a_feature"]})
    )
    os.utime(tmp_path / "m_v3.onnx", (2_100_000_000, 2_100_000_000))
    assert store.refresh() is False
    assert store.current.version == "v2" and store.last_error and "m_v3" in store.last_error


# --------------------------------------------------------------------------------------
# edge/main.py
# --------------------------------------------------------------------------------------


@pytest.fixture
def edge_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, object, np.ndarray]]:
    rf, X = _train_rf(seed=3)
    _export(rf, tmp_path / "thor-bearing-classifier_v7.onnx", "v7", True)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("EDGE_MQTT_ENABLED", "0")
    monkeypatch.setenv("CONTROL_PLANE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EDGE_MODEL_POLL_S", "0.2")
    import edge.main as edge_main

    edge_main = importlib.reload(edge_main)
    with TestClient(edge_main.app) as c:
        yield c, rf, X


def test_edge_health_and_predict(edge_client: tuple[TestClient, object, np.ndarray]) -> None:
    c, rf, X = edge_client
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == "v7" and h["n_assets_seen"] == 0
    assert h["uptime_s"] >= 0 and h["mqtt_connected"] is False
    feats = dict(zip(ef.FEATURE_NAMES, X[0].tolist(), strict=True))
    r = c.post("/predict", json={"features": feats})
    assert r.status_code == 200
    p = r.json()["failure_probability"]
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(rf.predict_proba(X[:1])[0, 1], abs=1e-5)
    r = c.request("GET", "/predict", json={"features": feats})  # spec says GET with a body
    assert r.status_code == 200 and r.json()["failure_probability"] == pytest.approx(p)
    r = c.post("/predict", json={"features": {"vib_rms_mean": 1.0}})
    assert r.status_code == 422
    assert c.get("/features").json()["available"] == ef.FEATURE_NAMES


def test_edge_ingest_produces_prediction_after_full_window(
    edge_client: tuple[TestClient, object, np.ndarray],
) -> None:
    c, _, _ = edge_client
    df, _ = generate_fleet(days=1, seed=2)
    rows = df[df["asset_id"] == "MTR-042"].head(14).copy()
    rows["ts"] = rows["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = json.loads(rows.to_json(orient="records"))
    r = c.post("/ingest", json=payload[:11])
    assert r.status_code == 200 and r.json() == {"accepted": 11, "predictions": []}
    r = c.post("/ingest", json=payload[11:])
    body = r.json()
    assert body["accepted"] == 3 and len(body["predictions"]) == 3
    pred = body["predictions"][-1]
    assert pred["asset_id"] == "MTR-042" and pred["model_version"] == "v7"
    assert 0.0 <= pred["failure_probability"] <= 1.0
    assert pred["ts"] == payload[-1]["ts"]
    h = c.get("/health").json()
    assert h["n_assets_seen"] == 1 and h["n_predictions"] == 3 and h["n_rows"] == 14
    assert c.post("/ingest", json={"asset_id": "MTR-001"}).json()["predictions"] == []


def test_edge_without_model_is_healthy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("EDGE_MQTT_ENABLED", "0")
    monkeypatch.setenv("CONTROL_PLANE_URL", "http://127.0.0.1:1")
    import edge.main as edge_main

    edge_main = importlib.reload(edge_main)
    with TestClient(edge_main.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["model_version"] is None
        assert c.post("/predict", json={"features": {}}).status_code == 503
        r = c.post(
            "/ingest", json=[{"asset_id": "MTR-001", **{k: 1.0 for k in ef.RAW_COLUMNS}}] * 20
        )
        assert r.json()["predictions"] == []


def test_edge_never_imports_control_plane() -> None:
    src = "".join(p.read_text(encoding="utf-8") for p in Path("edge").glob("*.py"))
    assert "from apps" not in src and "import apps" not in src
    assert "from agents" not in src and "import agents" not in src
