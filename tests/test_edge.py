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


def _export(
    rf, path: Path, version: str, zipmap: bool, baseline: dict | None = None, calibration=None
) -> None:
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
        "calibration": {"method": "none"} if calibration is None else calibration,
    }
    if baseline is not None:
        sidecar["baseline"] = baseline
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")


def _pandas_reference(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Independent pandas implementation of the ten baseline-free formulas (one asset)."""
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
    return out.dropna().reset_index(drop=True)


@pytest.fixture(scope="module")
def fleet_df() -> pd.DataFrame:
    df, _ = generate_fleet(days=2, seed=5)
    return df


@pytest.fixture(scope="module")
def asset_df(fleet_df: pd.DataFrame) -> pd.DataFrame:
    return fleet_df[fleet_df["asset_id"] == "MTR-042"].reset_index(drop=True)


@pytest.fixture(scope="module")
def baseline(fleet_df: pd.DataFrame) -> dict:
    """The baseline exactly as training produces it (per-regime stats + centroids) --
    the format the sidecar ships, never a hand-made flat table."""
    from agents.data_agent.profiling import detect_regime
    from agents.ml_architect.features import baseline_for

    train = sorted(fleet_df["asset_id"].unique())[:18]
    regimes = detect_regime(fleet_df, fit_assets=train)
    return baseline_for(fleet_df, regimes, fit_assets=train)


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
    # a baseline without regime centroids uses its pooled mean / std
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


def test_regime_is_inferred_from_rpm_and_load_not_from_a_label() -> None:
    """Training-format baseline: the window's regime comes from its rpm / load against the
    centroids; a `regime` argument or row key is ignored (the edge never sees the
    simulator's label)."""
    base = {
        s: {"mean": 0.0, "std": 1.0, "R1_mean": 0.0, "R1_std": 1.0, "R2_mean": 2.0, "R2_std": 0.5}
        for s in ef.RAW_COLUMNS
    }
    base["bearing_temp_c"] = {
        "mean": 0.0, "std": 1.0, "R1_mean": 30.0, "R1_std": 1.0, "R2_mean": 50.0, "R2_std": 2.0
    }
    base[ef.CENTROIDS_KEY] = {
        "R1_rpm": 600.0, "R1_load_pct": 10.0, "R2_rpm": 1480.0, "R2_load_pct": 55.0,
        "rpm_scale": 400.0, "load_pct_scale": 30.0,
    }
    buf = np.zeros((4, 8))
    buf[:, 0] = [1.0, 2.0, 3.0, 4.0]
    buf[:, 3] = [50.0, 50.0, 52.0, 52.0]
    buf[:, 4] = 40.0
    buf[:, 6], buf[:, 7] = 1480.0, 55.0  # sits on the R2 centroid
    f = dict(zip(ef.FEATURE_NAMES, ef.compute_features_np(buf, base, regime="R1"), strict=True))
    assert f["vib_rms_z"] == pytest.approx((2.5 - 2.0) / 0.5)
    assert f["bearing_temp_z"] == pytest.approx((51.0 - 50.0) / 2.0)
    assert f["vib_rms_resid"] == pytest.approx(0.5)
    # move the window onto the R1 centroid: the R1 baseline applies, whatever any label says
    buf[:, 6], buf[:, 7] = 600.0, 10.0
    g = dict(zip(ef.FEATURE_NAMES, ef.compute_features_np(buf, base, regime="R2"), strict=True))
    assert g["vib_rms_z"] == pytest.approx(2.5) and g["bearing_temp_z"] == pytest.approx(21.0)
    # a window straddling both regimes gets the mean of the per-row baselines
    buf[:2, 6], buf[:2, 7] = 1480.0, 55.0
    h = dict(zip(ef.FEATURE_NAMES, ef.compute_features_np(buf, base), strict=True))
    assert h["vib_rms_resid"] == pytest.approx(2.5 - 1.0)
    rows = [{c: 1.0 for c in ef.RAW_COLUMNS} | {"rpm": 1480.0, "load_pct": 55.0, "regime": "R1"}] * 4
    assert ef.compute_features_dict(rows, base)["vib_rms_resid"] == pytest.approx(1.0 - 2.0)
    assert ef.regime_ids(base) == ["R1", "R2"] and ef.regime_ids(None) == []


def test_rolling_matches_single_window_and_pandas(asset_df: pd.DataFrame, baseline: dict) -> None:
    window = 12
    ref = _pandas_reference(asset_df, window)
    got = ef.rolling_features(asset_df[ef.RAW_COLUMNS].to_numpy(), window, baseline)
    assert got.shape == (len(asset_df) - window + 1, N_FEATURES)
    np.testing.assert_allclose(got[:, :10], ref.to_numpy(), rtol=1e-9, atol=1e-9)
    # single-window path agrees with the vectorized path, row by row (spot check)
    for i in (0, 17, 100, len(got) - 1):
        one = ef.compute_features_np(asset_df[ef.RAW_COLUMNS].to_numpy()[i : i + window], baseline)
        np.testing.assert_allclose(one, got[i], rtol=1e-9, atol=1e-9)
    assert ef.rolling_features(np.zeros((5, 8)), 12).shape == (0, N_FEATURES)


def test_parity_with_ml_architect_reference(fleet_df: pd.DataFrame, baseline: dict) -> None:
    """edge.features must equal agents.ml_architect.features on the training-format baseline,
    including windows that straddle a regime change on a held-out motor."""
    from agents.ml_architect.features import FEATURE_NAMES, compute_features_np

    assert list(FEATURE_NAMES) == ef.FEATURE_NAMES
    window = 12
    for asset in ("MTR-042", "MTR-024", "MTR-001"):
        g = fleet_df[fleet_df["asset_id"] == asset].reset_index(drop=True)
        raw = g[ef.RAW_COLUMNS].to_numpy()
        regime_change = np.flatnonzero(g["regime"].to_numpy()[1:] != g["regime"].to_numpy()[:-1])
        starts = [0, len(raw) // 2, len(raw) - window] + [
            int(i) - window // 2 for i in regime_change[:3] if i >= window
        ]
        for start in starts:
            buf = raw[start : start + window]
            ours = ef.compute_features_np(buf, baseline)
            theirs = np.asarray(compute_features_np(buf, baseline), dtype=float).reshape(-1)
            np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)
    # the rolling path agrees with the training pipeline on a whole held-out motor
    from agents.ml_architect.features import build_feature_pipeline

    fdf, _ = build_feature_pipeline(fleet_df, None, horizon_h=48.0, baseline=baseline)
    g = fleet_df[fleet_df["asset_id"] == "MTR-024"].reset_index(drop=True)
    edge_rows = ef.rolling_features(g[ef.RAW_COLUMNS].to_numpy(), window, baseline)
    centre_rows = fdf[fdf["asset_id"] == "MTR-024"][ef.FEATURE_NAMES].to_numpy(dtype=float)
    np.testing.assert_allclose(edge_rows, centre_rows, rtol=1e-9, atol=1e-9)


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


def test_edge_applies_exported_calibration(tmp_path: Path) -> None:
    """The sidecar's calibration transform (sigmoid or isotonic) makes the edge reproduce the
    centre's CalibratedClassifierCV probabilities; a legacy "none" string still loads."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    from agents.validation.checks import apply_calibration, calibration_transform

    rf, X = _train_rf(seed=4)
    rng = np.random.default_rng(9)
    X_cal = rng.normal(size=(250, N_FEATURES)).astype(np.float32)
    y_cal = (X_cal[:, 0] + X_cal[:, 10] + 0.3 * rng.normal(size=250) > 0).astype(int)
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(FrozenEstimator(rf), method=method).fit(X_cal, y_cal)
        transform = calibration_transform(cal)
        assert transform["method"] == method
        _export(rf, tmp_path / f"m_{method}.onnx", method, False, calibration=transform)
        store = ModelStore(tmp_path)
        store.refresh()
        model = store.current
        assert model is not None and model.calibration["method"] == method
        expected = cal.predict_proba(X[:30])[:, 1]
        got = np.array(
            [predict_one(model, dict(zip(ef.FEATURE_NAMES, X[i].tolist(), strict=True))) for i in range(30)]
        )
        np.testing.assert_allclose(got, expected, atol=1e-4)
        # and the two reference implementations agree with each other
        raw = rf.predict_proba(X[:30])[:, 1]
        np.testing.assert_allclose(apply_calibration(raw, transform), expected, atol=1e-9)
        (tmp_path / f"m_{method}.onnx").unlink()
        (tmp_path / f"m_{method}.json").unlink()
    _export(rf, tmp_path / "legacy.onnx", "legacy", False, calibration="none")
    store = ModelStore(tmp_path)
    assert store.refresh() is True and store.current.calibration == {"method": "none"}


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
