"""Thor edge inference service -- standalone FastAPI app on :8001.

Independent of the control plane: only fastapi / uvicorn / onnxruntime / paho-mqtt / numpy /
httpx / pydantic. Never imports from `apps/` or `agents/`.

* Watches MODELS_DIR (default ./models) for `*.onnx` + `*.json` sidecars; loads the newest.
* Subscribes to `telemetry/#`, keeps a per-asset rolling buffer of raw rows, computes the
  canonical feature set with `edge.features`, runs onnxruntime and publishes
  `predictions/{asset_id}` = {asset_id, ts, failure_probability, model_version}; the same
  payload is POSTed best-effort to `{CONTROL_PLANE_URL}/predictions`.
* GET  /health   -> {status, model_version, n_assets_seen, uptime_s, ...}
* GET|POST /predict  body {"features": {name: value}} -> {"failure_probability": p}
* POST /ingest  body row | [rows] -> predictions produced (dev/test path without a broker)

Environment: MQTT_BROKER_URL (mqtt://localhost:1883), MODELS_DIR, CONTROL_PLANE_URL
(http://localhost:8000), EDGE_MQTT_ENABLED (1), EDGE_MODEL_POLL_S (10), EDGE_BUFFER_ROWS (64).
Without a model the service is healthy, logs "no model" and produces no predictions.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException
from paho.mqtt import client as mqtt
from pydantic import BaseModel, Field

from edge.features import FEATURE_NAMES, RAW_COLUMNS, compute_features_dict
from edge.inference import ModelStore, predict_one

log = logging.getLogger("thor.edge")
logging.basicConfig(level=os.getenv("EDGE_LOG_LEVEL", "INFO"))

TELEMETRY_TOPIC = "telemetry/#"
PREDICTION_TOPIC_FMT = "predictions/{asset_id}"


def parse_broker_url(url: str) -> tuple[str, int]:
    """'mqtt://host:1883' -> ('host', 1883)."""
    tail = url.replace("mqtt://", "").replace("tcp://", "").strip("/")
    host, _, port = tail.partition(":")
    return host or "localhost", int(port) if port else 1883


class EdgeConfig(BaseModel):
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_enabled: bool = True
    models_dir: Path = Path("./models")
    control_plane_url: str = "http://localhost:8000"
    model_poll_s: float = 10.0
    buffer_rows: int = 64
    retry_delay_s: float = 5.0

    @classmethod
    def from_env(cls) -> EdgeConfig:
        host, port = parse_broker_url(os.getenv("MQTT_BROKER_URL", "mqtt://localhost:1883"))
        return cls(
            mqtt_host=host,
            mqtt_port=port,
            mqtt_enabled=os.getenv("EDGE_MQTT_ENABLED", "1") not in ("0", "false", "False", ""),
            models_dir=Path(os.getenv("MODELS_DIR", "./models")),
            control_plane_url=os.getenv("CONTROL_PLANE_URL", "http://localhost:8000"),
            model_poll_s=float(os.getenv("EDGE_MODEL_POLL_S", "10")),
            buffer_rows=int(os.getenv("EDGE_BUFFER_ROWS", "64")),
            retry_delay_s=float(os.getenv("EDGE_MQTT_RETRY_S", "5")),
        )


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(description="{feature_name: value} for the sidecar features")


class EdgeState:
    """Runtime state: model store, per-asset raw buffers, MQTT + poster threads."""

    def __init__(self, cfg: EdgeConfig) -> None:
        self.cfg = cfg
        self.store = ModelStore(cfg.models_dir)
        self.buffers: dict[str, deque[dict[str, Any]]] = {}
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.n_predictions = 0
        self.n_rows = 0
        self.last_prediction: dict[str, dict[str, Any]] = {}
        self.mqtt_connected = False
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._client: mqtt.Client | None = None
        self._post_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10_000)
        self._post_failures = 0

    # -- core path ----------------------------------------------------------------------------

    def process_row(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Buffer one raw telemetry row and score it when a full window is available.

        Input: telemetry dict with at least asset_id, ts and the 8 RAW_COLUMNS. Output: the
        prediction payload {asset_id, ts, failure_probability, model_version} or None (no model
        yet, window not full, or malformed row).
        """
        try:
            asset_id = str(payload["asset_id"])
            row = {c: float(payload[c]) for c in RAW_COLUMNS}
        except (KeyError, TypeError, ValueError) as exc:
            self.last_error = f"bad row: {exc}"
            log.debug("dropping malformed telemetry row: %s", exc)
            return None
        row["ts"] = payload.get("ts")
        row["regime"] = payload.get("regime")
        with self.lock:
            buf = self.buffers.get(asset_id)
            if buf is None:
                buf = deque(maxlen=self.cfg.buffer_rows)
                self.buffers[asset_id] = buf
            buf.append(row)
            self.n_rows += 1
            model = self.store.current
            if model is None or len(buf) < model.window_rows:
                return None
            window = list(buf)[-model.window_rows :]
        try:
            feats = compute_features_dict(window, model.baseline, regime=window[-1].get("regime"))
            p = predict_one(model, feats)
        except Exception as exc:  # noqa: BLE001 - never let one bad window kill the stream
            self.last_error = f"inference failed: {exc}"
            log.warning("inference failed for %s: %s", asset_id, exc)
            return None
        ts = row["ts"] or datetime.now(UTC).isoformat()
        pred = {
            "asset_id": asset_id,
            "ts": ts if isinstance(ts, str) else str(ts),
            "failure_probability": round(float(p), 6),
            "model_version": model.version,
            "source": "edge",
        }
        with self.lock:
            self.n_predictions += 1
            self.last_prediction[asset_id] = pred
        return pred

    def emit(self, pred: dict[str, Any]) -> None:
        """Publish a prediction to MQTT (if connected) and queue the control-plane POST."""
        client = self._client
        if client is not None and self.mqtt_connected:
            try:
                client.publish(
                    PREDICTION_TOPIC_FMT.format(asset_id=pred["asset_id"]), json.dumps(pred), qos=0
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("MQTT publish failed: %s", exc)
        try:
            self._post_queue.put_nowait(pred)
        except queue.Full:
            log.debug("prediction POST queue full; dropping")

    # -- threads --------------------------------------------------------------------------------

    def _on_connect(
        self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        if getattr(reason_code, "is_failure", False):
            self.mqtt_connected = False
            log.warning("MQTT connect refused: %s", reason_code)
            return
        self.mqtt_connected = True
        client.subscribe(TELEMETRY_TOPIC, qos=0)
        log.info(
            "edge subscribed to %s on %s:%s",
            TELEMETRY_TOPIC,
            self.cfg.mqtt_host,
            self.cfg.mqtt_port,
        )

    def _on_disconnect(
        self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        self.mqtt_connected = False

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        pred = self.process_row(payload)
        if pred is not None:
            self.emit(pred)

    def _mqtt_loop(self) -> None:
        while not self._stop.is_set():
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=f"thor-edge-{os.getpid()}"
            )
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            client.reconnect_delay_set(
                min_delay=int(self.cfg.retry_delay_s), max_delay=int(self.cfg.retry_delay_s)
            )
            self._client = client
            try:
                client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=30)
                client.loop_forever(retry_first_connection=False)
            except Exception as exc:  # noqa: BLE001 - broker down is expected locally
                self.mqtt_connected = False
                log.warning(
                    "MQTT broker %s:%s unavailable (%s); retrying in %.0fs",
                    self.cfg.mqtt_host,
                    self.cfg.mqtt_port,
                    exc,
                    self.cfg.retry_delay_s,
                )
            finally:
                self._client = None
            self._stop.wait(self.cfg.retry_delay_s)

    def _poster_loop(self) -> None:
        url = self.cfg.control_plane_url.rstrip("/") + "/predictions"
        with httpx.Client(timeout=2.0) as http:
            while not self._stop.is_set():
                try:
                    pred = self._post_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    http.post(url, json={k: v for k, v in pred.items()})
                    self._post_failures = 0
                except (httpx.HTTPError, OSError) as exc:
                    self._post_failures += 1
                    if self._post_failures in (1, 10, 100) or self._post_failures % 1000 == 0:
                        log.warning("POST %s failed (%d so far): %s", url, self._post_failures, exc)

    def _model_watch_loop(self) -> None:
        while not self._stop.wait(self.cfg.model_poll_s):
            try:
                self.store.refresh()
            except Exception as exc:  # noqa: BLE001
                log.warning("model refresh failed: %s", exc)

    def start(self) -> None:
        self.store.refresh()
        if self.store.current is None:
            log.info("no model in %s -- edge is healthy but idle", self.cfg.models_dir)
        targets = [
            ("thor-edge-models", self._model_watch_loop),
            ("thor-edge-post", self._poster_loop),
        ]
        if self.cfg.mqtt_enabled:
            targets.append(("thor-edge-mqtt", self._mqtt_loop))
        for name, target in targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        client = self._client
        if client is not None:
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001
                pass
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()

    def health(self) -> dict[str, Any]:
        model = self.store.current
        return {
            "status": "ok",
            "model_version": model.version if model else None,
            "model_name": model.model_name if model else None,
            "n_assets_seen": len(self.buffers),
            "uptime_s": round(time.time() - self.started_at, 1),
            "n_rows": self.n_rows,
            "n_predictions": self.n_predictions,
            "mqtt_connected": self.mqtt_connected,
            "models_dir": str(self.cfg.models_dir),
            "last_error": self.last_error or self.store.last_error,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = EdgeState(EdgeConfig.from_env())
    app.state.edge = state
    state.start()
    try:
        yield
    finally:
        state.stop()


app = FastAPI(title="Thor Edge Inference", version="0.1.0", lifespan=lifespan)


def _state() -> EdgeState:
    state = getattr(app.state, "edge", None)
    if state is None:  # app used without lifespan (plain import); start lazily
        state = EdgeState(EdgeConfig.from_env())
        app.state.edge = state
        state.start()
    return state


@app.get("/health")
def health() -> dict[str, Any]:
    """{status, model_version, n_assets_seen, uptime_s, ...}."""
    return _state().health()


@app.api_route("/predict", methods=["GET", "POST"])
def predict(req: PredictRequest = Body(...)) -> dict[str, Any]:
    """Score a ready-made feature dict. 503 when no model is loaded, 422 on missing features."""
    model = _state().store.current
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    try:
        p = predict_one(model, req.features)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"failure_probability": p, "model_version": model.version}


@app.post("/ingest")
def ingest(payload: list[dict[str, Any]] | dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Push raw telemetry rows directly (no broker). Output: {"accepted": n, "predictions": [...]}."""
    state = _state()
    rows = payload if isinstance(payload, list) else [payload]
    preds: list[dict[str, Any]] = []
    for row in rows:
        pred = state.process_row(row)
        if pred is not None:
            state.emit(pred)
            preds.append(pred)
    return {"accepted": len(rows), "predictions": preds}


@app.get("/features")
def features() -> dict[str, Any]:
    """Feature names the edge can compute, and the loaded model's expected subset."""
    model = _state().store.current
    return {"available": FEATURE_NAMES, "model": model.features if model else None}


@app.post("/models/refresh")
def refresh_models() -> dict[str, Any]:
    """Force a MODELS_DIR rescan (the watcher also does this every EDGE_MODEL_POLL_S)."""
    state = _state()
    reloaded = state.store.refresh()
    return {"reloaded": reloaded, **state.health()}
