"""ONNX model discovery + inference for the edge container (numpy + onnxruntime only).

A deployed model is a pair of files in MODELS_DIR written by `agents/mlops/lifecycle.deploy_edge`:

    <name>_<version>.onnx
    <name>_<version>.json      sidecar:
        {
          "model_name": "thor-bearing-classifier",
          "version": "v12345",
          "features": ["vib_rms_mean", ...],        # subset/order of edge.features.FEATURE_NAMES
          "window_rows": 12,
          "calibration": "none" | "isotonic" | "sigmoid",   # optional, informational
          "baseline": {...}                          # optional, see edge.features
        }

The newest pair (by .onnx mtime) wins. Probability extraction handles both skl2onnx's
`output_probability` (sequence of {class: prob} maps, ZipMap) and plain (n, 2) tensors.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from edge.features import FEATURE_NAMES

log = logging.getLogger("thor.edge.inference")


@dataclass
class LoadedModel:
    """A loaded ONNX session plus its sidecar metadata."""

    model_name: str
    version: str
    features: list[str]
    window_rows: int
    onnx_path: Path
    sidecar_path: Path
    session: Any
    input_name: str
    input_dtype: Any
    calibration: str = "none"
    baseline: dict[str, Any] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)
    mtime: float = 0.0

    @property
    def key(self) -> tuple[str, float]:
        return (str(self.onnx_path), self.mtime)


def find_newest_model(models_dir: Path) -> tuple[Path, Path] | None:
    """Return (onnx_path, sidecar_path) for the newest *.onnx that has a matching *.json.

    Inputs: models directory (may not exist). Output: the pair, or None if nothing deployable.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return None
    candidates = [p for p in models_dir.glob("*.onnx") if p.with_suffix(".json").is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest, newest.with_suffix(".json")


def load_model(onnx_path: Path, sidecar_path: Path) -> LoadedModel:
    """Create an onnxruntime session and read the sidecar. Raises ValueError on a bad sidecar."""
    import onnxruntime as ort

    meta = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    features = list(meta.get("features") or FEATURE_NAMES)
    unknown = [f for f in features if f not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"sidecar lists features the edge cannot compute: {unknown}")
    window_rows = int(meta.get("window_rows", 12))
    if window_rows < 1:
        raise ValueError("window_rows must be >= 1")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    dtype = np.float64 if "double" in inp.type else np.float32
    return LoadedModel(
        model_name=str(meta.get("model_name", Path(onnx_path).stem)),
        version=str(meta.get("version", "unknown")),
        features=features,
        window_rows=window_rows,
        onnx_path=Path(onnx_path),
        sidecar_path=Path(sidecar_path),
        session=session,
        input_name=inp.name,
        input_dtype=dtype,
        calibration=str(meta.get("calibration", "none")),
        baseline=dict(meta.get("baseline") or {}),
        mtime=Path(onnx_path).stat().st_mtime,
    )


def extract_probability(outputs: list[Any], output_names: list[str]) -> np.ndarray:
    """Positive-class probability (n,) from onnxruntime outputs.

    Handles: ZipMap sequence-of-maps ({0: p0, 1: p1} per row), (n, 2+) probability tensors
    (second column), (n, 1)/(n,) float tensors (used as-is). Outputs whose name contains
    "probab" are preferred; integer label outputs are skipped.
    """
    order = sorted(
        range(len(outputs)), key=lambda i: 0 if "probab" in output_names[i].lower() else 1
    )
    for i in order:
        out = outputs[i]
        if isinstance(out, list):
            if not out:
                continue
            if isinstance(out[0], dict):
                probs = []
                for row in out:
                    if 1 in row:
                        probs.append(float(row[1]))
                    elif "1" in row:
                        probs.append(float(row["1"]))
                    else:
                        vals = [row[k] for k in sorted(row)]
                        probs.append(float(vals[1] if len(vals) > 1 else vals[0]))
                return np.asarray(probs, dtype=float)
            out = np.asarray(out)
        arr = np.asarray(out)
        if arr.dtype.kind not in "fc":
            continue  # integer labels
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1].astype(float)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        if arr.ndim == 1:
            return arr.astype(float)
    raise ValueError(f"no probability output found among {output_names}")


def predict_proba(model: LoadedModel, X: np.ndarray) -> np.ndarray:
    """Run the session on X of shape (n, len(model.features)). Output: (n,) probabilities in [0, 1]."""
    X = np.asarray(X, dtype=model.input_dtype).reshape(-1, len(model.features))
    outputs = model.session.run(None, {model.input_name: X})
    names = [o.name for o in model.session.get_outputs()]
    p = extract_probability(outputs, names)
    return np.clip(p, 0.0, 1.0)


def predict_one(model: LoadedModel, features: dict[str, float]) -> float:
    """Score one feature dict (must contain every name in model.features). Output: probability."""
    missing = [f for f in model.features if f not in features]
    if missing:
        raise KeyError(f"missing features: {missing}")
    vec = np.array([[float(features[f]) for f in model.features]])
    return float(predict_proba(model, vec)[0])


class ModelStore:
    """Holds the currently loaded model and hot-reloads when a newer pair appears."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.current: LoadedModel | None = None
        self.last_error: str | None = None
        self._last_pair_key: tuple[str, float] | None = None

    def refresh(self) -> bool:
        """Scan MODELS_DIR; load the newest model if it changed. Output: True if (re)loaded."""
        pair = find_newest_model(self.models_dir)
        if pair is None:
            if self.current is None and self._last_pair_key is None:
                log.info("no model in %s -- edge idle", self.models_dir)
                self._last_pair_key = ("", 0.0)
            return False
        onnx_path, sidecar = pair
        key = (str(onnx_path), onnx_path.stat().st_mtime)
        if key == self._last_pair_key:
            return False
        try:
            model = load_model(onnx_path, sidecar)
        except Exception as exc:  # noqa: BLE001 - a broken export must not kill the edge
            self.last_error = f"{onnx_path.name}: {exc}"
            log.warning("failed to load %s: %s", onnx_path.name, exc)
            self._last_pair_key = key  # do not retry the same broken file every poll
            return False
        self.current = model
        self.last_error = None
        self._last_pair_key = key
        log.info(
            "loaded %s %s from %s (features=%d window=%d)",
            model.model_name,
            model.version,
            onnx_path.name,
            len(model.features),
            model.window_rows,
        )
        return True
