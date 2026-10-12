"""Labels from what the technician found (Phase A.3, CLAUDE.md section 9).

The seeded history carries the simulator's completed failures (``failure_within_h``), which
stand in for imported CMMS records. A real plant has no such column: its labels live in work
orders. A person's **Confirmed: bearing damage** finding on a completed work order turns that
motor's pre-repair history into a labelled failure for the next ``mode=train`` run:

* the event time is the intervention (when the motor went offline), not a failure -- the
  motor never actually failed, so the label is conservative and may sit earlier than a true
  failure would have (see docs/finals/model-card.md);
* rows from the motor's previous completed repair (or the start of its history) up to the
  event get ``failure_within_h = hours until the event``; nothing after it is labelled;
* the motor joins the TRAIN group only: the split is computed exactly as before (same seed,
  same simulator-history positives) and the motor is then moved out of validation,
  calibration or test with a note, so the test motors behind the validated metrics stay
  the same otherwise.

Everything here is pure pandas over frames the caller fetched; nothing reads the simulator's
``health`` or ``failure_within_h`` of the confirmed motor -- those values are overwritten by
the finding. With no events every function returns its input unchanged (the same object), so
a fleet without findings trains exactly as before.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from agents.ml_architect.split import COMPLETED_FAILURE_MAX_H
from apps.api.schemas import AssetSplit, FieldFailureEvent, LabelSource

log = logging.getLogger("thor.labels")


def label_rows_for_event(rows: pd.DataFrame, event: FieldFailureEvent) -> pd.DataFrame:
    """Rows of one motor labelled by one confirmed finding.

    Input: the motor's telemetry rows (any range; ``ts`` tz-aware UTC) and the event. Output:
    a copy of the rows with ``labelled_from < ts < event_ts`` and ``failure_within_h`` set to
    the hours until ``event_ts`` (the simulator's value, if any, is replaced). Empty frame when
    nothing falls inside the window.
    """
    if rows.empty:
        return rows.copy()
    ts = pd.to_datetime(rows["ts"], utc=True)
    mask = ts < pd.Timestamp(event.event_ts)
    if event.labelled_from is not None:
        mask &= ts > pd.Timestamp(event.labelled_from)
    out = rows.loc[mask].copy()
    if out.empty:
        return out
    hours = (
        pd.Timestamp(event.event_ts) - pd.to_datetime(out["ts"], utc=True)
    ).dt.total_seconds() / 3600.0
    out["failure_within_h"] = hours.astype(float).to_numpy()
    return out


def completes_failure(labelled: pd.DataFrame) -> bool:
    """True when the labelled rows reach the intervention closely enough to count as a
    completed failure under the same rule as the simulator's history
    (``COMPLETED_FAILURE_MAX_H``). A data gap before the outage larger than that means the
    finding cannot stand in for a completed failure and is left out, with a note."""
    if labelled.empty or "failure_within_h" not in labelled.columns:
        return False
    return bool(
        pd.to_numeric(labelled["failure_within_h"], errors="coerce").min()
        <= COMPLETED_FAILURE_MAX_H
    )


def apply_field_labels(
    history: pd.DataFrame,
    events: Iterable[FieldFailureEvent],
    rows_by_asset: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[FieldFailureEvent]]:
    """Merge confirmed-finding labels into the training frame.

    Inputs: the recorded history (rows with stored ground truth, as ``_telemetry`` loads it),
    the events, and per-asset telemetry (history plus live rows) for every event asset.
    Output: ``(frame, applied)``. For each event whose labelled rows complete a failure, the
    motor's history rows are REPLACED by its labelled rows (so none of its labels come from
    the simulator); the frame is re-sorted by ``(asset_id, ts)``. Events that do not qualify
    are skipped with a warning. With no events the input frame is returned as-is (the same
    object), so training without findings is unchanged.
    """
    events = list(events)
    if not events:
        return history, []
    applied: list[FieldFailureEvent] = []
    parts: list[pd.DataFrame] = []
    drop: set[str] = set()
    for ev in events:
        rows = rows_by_asset.get(ev.asset_id)
        if rows is None or rows.empty:
            log.warning("finding %s: no telemetry rows for %s; skipped", ev.finding_id, ev.asset_id)
            continue
        labelled = label_rows_for_event(rows, ev)
        if not completes_failure(labelled):
            log.warning(
                "finding %s on %s: labelled rows do not reach the intervention within %g h; skipped",
                ev.finding_id,
                ev.asset_id,
                COMPLETED_FAILURE_MAX_H,
            )
            continue
        drop.add(ev.asset_id)
        parts.append(labelled)
        applied.append(ev)
    if not applied:
        return history, []
    kept = history.loc[~history["asset_id"].astype(str).isin(drop)]
    cols = list(history.columns)
    frame = pd.concat([kept, *[p.reindex(columns=cols) for p in parts]], ignore_index=True)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.sort_values(["asset_id", "ts"]).reset_index(drop=True), applied


def adopt_into_train(split: AssetSplit, events: Iterable[FieldFailureEvent]) -> AssetSplit:
    """Move every field-confirmed motor into the train group, with a note.

    Input: the split computed exactly as before (same seed, simulator-history positives) and
    the applied events. Output: the same object when nothing moves; otherwise a copy where
    each event asset is removed from validation / calibration / test (and ``forced_test``),
    added to ``train`` and to ``positives["train"]``, with one note per move. The remaining
    test motors are untouched.
    """
    events = list(events)
    if not events:
        return split
    train = list(split.train)
    validation = list(split.validation)
    calibration = list(split.calibration)
    test = list(split.test)
    forced = list(split.forced_test)
    positives = {g: list(v) for g, v in split.positives.items()}
    notes = list(split.notes)
    moved = False
    for ev in events:
        a = ev.asset_id
        came_from = [
            g
            for g, group in (
                ("validation", validation),
                ("calibration", calibration),
                ("test", test),
            )
            if a in group
        ]
        for group in (validation, calibration, test, forced):
            while a in group:
                group.remove(a)
        for g in ("validation", "calibration", "test"):
            positives[g] = [x for x in positives.get(g, []) if x != a]
        if a not in train:
            train.append(a)
        if a not in positives.setdefault("train", []):
            positives["train"].append(a)
        origin = " and ".join(came_from) if came_from else "train"
        notes.append(
            f"{a} trains as a completed failure confirmed by work order {ev.contract_id} "
            f"(recorded by {ev.recorded_by}, intervention {ev.event_ts.isoformat()})"
            + (
                f"; moved from {origin} to train, so its score is no longer out-of-sample"
                if came_from
                else ""
            )
        )
        moved = True
    if not moved:
        return split
    return split.model_copy(
        update={
            "train": sorted(train),
            "validation": sorted(validation),
            "calibration": sorted(calibration),
            "test": sorted(test),
            "forced_test": sorted(forced),
            "positives": {g: sorted(v) for g, v in positives.items()},
            "notes": notes,
        }
    )


def label_sources_for(
    split: AssetSplit,
    events: Iterable[FieldFailureEvent],
    n_rows: dict[str, int] | None = None,
) -> list[LabelSource]:
    """The label ledger of one run: every completed failure it trains on and its source.

    Input: the final split (after ``adopt_into_train``), the applied events and, optionally,
    the number of rows each finding labelled. Output: one ``LabelSource`` per positive motor,
    train group first; simulator-history motors are marked as standing in for imported CMMS
    records, field-confirmed motors carry the work order, the person and the intervention.
    """
    by_asset = {ev.asset_id: ev for ev in events}
    out: list[LabelSource] = []
    for group in ("train", "validation", "calibration", "test"):
        for a in split.positives.get(group, []):
            ev = by_asset.get(a)
            if ev is None:
                out.append(LabelSource(asset_id=a, source="simulator_history", group=group))
            else:
                out.append(
                    LabelSource(
                        asset_id=a,
                        source="work_order_finding",
                        group=group,
                        contract_id=ev.contract_id,
                        finding_id=ev.finding_id,
                        recorded_by=ev.recorded_by,
                        event_ts=ev.event_ts,
                        labelled_from=ev.labelled_from,
                        n_rows=(n_rows or {}).get(a),
                    )
                )
    return out
