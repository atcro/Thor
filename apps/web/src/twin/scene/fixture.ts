/**
 * A realistic TwinSnapshot for layout tests and the dev harness. Mirrors the real plant
 * mapping in data/simulator/layout.py (Packaging Lines 1-5 of four motors each plus the
 * Compressor Hall), with MTR-042 as the live critical filler on Line 3, MTR-021 offline
 * under a work order, and MTR-016 reporting no data.
 *
 * Fixture only: the app never renders it. Every value shape matches apps/api/schemas.py.
 */
import type {
  AlarmLevel,
  Asset,
  DecisionState,
  ObservedTelemetryRow,
  TwinAsset,
  TwinLine,
  TwinSnapshot,
  TwinStatus,
  WorkOrder,
} from "../../api/types";

const PLANT_TS = "2025-09-22T14:03:30Z";
const RECEIVED_AT = "2026-10-11T18:20:04.120Z";

const PACKAGING_ROLES = ["infeed conveyor", "filler", "capper/labeler", "case packer"] as const;

/** line number -> [infeed, filler, capper/labeler, case packer] (data/simulator/layout.py). */
const PACKAGING: Record<number, [string, string, string, string]> = {
  1: ["MTR-022", "MTR-001", "MTR-005", "MTR-019"],
  2: ["MTR-024", "MTR-011", "MTR-006", "MTR-010"],
  3: ["MTR-013", "MTR-042", "MTR-021", "MTR-009"],
  4: ["MTR-018", "MTR-003", "MTR-017", "MTR-023"],
  5: ["MTR-016", "MTR-015", "MTR-007", "MTR-008"],
};
const COMPRESSORS: [string, string][] = [
  ["MTR-012", "K-1"],
  ["MTR-002", "K-2"],
  ["MTR-014", "K-3"],
  ["MTR-020", "K-4"],
];

const RATED_KW: Record<string, number> = {
  "MTR-001": 55, "MTR-002": 90, "MTR-003": 55, "MTR-042": 75, "MTR-005": 30, "MTR-006": 45,
  "MTR-007": 15, "MTR-008": 30, "MTR-009": 45, "MTR-010": 90, "MTR-011": 55, "MTR-012": 110,
  "MTR-013": 15, "MTR-014": 45, "MTR-015": 45, "MTR-016": 15, "MTR-017": 30, "MTR-018": 15,
  "MTR-019": 90, "MTR-020": 55, "MTR-021": 30, "MTR-022": 30, "MTR-023": 45, "MTR-024": 15,
};

function latestRow(
  assetId: string,
  rpm: number,
  load: number,
  overrides: Partial<ObservedTelemetryRow> = {},
): ObservedTelemetryRow {
  const kw = RATED_KW[assetId] ?? 30;
  return {
    asset_id: assetId,
    ts: PLANT_TS,
    vibration_rms: 2.1,
    vibration_kurtosis: 3.0,
    vibration_crest: 3.4,
    bearing_temp_c: 51.0,
    motor_temp_c: 58.0,
    current_a: Number(((kw * 1000) / (400 * Math.sqrt(3) * 0.88) * (load / 100)).toFixed(1)),
    rpm,
    load_pct: load,
    ...overrides,
  };
}

interface Spec {
  id: string;
  line: string;
  role: string;
  name: string;
  status?: TwinStatus;
  p?: number | null;
  regime?: string | null;
  decision?: DecisionState;
  latest?: ObservedTelemetryRow | null;
  dataState?: TwinAsset["data_state"];
  workOrder?: WorkOrder | null;
  alarm?: AlarmLevel | null;
  alarmSince?: string | null;
  alarmReason?: string | null;
}

function asset(spec: Spec): TwinAsset {
  const status = spec.status ?? "healthy";
  const p = spec.p === undefined ? 0.04 : spec.p;
  const latest =
    spec.latest === undefined ? latestRow(spec.id, 1478, 56) : spec.latest;
  const a: Asset = {
    asset_id: spec.id,
    name: spec.name,
    site: "Ludvika",
    line: spec.line,
    asset_type: "induction motor",
    rated_kw: RATED_KW[spec.id] ?? 30,
    criticality: spec.id === "MTR-042" ? "high" : "medium",
    role: spec.role,
  };
  return {
    asset: a,
    status,
    data_state: spec.dataState ?? (latest ? "live" : "none"),
    decision_state: spec.decision ?? "none",
    failure_probability: p,
    health_score: p === null ? null : Math.round(100 * (1 - p)),
    prediction_ts: p === null ? null : PLANT_TS,
    model_version: p === null ? null : "3",
    n_prediction_rows: p === null ? null : 3,
    regime: spec.regime === undefined ? "R2" : spec.regime,
    latest,
    last_ts: latest ? latest.ts : null,
    last_received_at: latest ? RECEIVED_AT : null,
    msgs_last_60s: latest ? 20 : 0,
    open_contract_id: spec.decision === "contract_pending" ? "dc-0042" : null,
    work_order: spec.workOrder ?? null,
    // Latched alarm behind the status (apps/api/alarms.py): critical / warning carry the plant
    // time they were raised and the rule; healthy motors have none.
    alarm: spec.alarm === undefined ? (status === "critical" || status === "warning" ? status : null) : spec.alarm,
    alarm_since: spec.alarmSince === undefined ? (status === "critical" || status === "warning" ? "2025-09-22T11:30:00Z" : null) : spec.alarmSince,
    alarm_reason:
      spec.alarmReason === undefined
        ? status === "critical"
          ? "sustained p >= 0.5 over 3 rows"
          : status === "warning"
            ? "sustained p >= 0.3 over 3 rows"
            : null
        : spec.alarmReason,
  };
}

const assets: TwinAsset[] = [];
const lines: TwinLine[] = [];

for (const lineNo of [1, 2, 3, 4, 5]) {
  const ids = PACKAGING[lineNo];
  const line = `Packaging Line ${lineNo}`;
  ids.forEach((id, i) => {
    const role = PACKAGING_ROLES[i];
    const name = `Line ${lineNo} ${role} drive motor`;
    if (id === "MTR-042") {
      assets.push(
        asset({
          id, line, role, name,
          status: "critical",
          p: 0.64,
          regime: "R2",
          decision: "contract_pending",
          latest: latestRow(id, 1472, 61, {
            vibration_rms: 6.9,
            vibration_kurtosis: 5.8,
            vibration_crest: 5.1,
            bearing_temp_c: 83.4,
            motor_temp_c: 71.2,
          }),
        }),
      );
    } else if (id === "MTR-021") {
      assets.push(
        asset({
          id, line, role, name,
          status: "offline",
          p: 0.41,
          regime: null,
          decision: "work_order_in_progress",
          latest: latestRow(id, 0, 0, { ts: "2025-09-22T13:40:00Z", current_a: 0 }),
          dataState: "stale",
          workOrder: {
            contract_id: "dc-0021",
            asset_id: id,
            recommendation: "maintain_now",
            window_start: "2025-09-22T13:40:00Z",
            window_end: "2025-09-22T17:40:00Z",
            planned_downtime_h: 4,
            approved_at: "2026-10-11T18:12:40Z",
            approver: "demo-prep@thor",
            status: "in_progress",
            started_ts: "2025-09-22T13:40:00Z",
            completed_ts: null,
          },
        }),
      );
    } else if (id === "MTR-016") {
      assets.push(
        asset({
          id, line, role, name,
          status: "no_data",
          p: null,
          regime: null,
          latest: null,
          dataState: "none",
        }),
      );
    } else if (id === "MTR-017") {
      assets.push(
        asset({ id, line, role, name, status: "warning", p: 0.34, regime: "R3",
          latest: latestRow(id, 1490, 88, { bearing_temp_c: 66.5, vibration_rms: 3.2 }) }),
      );
    } else if (id === "MTR-018") {
      assets.push(
        asset({ id, line, role, name, status: "healthy", p: 0.07, regime: "R3",
          latest: latestRow(id, 1485, 84) }),
      );
    } else if (id === "MTR-007" || id === "MTR-024") {
      assets.push(
        asset({ id, line, role, name, status: "healthy", p: 0.03, regime: "R1",
          latest: latestRow(id, 610, 18) }),
      );
    } else if (id === "MTR-015") {
      assets.push(
        asset({ id, line, role, name, status: "no_prediction", p: null, regime: "R2" }),
      );
    } else {
      assets.push(asset({ id, line, role, name }));
    }
  });
  lines.push({ line, kind: "packaging", index: lineNo - 1, assets: [...ids], mes: {
    line,
    sku: `SKU-${1000 + lineNo}`,
    container: "PET 500 ml",
    quantity_produced: 1234 * lineNo,
    uom: "bottles",
    shift: "B",
    plant_ts: PLANT_TS,
    source: "simulated MES feed",
    basis: "fixture",
  } });
}

for (const [id, k] of COMPRESSORS) {
  assets.push(
    asset({
      id,
      line: "Compressor Hall",
      role: "air compressor",
      name: `Air compressor ${k} drive motor`,
      regime: id === "MTR-012" ? "R3" : "R2",
      latest: latestRow(id, 2960, id === "MTR-012" ? 82 : 58),
    }),
  );
}
lines.push({
  line: "Compressor Hall",
  kind: "compressor",
  index: 5,
  assets: COMPRESSORS.map(([id]) => id),
  mes: null,
});

export const SCENE_FIXTURE: TwinSnapshot = {
  site: "Packaging",
  sites: [
    { name: "Packaging", available: true, note: "" },
    { name: "Molding", available: false, note: "site tab only -- not built" },
    { name: "Batching", available: false, note: "site tab only -- not built" },
  ],
  plant_time: PLANT_TS,
  thresholds: { classification: 0.5, early_warning: 0.3, sustained_rows: 3 },
  lines,
  assets,
  model: {
    name: "thor-pdm",
    version: "3",
    family: "lightgbm",
    stage: "production",
    ims_total: 0.81,
    registered_at: "2026-10-11T17:58:00Z",
    approver: "demo-prep@thor",
    approved_at: "2026-10-11T17:59:30Z",
    promotion_id: "pr-0003",
    validation_passed: true,
    parity_passed: true,
    parity_max_abs_diff: 0.0007,
    lead_time_median_h: 31.5,
    calibration_method: "isotonic",
    onnx_path: "models/thor-pdm-3.onnx",
  },
  live: {
    server_time: RECEIVED_AT,
    window_s: 10,
    msgs_per_s: 28.4,
    preds_per_s: 27.9,
    messages_total: 184_220,
    predictions_total: 183_902,
    edge_inference_ms_p50: 1.9,
    edge_inference_ms_p95: 3.4,
    path_latency_ms_p50: 14.2,
    path_latency_ms_p95: 29.8,
    n_latency_samples: 2000,
    stale_after_s: 15,
    mqtt_connected: true,
    edge_up: true,
  },
};
