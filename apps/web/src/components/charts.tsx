import type { ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtDateTime, fmtNum, fmtTime } from "../lib/format";

export const CHART = {
  ember: "#e3793b",
  steel: "#84acc7",
  green: "#4fb37a",
  amber: "#d9a441",
  grid: "#26313d",
  axis: "#4e5a66",
  tick: "#74818e",
  /** single-hue steel ramp, light -> dark, for stacked magnitude segments */
  steelRamp: ["#c9dce8", "#a9c7db", "#84acc7", "#5f8aa8", "#42667f"],
} as const;

export const AXIS_TICK = { fill: CHART.tick, fontSize: 10, fontFamily: "var(--font-mono)" };

interface TipItem {
  name?: unknown;
  value?: unknown;
  color?: string;
  dataKey?: unknown;
}

/** Shared tooltip body; recharts clones this element with active/payload/label. */
export function ChartTip({
  active,
  payload,
  label,
  labelFormatter,
  valueFormatter,
}: {
  active?: boolean;
  payload?: ReadonlyArray<TipItem>;
  label?: unknown;
  labelFormatter?: (l: unknown) => ReactNode;
  valueFormatter?: (v: unknown, name: unknown) => ReactNode;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="tip">
      <div className="t">{labelFormatter ? labelFormatter(label) : String(label ?? "")}</div>
      {payload.map((p, i) => (
        <div key={i} className="row" style={{ gap: 6 }}>
          <span className="dot" style={{ background: p.color ?? CHART.steel, width: 6, height: 6 }} />
          <span className="muted">{String(p.name ?? p.dataKey ?? "")}</span>
          <span>
            {valueFormatter
              ? valueFormatter(p.value, p.name)
              : typeof p.value === "number"
                ? fmtNum(p.value, 3)
                : String(p.value ?? "—")}
          </span>
        </div>
      ))}
    </div>
  );
}

export interface TimePoint {
  t: number;
  v: number | null;
}

/**
 * One series, one axis — small multiple for a time-series. Different units never share an axis.
 */
export function TimeSeriesChart({
  title,
  unit,
  data,
  color,
  height = 130,
  yDomain,
  threshold,
  digits = 2,
}: {
  title: string;
  unit?: string;
  data: TimePoint[];
  color: string;
  height?: number;
  yDomain?: [number | "auto", number | "auto"];
  threshold?: number;
  digits?: number;
}) {
  const last = [...data].reverse().find((d) => d.v !== null);
  return (
    <div>
      <div className="chart-title">
        <span>
          <b>{title}</b>
          {unit && <span className="dim"> {unit}</span>}
        </span>
        <span className="mono">{last ? fmtNum(last.v, digits) : "—"}</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => fmtTime(new Date(v).toISOString())}
            tick={AXIS_TICK}
            axisLine={{ stroke: CHART.axis }}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            domain={yDomain ?? ["auto", "auto"]}
            tick={AXIS_TICK}
            axisLine={false}
            tickLine={false}
            width={52}
            tickFormatter={(v: number) => fmtNum(v, digits > 1 ? 1 : digits)}
          />
          <Tooltip
            content={
              <ChartTip
                labelFormatter={(l) => fmtDateTime(new Date(Number(l)).toISOString())}
                valueFormatter={(v) => (typeof v === "number" ? fmtNum(v, digits) : "—")}
              />
            }
            cursor={{ stroke: CHART.axis, strokeDasharray: "3 3" }}
          />
          {threshold !== undefined && (
            <ReferenceLine y={threshold} stroke={CHART.amber} strokeDasharray="4 3" />
          )}
          <Line
            type="monotone"
            dataKey="v"
            name={title}
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
