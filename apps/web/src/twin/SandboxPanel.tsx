/**
 * Sandbox panel (3-D Digital Twin): lets the presenter override one motor's recorded readings
 * (rpm, bearing temperature), inject the simulator's bearing fault, shake up operations or
 * reset. Every action is a POST to /sandbox/* and every number shown comes from the
 * SandboxState the API returned; the readings themselves are produced by the replay and scored
 * like plant data, tagged sandbox all the way through. Nothing here scores or decides.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { getSandbox, sandboxFault, sandboxRandomFault, sandboxReset, sandboxSet, sandboxShake } from "../api/client";
import type { SandboxOverride, SandboxState, TwinAsset } from "../api/types";
import { fmtDateTime, fmtNum, fmtShort } from "../lib/format";

export const SANDBOX_DEFAULT_MOTOR = "MTR-042";
/** Slider changes are sent ~300 ms after the last movement (one POST per release). */
const DEBOUNCE_MS = 300;
const RPM_MIN = 0;
const RPM_MAX = 1800;
const RPM_STEP = 10;
const TEMP_MIN = 20;
const TEMP_MAX = 120;
const TEMP_STEP = 1;

function overrideFor(state: SandboxState | null, assetId: string): SandboxOverride | null {
  return state?.overrides.find((o) => o.asset_id === assetId) ?? null;
}

export function SandboxPanel({
  assetId,
  asset,
  state,
  onState,
}: {
  /** The motor the sliders act on (the twin's selection, default MTR-042). */
  assetId: string;
  /** Its /twin entry, for the recorded rpm / temperature the sliders start from. */
  asset: TwinAsset | null;
  /** GET /sandbox as the screen polls it; POST responses replace it through onState. */
  state: SandboxState | null;
  onState: (s: SandboxState) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const override = overrideFor(state, assetId);

  // Slider positions: the active override's setpoint, else the motor's latest recorded value.
  const [rpm, setRpm] = useState<number | null>(null);
  const [temp, setTemp] = useState<number | null>(null);
  const rpmShown = rpm ?? override?.rpm ?? asset?.latest?.rpm ?? 1480;
  const tempShown = temp ?? override?.bearing_temp_c ?? asset?.latest?.bearing_temp_c ?? 50;
  const timers = useRef<{ rpm?: number; temp?: number }>({});

  useEffect(() => {
    // A new motor: forget the local slider positions.
    setRpm(null);
    setTemp(null);
  }, [assetId]);

  useEffect(() => {
    if (!flash) return;
    const id = window.setTimeout(() => setFlash(null), 4000);
    return () => window.clearTimeout(id);
  }, [flash]);

  const run = useCallback(
    async (label: string, fn: () => Promise<SandboxState>) => {
      setBusy(label);
      setError(null);
      try {
        onState(await fn());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [onState],
  );

  const sendRpm = (value: number) => {
    setRpm(value);
    window.clearTimeout(timers.current.rpm);
    timers.current.rpm = window.setTimeout(() => void run("rpm", () => sandboxSet({ asset_id: assetId, rpm: value })), DEBOUNCE_MS);
  };
  const sendTemp = (value: number) => {
    setTemp(value);
    window.clearTimeout(timers.current.temp);
    timers.current.temp = window.setTimeout(
      () => void run("temp", () => sandboxSet({ asset_id: assetId, bearing_temp_c: value })),
      DEBOUNCE_MS,
    );
  };
  useEffect(() => {
    const t = timers.current;
    return () => {
      window.clearTimeout(t.rpm);
      window.clearTimeout(t.temp);
    };
  }, []);

  const reset = () =>
    void run("reset", async () => {
      const r = await sandboxReset();
      setRpm(null);
      setTemp(null);
      setFlash(r.alarms_cleared.length > 0 ? `alarms cleared: ${r.alarms_cleared.join(", ")}` : "alarms cleared: none");
      return r;
    });

  return (
    <aside className="xray-frame twin-sandbox" data-testid="sandbox-panel" aria-label="Sandbox">
      <h3>Sandbox · simulated readings</h3>
      <div>
        motor <span className="motor">{assetId}</span>
        {asset?.asset.name ? <span className="twin-dim"> · {asset.asset.name}</span> : null}
      </div>
      <div className="note">select a motor in the scene to retarget; sliders apply on release</div>

      <div className="slider">
        <label htmlFor="sandbox-rpm">speed (rpm)</label>
        <span className="val" data-testid="sandbox-rpm-value">{fmtNum(rpmShown, 0)}</span>
        <input
          id="sandbox-rpm"
          type="range"
          min={RPM_MIN}
          max={RPM_MAX}
          step={RPM_STEP}
          value={rpmShown}
          onChange={(e) => sendRpm(Number(e.target.value))}
          aria-valuetext={`${fmtNum(rpmShown, 0)} rpm`}
        />
      </div>
      <div className="slider">
        <label htmlFor="sandbox-temp">bearing temperature (°C)</label>
        <span className="val" data-testid="sandbox-temp-value">{fmtNum(tempShown, 0)}</span>
        <input
          id="sandbox-temp"
          type="range"
          min={TEMP_MIN}
          max={TEMP_MAX}
          step={TEMP_STEP}
          value={tempShown}
          onChange={(e) => sendTemp(Number(e.target.value))}
          aria-valuetext={`${fmtNum(tempShown, 0)} °C`}
        />
      </div>

      <div className="actions">
        <button type="button" className="xray-btn primary" disabled={busy !== null} onClick={() => void run("fault", () => sandboxFault({ asset_id: assetId }))} title="Inject the simulator's drive-end bearing fault on this motor (POST /sandbox/fault)">
          Troubleshoot {assetId}
        </button>
        <button type="button" className="xray-btn" disabled={busy !== null} onClick={() => void run("random", sandboxRandomFault)} title="The same fault on a random motor that is not offline or already faulted (POST /sandbox/random-fault)">
          Random fault
        </button>
        <button type="button" className="xray-btn" disabled={busy !== null} onClick={() => void run("shake", sandboxShake)} title="Random rpm / load / temperature within normal ranges on several motors (POST /sandbox/shake)">
          Shake up operations
        </button>
        <button type="button" className="xray-btn" disabled={busy !== null} onClick={reset} title="Remove every override and clear sandbox-only alarms (POST /sandbox/reset)">
          Reset sandbox
        </button>
      </div>
      {flash && (
        <div className="flash" role="status" data-testid="sandbox-flash">
          {flash}
        </div>
      )}
      {error && (
        <div className="err" role="alert">
          {error}
        </div>
      )}

      <div className="overrides" data-testid="sandbox-overrides">
        <div className="twin-dim">
          active overrides: {state ? state.overrides.length : "—"}
          {state?.last_reset_at ? ` · last reset ${fmtShort(state.last_reset_at)}` : ""}
        </div>
        {(state?.overrides ?? []).map((o) => (
          <div className="row" key={o.asset_id} title={o.note || undefined}>
            <span>{o.asset_id}</span>
            <span>
              {o.rpm !== null ? `${fmtNum(o.rpm, 0)} rpm` : "rpm —"} · {o.bearing_temp_c !== null ? `${fmtNum(o.bearing_temp_c, 0)} °C` : "temp —"}
              {o.load_pct !== null ? ` · ${fmtNum(o.load_pct, 0)} %` : ""}
              {o.fault ? (
                <span className="xray-warn" title={o.fault_since ? `fault since ${fmtDateTime(o.fault_since)} (plant time), ramp ${fmtNum(o.fault_ramp_h, 1)} h` : undefined}>
                  {" "}
                  · fault since {fmtShort(o.fault_since)}
                </span>
              ) : null}
            </span>
          </div>
        ))}
      </div>
      <div className="note">readings under an override are tagged sandbox and scored like plant data; a reset clears sandbox-only alarms</div>
    </aside>
  );
}

export function useSandboxState(enabled: boolean): { state: SandboxState | null; setState: (s: SandboxState) => void } {
  const [state, setState] = useState<SandboxState | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getSandbox();
        if (alive) setState(s);
      } catch {
        /* the next poll retries */
      }
    };
    void tick();
    // The twin keeps a light 2 s poll even while the panel is closed so the banner can show a
    // sandbox someone else left on; the panel itself only adds the actions.
    const id = window.setInterval(() => void tick(), enabled ? 2000 : 10000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [enabled]);
  return { state, setState };
}
