/**
 * The X-Ray toggle: a segmented button usable in any toolbar (Twin toolbar, Dashboard head).
 * State comes from useXRay(), so the Twin and the Dashboard share one setting.
 */
import { useXRay } from "./useXRay";

export function XRayToggle({ className = "" }: { className?: string }) {
  const xray = useXRay();
  return (
    <div className={`seg xray-toggle${xray.enabled ? " on" : ""}${className ? ` ${className}` : ""}`} role="group" aria-label="X-Ray mode">
      <button
        type="button"
        className={`seg-btn${xray.enabled ? " on" : ""}`}
        aria-pressed={xray.enabled}
        title="X-Ray: hover an asset for its status and key facts, click for the HUD"
        onClick={xray.toggle}
      >
        <span className="xray-toggle-led" aria-hidden="true" />
        X-Ray
      </button>
    </div>
  );
}
