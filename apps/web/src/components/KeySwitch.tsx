import { useState } from "react";
import { getKeyState, keyDown, keyUp } from "../api/client";
import type { LlmKeyState } from "../api/types";
import { usePolling } from "../hooks/usePolling";
import { Dot } from "./ui";

/**
 * API Key Up / API Key Down: the one switch for every LLM path in Thor, shown in the header
 * of every screen. It polls GET /llm/key (the state only, never the key) and flips it with
 * POST /llm/key/up | /down. Up: the API re-reads the provider key from its source into
 * memory and Bolt, the contract explanation and the routing call use the model. Down: the
 * API drops the key from memory and every path falls back to templates. Thor starts Down.
 */
export function KeySwitch() {
  const state = usePolling(getKeyState, 3000, "llm-key");
  const [busy, setBusy] = useState(false);
  const [override, setOverride] = useState<{ state: LlmKeyState; at: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const s = override && (state.lastUpdated ?? 0) < override.at ? override.state : (state.data ?? override?.state ?? null);
  if (!s) return null;
  const up = s.key === "up";
  const flip = async () => {
    setBusy(true);
    setErr(null);
    try {
      const next = await (up ? keyDown() : keyUp());
      setOverride({ state: next, at: Date.now() });
      void state.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const provider = s.provider === "none" ? "no key configured at the source" : `provider ${s.provider}`;
  const title = up
    ? `API Key Up: the API holds the provider key in memory (${provider}); Bolt, the contract explanation and the routing call use the model. Click to drop it (API Key Down).`
    : `API Key Down: the API holds no provider key; every path uses deterministic templates. Click to re-read the key from its source (API Key Up).`;
  return (
    <span className="keyswitch" data-testid="key-switch" title={title}>
      <button
        type="button"
        className={`btn sm${up ? " steel" : ""}`}
        aria-pressed={up}
        disabled={busy}
        onClick={() => void flip()}
      >
        <Dot on={up} pulse={busy} />
        {up ? "API Key Up" : "API Key Down"}
      </button>
      {up && s.provider !== "none" && <span className="mono small dim">{s.provider}</span>}
      {up && s.provider === "none" && <span className="small muted">no key at source</span>}
      {err && <span className="small" style={{ color: "var(--ember-2)" }}>{err}</span>}
    </span>
  );
}
