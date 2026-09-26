import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { copilotChat, getFleet } from "../api/client";
import type { CopilotMessage, CopilotToolCall, ExplanationSource } from "../api/types";
import { Panel, SourceChip } from "../components/ui";
import { usePolling } from "../hooks/usePolling";

interface ChatEntry extends CopilotMessage {
  tool_calls?: CopilotToolCall[];
  source?: ExplanationSource;
  error?: boolean;
}

const SUGGESTIONS = [
  "Which assets are at highest risk right now?",
  "Why is MTR-042 flagged?",
  "What would the failure probability be at 40% load?",
  "Summarise the open decision contract.",
];

function toolName(t: CopilotToolCall): string {
  return String(t.name ?? t.tool ?? "tool");
}

function toolArgs(t: CopilotToolCall): string {
  const a = t.args ?? t.input;
  if (!a || Object.keys(a).length === 0) return "";
  return Object.entries(a)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(", ");
}

export function Copilot() {
  const [params] = useSearchParams();
  const fleet = usePolling(getFleet, 30000, "fleet-for-copilot");
  const [assetId, setAssetId] = useState<string>(params.get("asset") ?? "");
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [entries, busy]);

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || busy) return;
    const userMsg: ChatEntry = { role: "user", content };
    const history: CopilotMessage[] = [...entries, userMsg]
      .filter((e) => !e.error)
      .map(({ role, content: c }) => ({ role, content: c }));
    setEntries((prev) => [...prev, userMsg]);
    setDraft("");
    setBusy(true);
    try {
      const res = await copilotChat({ messages: history, asset_id: assetId || null });
      setEntries((prev) => [...prev, { role: "assistant", content: res.reply, tool_calls: res.tool_calls ?? [], source: res.source }]);
    } catch (e) {
      setEntries((prev) => [...prev, { role: "assistant", content: e instanceof Error ? e.message : String(e), error: true }]);
    } finally {
      setBusy(false);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(draft);
    }
  };

  const assets = fleet.data ? [...fleet.data].sort((a, b) => a.asset.asset_id.localeCompare(b.asset.asset_id)) : [];

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Bolt</h1>
          <div className="sub">Read-only copilot over deterministic tools (get_fleet, get_asset, get_contract, get_run, list_pending_approvals, search_manuals). It explains numbers and quotes retrieved manual passages; it never produces either.</div>
        </div>
        <div className="row">
          <label className="field">
            asset context
            <select className="select mono" value={assetId} onChange={(e) => setAssetId(e.target.value)} style={{ minWidth: 140 }}>
              <option value="">— none —</option>
              {assets.map((a) => (
                <option key={a.asset.asset_id} value={a.asset.asset_id}>
                  {a.asset.asset_id}
                </option>
              ))}
            </select>
          </label>
          <button className="btn ghost sm" onClick={() => setEntries([])} disabled={entries.length === 0 || busy}>
            Clear
          </button>
        </div>
      </div>

      <Panel tight>
        <div className="chat">
          <div className="chat-log" ref={logRef}>
            {entries.length === 0 && (
              <div className="state" style={{ marginTop: 40 }}>
                <div style={{ marginBottom: 10 }}>Ask about fleet risk, a specific asset, or an open decision contract.</div>
                <div className="row wrap" style={{ justifyContent: "center" }}>
                  {SUGGESTIONS.map((s) => (
                    <button key={s} className="btn sm" onClick={() => void send(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {entries.map((m, i) => (
              <div key={i} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                {m.tool_calls && m.tool_calls.length > 0 && (
                  <div className="meta">
                    {m.tool_calls.map((t, j) => (
                      <span key={j} className="chip steel" title={JSON.stringify(t)}>
                        ⚙ {toolName(t)}
                        {toolArgs(t) && <span className="dim">({toolArgs(t)})</span>}
                      </span>
                    ))}
                  </div>
                )}
                <div className="bubble" style={m.error ? { borderColor: "rgba(227,121,59,0.5)", color: "var(--ember-2)" } : undefined}>
                  {m.content}
                </div>
                <div className="meta">
                  <span>{m.role === "user" ? "you" : "bolt"}</span>
                  {m.source && <SourceChip source={m.source} />}
                </div>
              </div>
            ))}
            {busy && (
              <div className="msg assistant">
                <div className="bubble muted">
                  <span className="dot pulse" style={{ background: "var(--steel)", marginRight: 6 }} />
                  thinking…
                </div>
              </div>
            )}
          </div>
          <div className="chat-input">
            <textarea
              className="textarea"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKey}
              placeholder={assetId ? `Ask about ${assetId}… (Enter to send)` : "Ask Bolt… (Enter to send, Shift+Enter for newline)"}
              disabled={busy}
            />
            <button className="btn primary" onClick={() => void send(draft)} disabled={busy || !draft.trim()}>
              Send
            </button>
          </div>
        </div>
      </Panel>
    </div>
  );
}
