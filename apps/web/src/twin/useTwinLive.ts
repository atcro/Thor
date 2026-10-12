/**
 * Live data for the 3-D Digital Twin.
 *
 *  - `snapshot`: GET /twin polled every 2 s (status, regime, sensors, decisions, provenance).
 *  - `live`: the control plane's measured counters (message rate, prediction rate, edge
 *    inference and path latency), refreshed with every event frame.
 *  - `pulseQueue`: every ingest/prediction arrival the control plane reports, pushed as a
 *    PulseEvent for the scene to drain. Events come over WS /ws/twin; if the socket is
 *    unavailable the hook short-polls GET /twin/events?since= every 500 ms. Nothing here
 *    generates events on its own: an idle plant produces no pulses.
 */
import { useEffect, useRef, useState } from "react";
import { getTwin, getTwinEvents, twinWsUrl } from "../api/client";
import type { LiveCounters, TwinEventBatch, TwinSnapshot } from "../api/types";
import { usePolling } from "../hooks/usePolling";
import type { PulseEvent } from "./sceneTypes";

export type LiveTransport = "ws" | "poll" | "idle";

export interface TwinLive {
  snapshot: TwinSnapshot | null;
  snapshotError: string | null;
  snapshotLoading: boolean;
  refresh: () => Promise<void>;
  live: LiveCounters | null;
  transport: LiveTransport;
  /** Events received since the page opened (sum of `count`). */
  eventsSeen: number;
  pulseQueue: React.MutableRefObject<PulseEvent[]>;
}

const SNAPSHOT_MS = 2000;
const POLL_MS = 500;
const WS_RETRY_MS = 10000;
/** Never let an unread queue grow without bound if the scene is not mounted. */
const QUEUE_CAP = 2000;

export function useTwinLive(enabled: boolean = true): TwinLive {
  const snap = usePolling(getTwin, SNAPSHOT_MS, "twin", enabled);
  const [live, setLive] = useState<LiveCounters | null>(null);
  const [transport, setTransport] = useState<LiveTransport>("idle");
  const [eventsSeen, setEventsSeen] = useState(0);
  const pulseQueue = useRef<PulseEvent[]>([]);
  const sinceRef = useRef(0);
  const seenRef = useRef(0);
  const lastStateAt = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let ws: WebSocket | null = null;
    let pollTimer: number | undefined;
    let retryTimer: number | undefined;
    let polling = false;

    const absorb = (batch: TwinEventBatch) => {
      const at = performance.now();
      const q = pulseQueue.current;
      for (const e of batch.events) {
        q.push({ seq: e.seq, kind: e.kind, source: e.source ?? null, asset_id: e.asset_id, count: e.count, at });
        seenRef.current += e.count;
      }
      if (q.length > QUEUE_CAP) q.splice(0, q.length - QUEUE_CAP);
      if (batch.next > sinceRef.current) sinceRef.current = batch.next;
      // Counters re-render at most twice a second; pulses never wait for React.
      if (at - lastStateAt.current > 500 || batch.events.length === 0) {
        lastStateAt.current = at;
        setLive(batch.live);
        setEventsSeen(seenRef.current);
      }
    };

    const startPolling = () => {
      if (polling || !alive) return;
      polling = true;
      setTransport("poll");
      const tick = async () => {
        if (!alive || !polling) return;
        try {
          if (typeof document === "undefined" || !document.hidden) {
            absorb(await getTwinEvents(sinceRef.current));
          }
        } catch {
          /* control plane down: keep polling; the snapshot error is shown by the page */
        }
        if (alive && polling) pollTimer = window.setTimeout(tick, POLL_MS);
      };
      void tick();
    };

    const stopPolling = () => {
      polling = false;
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
    };

    const connect = () => {
      if (!alive) return;
      if (typeof WebSocket === "undefined") {
        startPolling();
        return;
      }
      try {
        ws = new WebSocket(twinWsUrl());
      } catch {
        startPolling();
        return;
      }
      ws.onopen = () => {
        stopPolling();
        setTransport("ws");
        ws?.send(JSON.stringify({ since: sinceRef.current }));
      };
      ws.onmessage = (ev) => {
        try {
          absorb(JSON.parse(String(ev.data)) as TwinEventBatch);
        } catch {
          /* malformed frame: ignore */
        }
      };
      const fallback = () => {
        ws = null;
        if (!alive) return;
        startPolling();
        retryTimer = window.setTimeout(connect, WS_RETRY_MS);
      };
      ws.onerror = () => {
        /* onclose follows */
      };
      ws.onclose = fallback;
    };

    connect();
    return () => {
      alive = false;
      stopPolling();
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [enabled]);

  return {
    snapshot: snap.data,
    snapshotError: snap.error,
    snapshotLoading: snap.loading,
    refresh: snap.refresh,
    live: live ?? snap.data?.live ?? null,
    transport,
    eventsSeen,
    pulseQueue,
  };
}
