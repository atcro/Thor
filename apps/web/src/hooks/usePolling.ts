import { useCallback, useEffect, useRef, useState } from "react";

export interface PollState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  /** Re-run the fetch immediately (does not reset the interval). */
  refresh: () => Promise<void>;
  lastUpdated: number | null;
}

/**
 * Poll an async loader on a fixed interval. `key` is a stable dependency key
 * (e.g. the asset id) — when it changes the data resets and polling restarts.
 * Pass `intervalMs = 0` for a single fetch. `enabled = false` pauses polling.
 */
export function usePolling<T>(
  loader: () => Promise<T>,
  intervalMs: number,
  key: string = "",
  enabled: boolean = true,
): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const alive = useRef(true);

  const run = useCallback(async () => {
    try {
      const result = await loaderRef.current();
      if (!alive.current) return;
      setData(result);
      setError(null);
      setLastUpdated(Date.now());
    } catch (e) {
      if (!alive.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    setData(null);
    setError(null);
    setLoading(true);
    if (!enabled) {
      setLoading(false);
      return () => {
        alive.current = false;
      };
    }
    void run();
    // Skip ticks while the tab is hidden (five screens polling in background tabs would
    // otherwise keep the control plane busy for nobody); refresh as soon as it is visible again.
    const tick = () => {
      if (typeof document !== "undefined" && document.hidden) return;
      void run();
    };
    const onVisible = () => {
      if (typeof document !== "undefined" && !document.hidden) void run();
    };
    const id = intervalMs > 0 ? window.setInterval(tick, intervalMs) : undefined;
    if (id !== undefined) document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive.current = false;
      if (id !== undefined) {
        window.clearInterval(id);
        document.removeEventListener("visibilitychange", onVisible);
      }
    };
  }, [run, intervalMs, key, enabled]);

  return { data, error, loading, refresh: run, lastUpdated };
}
