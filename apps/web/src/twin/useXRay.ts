/**
 * X-Ray mode state: the toggle (persisted per viewer in localStorage `thor.xray`) and which
 * asset's HUD is open. Shared by the 3-D Digital Twin and the Fleet.
 *
 * The state lives in a tiny external store so every `useXRay()` caller sees the same value.
 * `XRayProvider` scopes a fresh store to a subtree (App.tsx mounts one app-wide; tests mount
 * one per render). Without a provider the hook falls back to a module-level default store, so
 * a screen rendered on its own still works.
 */
import { createContext, createElement, useContext, useMemo, useSyncExternalStore, type ReactNode } from "react";

export const XRAY_STORAGE_KEY = "thor.xray";

export interface XRayState {
  enabled: boolean;
  openedAsset: string | null;
}

export interface XRayApi extends XRayState {
  toggle: () => void;
  set: (enabled: boolean) => void;
  openAsset: (assetId: string) => void;
  close: () => void;
}

interface XRayStore {
  get: () => XRayState;
  subscribe: (cb: () => void) => () => void;
  api: Omit<XRayApi, keyof XRayState>;
}

function readEnabled(): boolean {
  try {
    return localStorage.getItem(XRAY_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function saveEnabled(on: boolean): void {
  try {
    localStorage.setItem(XRAY_STORAGE_KEY, on ? "1" : "0");
  } catch {
    /* private mode or blocked storage: the toggle still works for this page view */
  }
}

export function createXRayStore(initial?: Partial<XRayState>): XRayStore {
  let state: XRayState = { enabled: initial?.enabled ?? readEnabled(), openedAsset: initial?.openedAsset ?? null };
  const listeners = new Set<() => void>();
  const update = (patch: Partial<XRayState>) => {
    state = { ...state, ...patch };
    for (const l of listeners) l();
  };
  const set = (enabled: boolean) => {
    saveEnabled(enabled);
    // Turning X-Ray off also closes the HUD: the HUD is an X-Ray surface.
    update(enabled ? { enabled } : { enabled, openedAsset: null });
  };
  return {
    get: () => state,
    subscribe: (cb) => {
      listeners.add(cb);
      return () => {
        listeners.delete(cb);
      };
    },
    api: {
      set,
      toggle: () => set(!state.enabled),
      openAsset: (assetId: string) => update({ openedAsset: assetId }),
      close: () => update({ openedAsset: null }),
    },
  };
}

let defaultStore: XRayStore | null = null;
function getDefaultStore(): XRayStore {
  if (!defaultStore) defaultStore = createXRayStore();
  return defaultStore;
}

const XRayContext = createContext<XRayStore | null>(null);

export function XRayProvider({ children, initial }: { children: ReactNode; initial?: Partial<XRayState> }) {
  const store = useMemo(() => createXRayStore(initial), [initial]);
  return createElement(XRayContext.Provider, { value: store }, children);
}

export function useXRay(): XRayApi {
  const ctx = useContext(XRayContext);
  const store = ctx ?? getDefaultStore();
  const state = useSyncExternalStore(store.subscribe, store.get, store.get);
  return useMemo(() => ({ ...state, ...store.api }), [state, store]);
}
