/**
 * Theme root for the 3-D Digital Twin and the X-Ray HUD.
 *
 * Sets the `--tw-*` custom properties from src/twin/palette.ts on a wrapper element with the
 * class `twin-theme`, so everything inside (scene chrome, tooltip, HUD, rewind bar) picks up
 * the twin's colour scheme while the rest of the app keeps its own (src/index.css). Switching
 * the twin back to the app's colours is the one-line `TWIN_THEME` change in palette.ts.
 */
import { useMemo, type CSSProperties, type ReactNode } from "react";
import { paletteCssVars, twinPalette } from "./palette";
import "./xray.css";

export function TwinThemeRoot({
  className = "",
  style,
  children,
  ...rest
}: {
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
  "data-testid"?: string;
  role?: string;
  "aria-label"?: string;
}) {
  const vars = useMemo(() => paletteCssVars(twinPalette()) as CSSProperties, []);
  return (
    <div className={`twin-theme${className ? ` ${className}` : ""}`} style={{ ...vars, ...style }} {...rest}>
      {children}
    </div>
  );
}
