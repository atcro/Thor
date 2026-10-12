/** WebGL availability probe. The twin falls back to the Dashboard when this is false. */
export function supportsWebGL(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    return gl !== null && gl !== undefined;
  } catch {
    return false;
  }
}
