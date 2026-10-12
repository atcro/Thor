/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Built files go under /static/: the SPA route /assets/:id (Asset 360) must survive a
  // browser refresh, and nginx serves /assets/ as files only when it is the asset directory.
  build: { assetsDir: "static" },
  server: { port: 5173 },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: false,
  },
});
