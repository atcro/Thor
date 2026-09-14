import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { LoadingState } from "./components/ui";
import { Fleet } from "./screens/Fleet";

// Chart-heavy screens are lazy so recharts is not in the Fleet/Copilot critical path.
const Asset360 = lazy(() => import("./screens/Asset360").then((m) => ({ default: m.Asset360 })));
const Studio = lazy(() => import("./screens/Studio").then((m) => ({ default: m.Studio })));
const ModelOps = lazy(() => import("./screens/ModelOps").then((m) => ({ default: m.ModelOps })));
const Copilot = lazy(() => import("./screens/Copilot").then((m) => ({ default: m.Copilot })));

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingState />}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Fleet />} />
            <Route path="/assets/:id" element={<Asset360 />} />
            <Route path="/studio" element={<Studio />} />
            <Route path="/modelops" element={<ModelOps />} />
            <Route path="/copilot" element={<Copilot />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
