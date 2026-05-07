import React, { useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import MotifExplorer from "./pages/MotifExplorer";
import EnergyAnalysis from "./pages/EnergyAnalysis";
import SimilaritySearch from "./pages/SimilaritySearch";
import StaticGallery from "./pages/StaticGallery";
import * as $3Dmol from "3dmol";

const API_BASE = "http://127.0.0.1:8000";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function fetchText(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const text = await response.text();

  if (!response.ok) {
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return text;
}

function Card({ children, className = "" }) {
  return <div className={`glass-card ${className}`}>{children}</div>;
}

function CardHeader({ children, className = "" }) {
  return <div className={`border-b border-slate-200/50 p-5 ${className}`}>{children}</div>;
}

function CardTitle({ children, className = "" }) {
  return <h2 className={`font-display text-lg font-semibold text-slate-800 tracking-tight ${className}`}>{children}</h2>;
}

function CardContent({ children, className = "" }) {
  return <div className={`p-5 ${className}`}>{children}</div>;
}

function Button({ children, className = "", variant = "primary", ...props }) {
  const baseClass = variant === "primary" ? "btn-primary" : "btn-secondary";
  return (
    <button
      {...props}
      className={`${baseClass} ${className}`}
    >
      {children}
    </button>
  );
}

function Input(props) {
  return <input {...props} className="input-field" />;
}

function StatCard({ title, value, hint, badge }) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-slate-500 uppercase tracking-wider">{title}</p>
            <p className="mt-2 font-display text-3xl font-bold tracking-tight text-slate-900">{value}</p>
            {hint ? <p className="mt-1.5 text-xs font-medium text-slate-400">{hint}</p> : null}
          </div>
          <div className="stat-card-badge">
            {badge}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}


function SimpleScatter({ points, title, summary="This scatter plot visualizes the dataset representation in a 2-dimensional latent space. Clusters highlight groups of similar structural arrangements." }) {
  const width = 760;
  const height = 380;
  const padding = 50;

  const valid = (points || []).filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y)
  );

  let minX = 0;
  let maxX = 1;
  let minY = 0;
  let maxY = 1;

  if (valid.length > 0) {
    minX = Math.min(...valid.map((p) => p.x));
    maxX = Math.max(...valid.map((p) => p.x));
    minY = Math.min(...valid.map((p) => p.y));
    maxY = Math.max(...valid.map((p) => p.y));
  }

  const scaleX = (x) => {
    const denom = maxX - minX || 1;
    return padding + ((x - minX) / denom) * (width - padding * 2);
  };

  const scaleY = (y) => {
    const denom = maxY - minY || 1;
    return height - padding - ((y - minY) / denom) * (height - padding * 2);
  };

  return (
    <Card className="group relative">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto min-h-[400px] relative">
        <svg width={width} height={height} className="rounded-xl border border-slate-200/50 bg-white/50 backdrop-blur-sm">
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cbd5e1" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#cbd5e1" />
          
          <text x={width / 2} y={height - 15} textAnchor="middle" className="text-xs fill-slate-500 font-medium tracking-wide shadow-sm">Dimension 1</text>
          <text x={18} y={height / 2} textAnchor="middle" transform={`rotate(-90, 18, ${height/2})`} className="text-xs fill-slate-500 font-medium tracking-wide shadow-sm">Dimension 2</text>

          {valid.map((p, i) => (
            <circle
              key={`${p.structure_id || "point"}-${i}`}
              cx={scaleX(p.x)}
              cy={scaleY(p.y)}
              r="3.5"
              fill="rgba(124, 58, 237, 0.6)"
              className="transition-all hover:r-5 hover:fill-brand-600 hover:opacity-100 cursor-pointer"
            >
              <title>{`${p.structure_id || "point"} (${p.method || "map"})`}</title>
            </circle>
          ))}
        </svg>

        <div className="absolute top-8 right-8 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-10">
           <div className="bg-white/95 backdrop-blur-md p-3.5 rounded-xl shadow-glass border border-slate-200/60 max-w-[240px]">
              <p className="font-semibold text-slate-800 text-xs tracking-tight">Graph Overview</p>
              <p className="text-[11px] text-slate-600 mt-1 leading-relaxed">{summary}</p>
           </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AppShell({ page, setPage, children }) {
  const tabs = [
    ["dashboard", "Dashboard"],
    ["structure", "Structure Explorer"],
    ["motif", "Motif Explorer"],
    ["similarity", "Similarity Search"],
    ["analysis", "Energy Analysis"],
    ["gallery", "Visual Gallery"],
  ];

  return (
    <div className="min-h-[100svh] relative overflow-hidden bg-slate-50">
      {/* Decorative gradient blobs */}
      <div className="pointer-events-none absolute -top-[40%] -left-[20%] w-[70%] h-[70%] rounded-full bg-brand-400/20 mix-blend-multiply blur-[120px] animate-float opacity-70"></div>
      <div className="pointer-events-none absolute -top-[60%] -right-[10%] w-[60%] h-[60%] rounded-full bg-indigo-400/20 mix-blend-multiply blur-[120px] animate-float opacity-70" style={{animationDelay: '2s'}}></div>
      
      <div className="dashboard-container relative z-10 space-y-8">
        <header className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between pb-6 border-b border-slate-200/60">
          <div>
            <h1 className="hero-title">
              Polymer Discovery
            </h1>
            <p className="mt-2 text-base text-slate-600 font-medium">
              Advanced structure embeddings, motifs, and energy analysis platform.
            </p>
          </div>

          <nav className="flex flex-wrap gap-2 rounded-2xl bg-white/60 p-1.5 backdrop-blur-md shadow-sm ring-1 ring-slate-200/50">
            {tabs.map(([key, label]) => (
              <button
                key={key}
                onClick={() => setPage(key)}
                className={`px-4 py-2 text-sm font-semibold rounded-xl transition-all duration-200 ${
                  page === key 
                    ? "bg-white text-brand-700 shadow-sm ring-1 ring-slate-200" 
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-50/50 hover:text-brand-600"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </header>

        <main className="animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
          {children}
        </main>
      </div>
    </div>
  );
}

function DashboardPage() {
  const [searchRows, setSearchRows] = useState([]);
  const [umap, setUmap] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        setLoading(true);
        const [rows, map] = await Promise.all([
          fetchJson("/search?limit=500"),
          fetchJson("/embedding-map?method=umap&limit=3000"),
        ]);

        if (!active) return;

        setSearchRows(rows);
        setUmap(map);
        setError("");
      } catch (err) {
        if (!active) return;
        setError(err.message || "Failed to load dashboard");
      } finally {
        if (active) setLoading(false);
      }
    }

    loadDashboard();

    return () => {
      active = false;
    };
  }, []);

  const clusterCount = new Set(searchRows.map((r) => r.cluster_label)).size;
  const avgDelta =
    searchRows.length > 0
      ? (
        searchRows.reduce((sum, r) => sum + (r.delta_energy ?? 0), 0) /
        searchRows.length
      ).toFixed(4)
      : "-";

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Structures"
          value={searchRows.length || "-"}
          hint="Loaded from /search"
          badge="DB"
        />
        <StatCard
          title="Clusters"
          value={clusterCount || "-"}
          hint="Distinct cluster labels"
          badge="CL"
        />
        <StatCard
          title="Avg ΔE"
          value={avgDelta}
          hint="Mean delta energy"
          badge="DE"
        />
        <StatCard
          title="Map Points"
          value={umap.length || "-"}
          hint="UMAP coordinates"
          badge="MAP"
        />
      </div>

      {loading ? <p className="text-sm text-slate-500">Loading dashboard...</p> : null}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}

      <div className="grid gap-6 xl:grid-cols-[2fr_1fr]">
        <SimpleScatter points={umap} title="UMAP Embedding Map" />

        <Card>
          <CardHeader>
            <CardTitle>Quick Insights</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-slate-600">
            <p>This page gives a quick overview of the embedding space and dataset.</p>
            <p>Use Structure Explorer for single-structure inspection.</p>
            <p>Use Similarity Search for top-k nearest structures.</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function StructureExplorerPage() {
  const [structureId, setStructureId] = useState("L0_D0_U0");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [cifText, setCifText] = useState("");
  const [cifError, setCifError] = useState("");
  const [viewerMode, setViewerMode] = useState("atom");

  const viewerContainerRef = useRef(null);
  const viewerInstanceRef = useRef(null);

  useEffect(() => {
    if (!viewerContainerRef.current) return;

    if (!viewerInstanceRef.current) {
      viewerInstanceRef.current = $3Dmol.createViewer(viewerContainerRef.current, {
        backgroundColor: "white",
      });
    }

    const viewer = viewerInstanceRef.current;
    viewer.clear();

    if (!cifText || !cifText.trim()) {
      viewer.render();
      return;
    }

    try {
      viewer.addModel(cifText, "cif");
      const model = viewer.getModel();

      if (viewerMode === "layer") {
        const atoms = model?.selectedAtoms({}) || [];

        if (atoms.length > 0) {
          const sortedZ = [...atoms].map((a) => a.z).sort((a, b) => a - b);
          const medianZ = sortedZ[Math.floor(sortedZ.length / 2)];

          atoms.forEach((atom) => {
            const color = atom.z <= medianZ ? "#2563eb" : "#f97316";
            viewer.setStyle(
              { serial: atom.serial },
              {
                stick: { color, radius: 0.16 },
                sphere: { color, scale: 0.3 },
              }
            );
          });
        } else {
          viewer.setStyle({}, { stick: { radius: 0.16 }, sphere: { scale: 0.3 } });
        }
      } else {
        viewer.setStyle(
          { elem: "H" },
          { stick: { color: "#e5e7eb", radius: 0.1 }, sphere: { color: "#e5e7eb", scale: 0.18 } }
        );
        viewer.setStyle(
          { elem: "C" },
          { stick: { color: "#4b5563", radius: 0.16 }, sphere: { color: "#4b5563", scale: 0.28 } }
        );
        viewer.setStyle(
          { elem: "O" },
          { stick: { color: "#ef4444", radius: 0.16 }, sphere: { color: "#ef4444", scale: 0.32 } }
        );
      }

      viewer.zoomTo();
      viewer.render();
    } catch (err) {
      console.error("3D viewer render error:", err);
    }
  }, [cifText, viewerMode]);

  useEffect(() => {
    return () => {
      if (viewerInstanceRef.current) {
        try {
          viewerInstanceRef.current.clear();
          viewerInstanceRef.current.render();
        } catch {
          // ignore cleanup errors
        }
      }
    };
  }, []);

  async function loadStructure() {
    try {
      setLoading(true);
      setError("");
      setCifError("");
      setCifText("");

      const result = await fetchJson(`/structure-view/${encodeURIComponent(structureId)}`);
      setData(result);

      try {
        const cifTextData = await fetchText(`/structure/${encodeURIComponent(structureId)}/cif`);
        setCifText(cifTextData);
      } catch (cifErr) {
        setCifText("");
        setCifError(
          "CIF viewer endpoint is not available yet. Add the backend route /structure/{structure_id}/cif to enable 3D visualization."
        );
      }
    } catch (err) {
      setData(null);
      setCifText("");
      setError(err.message || "Failed to load structure");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadStructure();
  }, []);

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="flex flex-col gap-3 p-5 md:flex-row">
          <Input
            value={structureId}
            onChange={(e) => setStructureId(e.target.value)}
            placeholder="Enter structure ID"
          />
          <Button onClick={loadStructure}>
            <span className="mr-2">🔍</span>
            Load Structure
          </Button>
        </CardContent>
      </Card>

      {loading ? <p className="text-sm text-slate-500">Loading structure...</p> : null}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}

      {data ? (
        <>
          <div className="grid gap-6 xl:grid-cols-[1fr_1.4fr_1fr]">
            <Card>
              <CardHeader>
                <CardTitle>Metadata</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <p><span className="font-medium">Structure ID:</span> {data.structure_id}</p>
                <p><span className="font-medium">CIF Path:</span> {data.relative_cif_path}</p>
                <p><span className="font-medium">Lower Rotation:</span> {data.lower_rotation}</p>
                <p><span className="font-medium">Displacement:</span> {data.displacement}</p>
                <p><span className="font-medium">Upper Rotation:</span> {data.upper_rotation}</p>
                <p><span className="font-medium">Energy:</span> {data.energy}</p>
                <p><span className="font-medium">Stable Energy:</span> {data.stable_energy}</p>
                <p><span className="font-medium">ΔE:</span> {data.delta_energy}</p>
                <p><span className="font-medium">Cluster:</span> {String(data.cluster_label)}</p>
                <p><span className="font-medium">Confidence:</span> {String(data.cluster_confidence)}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>3D CIF Viewer</CardTitle>
              </CardHeader>

              <CardContent>
                <div className="mb-3 flex items-center justify-between gap-3">
                  <select
                    value={viewerMode}
                    onChange={(e) => setViewerMode(e.target.value)}
                    className="rounded-lg border px-3 py-2 text-sm"
                  >
                    <option value="atom">Atom Type</option>
                    <option value="layer">Lower vs Upper Layer</option>
                  </select>

                  <Button
                    variant="secondary"
                    onClick={() => {
                      if (viewerInstanceRef.current) {
                        viewerInstanceRef.current.zoomTo();
                        viewerInstanceRef.current.render();
                      }
                    }}
                  >
                    Reset View
                  </Button>
                </div>

                {cifError ? (
                  <div className="flex min-h-[400px] items-center justify-center rounded-xl border border-dashed p-6 text-center text-sm text-slate-500">
                    {cifError}
                  </div>
                ) : (
                  <div
                    ref={viewerContainerRef}
                    className="relative w-full overflow-hidden rounded-xl border"
                    style={{ height: "400px" }}
                  />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Projection Coordinates</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {(data.map_points || []).map((p) => (
                  <div key={p.method} className="rounded-xl border p-3">
                    <p className="font-medium uppercase tracking-wide text-slate-500">{p.method}</p>
                    <p>x: {p.x}</p>
                    <p>y: {p.y}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Nearest Neighbors</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-slate-500">
                    <th className="py-2 pr-4">Neighbor</th>
                    <th className="py-2 pr-4">Rank</th>
                    <th className="py-2 pr-4">Score</th>
                    <th className="py-2 pr-4">Cluster</th>
                    <th className="py-2 pr-4">ΔE</th>
                    <th className="py-2 pr-4">CIF Path</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.neighbors || []).map((n) => (
                    <tr key={`${n.neighbor_structure_id}-${n.rank}`} className="border-b last:border-0">
                      <td className="py-2 pr-4">{n.neighbor_structure_id}</td>
                      <td className="py-2 pr-4">{n.rank}</td>
                      <td className="py-2 pr-4">{n.similarity_score}</td>
                      <td className="py-2 pr-4">{String(n.cluster_label)}</td>
                      <td className="py-2 pr-4">{String(n.delta_energy)}</td>
                      <td className="py-2 pr-4">{n.relative_cif_path}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("dashboard");

  return (
    <AppShell page={page} setPage={setPage}>
      <div key={page}>
        {page === "dashboard" && <DashboardPage />}
        {page === "structure" && <StructureExplorerPage />}
        {page === "motif" && <MotifExplorer />}
        {page === "similarity" && <SimilaritySearch />}
        {page === "analysis" && <EnergyAnalysis />}
        {page === "gallery" && <StaticGallery />}
      </div>
    </AppShell>
  );
}