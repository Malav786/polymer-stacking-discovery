import { useEffect, useMemo, useState } from "react";
import PlotlyComponent from "react-plotly.js";

const Plot = PlotlyComponent.default || PlotlyComponent;

const API_BASE_URL = "http://127.0.0.1:8000";

async function fetchJson(path) {
    const response = await fetch(`${API_BASE_URL}${path}`);

    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }

    return response.json();
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

function MetricCard({ label, value, hint }) {
    return (
        <Card>
            <CardContent className="p-6">
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wider">{label}</p>
                <p className="mt-2 font-display text-3xl font-bold tracking-tight text-slate-900">{value}</p>
                {hint ? <p className="mt-1.5 text-xs font-medium text-slate-400">{hint}</p> : null}
            </CardContent>
        </Card>
    );
}

function finiteNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function mean(values) {
    if (!values.length) return null;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function quantile(sortedValues, q) {
    if (!sortedValues.length) return null;
    if (sortedValues.length === 1) return sortedValues[0];

    const pos = (sortedValues.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;

    if (sortedValues[base + 1] !== undefined) {
        return sortedValues[base] + rest * (sortedValues[base + 1] - sortedValues[base]);
    }

    return sortedValues[base];
}

function formatNumber(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return Number(value).toFixed(digits);
}

function buildHistogram(values, binCount = 20) {
    if (!values.length) return [];

    const min = Math.min(...values);
    const max = Math.max(...values);

    if (min === max) {
        return [
            {
                x0: min,
                x1: max,
                count: values.length,
            },
        ];
    }

    const width = (max - min) / binCount;
    const bins = Array.from({ length: binCount }, (_, i) => ({
        x0: min + i * width,
        x1: min + (i + 1) * width,
        count: 0,
    }));

    for (const value of values) {
        let idx = Math.floor((value - min) / width);
        if (idx === binCount) idx = binCount - 1;
        bins[idx].count += 1;
    }

    return bins;
}

function getEnergyColor(value, minValue, maxValue) {
    if (!Number.isFinite(value) || !Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
        return "rgb(100 116 139)";
    }

    const span = maxValue - minValue || 1;
    const t = Math.max(0, Math.min(1, (value - minValue) / span));

    const r = Math.round(59 + t * (239 - 59));
    const g = Math.round(130 + t * (68 - 130));
    const b = Math.round(246 + t * (68 - 246));

    return `rgb(${r} ${g} ${b})`;
}

function HistogramChart({ values, title, summary = "Displays the frequency distribution of stabilization energy (ΔE) across the dataset. A peak indicates the most typical stability range." }) {
    return (
        <Card className="group relative">
            <CardHeader>
                <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent className="relative min-h-[340px] p-2">
                <Plot
                    data={[
                        {
                            x: values,
                            type: 'histogram',
                            marker: { color: 'rgba(124, 58, 237, 0.75)' }
                        }
                    ]}
                    layout={{
                        autosize: true,
                        margin: { l: 50, r: 20, t: 20, b: 40 },
                        xaxis: { title: 'ΔE (Energy Difference)' },
                        yaxis: { title: 'Number of Structures' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '320px' }}
                    config={{ displayModeBar: false }}
                />
            </CardContent>
        </Card>
    );
}

function EnergyScatter({ points, title, summary = "Visualizes structural embeddings colored heavily by their delta energy. Darker blue indicates higher stability (low ΔE), while darker red reveals correspondingly lower stability." }) {
    const valid = (points || []).filter(
        (p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.delta_energy)
    );

    return (
        <Card className="group relative">
            <CardHeader>
                <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent className="relative min-h-[440px] p-2">
                <Plot
                    data={[
                        {
                            x: valid.map(p => p.x),
                            y: valid.map(p => p.y),
                            text: valid.map(p => `${p.structure_id}<br>ΔE: ${p.delta_energy?.toFixed(4)}<br>Cluster: ${p.cluster_label}`),
                            mode: 'markers',
                            type: 'scatter',
                            marker: {
                                size: 6,
                                color: valid.map(p => p.delta_energy),
                                colorscale: [
                                    [0, 'rgb(59, 130, 246)'],
                                    [1, 'rgb(239, 68, 68)']
                                ],
                                colorbar: {
                                    title: 'ΔE',
                                    thickness: 15,
                                    len: 0.8
                                },
                                opacity: 0.85
                            },
                            hoverinfo: 'text'
                        }
                    ]}
                    layout={{
                        autosize: true,
                        margin: { l: 50, r: 20, t: 20, b: 40 },
                        xaxis: { title: 'Projection Dimension 1' },
                        yaxis: { title: 'Projection Dimension 2' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '420px' }}
                    config={{ displayModeBar: false }}
                />
            </CardContent>
        </Card>
    );
}

function ClusterBoxplot({ rows, topN = 10, summary = "Details the spread and median of energy values for highly populated clustering motifs, establishing which structural families are statistically most stable." }) {
    const clusterStats = Object.values(
        rows.reduce((acc, row) => {
            const label = String(row.cluster_label ?? "unknown");
            const delta = finiteNumber(row.delta_energy);
            if (delta === null) return acc;
            if (!acc[label]) acc[label] = { label, values: [] };
            acc[label].values.push(delta);
            return acc;
        }, {})
    ).sort((a, b) => mean(a.values) - mean(b.values)).slice(0, topN);

    const plotData = clusterStats.map(c => ({
        y: c.values,
        type: 'box',
        name: `Cluster ${c.label}`,
        boxpoints: false,
        marker: { color: '#3b82f6' }
    }));

    return (
        <Card className="group relative">
            <CardHeader>
                <CardTitle>Cluster vs ΔE Boxplots</CardTitle>
            </CardHeader>
            <CardContent className="relative min-h-[400px] p-2">
                <Plot
                    data={plotData}
                    layout={{
                        autosize: true,
                        margin: { l: 50, r: 20, t: 20, b: 50 },
                        xaxis: { title: 'Cluster Label (Motif Families)' },
                        yaxis: { title: 'ΔE (Energy Difference)' },
                        showlegend: false,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '380px' }}
                    config={{ displayModeBar: false }}
                />
            </CardContent>
        </Card>
    );
}

function TopMotifCards({ clusters }) {
    return (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {clusters.map((cluster) => (
                <Card key={cluster.cluster_label}>
                    <CardContent className="p-5">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-sm text-slate-500">Motif Cluster</p>
                                <p className="mt-2 text-2xl font-semibold text-slate-900">
                                    {cluster.cluster_label}
                                </p>
                            </div>
                            <div className="rounded-xl bg-slate-100 px-3 py-2 text-xs font-semibold text-slate-700">
                                Top Stable
                            </div>
                        </div>

                        <div className="mt-4 space-y-1 text-sm text-slate-600">
                            <p>Mean ΔE: {formatNumber(cluster.mean_delta)}</p>
                            <p>Best ΔE: {formatNumber(cluster.min_delta)}</p>
                            <p>Members: {cluster.count}</p>
                        </div>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}

export default function EnergyAnalysis() {
    const [method, setMethod] = useState("umap");
    const [rows, setRows] = useState([]);
    const [mapRows, setMapRows] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        let active = true;

        async function loadData() {
            try {
                setLoading(true);
                setError("");

                const [searchRows, embeddingRows] = await Promise.all([
                    fetchJson("/search?limit=500"),
                    fetchJson(`/embedding-map?method=${method}&limit=3000`),
                ]);

                if (!active) return;

                setRows(searchRows);
                setMapRows(embeddingRows);
            } catch (err) {
                if (!active) return;
                setRows([]);
                setMapRows([]);
                setError(err.message || "Failed to load energy dashboard");
            } finally {
                if (active) setLoading(false);
            }
        }

        loadData();

        return () => {
            active = false;
        };
    }, [method]);

    const stats = useMemo(() => {
        const deltaValues = rows.map((r) => finiteNumber(r.delta_energy)).filter((v) => v !== null);
        const energyValues = rows.map((r) => finiteNumber(r.energy)).filter((v) => v !== null);

        const meanDelta = deltaValues.length ? mean(deltaValues) : null;
        const minDelta = deltaValues.length ? Math.min(...deltaValues) : null;
        const maxDelta = deltaValues.length ? Math.max(...deltaValues) : null;
        const meanEnergy = energyValues.length ? mean(energyValues) : null;

        const clusterGroups = Object.values(
            rows.reduce((acc, row) => {
                const label = String(row.cluster_label ?? "unknown");
                const delta = finiteNumber(row.delta_energy);

                if (delta === null) return acc;

                if (!acc[label]) {
                    acc[label] = {
                        cluster_label: label,
                        values: [],
                    };
                }

                acc[label].values.push(delta);
                return acc;
            }, {})
        )
            .map((group) => ({
                cluster_label: group.cluster_label,
                count: group.values.length,
                mean_delta: mean(group.values),
                min_delta: Math.min(...group.values),
            }))
            .sort((a, b) => a.mean_delta - b.mean_delta);

        return {
            structureCount: rows.length,
            meanEnergy,
            meanDelta,
            minDelta,
            maxDelta,
            topClusters: clusterGroups.slice(0, 4),
            deltaValues,
        };
    }, [rows]);

    const coloredProjection = useMemo(() => {
        const byId = new Map(rows.map((row) => [row.structure_id, row]));

        return mapRows
            .map((point) => {
                const match = byId.get(point.structure_id);
                return {
                    ...point,
                    delta_energy: finiteNumber(match?.delta_energy),
                    cluster_label: match?.cluster_label,
                };
            })
            .filter((point) => Number.isFinite(point.delta_energy));
    }, [rows, mapRows]);

    return (
        <div className="space-y-6">
            <Card>
                <CardContent className="flex flex-col gap-3 p-5 md:flex-row md:items-center">
                    <select
                        value={method}
                        onChange={(e) => setMethod(e.target.value)}
                        className="input-field max-w-xs"
                    >
                        <option value="umap">UMAP</option>
                        <option value="pca">PCA</option>
                        <option value="tsne">t-SNE</option>
                    </select>

                    <p className="text-sm text-slate-500">
                        Change the projection method to inspect energy organization in latent space.
                    </p>
                </CardContent>
            </Card>

            {loading ? <p className="text-sm text-slate-500">Loading energy dashboard...</p> : null}
            {error ? <p className="text-sm text-red-600">{error}</p> : null}

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                    label="Structures Loaded"
                    value={stats.structureCount}
                    hint="Rows from /search"
                />
                <MetricCard
                    label="Average Energy"
                    value={formatNumber(stats.meanEnergy)}
                    hint="Mean raw energy"
                />
                <MetricCard
                    label="Average ΔE"
                    value={formatNumber(stats.meanDelta)}
                    hint="Mean stability value"
                />
                <MetricCard
                    label="ΔE Range"
                    value={`${formatNumber(stats.minDelta, 3)} to ${formatNumber(stats.maxDelta, 3)}`}
                    hint="Min to max delta energy"
                />
            </div>

            <TopMotifCards clusters={stats.topClusters} />

            <div className="grid gap-6 xl:grid-cols-[1.15fr_1fr]">
                <HistogramChart
                    values={stats.deltaValues}
                    title="Histogram of ΔE"
                />

                <Card>
                    <CardHeader>
                        <CardTitle>Low-Energy Motif Summary</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3 text-sm text-slate-600">
                        {stats.topClusters.length === 0 ? (
                            <p>No cluster summaries available.</p>
                        ) : (
                            stats.topClusters.map((cluster) => (
                                <div key={cluster.cluster_label} className="rounded-xl border p-3">
                                    <p className="font-medium text-slate-900">
                                        Cluster {cluster.cluster_label}
                                    </p>
                                    <p className="mt-1">
                                        This motif currently looks among the most stable groups based on mean ΔE.
                                    </p>
                                    <p className="mt-1 text-xs text-slate-500">
                                        Mean ΔE {formatNumber(cluster.mean_delta)} · Best ΔE {formatNumber(cluster.min_delta)} · Members {cluster.count}
                                    </p>
                                </div>
                            ))
                        )}
                    </CardContent>
                </Card>
            </div>

            <ClusterBoxplot rows={rows} topN={10} />

            <EnergyScatter
                points={coloredProjection}
                title={`${method.toUpperCase()} Colored by ΔE`}
            />
        </div>
    );
}