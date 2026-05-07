import React from "react";

function Card({ children, className = "" }) {
    return <div className={`glass-card overflow-hidden ${className}`}>{children}</div>;
}

function CardContent({ children, className = "" }) {
    return <div className={`p-5 ${className}`}>{children}</div>;
}

const GALLERY_IMAGES = [
    {
        id: "loss-curve",
        src: "/assets/graphs/training_loss_curve.png",
        title: "Model Training Loss",
        description: "Convergence of the Self-Supervised Graph Neural Network over training epochs, demonstrating stable learning of the polymer structural embeddings.",
    },
    {
        id: "umap-delta",
        src: "/assets/graphs/umap_colored_by_delta_energy.png",
        title: "UMAP Projection (Colored by ΔE)",
        description: "A 2D manifold of all polymer structures. Points are colored by their stabilization energy, revealing clear 'basins' of low-energy motifs.",
    },
    {
        id: "delta-energy-histogram",
        src: "/assets/graphs/task31_delta_energy_histogram.png",
        title: "Delta Energy Distribution",
        description: "A histogram displaying the overall distribution of stabilization energy (ΔE) across the dataset, highlighting the prevalence of stable configurations.",
    },
    {
        id: "pca-projection",
        src: "/assets/graphs/pca_colored_by_delta_energy.png",
        title: "PCA Projection (Baseline)",
        description: "A Principal Component Analysis projection colored by stabilization energy, serving as a baseline linear manifold to compare against the more expressive UMAP embeddings.",
    },
];

export default function StaticGallery() {
    return (
        <div className="space-y-8">
            <div className="max-w-3xl">
                <h1 className="text-3xl font-display font-bold tracking-tight text-slate-900">
                    Analysis Visuals Gallery
                </h1>
                <p className="mt-4 text-lg text-slate-600 leading-relaxed">
                    A collection of key static figures and analytical plots generated during the motif discovery pipeline.
                </p>
            </div>

            <div className="grid gap-8 md:grid-cols-2">
                {GALLERY_IMAGES.map((img) => (
                    <Card key={img.id}>
                        <div className="aspect-[4/3] w-full bg-slate-50 flex items-center justify-center p-4 border-b border-slate-100">
                            <img
                                src={img.src}
                                alt={img.title}
                                className="max-w-full max-h-full object-contain rounded drop-shadow-sm mix-blend-multiply"
                            />
                        </div>
                        <CardContent>
                            <h3 className="text-lg font-semibold text-slate-800">{img.title}</h3>
                            <p className="mt-2 text-sm text-slate-600 leading-relaxed">
                                {img.description}
                            </p>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    );
}
