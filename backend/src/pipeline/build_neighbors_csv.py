from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDINGS_CSV = Path("outputs/graph_embeddings/embeddings/structure_embeddings.csv")
OUTPUT_CSV = Path("outputs/similarity_search/neighbors.csv")
TOP_K = 10


def main() -> None:
    df = pd.read_csv(EMBEDDINGS_CSV)

    if "structure_id" not in df.columns:
        raise ValueError("structure_embeddings.csv must contain structure_id")

    embedding_cols = sorted([c for c in df.columns if c.startswith("emb_")])
    if not embedding_cols:
        raise ValueError("No embedding columns found")

    structure_ids = df["structure_id"].astype(str).tolist()
    X = df[embedding_cols].values

    sim = cosine_similarity(X)

    rows = []
    for i, query_id in enumerate(structure_ids):
        sim_scores = sim[i].copy()

        # remove self-match
        sim_scores[i] = -1.0

        top_idx = sim_scores.argsort()[::-1][:TOP_K]

        for rank, j in enumerate(top_idx, start=1):
            rows.append(
                {
                    "query_structure_id": query_id,
                    "neighbor_structure_id": structure_ids[j],
                    "rank": rank,
                    "similarity_score": float(sim_scores[j]),
                    "distance_metric": "cosine",
                }
            )

    out_df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()