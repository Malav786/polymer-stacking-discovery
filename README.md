# 🔬 Polymer Stacking Discovery Platform

> **Self-Supervised Graph Neural Network Pipeline for Structural Motif Discovery in Polymer Crystal Packing**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C.svg)](https://pytorch.org/)

---

## 📖 Abstract

This project presents a **full-stack computational materials science platform** for discovering structural motifs in polymer crystal packing configurations. The pipeline converts **Crystallographic Information Files (CIF)** into molecular graphs, trains a **GINEConv-based Self-Supervised Graph Neural Network** using contrastive learning (NT-Xent loss), and generates rich 128-dimensional embeddings that capture structural similarity.

Key capabilities include:
- **CIF → Graph Conversion**: Automated parsing of crystal structures into PyTorch Geometric graph objects with atom-level node features and bond-level edge attributes
- **Self-Supervised Learning**: Contrastive pre-training using augmented graph pairs — no labeled data required
- **Structural Motif Discovery**: UMAP dimensionality reduction + HDBSCAN clustering to identify recurring packing patterns
- **Energy-Aware Analysis**: Correlation of discovered motifs with computed stacking energies (ΔE) to validate physical relevance
- **Similarity Search**: Cosine-similarity-based retrieval of structurally analogous polymers from the learned embedding space
- **Interactive Dashboard**: React-based frontend with 3D CIF visualization, Plotly charts, and real-time API-driven exploration

---

## 🏗️ Architecture

```
masters_project/
├── backend/                    # FastAPI + PyTorch backend
│   ├── pyproject.toml          # Python dependencies (uv/pip)
│   ├── .env.example            # Environment variable template
│   └── src/
│       ├── api/                # REST API (FastAPI)
│       │   ├── main.py         # App entry point
│       │   ├── schemas.py      # Pydantic models
│       │   └── routes/         # Endpoint handlers
│       ├── core/config.py      # Settings & configuration
│       ├── db/                 # SQLAlchemy data layer
│       ├── models/             # GNN encoder, inference, similarity
│       └── pipeline/           # Data processing scripts
├── frontend/                   # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx             # Main app with routing
│   │   ├── pages/              # Dashboard views
│   │   ├── components/         # Reusable UI (3D viewer, toolbar)
│   │   └── lib/api.js          # API client
│   └── public/assets/          # Static visualization assets
├── notebooks/                  # Jupyter research notebooks (01–08)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **Node.js 18+** & **npm**
- **uv** (recommended) or **pip** for Python dependency management
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/Malav786/polymer-stacking-discovery.git
cd polymer-stacking-discovery
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment and install dependencies (using uv — recommended)
uv venv
uv sync

# OR using pip
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -e .

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your local paths if needed
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd ../frontend

# Install Node.js dependencies
npm install
```

### 4. Data Setup

> **Note:** The raw CIF dataset and pre-computed outputs are **not included** in this repository due to size constraints. Contact the author or refer to the original dataset source to obtain the `data/` directory. Place it at the project root:
>
> ```
> masters_project/
> ├── data/
> │   ├── file_energy.csv
> │   ├── polymer_metadata.db
> │   └── r0/ r20/ r40/ ... r340/   ← CIF structure folders
> ```

---

## ▶️ Running the Application

Open **two terminals** and run:

**Terminal 1 — Backend API:**
```bash
cd backend
uv run uvicorn src.api.main:app --reload
```
> Backend runs at `http://localhost:8000` · API docs at `http://localhost:8000/docs`

**Terminal 2 — Frontend Dev Server:**
```bash
cd frontend
npm run dev
```
> Frontend runs at `http://localhost:5173`

---

## 📓 Research Notebooks

The `notebooks/` directory contains the complete research pipeline:

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `build_dataset_inventory` | Dataset cataloging & inventory |
| 02 | `cif_parse_to_graph` | CIF → PyG graph conversion |
| 03 | `EDA_visuals` | Exploratory data analysis |
| 04 | `features` | SSL-GNN training & embedding generation |
| 05 | `structural_motif` | UMAP + HDBSCAN motif discovery |
| 06 | `energy_aware_scientific_interpretation` | Energy–structure correlation |
| 07 | `similarity_search_and_discovery` | Cosine similarity retrieval |
| 08 | `experiments_validation` | Full experimental validation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch, PyTorch Geometric |
| **GNN Architecture** | GINEConv (Graph Isomorphism Network with Edge features) |
| **SSL Objective** | NT-Xent (Normalized Temperature-scaled Cross-Entropy) |
| **Clustering** | UMAP + HDBSCAN |
| **Backend API** | FastAPI, SQLAlchemy, Pydantic |
| **Frontend** | React 19, Vite, Tailwind CSS |
| **Visualization** | Plotly.js, 3Dmol.js |
| **Data** | CIF files, SQLite, Pandas |

---

## 📄 License

This project was developed as a **Master's Capstone Project**. All rights reserved.

---

## 👤 Author

**Malav Champaneria**
Master's in Computer Science
