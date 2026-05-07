from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm
from ase.io import read


PROJECT_ROOT = Path.cwd().resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPECTED_CIF_COUNT = 2916
STABLE_ENERGY = -936.6398

INVENTORY_CSV = OUTPUT_DIR / "dataset_inventory.csv"
MASTER_CSV = OUTPUT_DIR / "dataset_master.csv"
ENERGY_CSV = DATA_ROOT / "file_energy.csv"


def rel_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_cif_safe(cif_path: Path):
    try:
        atoms = read(str(cif_path))
        if atoms is None or len(atoms) == 0:
            return None, "empty_or_unparsed"
        return atoms, "ok"
    except Exception as e:
        return None, f"parse_error: {type(e).__name__}: {e}"


def parse_cif_metadata(cif_path: Path, data_root: Path) -> dict[str, float | None]:
    rel_parts = cif_path.resolve().relative_to(data_root.resolve()).parts
    lower_rotation = displacement = upper_rotation = None

    if len(rel_parts) >= 3:
        r_folder = rel_parts[-3]
        t_folder = rel_parts[-2]
        filename = Path(rel_parts[-1]).stem

        m_r = re.fullmatch(r"r(-?\d+(?:\.\d+)?)", r_folder, flags=re.IGNORECASE)
        m_t = re.fullmatch(r"t(-?\d+(?:\.\d+)?)", t_folder, flags=re.IGNORECASE)
        m_f = re.fullmatch(
            r"t(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)",
            filename,
            flags=re.IGNORECASE,
        )

        if m_r:
            lower_rotation = float(m_r.group(1))
        if m_t:
            displacement = float(m_t.group(1))
        if m_f:
            if displacement is None:
                displacement = float(m_f.group(1))
            upper_rotation = float(m_f.group(2))

    return {
        "lower_rotation": lower_rotation,
        "displacement": displacement,
        "upper_rotation": upper_rotation,
    }


def make_structure_id(
    lower_rotation: float | None,
    displacement: float | None,
    upper_rotation: float | None,
    idx: int,
) -> str:
    if None not in (lower_rotation, displacement, upper_rotation):
        return f"L{lower_rotation:g}_D{displacement:g}_U{upper_rotation:g}"
    return f"STRUCT_{idx:04d}"


def build_dataset_inventory(data_root: Path) -> pd.DataFrame:
    if not data_root.exists():
        raise FileNotFoundError(f"Data folder not found: {data_root}")

    cif_files = sorted(data_root.rglob("*.cif"))
    inventory_rows = []

    for idx, cif_path in enumerate(tqdm(cif_files, desc="Task 6: scanning CIFs"), start=1):
        _, _ = read_cif_safe(cif_path)
        meta = parse_cif_metadata(cif_path, data_root)

        inventory_rows.append(
            {
                "structure_id": make_structure_id(
                    meta["lower_rotation"],
                    meta["displacement"],
                    meta["upper_rotation"],
                    idx,
                ),
                "relative_cif_path": rel_to(cif_path, data_root),
                "lower_rotation": meta["lower_rotation"],
                "displacement": meta["displacement"],
                "upper_rotation": meta["upper_rotation"],
            }
        )

    df_inventory = pd.DataFrame(inventory_rows).sort_values(
        ["lower_rotation", "displacement", "upper_rotation"],
        na_position="last",
    ).reset_index(drop=True)

    return df_inventory


def build_dataset_master(
    df_inventory: pd.DataFrame,
    energy_csv: Path,
    stable_energy: float,
) -> pd.DataFrame:
    df_energy = pd.read_csv(energy_csv).rename(
        columns={
            "r_bottom": "lower_rotation",
            "r_upper": "upper_rotation",
            "energy_values_experimental": "energy",
        }
    )

    if "Unnamed: 0" in df_energy.columns:
        df_energy = df_energy.drop(columns="Unnamed: 0")

    for col in ["lower_rotation", "displacement", "upper_rotation", "energy"]:
        df_energy[col] = pd.to_numeric(df_energy[col], errors="coerce")

    df_energy["delta_energy"] = df_energy["energy"] - stable_energy

    df_master = df_inventory.merge(
        df_energy[
            ["lower_rotation", "displacement", "upper_rotation", "energy", "delta_energy"]
        ],
        on=["lower_rotation", "displacement", "upper_rotation"],
        how="left",
        validate="one_to_one",
    )

    df_master = df_master[
        [
            "structure_id",
            "relative_cif_path",
            "lower_rotation",
            "displacement",
            "upper_rotation",
            "energy",
            "delta_energy",
        ]
    ]

    return df_master


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_inventory = build_dataset_inventory(DATA_ROOT)
    df_inventory.to_csv(INVENTORY_CSV, index=False)

    print("=" * 60)
    print("DATASET INVENTORY SUMMARY")
    print("=" * 60)
    print(f"Total CIF files found  : {len(df_inventory)}")
    print(f"Expected CIF count     : {EXPECTED_CIF_COUNT}")
    print(f"Count match            : {len(df_inventory) == EXPECTED_CIF_COUNT}")
    print(f"Saved inventory to     : {INVENTORY_CSV}")
    print("=" * 60)

    df_master = build_dataset_master(df_inventory, ENERGY_CSV, STABLE_ENERGY)
    df_master.to_csv(MASTER_CSV, index=False)

    print("=" * 60)
    print("DATASET MASTER SUMMARY")
    print("=" * 60)
    print(f"Rows                    : {len(df_master)}")
    print(f"Missing energy rows     : {df_master['energy'].isna().sum()}")
    print(f"Missing delta rows      : {df_master['delta_energy'].isna().sum()}")
    print(f"Saved master to         : {MASTER_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()