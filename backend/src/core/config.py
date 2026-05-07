from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

# We assume the backend directory is masters_project/backend
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

class Settings(BaseSettings):
    api_title: str = "Polymer Stacking Discovery API"
    api_version: str = "1.0.0"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    
    # Paths
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    outputs_dir: Path = Field(default=PROJECT_ROOT / "outputs")
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "polymer_metadata.db")
    
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

settings = Settings()
