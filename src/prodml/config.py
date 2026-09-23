from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    data_path: Path = PROJECT_ROOT / "data" / "green_tripdata_2026-01.parquet"
    model_path: Path = PROJECT_ROOT / "models" / "model.pkl"

    validation_size: float = 0.2
    random_state: int = 42

    min_duration: float = 1.0
    max_duration: float = 60.0

    min_trip_distance: float = 0.0
    max_trip_distance: float = 50.0

    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
