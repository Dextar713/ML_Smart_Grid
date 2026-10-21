from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Define your environment variables here with type hints
    app_name: str = "Smart Grid Load Forecasting"
    debug: bool = True
    entsoe_api_key: str | None = None

    # model_config tells Pydantic to read from the .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        # Ignore extra variables in the .env file that aren't defined in this model
        extra="ignore" 
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Instantiates and caches the Settings object.
    Subsequent calls will return the cached instance instead of re-reading the .env file.
    """
    return Settings()