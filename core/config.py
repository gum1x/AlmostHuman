from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://ci_user:ci_pass@localhost:5432/telegram_ci"
    redis_url: str = "redis://localhost:6379/0"

    tg_api_id: int = 0
    tg_api_hash: str = ""
    tg_phone: str = ""
    tg_session_name: str = "session"
    monitored_chat_ids: str = ""
    monitor_private_dms: bool = True

    redis_stream_key: str = "ci:events"
    redis_consumer_group: str = "ci_workers"
    redis_batch_size: int = 50
    redis_block_ms: int = 5000
    redis_stream_maxlen: int = 100_000
    redis_autoclaim_interval_s: int = 30
    redis_autoclaim_min_idle_ms: int = 60_000

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def chat_ids(self) -> list[int]:
        if not self.monitored_chat_ids:
            return []
        return [int(cid.strip()) for cid in self.monitored_chat_ids.split(",") if cid.strip()]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
