import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone

correlation_id_var: ContextVar[str] = ContextVar(
    "correkation_id",
    default="-",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }

        if record.exc_info:
            log_record["execution"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def set_correlation_id(correlation_id: str) -> Token:
    return correlation_id_var.set(correlation_id)


def reset_correlation_id(token: Token) -> None:
    correlation_id_var.reset(token)


def get_correlation_id() -> str:
    return correlation_id_var.get()
