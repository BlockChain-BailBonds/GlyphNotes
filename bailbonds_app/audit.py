from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    timestamp: datetime
    actor: str
    action: str
    metadata: dict[str, Any]


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def list_entries(self) -> list[AuditEntry]:
        return list(self._entries)
