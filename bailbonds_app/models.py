from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CustodyState(str, Enum):
    UNKNOWN = "unknown"
    NOT_IN_CUSTODY = "not_in_custody"
    IN_CUSTODY = "in_custody"
    RELEASED = "released"


@dataclass(frozen=True)
class DefendantInfo:
    first_name: str
    last_name: str
    date_of_birth: str
    phone: str
    email: str


@dataclass(frozen=True)
class CourtInfo:
    jurisdiction: str
    court_name: str
    county: str
    state: str


@dataclass(frozen=True)
class BondRequest:
    defendant: DefendantInfo
    court: CourtInfo
    bond_amount: int
    device_context: Mapping[str, str] = field(default_factory=dict)
    profile_context: Mapping[str, str] = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    request: BondRequest
    created_at: datetime
    status: str = "provisional"
    jurisdiction_detected: str | None = None
    booking_number: str | None = None
    case_number: str | None = None
    next_court_date: datetime | None = None
    charges: list[str] = field(default_factory=list)
    contract_hash: str | None = None


@dataclass(frozen=True)
class CourtEvent:
    case_id: str
    event_time: datetime
    event_type: str
    description: str
    source: str


@dataclass(frozen=True)
class InmateStatus:
    case_id: str
    status_time: datetime
    custody_state: CustodyState
    facility: str | None
    booking_number: str | None
    booking_datetime: datetime | None
    release_datetime: datetime | None
    source: str


@dataclass(frozen=True)
class ComplianceResult:
    jurisdiction: str
    is_compliant: bool
    notes: list[str]
    max_premium_pct: float


@dataclass(frozen=True)
class RiskAssessment:
    case_id: str
    risk_score: float
    risk_band: str
    explanation: str
    model_version: str


@dataclass(frozen=True)
class ContractPacket:
    case_id: str
    agreement_text: str
    on_chain_hash: str
    generated_at: datetime


@dataclass
class CaseSnapshot:
    case: Case
    court_events: Iterable[CourtEvent]
    inmate_status: InmateStatus | None
    compliance: ComplianceResult | None
    risk_assessment: RiskAssessment | None
    contract_packet: ContractPacket | None
    recovery_authorization: RecoveryAuthorization | None


@dataclass(frozen=True)
class RecoveryAuthorization:
    case_id: str
    authorized: bool
    court_order_id: str | None
    authorized_at: datetime | None


@dataclass(frozen=True)
class CaseResponse:
    case_id: str
    status: str
    message: str


def new_case_id() -> str:
    return f"CASE-{uuid4().hex[:12].upper()}"
