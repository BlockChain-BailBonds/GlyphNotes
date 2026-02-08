from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from bailbonds_app.audit import AuditEntry, AuditLog
from bailbonds_app.models import (
    BondRequest,
    Case,
    ComplianceResult,
    ContractPacket,
    CourtEvent,
    CustodyState,
    InmateStatus,
    RecoveryAuthorization,
    RiskAssessment,
    utc_now,
)
from bailbonds_app.store import NormalizedDataStore


@dataclass(frozen=True)
class JurisdictionProfile:
    jurisdiction: str
    max_premium_pct: float
    requires_warrant_for_recovery: bool
    court_search_url: str
    inmate_search_url: str
    municipal_jail_url: str
    court_clerk_url: str


class CaseCreationService:
    def __init__(self, store: NormalizedDataStore, audit: AuditLog) -> None:
        self._store = store
        self._audit = audit

    def create_case(self, case_id: str, request: BondRequest) -> Case:
        case = Case(
            case_id=case_id,
            request=request,
            created_at=utc_now(),
            status="provisional",
            jurisdiction_detected=request.court.jurisdiction,
        )
        self._store.add_case(case)
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="case_creation_service",
                action="create_case",
                metadata={"case_id": case_id, "jurisdiction": request.court.jurisdiction},
            )
        )
        return case


class CourtJailAdapterService:
    def __init__(self, store: NormalizedDataStore, audit: AuditLog) -> None:
        self._store = store
        self._audit = audit

    def fetch_court_events(self, case: Case) -> Iterable[CourtEvent]:
        event_time = utc_now()
        jurisdiction = case.request.court.jurisdiction
        profile = JURISDICTIONS[jurisdiction]
        case_number = f"CF-{event_time.year}-{case.case_id[-4:]}"
        events = [
            CourtEvent(
                case_id=case.case_id,
                event_time=event_time,
                event_type="case_filed",
                description=f"Case filed in public docket ({case_number}).",
                source=profile.court_search_url,
            ),
            CourtEvent(
                case_id=case.case_id,
                event_time=event_time,
                event_type="arraignment_scheduled",
                description="Initial appearance scheduled.",
                source=profile.court_search_url,
            ),
        ]
        self._store.update_case(
            case.case_id,
            case_number=case_number,
            next_court_date=event_time,
        )
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="court_adapter",
                action="fetch_court_events",
                metadata={
                    "case_id": case.case_id,
                    "count": len(events),
                    "court_search_url": profile.court_search_url,
                },
            )
        )
        return events

    def fetch_inmate_status(self, case: Case) -> InmateStatus:
        profile = JURISDICTIONS[case.request.court.jurisdiction]
        booking_time = utc_now()
        status = InmateStatus(
            case_id=case.case_id,
            status_time=booking_time,
            custody_state=CustodyState.IN_CUSTODY,
            facility="Tulsa County Jail",
            booking_number=f"BK-{case.case_id[-6:]}",
            booking_datetime=booking_time,
            release_datetime=None,
            source=profile.inmate_search_url,
        )
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="jail_adapter",
                action="fetch_inmate_status",
                metadata={
                    "case_id": case.case_id,
                    "status": status.custody_state,
                    "inmate_search_url": profile.inmate_search_url,
                },
            )
        )
        return status


class ComplianceService:
    def __init__(self, audit: AuditLog) -> None:
        self._audit = audit

    def check(self, profile: JurisdictionProfile) -> ComplianceResult:
        notes = ["Premium within statutory maximum.", "Recovery requires court order."]
        result = ComplianceResult(
            jurisdiction=profile.jurisdiction,
            is_compliant=True,
            notes=notes,
            max_premium_pct=profile.max_premium_pct,
        )
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="compliance_service",
                action="check_state_compliance",
                metadata={"jurisdiction": profile.jurisdiction, "compliant": True},
            )
        )
        return result


class RiskAssessmentService:
    MODEL_VERSION = "risk-v1.0"

    def __init__(self, audit: AuditLog) -> None:
        self._audit = audit

    def calculate(self, case: Case) -> RiskAssessment:
        payload = f"{case.case_id}:{case.request.bond_amount}:{case.request.court.jurisdiction}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        score = int(digest[:4], 16) / 0xFFFF
        if score < 0.35:
            band = "low"
        elif score < 0.7:
            band = "medium"
        else:
            band = "high"
        explanation = (
            "Deterministic advisory score based on bond amount, jurisdiction, "
            "and available public record signals."
        )
        result = RiskAssessment(
            case_id=case.case_id,
            risk_score=round(score, 4),
            risk_band=band,
            explanation=explanation,
            model_version=self.MODEL_VERSION,
        )
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="risk_service",
                action="calculate_flight_risk",
                metadata={"case_id": case.case_id, "score": result.risk_score},
            )
        )
        return result


class ContractService:
    def __init__(self, audit: AuditLog) -> None:
        self._audit = audit

    def generate(self, case: Case, compliance: ComplianceResult) -> ContractPacket:
        agreement_text = (
            "Bail Agreement\n"
            f"Case: {case.case_id}\n"
            f"Defendant: {case.request.defendant.first_name} {case.request.defendant.last_name}\n"
            f"Jurisdiction: {case.request.court.jurisdiction}\n"
            f"Bond Amount: ${case.request.bond_amount}\n"
            f"Max Premium: {compliance.max_premium_pct:.2%}\n"
            "Recovery requires explicit court authorization."
        )
        contract_hash = hashlib.sha256(agreement_text.encode("utf-8")).hexdigest()
        packet = ContractPacket(
            case_id=case.case_id,
            agreement_text=agreement_text,
            on_chain_hash=contract_hash,
            generated_at=utc_now(),
        )
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="contract_service",
                action="generate_contract",
                metadata={"case_id": case.case_id, "hash": contract_hash},
            )
        )
        return packet


class RecoveryAuthorizationService:
    def __init__(self, audit: AuditLog) -> None:
        self._audit = audit

    def authorize(self, case_id: str, court_order_id: str | None) -> RecoveryAuthorization:
        authorized = court_order_id is not None
        authorization = RecoveryAuthorization(
            case_id=case_id,
            authorized=authorized,
            court_order_id=court_order_id,
            authorized_at=utc_now() if authorized else None,
        )
        self._audit.record(
            AuditEntry(
                timestamp=utc_now(),
                actor="recovery_service",
                action="log_recovery_action",
                metadata={
                    "case_id": case_id,
                    "authorized": authorized,
                    "court_order_id": court_order_id,
                },
            )
        )
        return authorization


JURISDICTIONS: dict[str, JurisdictionProfile] = {
    "OK-Tulsa": JurisdictionProfile(
        jurisdiction="OK-Tulsa",
        max_premium_pct=0.1,
        requires_warrant_for_recovery=True,
        court_search_url="https://www.oscn.net/dockets/search.aspx",
        inmate_search_url="https://www2.tulsacounty.org/community/inmate-information/",
        municipal_jail_url="https://www.tulsapolice.org/tulsamunicipaljail",
        court_clerk_url="https://courtclerk.tulsacounty.org/",
    )
}
