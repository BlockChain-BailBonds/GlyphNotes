from __future__ import annotations

from bailbonds_app.services import ComplianceService, RiskAssessmentService
from bailbonds_app.store import NormalizedDataStore


class MCPServer:
    def __init__(
        self,
        store: NormalizedDataStore,
        compliance_service: ComplianceService,
        risk_service: RiskAssessmentService,
    ) -> None:
        self._store = store
        self._compliance_service = compliance_service
        self._risk_service = risk_service

    def check_state_compliance(self, jurisdiction: str) -> dict[str, object]:
        result = self._compliance_service.check(profile=self._lookup_profile(jurisdiction))
        return {
            "jurisdiction": result.jurisdiction,
            "is_compliant": result.is_compliant,
            "notes": result.notes,
            "max_premium_pct": result.max_premium_pct,
        }

    def calculate_flight_risk(self, case_id: str) -> dict[str, object]:
        case = self._store.get_case(case_id)
        assessment = self._risk_service.calculate(case)
        return {
            "case_id": assessment.case_id,
            "risk_score": assessment.risk_score,
            "risk_band": assessment.risk_band,
            "explanation": assessment.explanation,
            "model_version": assessment.model_version,
        }

    def get_latest_court_events(self, case_id: str) -> list[dict[str, object]]:
        snapshot = self._store.get_case_snapshot(case_id)
        return [
            {
                "event_time": event.event_time.isoformat(),
                "event_type": event.event_type,
                "description": event.description,
                "source": event.source,
            }
            for event in snapshot.court_events
        ]

    def get_inmate_custody_status(self, case_id: str) -> dict[str, object] | None:
        snapshot = self._store.get_case_snapshot(case_id)
        status = snapshot.inmate_status
        if status is None:
            return None
        return {
            "status_time": status.status_time.isoformat(),
            "custody_state": status.custody_state.value,
            "facility": status.facility,
            "booking_number": status.booking_number,
            "booking_datetime": status.booking_datetime.isoformat() if status.booking_datetime else None,
            "release_datetime": status.release_datetime.isoformat() if status.release_datetime else None,
            "source": status.source,
        }

    def get_recovery_authorization(self, case_id: str) -> dict[str, object] | None:
        snapshot = self._store.get_case_snapshot(case_id)
        authorization = snapshot.recovery_authorization
        if authorization is None:
            return None
        return {
            "authorized": authorization.authorized,
            "court_order_id": authorization.court_order_id,
            "authorized_at": authorization.authorized_at.isoformat() if authorization.authorized_at else None,
        }

    def _lookup_profile(self, jurisdiction: str):
        from bailbonds_app.services import JURISDICTIONS

        return JURISDICTIONS.get(jurisdiction, JURISDICTIONS["OK-Tulsa"])
