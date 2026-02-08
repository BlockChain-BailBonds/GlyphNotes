from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from bailbonds_app.models import (
    Case,
    CaseSnapshot,
    ComplianceResult,
    ContractPacket,
    CourtEvent,
    InmateStatus,
    RecoveryAuthorization,
    RiskAssessment,
)


class NormalizedDataStore:
    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}
        self._events: dict[str, list[CourtEvent]] = {}
        self._inmate_status: dict[str, InmateStatus] = {}
        self._compliance: dict[str, ComplianceResult] = {}
        self._risk: dict[str, RiskAssessment] = {}
        self._contracts: dict[str, ContractPacket] = {}
        self._recovery: dict[str, RecoveryAuthorization] = {}

    def add_case(self, case: Case) -> None:
        self._cases[case.case_id] = case

    def update_case(self, case_id: str, **changes: object) -> None:
        case = self._cases[case_id]
        self._cases[case_id] = replace(case, **changes)

    def add_court_events(self, case_id: str, events: Iterable[CourtEvent]) -> None:
        self._events.setdefault(case_id, [])
        self._events[case_id].extend(events)

    def set_inmate_status(self, status: InmateStatus) -> None:
        self._inmate_status[status.case_id] = status

    def set_compliance(self, result: ComplianceResult) -> None:
        self._compliance[result.jurisdiction] = result

    def set_risk(self, result: RiskAssessment) -> None:
        self._risk[result.case_id] = result

    def set_contract(self, packet: ContractPacket) -> None:
        self._contracts[packet.case_id] = packet

    def set_recovery_authorization(self, authorization: RecoveryAuthorization) -> None:
        self._recovery[authorization.case_id] = authorization

    def get_case(self, case_id: str) -> Case:
        return self._cases[case_id]

    def get_case_snapshot(self, case_id: str) -> CaseSnapshot:
        case = self._cases[case_id]
        return CaseSnapshot(
            case=case,
            court_events=list(self._events.get(case_id, [])),
            inmate_status=self._inmate_status.get(case_id),
            compliance=self._compliance.get(case.request.court.jurisdiction),
            risk_assessment=self._risk.get(case_id),
            contract_packet=self._contracts.get(case_id),
            recovery_authorization=self._recovery.get(case_id),
        )
