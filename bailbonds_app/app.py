from __future__ import annotations

import asyncio
from dataclasses import asdict

from bailbonds_app.audit import AuditLog
from bailbonds_app.models import BondRequest, CaseResponse, CourtInfo, DefendantInfo, new_case_id
from bailbonds_app.services import (
    CaseCreationService,
    ComplianceService,
    ContractService,
    CourtJailAdapterService,
    RiskAssessmentService,
    RecoveryAuthorizationService,
    JURISDICTIONS,
)
from bailbonds_app.store import NormalizedDataStore


class BailBondsApp:
    def __init__(self) -> None:
        self.store = NormalizedDataStore()
        self.audit = AuditLog()
        self.case_service = CaseCreationService(self.store, self.audit)
        self.adapter_service = CourtJailAdapterService(self.store, self.audit)
        self.compliance_service = ComplianceService(self.audit)
        self.risk_service = RiskAssessmentService(self.audit)
        self.contract_service = ContractService(self.audit)
        self.recovery_service = RecoveryAuthorizationService(self.audit)

    async def press_im_going_to_jail_button(self, request: BondRequest) -> CaseResponse:
        case_id = new_case_id()
        case = self.case_service.create_case(case_id, request)
        asyncio.create_task(self._async_enrich(case.case_id))
        return CaseResponse(
            case_id=case.case_id,
            status=case.status,
            message="Case created. Court/jail enrichment is running asynchronously.",
        )

    async def _async_enrich(self, case_id: str) -> None:
        case = self.store.get_case(case_id)
        events = self.adapter_service.fetch_court_events(case)
        status = self.adapter_service.fetch_inmate_status(case)
        self.store.add_court_events(case_id, events)
        self.store.set_inmate_status(status)

        profile = JURISDICTIONS[case.request.court.jurisdiction]
        compliance = self.compliance_service.check(profile)
        risk = self.risk_service.calculate(case)
        contract = self.contract_service.generate(case, compliance)
        recovery = self.recovery_service.authorize(case_id, court_order_id=None)

        self.store.set_compliance(compliance)
        self.store.set_risk(risk)
        self.store.set_contract(contract)
        self.store.set_recovery_authorization(recovery)
        self.store.update_case(
            case_id,
            booking_number=status.booking_number,
            charges=["Failure to appear"],
            contract_hash=contract.on_chain_hash,
            status="active",
        )


async def demo_run() -> None:
    app = BailBondsApp()
    request = BondRequest(
        defendant=DefendantInfo(
            first_name="Jordan",
            last_name="Lee",
            date_of_birth="1991-06-12",
            phone="555-0199",
            email="jordan.lee@example.com",
        ),
        court=CourtInfo(
            jurisdiction="OK-Tulsa",
            court_name="Tulsa County District Court",
            county="Tulsa",
            state="OK",
        ),
        bond_amount=3500,
        device_context={"device_id": "device-123", "locale": "en-US"},
        profile_context={"member_id": "member-4488"},
    )

    response = await app.press_im_going_to_jail_button(request)
    print("Immediate response:")
    print(asdict(response))

    await asyncio.sleep(0)
    snapshot = app.store.get_case_snapshot(response.case_id)
    print("\nSnapshot after async enrichment:")
    print(
        {
            "case": snapshot.case.case_id,
            "status": snapshot.case.status,
            "booking_number": snapshot.case.booking_number,
            "case_number": snapshot.case.case_number,
            "next_court_date": snapshot.case.next_court_date.isoformat() if snapshot.case.next_court_date else None,
            "court_events": len(list(snapshot.court_events)),
            "custody_state": snapshot.inmate_status.custody_state.value if snapshot.inmate_status else None,
            "risk_band": snapshot.risk_assessment.risk_band if snapshot.risk_assessment else None,
            "contract_hash": snapshot.contract_packet.on_chain_hash if snapshot.contract_packet else None,
            "recovery_authorized": snapshot.recovery_authorization.authorized if snapshot.recovery_authorization else None,
        }
    )


if __name__ == "__main__":
    asyncio.run(demo_run())
