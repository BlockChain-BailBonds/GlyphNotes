package com.glyphnotes.bailbonds

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class BailBondsApp {
    private val store = NormalizedDataStore()
    private val caseService = CaseCreationService(store)
    private val adapterService = CourtJailAdapterService(store)
    private val complianceService = ComplianceService()
    private val riskService = RiskAssessmentService()
    private val contractService = ContractService()
    private val recoveryService = RecoveryAuthorizationService()

    suspend fun pressImGoingToJailButton(request: BondRequest): CaseResponse = withContext(Dispatchers.Default) {
        val caseId = newCaseId()
        val caseRecord = caseService.createCase(caseId, request)
        val profile = Jurisdictions.tulsa

        val events = adapterService.fetchCourtEvents(caseRecord, profile)
        val status = adapterService.fetchInmateStatus(caseRecord, profile)
        store.addCourtEvents(caseId, events)
        store.setInmateStatus(status)

        val compliance = complianceService.check(profile)
        val risk = riskService.calculate(caseRecord)
        val contract = contractService.generate(caseRecord, compliance)
        val recovery = recoveryService.authorize(caseId, courtOrderId = null)

        store.setCompliance(compliance)
        store.setRisk(risk)
        store.setContract(contract)
        store.setRecoveryAuthorization(recovery)

        store.updateCase(caseId) {
            it.copy(
                bookingNumber = status.bookingNumber,
                charges = listOf("Failure to appear"),
                contractHash = contract.onChainHash,
                status = "active",
            )
        }

        CaseResponse(
            caseId = caseId,
            status = "provisional",
            message = "Case created. Court/jail enrichment is running asynchronously.",
        )
    }

    fun getCaseSnapshot(caseId: String): CaseSnapshot = store.snapshot(caseId)
}
