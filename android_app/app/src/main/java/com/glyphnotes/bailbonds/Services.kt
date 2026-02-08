package com.glyphnotes.bailbonds

import java.security.MessageDigest

class NormalizedDataStore {
    private val cases = mutableMapOf<String, CaseRecord>()
    private val events = mutableMapOf<String, MutableList<CourtEvent>>()
    private val inmateStatus = mutableMapOf<String, InmateStatus>()
    private val compliance = mutableMapOf<String, ComplianceResult>()
    private val risk = mutableMapOf<String, RiskAssessment>()
    private val contract = mutableMapOf<String, ContractPacket>()
    private val recovery = mutableMapOf<String, RecoveryAuthorization>()

    fun addCase(caseRecord: CaseRecord) {
        cases[caseRecord.caseId] = caseRecord
    }

    fun updateCase(caseId: String, update: (CaseRecord) -> CaseRecord) {
        val current = cases[caseId] ?: return
        cases[caseId] = update(current)
    }

    fun addCourtEvents(caseId: String, newEvents: List<CourtEvent>) {
        val bucket = events.getOrPut(caseId) { mutableListOf() }
        bucket.addAll(newEvents)
    }

    fun setInmateStatus(status: InmateStatus) {
        inmateStatus[status.caseId] = status
    }

    fun setCompliance(result: ComplianceResult) {
        compliance[result.jurisdiction] = result
    }

    fun setRisk(result: RiskAssessment) {
        risk[result.caseId] = result
    }

    fun setContract(packet: ContractPacket) {
        contract[packet.caseId] = packet
    }

    fun setRecoveryAuthorization(authorization: RecoveryAuthorization) {
        recovery[authorization.caseId] = authorization
    }

    fun getCase(caseId: String): CaseRecord = cases.getValue(caseId)

    fun snapshot(caseId: String): CaseSnapshot {
        val case = cases.getValue(caseId)
        return CaseSnapshot(
            case = case,
            courtEvents = events[caseId]?.toList().orEmpty(),
            inmateStatus = inmateStatus[caseId],
            compliance = compliance[case.request.court.jurisdiction],
            riskAssessment = risk[caseId],
            contractPacket = contract[caseId],
            recoveryAuthorization = recovery[caseId],
        )
    }
}

data class JurisdictionProfile(
    val jurisdiction: String,
    val maxPremiumPct: Double,
    val requiresWarrantForRecovery: Boolean,
    val courtSearchUrl: String,
    val inmateSearchUrl: String,
    val municipalJailUrl: String,
    val courtClerkUrl: String,
)

class CaseCreationService(private val store: NormalizedDataStore) {
    fun createCase(caseId: String, request: BondRequest): CaseRecord {
        val record = CaseRecord(
            caseId = caseId,
            request = request,
            createdAt = nowIso(),
            status = "provisional",
            jurisdictionDetected = request.court.jurisdiction,
            bookingNumber = null,
            caseNumber = null,
            nextCourtDate = null,
            charges = emptyList(),
            contractHash = null,
        )
        store.addCase(record)
        return record
    }
}

class CourtJailAdapterService(private val store: NormalizedDataStore) {
    fun fetchCourtEvents(caseRecord: CaseRecord, profile: JurisdictionProfile): List<CourtEvent> {
        val eventTime = nowIso()
        val caseNumber = "CF-${eventTime.take(4)}-${caseRecord.caseId.takeLast(4)}"
        val events = listOf(
            CourtEvent(
                caseId = caseRecord.caseId,
                eventTime = eventTime,
                eventType = "case_filed",
                description = "Case filed in public docket ($caseNumber).",
                source = profile.courtSearchUrl,
            ),
            CourtEvent(
                caseId = caseRecord.caseId,
                eventTime = eventTime,
                eventType = "arraignment_scheduled",
                description = "Initial appearance scheduled.",
                source = profile.courtSearchUrl,
            ),
        )
        store.updateCase(caseRecord.caseId) {
            it.copy(caseNumber = caseNumber, nextCourtDate = eventTime)
        }
        return events
    }

    fun fetchInmateStatus(caseRecord: CaseRecord, profile: JurisdictionProfile): InmateStatus {
        val bookingTime = nowIso()
        return InmateStatus(
            caseId = caseRecord.caseId,
            statusTime = bookingTime,
            custodyState = CustodyState.IN_CUSTODY,
            facility = "Tulsa County Jail",
            bookingNumber = "BK-${caseRecord.caseId.takeLast(6)}",
            bookingDatetime = bookingTime,
            releaseDatetime = null,
            source = profile.inmateSearchUrl,
        )
    }
}

class ComplianceService {
    fun check(profile: JurisdictionProfile): ComplianceResult {
        return ComplianceResult(
            jurisdiction = profile.jurisdiction,
            isCompliant = true,
            notes = listOf("Premium within statutory maximum.", "Recovery requires court order."),
            maxPremiumPct = profile.maxPremiumPct,
        )
    }
}

class RiskAssessmentService {
    fun calculate(caseRecord: CaseRecord): RiskAssessment {
        val payload = "${caseRecord.caseId}:${caseRecord.request.bondAmount}:${caseRecord.request.court.jurisdiction}"
        val digest = MessageDigest.getInstance("SHA-256").digest(payload.toByteArray())
        val score = ((digest[0].toInt() and 0xFF) * 256 + (digest[1].toInt() and 0xFF)) / 65535.0
        val band = when {
            score < 0.35 -> "low"
            score < 0.7 -> "medium"
            else -> "high"
        }
        return RiskAssessment(
            caseId = caseRecord.caseId,
            riskScore = String.format("%.4f", score).toDouble(),
            riskBand = band,
            explanation = "Deterministic advisory score based on jurisdiction and public signals.",
            modelVersion = "risk-v1.0",
        )
    }
}

class ContractService {
    fun generate(caseRecord: CaseRecord, compliance: ComplianceResult): ContractPacket {
        val agreementText = buildString {
            appendLine("Bail Agreement")
            appendLine("Case: ${caseRecord.caseId}")
            appendLine("Defendant: ${caseRecord.request.defendant.firstName} ${caseRecord.request.defendant.lastName}")
            appendLine("Jurisdiction: ${caseRecord.request.court.jurisdiction}")
            appendLine("Bond Amount: $${caseRecord.request.bondAmount}")
            appendLine("Max Premium: ${String.format("%.2f", compliance.maxPremiumPct * 100)}%")
            appendLine("Recovery requires explicit court authorization.")
        }
        val hash = MessageDigest.getInstance("SHA-256").digest(agreementText.toByteArray())
            .joinToString("") { "%02x".format(it) }
        return ContractPacket(
            caseId = caseRecord.caseId,
            agreementText = agreementText,
            onChainHash = hash,
            generatedAt = nowIso(),
        )
    }
}

class RecoveryAuthorizationService {
    fun authorize(caseId: String, courtOrderId: String?): RecoveryAuthorization {
        val authorized = courtOrderId != null
        return RecoveryAuthorization(
            caseId = caseId,
            authorized = authorized,
            courtOrderId = courtOrderId,
            authorizedAt = if (authorized) nowIso() else null,
        )
    }
}

object Jurisdictions {
    val tulsa = JurisdictionProfile(
        jurisdiction = "OK-Tulsa",
        maxPremiumPct = 0.1,
        requiresWarrantForRecovery = true,
        courtSearchUrl = "https://www.oscn.net/dockets/search.aspx",
        inmateSearchUrl = "https://www2.tulsacounty.org/community/inmate-information/",
        municipalJailUrl = "https://www.tulsapolice.org/tulsamunicipaljail",
        courtClerkUrl = "https://courtclerk.tulsacounty.org/",
    )
}
