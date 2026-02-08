package com.glyphnotes.bailbonds

import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

private val formatter = DateTimeFormatter.ISO_OFFSET_DATE_TIME

fun nowIso(): String = formatter.format(Instant.now().atOffset(ZoneOffset.UTC))

fun newCaseId(): String = "CASE-${UUID.randomUUID().toString().replace("-", "").take(12).uppercase()}"

enum class CustodyState {
    UNKNOWN,
    NOT_IN_CUSTODY,
    IN_CUSTODY,
    RELEASED,
}

data class DefendantInfo(
    val firstName: String,
    val lastName: String,
    val dateOfBirth: String,
    val phone: String,
    val email: String,
)

data class CourtInfo(
    val jurisdiction: String,
    val courtName: String,
    val county: String,
    val state: String,
)

data class BondRequest(
    val defendant: DefendantInfo,
    val court: CourtInfo,
    val bondAmount: Int,
    val deviceContext: Map<String, String>,
    val profileContext: Map<String, String>,
) {
    companion object {
        fun sample(): BondRequest = BondRequest(
            defendant = DefendantInfo(
                firstName = "Jordan",
                lastName = "Lee",
                dateOfBirth = "1991-06-12",
                phone = "555-0199",
                email = "jordan.lee@example.com",
            ),
            court = CourtInfo(
                jurisdiction = "OK-Tulsa",
                courtName = "Tulsa County District Court",
                county = "Tulsa",
                state = "OK",
            ),
            bondAmount = 3500,
            deviceContext = mapOf("device_id" to "device-123", "locale" to "en-US"),
            profileContext = mapOf("member_id" to "member-4488"),
        )
    }
}

data class CaseRecord(
    val caseId: String,
    val request: BondRequest,
    val createdAt: String,
    val status: String,
    val jurisdictionDetected: String,
    val bookingNumber: String?,
    val caseNumber: String?,
    val nextCourtDate: String?,
    val charges: List<String>,
    val contractHash: String?,
)

data class CourtEvent(
    val caseId: String,
    val eventTime: String,
    val eventType: String,
    val description: String,
    val source: String,
)

data class InmateStatus(
    val caseId: String,
    val statusTime: String,
    val custodyState: CustodyState,
    val facility: String,
    val bookingNumber: String,
    val bookingDatetime: String,
    val releaseDatetime: String?,
    val source: String,
)

data class ComplianceResult(
    val jurisdiction: String,
    val isCompliant: Boolean,
    val notes: List<String>,
    val maxPremiumPct: Double,
)

data class RiskAssessment(
    val caseId: String,
    val riskScore: Double,
    val riskBand: String,
    val explanation: String,
    val modelVersion: String,
)

data class ContractPacket(
    val caseId: String,
    val agreementText: String,
    val onChainHash: String,
    val generatedAt: String,
)

data class RecoveryAuthorization(
    val caseId: String,
    val authorized: Boolean,
    val courtOrderId: String?,
    val authorizedAt: String?,
)

data class CaseSnapshot(
    val case: CaseRecord,
    val courtEvents: List<CourtEvent>,
    val inmateStatus: InmateStatus?,
    val compliance: ComplianceResult?,
    val riskAssessment: RiskAssessment?,
    val contractPacket: ContractPacket?,
    val recoveryAuthorization: RecoveryAuthorization?,
)

data class CaseResponse(
    val caseId: String,
    val status: String,
    val message: String,
)
