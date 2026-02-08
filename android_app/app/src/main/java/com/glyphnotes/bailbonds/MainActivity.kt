package com.glyphnotes.bailbonds

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.glyphnotes.bailbonds.databinding.ActivityMainBinding
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val app = BailBondsApp()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.triggerButton.setOnClickListener {
            binding.triggerButton.isEnabled = false
            binding.statusText.text = "Starting bail workflow..."

            lifecycleScope.launch {
                val request = BondRequest.sample()
                val response = app.pressImGoingToJailButton(request)
                binding.statusText.text = "Immediate response: ${response.message}\nCase: ${response.caseId}"

                delay(250)
                val snapshot = app.getCaseSnapshot(response.caseId)
                binding.statusText.text = buildString {
                    appendLine("Immediate response: ${response.message}")
                    appendLine("Case: ${response.caseId}")
                    appendLine("Status: ${snapshot.case.status}")
                    appendLine("Case number: ${snapshot.case.caseNumber}")
                    appendLine("Next court date: ${snapshot.case.nextCourtDate}")
                    appendLine("Custody: ${snapshot.inmateStatus?.custodyState}")
                    appendLine("Risk: ${snapshot.riskAssessment?.riskBand} (${snapshot.riskAssessment?.riskScore})")
                    appendLine("Contract hash: ${snapshot.contractPacket?.onChainHash}")
                    appendLine("Recovery authorized: ${snapshot.recoveryAuthorization?.authorized}")
                }
            }
        }
    }
}
