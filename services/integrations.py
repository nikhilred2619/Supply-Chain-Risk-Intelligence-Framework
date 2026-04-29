"""
services/integrations.py
─────────────────────────
Enterprise integration connectors for ERP/CRM platforms.

Demonstrates Layer 6 REST API compatibility with:
  1. Salesforce Platform Events (Agentforce-ready)
  2. SAP Business Technology Platform (BTP) via OData
  3. Oracle Integration Cloud via REST

Paper: Section VIII (Enterprise Deployment Architecture)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import time

from services.pipeline import PipelineResult
from utils.logger import get_logger

log = get_logger("integrations")


# ══════════════════════════════════════════════════════════════════════════════
# Base Connector
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectorConfig:
    platform:     str
    instance_url: str
    api_version:  str
    timeout_s:    int = 30
    retry_count:  int = 3


class BaseConnector:
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self._authenticated = False
        log.info("[%s] Connector initialised → %s",
                 config.platform, config.instance_url)

    def _format_risk_payload(self, result: PipelineResult) -> dict:
        """Standard risk payload format for all platform connectors."""
        return {
            "riskTier":            result.risk_tier,
            "rpnScore":            result.rpn_score,
            "stockoutProbability": round(result.stockout_probability, 4),
            "confidenceScore":     round(result.confidence_score, 3),
            "explanation":         result.explanation,
            "severityScore":       result.severity,
            "occurrenceScore":     result.occurrence,
            "detectionScore":      result.detection,
            "costImpactFraction":  round(result.cost_impact, 4),
            "serviceLevelDrop":    round(result.service_level_drop, 4),
            "isHighPriority":      result.is_high_priority,
            "pipelineVersion":     result.pipeline_version,
            "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Salesforce Integration
# Platform Events + Agentforce-compatible
# ══════════════════════════════════════════════════════════════════════════════

class SalesforceConnector(BaseConnector):
    """
    Salesforce Platform Events connector.

    Publishes SupplyChainRiskAlert__e Platform Events that can be
    consumed by Salesforce Flow automations and Agentforce AI agents.

    Authentication: OAuth 2.0 JWT Bearer Flow (enterprise standard)
    Endpoint: /services/data/v59.0/sobjects/SupplyChainRiskAlert__e/
    """

    EVENT_OBJECT = "SupplyChainRiskAlert__e"
    FLOW_TRIGGER_OBJECT = "SupplyChainRiskCase__c"

    def __init__(
        self,
        instance_url: str = "https://your-org.salesforce.com",
        access_token: str = "",
        api_version: str  = "59.0",
    ):
        super().__init__(ConnectorConfig(
            platform="Salesforce",
            instance_url=instance_url,
            api_version=api_version,
        ))
        self._token = access_token

    def publish_risk_event(self, result: PipelineResult) -> dict:
        """
        Publish supply chain risk assessment as a Salesforce Platform Event.

        The event triggers:
        - Automated case creation for Critical/High risks
        - Agentforce agent invocation for contextual investigation
        - Supply chain manager notification via Flow automation

        Returns
        -------
        dict with Salesforce API response metadata (simulated in mock mode)
        """
        payload = self._build_sf_payload(result)
        endpoint = (f"{self.config.instance_url}/services/data/"
                    f"v{self.config.api_version}/sobjects/{self.EVENT_OBJECT}/")

        log.info("[Salesforce] Publishing %s event | tier=%s rpn=%d",
                 self.EVENT_OBJECT, result.risk_tier, result.rpn_score)
        log.debug("[Salesforce] Endpoint: %s", endpoint)
        log.debug("[Salesforce] Payload: %s", json.dumps(payload, indent=2))

        # In production: httpx.post(endpoint, headers=self._auth_headers(), json=payload)
        return self._mock_sf_response(payload)

    def create_case_if_critical(self, result: PipelineResult,
                                 order_id: str = "") -> Optional[dict]:
        """Create a Salesforce Case record for Critical/High risk events."""
        if not result.is_high_priority:
            return None

        case_payload = {
            "Subject":       f"[{result.risk_tier}] Supply Chain Risk — RPN {result.rpn_score}",
            "Description":   result.explanation,
            "Priority":      "High" if result.risk_tier == "Critical" else "Medium",
            "Status":        "New",
            "Origin":        "LLM-FMEA Automated Detection",
            "Type":          "Supply Chain Risk",
            "Risk_Tier__c":  result.risk_tier,
            "RPN_Score__c":  result.rpn_score,
            "Order_ID__c":   order_id,
        }
        log.info("[Salesforce] Creating Case: %s", case_payload["Subject"])
        return {"id": "5001X000007MOCK", "success": True, "errors": []}

    def _build_sf_payload(self, result: PipelineResult) -> dict:
        base = self._format_risk_payload(result)
        return {
            "Risk_Tier__c":           base["riskTier"],
            "RPN_Score__c":           base["rpnScore"],
            "Stockout_Probability__c": base["stockoutProbability"],
            "Confidence_Score__c":    base["confidenceScore"],
            "Explanation__c":         base["explanation"][:255],  # SF field limit
            "Is_High_Priority__c":    base["isHighPriority"],
            "Cost_Impact__c":         base["costImpactFraction"],
            "Pipeline_Version__c":    base["pipelineVersion"],
        }

    def _mock_sf_response(self, payload: dict) -> dict:
        return {
            "id":       "e01XX0000000MOCK",
            "success":  True,
            "errors":   [],
            "_mock":    True,
            "event":    self.EVENT_OBJECT,
            "payload_keys": list(payload.keys()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SAP BTP Integration
# OData / REST via SAP Integration Suite
# ══════════════════════════════════════════════════════════════════════════════

class SAPBTPConnector(BaseConnector):
    """
    SAP Business Technology Platform connector.

    Exposes risk assessment results via OData service for consumption
    by SAP S/4HANA supply chain management modules.

    Integration pattern: SAP Integration Suite → OData Adapter → REST API
    """

    def __init__(
        self,
        api_host:   str = "https://your-iflow.cfapps.eu10.hana.ondemand.com",
        client_id:  str = "",
        client_secret: str = "",
    ):
        super().__init__(ConnectorConfig(
            platform="SAP-BTP",
            instance_url=api_host,
            api_version="v1",
        ))
        self._client_id     = client_id
        self._client_secret = client_secret

    def push_risk_assessment(self, result: PipelineResult,
                              material_number: str = "",
                              plant_code: str = "") -> dict:
        """
        Push risk assessment to SAP via Integration Suite OData endpoint.

        The OData entity corresponds to:
            SupplyChainRiskSet (EntitySet in ZMMD_SCRA service)
        """
        payload = self._build_sap_payload(result, material_number, plant_code)
        endpoint = f"{self.config.instance_url}/sap/opu/odata/sap/ZMMD_SCRA_SRV/SupplyChainRiskSet"

        log.info("[SAP-BTP] POST %s | tier=%s mat=%s plant=%s",
                 endpoint, result.risk_tier, material_number, plant_code)

        return {"d": {"results": payload, "__mock": True}}

    def _build_sap_payload(self, result: PipelineResult,
                            material: str, plant: str) -> dict:
        base = self._format_risk_payload(result)
        return {
            "MaterialNumber":    material,
            "PlantCode":         plant,
            "RiskTier":          base["riskTier"],
            "RiskPriorityNumber": str(base["rpnScore"]),
            "StockoutProbability": f"{base['stockoutProbability']:.4f}",
            "ConfidenceScore":   f"{base['confidenceScore']:.3f}",
            "Explanation":       base["explanation"][:500],
            "IsHighPriority":    "X" if base["isHighPriority"] else "",
            "CreatedOn":         base["timestamp"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Oracle Integration Cloud
# REST Trigger → OIC Flow → Oracle SCM Cloud
# ══════════════════════════════════════════════════════════════════════════════

class OracleICConnector(BaseConnector):
    """
    Oracle Integration Cloud connector.

    Invokes an OIC REST trigger that routes risk assessments to
    Oracle Supply Chain Management Cloud for automated procurement actions.
    """

    def __init__(
        self,
        oic_host:   str = "https://your-instance.integration.ocp.oraclecloud.com",
        username:   str = "",
        password:   str = "",
    ):
        super().__init__(ConnectorConfig(
            platform="Oracle-IC",
            instance_url=oic_host,
            api_version="v1",
        ))
        self._username = username
        self._password = password

    def trigger_risk_workflow(self, result: PipelineResult,
                               po_number: str = "") -> dict:
        """
        Trigger Oracle Integration Cloud supply chain risk workflow.

        The OIC flow:
        1. Receives risk assessment via REST trigger
        2. Evaluates against procurement policy rules
        3. Creates Oracle SCM purchase orders for Critical risks
        4. Notifies supply chain planners via Oracle HCM
        """
        payload = self._build_oracle_payload(result, po_number)
        endpoint = f"{self.config.instance_url}/ic/api/integration/v1/flows/rest/SCRA_RISK_FLOW/1.0/assess"

        log.info("[Oracle-IC] Triggering workflow | tier=%s po=%s",
                 result.risk_tier, po_number)

        return {"status": "TRIGGERED", "payload": payload, "__mock": True}

    def _build_oracle_payload(self, result: PipelineResult, po: str) -> dict:
        base = self._format_risk_payload(result)
        return {
            "purchaseOrderNumber": po,
            "riskTier":            base["riskTier"],
            "rpnScore":            base["rpnScore"],
            "stockoutProbability": base["stockoutProbability"],
            "explanation":         base["explanation"],
            "costImpact":          base["costImpactFraction"],
            "requiresEscalation":  base["isHighPriority"],
            "assessmentTimestamp": base["timestamp"],
            "pipelineVersion":     base["pipelineVersion"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Enterprise Integration Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class EnterpriseIntegrationOrchestrator:
    """
    Multi-platform integration orchestrator.

    Routes risk assessment results to all configured enterprise connectors
    based on risk tier and configured routing policy.

    Usage:
        orchestrator = EnterpriseIntegrationOrchestrator(
            sf=SalesforceConnector(...),
            sap=SAPBTPConnector(...),
            oracle=OracleICConnector(...),
        )
        orchestrator.route(pipeline_result)
    """

    def __init__(
        self,
        sf:     Optional[SalesforceConnector] = None,
        sap:    Optional[SAPBTPConnector]     = None,
        oracle: Optional[OracleICConnector]   = None,
        alert_threshold: str = "High",   # route events at this tier or above
    ):
        self.sf     = sf
        self.sap    = sap
        self.oracle = oracle
        self.alert_threshold = alert_threshold
        self._tier_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    def route(self, result: PipelineResult, context: dict = {}) -> dict:
        """
        Route risk assessment to configured enterprise connectors.

        Only routes events that meet or exceed the alert_threshold tier.
        Returns a summary of all connector responses.
        """
        responses = {}
        tier_idx  = self._tier_order.get(result.risk_tier, 0)
        threshold_idx = self._tier_order.get(self.alert_threshold, 2)

        if tier_idx < threshold_idx:
            log.info("Risk tier %s below alert threshold %s — routing suppressed",
                     result.risk_tier, self.alert_threshold)
            return {"routed": False, "risk_tier": result.risk_tier}

        if self.sf:
            responses["salesforce"] = self.sf.publish_risk_event(result)
            if result.risk_tier == "Critical":
                responses["salesforce_case"] = self.sf.create_case_if_critical(
                    result, order_id=context.get("order_id", ""))

        if self.sap:
            responses["sap"] = self.sap.push_risk_assessment(
                result,
                material_number=context.get("material_number", ""),
                plant_code=context.get("plant_code", ""),
            )

        if self.oracle:
            responses["oracle"] = self.oracle.trigger_risk_workflow(
                result, po_number=context.get("po_number", ""))

        log.info("Routed %s risk event to %d connectors",
                 result.risk_tier, len(responses))
        return {"routed": True, "risk_tier": result.risk_tier, "responses": responses}
