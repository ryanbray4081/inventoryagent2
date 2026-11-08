"""
Supply Chain Analytics Inventory Decision Agent
===============================================
Architecture Pipeline:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Implements end-to-end orchestration for:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (ROP - Continuous Review)
4. Fixed-Period Review (Order-Up-To Level)
5. Newsvendor Single-Period Model
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple
import re
import math
from scipy import stats

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_safety_stock_and_rop,
    calculate_fixed_period_review,
    calculate_newsvendor,
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
)


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock and Reorder Point (Continuous Review)"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model (Periodic Review)"
    NEWSVENDOR = "Newsvendor Model (Single-Period / Perishable)"
    UNKNOWN = "Unknown Model"


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    details: str
    severity: str = "INFO"  # INFO, WARNING, ERROR


@dataclass
class DecisionTrace:
    """Encapsulates the complete 7-stage architectural decision trace."""
    customer_question: str
    identified_model: InventoryModelType
    model_confidence: float
    model_rationale: str
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool_name: str
    calculation_result: Optional[Any] = None
    verification_passed: bool = False
    verification_checks: List[VerificationCheck] = field(default_factory=list)
    executive_summary: str = ""
    mathematical_derivation: str = ""
    cost_and_tradeoff_analysis: str = ""
    managerial_recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_question": self.customer_question,
            "identified_model": self.identified_model.value,
            "model_confidence": round(self.model_confidence, 2),
            "model_rationale": self.model_rationale,
            "extracted_inputs": self.extracted_inputs,
            "missing_inputs": self.missing_inputs,
            "selected_tool_name": self.selected_tool_name,
            "calculation_result": self.calculation_result.to_dict() if hasattr(self.calculation_result, "to_dict") else str(self.calculation_result),
            "verification_passed": self.verification_passed,
            "verification_checks": [
                {"name": c.name, "passed": c.passed, "details": c.details, "severity": c.severity}
                for c in self.verification_checks
            ],
            "executive_summary": self.executive_summary,
            "mathematical_derivation": self.mathematical_derivation,
            "cost_and_tradeoff_analysis": self.cost_and_tradeoff_analysis,
            "managerial_recommendations": self.managerial_recommendations,
            "error_message": self.error_message,
        }


class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Orchestrates natural language interpretation, model classification,
    parameter extraction, Python mathematical tool invocation, sanity verification,
    and strategic managerial explanation.
    """

    def __init__(self):
        self.tools = {
            InventoryModelType.EOQ: calculate_eoq,
            InventoryModelType.QUANTITY_DISCOUNT: calculate_quantity_discount,
            InventoryModelType.SAFETY_STOCK_ROP: calculate_safety_stock_and_rop,
            InventoryModelType.FIXED_PERIOD_REVIEW: calculate_fixed_period_review,
            InventoryModelType.NEWSVENDOR: calculate_newsvendor,
        }

    # =========================================================================
    # Step 1 & 2: Identify Model from Natural Language
    # =========================================================================

    def identify_model(self, question: str) -> Tuple[InventoryModelType, float, str]:
        """
        Classifies customer question into one of the 5 inventory models.
        Returns: (InventoryModelType, confidence_score, rationale)
        """
        q = question.lower()

        # Score weights for each model based on strong indicators
        scores = {
            InventoryModelType.QUANTITY_DISCOUNT: 0,
            InventoryModelType.NEWSVENDOR: 0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0,
            InventoryModelType.SAFETY_STOCK_ROP: 0,
            InventoryModelType.EOQ: 0,
        }

        # Quantity Discount cues
        if any(w in q for w in ["tier", "bracket", "volume discount", "quantity discount", "discount schedule", "price break"]):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 5
        if re.search(r"\b\d+\s*(?:to|-)\s*\d+\s*(?:units)?\s*(?:at|for|\$)", q):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 4

        # Newsvendor cues
        if any(w in q for w in ["newsvendor", "newsboy", "perishable", "seasonal", "fashion", "single-period", "single period"]):
            scores[InventoryModelType.NEWSVENDOR] += 6
        if any(w in q for w in ["salvage", "salvage value", "liquidate", "unsold", "critical fractile", "overage", "underage"]):
            scores[InventoryModelType.NEWSVENDOR] += 4

        # Fixed-Period Review cues
        if any(w in q for w in ["fixed-period", "fixed period", "periodic review", "order-up-to", "order up to", "target inventory", "target level"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 6
        if any(w in q for w in ["review period", "review every", "reviewed every", "on-hand and on-order", "protection interval"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 4
        if "on hand" in q and ("review" in q or "order-up-to" in q or "target" in q):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 3

        # Safety Stock & Reorder Point cues
        if any(w in q for w in ["reorder point", "rop", "safety stock", "lead time demand"]):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 5
        if any(w in q for w in ["service level", "cycle service level", "csl", "stockout risk"]) and scores[InventoryModelType.FIXED_PERIOD_REVIEW] == 0:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 3
        if "lead time" in q and ("reorder" in q or "buffer" in q):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 2

        # EOQ cues
        if any(w in q for w in ["eoq", "economic order quantity"]):
            scores[InventoryModelType.EOQ] += 6
        if any(w in q for w in ["annual demand", "setup cost", "ordering cost", "holding cost", "order quantity"]):
            scores[InventoryModelType.EOQ] += 3
        if "how many units should we order" in q or "optimal order quantity" in q or "batch size" in q:
            scores[InventoryModelType.EOQ] += 2

        # Determine winner
        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]

        if max_score == 0:
            return (InventoryModelType.UNKNOWN, 0.0, "Could not match query to any supported inventory model.")

        confidence = min(0.99, 0.50 + (max_score * 0.08))

        rationale_map = {
            InventoryModelType.QUANTITY_DISCOUNT: "Detected volume discount schedule, price tiers, and tier breakpoints.",
            InventoryModelType.NEWSVENDOR: "Detected single-period seasonal/perishable characteristics, salvage economics, and demand uncertainty.",
            InventoryModelType.FIXED_PERIOD_REVIEW: "Detected periodic review cycle, protection interval, order-up-to policy, and pipeline inventory.",
            InventoryModelType.SAFETY_STOCK_ROP: "Detected continuous review policy, lead time buffer requirements, and cycle service level targets.",
            InventoryModelType.EOQ: "Detected classic annual demand, order setup cost, and inventory holding cost trade-offs.",
        }

        return (best_model, confidence, rationale_map.get(best_model, "Pattern match"))

    # =========================================================================
    # Step 3: Extract Inputs from Natural Language
    # =========================================================================

    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts numerical parameters and operational constraints from text for the target model.
        Returns: (inputs_dict, missing_required_fields)
        """
        text = question.replace(",", "")  # Normalize comma-separated numbers like 12,000 -> 12000

        if model_type == InventoryModelType.EOQ:
            return self._extract_eoq_inputs(text)
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return self._extract_quantity_discount_inputs(text)
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return self._extract_safety_stock_rop_inputs(text)
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return self._extract_fixed_period_review_inputs(text)
        elif model_type == InventoryModelType.NEWSVENDOR:
            return self._extract_newsvendor_inputs(text)
        else:
            return ({}, ["unknown_model"])

    def _extract_eoq_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        inputs: Dict[str, Any] = {}
        missing = []

        # Demand
        m_demand = re.search(r"(?:annual\s*demand|demand|sells?)\s*(?:for\s+[\w\s]+?)?\s*(?:is|of|for|=|:)?\s*(?:about)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?", text, re.I)
        if m_demand:
            inputs["demand"] = float(m_demand.group(1))

        # Order Cost / Setup Cost
        m_order = re.search(r"(?:setup|order(?:ing)?)\s*cost\s*(?:is|of|per order)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if not m_order:
            m_order = re.search(r"\$(\d+(?:\.\d+)?)\s*(?:to place|per order|setup cost)", text, re.I)
        if m_order:
            inputs["order_cost"] = float(m_order.group(1))

        # Holding Cost per unit or rate
        m_hold_rate = re.search(r"holding\s*(?:cost\s*)?(?:rate)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        m_hold_unit = re.search(r"(?:holding(?:\s*inventory)?\s*costs?|holding\s*cost)\s*(?:is|of)?\s*\$?(\d+(?:\.\d+)?)\s*per unit", text, re.I)
        if not m_hold_unit:
            m_hold_unit = re.search(r"\$(\d+(?:\.\d+)?)\s*per unit\s*(?:per year)?\s*(?:holding|to hold)", text, re.I)
        if not m_hold_unit:
            m_hold_unit = re.search(r"holding\s*(?:inventory)?\s*(?:costs|cost\s*is)\s*\$?(\d+(?:\.\d+)?)", text, re.I)

        if m_hold_unit:
            inputs["holding_cost"] = float(m_hold_unit.group(1))
        elif m_hold_rate:
            inputs["holding_cost_rate"] = float(m_hold_rate.group(1)) / 100.0

        # Unit Cost (careful not to conflate with holding or order cost)
        m_unit_cost = re.search(r"(?:unit price|unit cost|purchase price|wholesale price)\s*(?:is|of)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if not m_unit_cost:
            m_unit_cost = re.search(r"costs?\s*us\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if m_unit_cost:
            inputs["unit_cost"] = float(m_unit_cost.group(1))

        # Working days
        m_days = re.search(r"(\d+)\s*(?:working days?|days a year|operating days?)", text, re.I)
        if m_days:
            inputs["working_days"] = int(m_days.group(1))
        else:
            inputs["working_days"] = 250

        # Validate required
        if "demand" not in inputs:
            missing.append("demand (annual units)")
        if "order_cost" not in inputs:
            missing.append("order_cost ($/order)")
        if "holding_cost" not in inputs and not ("holding_cost_rate" in inputs and "unit_cost" in inputs):
            missing.append("holding_cost ($/unit/year or holding_cost_rate %)")

        return inputs, missing

    def _extract_quantity_discount_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        inputs: Dict[str, Any] = {}
        missing = []

        # Demand
        m_demand = re.search(r"(?:annual\s*demand|demand|sells?)\s*(?:for\s+[\w\s]+?)?\s*(?:is|of|for|=|:)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_demand:
            inputs["demand"] = float(m_demand.group(1))

        # Order Cost
        m_order = re.search(r"(?:order(?:ing)?|setup)\s*cost\s*(?:is|of)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if not m_order:
            m_order = re.search(r"\$(\d+(?:\.\d+)?)\s*per order", text, re.I)
        if m_order:
            inputs["order_cost"] = float(m_order.group(1))

        # Holding cost rate
        m_rate = re.search(r"holding\s*cost\s*rate\s*(?:is|of)?\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        if not m_rate:
            m_rate = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of unit price|holding)", text, re.I)
        if m_rate:
            inputs["holding_cost_rate"] = float(m_rate.group(1)) / 100.0

        # Tiers extraction
        # Look for patterns like "1 to 499 units at $10.00" or "500-999 units at $9.50" or "1000+ units at $9.00"
        tiers: List[Dict[str, Any]] = []
        tier_matches = re.findall(r"(?:Tier\s*\d+:\s*)?(\d+)\s*(?:to|-)\s*(\d+)\s*(?:units)?\s*(?:at|for|\$)\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        for tm in tier_matches:
            tiers.append({
                "min_qty": float(tm[0]),
                "max_qty": float(tm[1]),
                "unit_price": float(tm[2]),
            })

        # Match open-ended top tier (e.g., 1000 units or more at $9.00 or 1000+ units at $9.00)
        top_tier = re.search(r"(\d+)\s*(?:units\s*)?(?:\+|or more|and up|and above)\s*(?:units)?\s*(?:at|for|\$)\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if top_tier:
            tiers.append({
                "min_qty": float(top_tier.group(1)),
                "max_qty": None,
                "unit_price": float(top_tier.group(2)),
            })

        if tiers:
            inputs["discount_tiers"] = tiers

        if "demand" not in inputs:
            missing.append("demand")
        if "order_cost" not in inputs:
            missing.append("order_cost")
        if "holding_cost_rate" not in inputs:
            missing.append("holding_cost_rate")
        if "discount_tiers" not in inputs or len(inputs["discount_tiers"]) < 2:
            missing.append("discount_tiers (at least 2 tiers)")

        return inputs, missing

    def _extract_safety_stock_rop_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        inputs: Dict[str, Any] = {}
        missing = []

        # Lead time
        m_lead = re.search(r"lead\s*time\s*(?:is|of)?\s*(\d+(?:\.\d+)?)\s*days?", text, re.I)
        if m_lead:
            inputs["lead_time_days"] = float(m_lead.group(1))

        # Stochastic lead time std
        m_lead_std = re.search(r"lead\s*time\s*std\w*\s*(?:dev|deviation)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_lead_std:
            inputs["lead_time_std_days"] = float(m_lead_std.group(1))
        else:
            inputs["lead_time_std_days"] = 0.0

        # Daily demand mean
        m_mean = re.search(r"(?:average|mean)?\s*daily\s*demand\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if not m_mean:
            m_mean = re.search(r"demand\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*units?\s*per\s*day", text, re.I)
        if m_mean:
            inputs["demand_mean_daily"] = float(m_mean.group(1))

        # Daily demand std
        m_std = re.search(r"(?:standard\s*deviation|std\s*dev)\s*(?:of\s*(?:daily\s*)?demand)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_std:
            inputs["demand_std_daily"] = float(m_std.group(1))

        # Service Level
        m_csl = re.search(r"(?:service\s*level|csl)\s*(?:is|of|target)?\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        if not m_csl:
            m_csl = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s*)?service\s*level", text, re.I)
        if m_csl:
            val = float(m_csl.group(1))
            inputs["service_level"] = val / 100.0 if val > 1.0 else val

        if "lead_time_days" not in inputs:
            missing.append("lead_time_days")
        if "demand_mean_daily" not in inputs:
            missing.append("demand_mean_daily")
        if "demand_std_daily" not in inputs:
            missing.append("demand_std_daily")
        if "service_level" not in inputs:
            missing.append("service_level")

        return inputs, missing

    def _extract_fixed_period_review_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        inputs: Dict[str, Any] = {}
        missing = []

        # Review period
        m_rev = re.search(r"reviews?\s*(?:inventory\s*)?(?:every|period\s*(?:is|of)?)?\s*(\d+(?:\.\d+)?)\s*days?", text, re.I)
        if not m_rev:
            m_rev = re.search(r"every\s*(\d+(?:\.\d+)?)\s*days", text, re.I)
        if m_rev:
            inputs["review_period_days"] = float(m_rev.group(1))

        # Lead time
        m_lead = re.search(r"lead\s*time\s*(?:is|of)?\s*(\d+(?:\.\d+)?)\s*days?", text, re.I)
        if m_lead:
            inputs["lead_time_days"] = float(m_lead.group(1))

        # Mean daily demand
        m_mean = re.search(r"(?:average|mean)?\s*daily\s*demand\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if not m_mean:
            m_mean = re.search(r"demand\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*units?\s*per\s*day", text, re.I)
        if m_mean:
            inputs["demand_mean_daily"] = float(m_mean.group(1))

        # Std daily demand
        m_std = re.search(r"(?:standard\s*deviation|std\s*dev)\s*(?:of\s*(?:daily\s*)?demand)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_std:
            inputs["demand_std_daily"] = float(m_std.group(1))

        # Service level
        m_csl = re.search(r"(?:service\s*level|target\s*service\s*level)\s*(?:is|of)?\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        if not m_csl:
            m_csl = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s*)?service\s*level", text, re.I)
        if m_csl:
            val = float(m_csl.group(1))
            inputs["service_level"] = val / 100.0 if val > 1.0 else val

        # On-hand
        m_onhand = re.search(r"(\d+(?:\.\d+)?)\s*(?:units\s*)?(?:are\s*)?on[-\s]*hand", text, re.I)
        if not m_onhand:
            m_onhand = re.search(r"on[-\s]*hand(?:\s*inventory)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        inputs["on_hand"] = float(m_onhand.group(1)) if m_onhand else 0.0

        # On-order
        m_onorder = re.search(r"(\d+(?:\.\d+)?)\s*(?:units\s*)?(?:are\s*)?on[-\s]*order", text, re.I)
        if not m_onorder:
            m_onorder = re.search(r"on[-\s]*order(?:\s*inventory)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        inputs["on_order"] = float(m_onorder.group(1)) if m_onorder else 0.0

        # Backorders
        m_bo = re.search(r"(\d+(?:\.\d+)?)\s*(?:units\s*)?(?:are\s*)?(?:on\s*)?back[-\s]*orders?", text, re.I)
        if not m_bo:
            m_bo = re.search(r"back[-\s]*orders?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        inputs["backorders"] = float(m_bo.group(1)) if m_bo else 0.0

        if "review_period_days" not in inputs:
            missing.append("review_period_days")
        if "lead_time_days" not in inputs:
            missing.append("lead_time_days")
        if "demand_mean_daily" not in inputs:
            missing.append("demand_mean_daily")
        if "demand_std_daily" not in inputs:
            missing.append("demand_std_daily")
        if "service_level" not in inputs:
            missing.append("service_level")

        return inputs, missing

    def _extract_newsvendor_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        inputs: Dict[str, Any] = {}
        missing = []

        # Selling price
        m_price = re.search(r"(?:selling\s*price|retail\s*price|sells?\s*for)\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if not m_price:
            m_price = re.search(r"price\s*(?:is|of)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if m_price:
            inputs["selling_price"] = float(m_price.group(1))

        # Unit cost
        m_cost = re.search(r"(?:costs?\s*us|acquisition\s*cost|wholesale\s*cost|unit\s*cost|purchase\s*cost)\s*(?:is|of)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if not m_cost:
            m_cost = re.search(r"costs?\s*\$?(\d+(?:\.\d+)?)\s*to (?:buy|purchase|produce)", text, re.I)
        if m_cost:
            inputs["unit_cost"] = float(m_cost.group(1))

        # Salvage value
        m_salvage = re.search(r"salvage\s*(?:value|price)?\s*(?:is|of|at)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if not m_salvage:
            m_salvage = re.search(r"liquidat\w*\s*(?:at|for)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if m_salvage:
            inputs["salvage_value"] = float(m_salvage.group(1))
        else:
            inputs["salvage_value"] = 0.0

        # Mean demand
        m_mean = re.search(r"(?:mean|average)\s*(?:demand|seasonal demand)?\s*(?:is|of)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if not m_mean:
            m_mean = re.search(r"demand\s*(?:follows\s*a\s*normal\s*distribution\s*with\s*mean|of)\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_mean:
            inputs["demand_mean"] = float(m_mean.group(1))

        # Std demand
        m_std = re.search(r"(?:standard\s*deviation|std\s*dev)\s*(?:of|is)?\s*(\d+(?:\.\d+)?)", text, re.I)
        if m_std:
            inputs["demand_std"] = float(m_std.group(1))

        if "selling_price" not in inputs:
            missing.append("selling_price")
        if "unit_cost" not in inputs:
            missing.append("unit_cost")
        if "demand_mean" not in inputs:
            missing.append("demand_mean")
        if "demand_std" not in inputs:
            missing.append("demand_std")

        return inputs, missing

    # =========================================================================
    # Step 4 & 5: Tool Selection and Calculation
    # =========================================================================

    def select_tool(self, model_type: InventoryModelType) -> Tuple[Callable, str]:
        """Resolves tool function and human-readable tool name."""
        tool_func = self.tools[model_type]
        return tool_func, f"inventory_tools.{tool_func.__name__}"

    def calculate(self, tool_func: Callable, inputs: Dict[str, Any]) -> Any:
        """Executes mathematical tool with precise Python numerical operations."""
        return tool_func(**inputs)

    # =========================================================================
    # Step 6: Verification
    # =========================================================================

    def verify(self, model_type: InventoryModelType, inputs: Dict[str, Any], result: Any) -> Tuple[bool, List[VerificationCheck]]:
        """
        Validates mathematical constraints, boundary physics, and managerial sanity.
        Returns: (all_passed, checks_list)
        """
        checks: List[VerificationCheck] = []

        if model_type == InventoryModelType.EOQ:
            # Check 1: Positive order quantity
            checks.append(VerificationCheck(
                name="Strictly Positive Order Quantity",
                passed=result.optimal_order_quantity > 0,
                details=f"Calculated Q* = {result.optimal_order_quantity:.2f} units."
            ))
            # Check 2: Cost parity check (|Ordering Cost - Holding Cost| / Total Cost <= 0.001)
            diff = abs(result.annual_ordering_cost - result.annual_holding_cost)
            rel_diff = diff / max(result.total_annual_inventory_cost, 1e-6)
            checks.append(VerificationCheck(
                name="Ordering vs Holding Cost Equivalence",
                passed=rel_diff < 0.001,
                details=f"Ordering Cost (${result.annual_ordering_cost:.2f}) vs Holding Cost (${result.annual_holding_cost:.2f}); relative delta = {rel_diff * 100:.4f}%."
            ))
            # Check 3: Practical order cycle sanity
            is_sane_cycle = 0.5 <= result.orders_per_year <= 365
            checks.append(VerificationCheck(
                name="Operational Frequency Sanity",
                passed=is_sane_cycle,
                details=f"Orders per year: {result.orders_per_year:.1f} (every {result.cycle_time_days:.1f} working days).",
                severity="WARNING" if not is_sane_cycle else "INFO"
            ))

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            # Check 1: Optimal tier cost <= base tier cost
            checks.append(VerificationCheck(
                name="Economic Dominance over Base Tier",
                passed=result.total_annual_cost <= result.base_tier_cost + 1e-4,
                details=f"Selected Tier {result.optimal_tier_index} Total Cost (${result.total_annual_cost:,.2f}) <= Base Tier (${result.base_tier_cost:,.2f})."
            ))
            # Check 2: Quantity meets tier minimum breakpoint
            optimal_eval = result.tier_evaluations[result.optimal_tier_index]
            checks.append(VerificationCheck(
                name="Tier Feasibility & Price Breakpoint",
                passed=result.optimal_quantity >= optimal_eval.min_qty,
                details=f"Realized Q ({result.optimal_quantity:.1f}) meets or exceeds tier breakpoint minimum ({optimal_eval.min_qty})."
            ))

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            # Check 1: Positive safety stock for CSL > 50%
            checks.append(VerificationCheck(
                name="Safety Stock Buffer Positivity",
                passed=result.safety_stock > 0,
                details=f"Safety stock is {result.safety_stock:.2f} units for {result.service_level * 100:.1f}% service level."
            ))
            # Check 2: ROP = DL + SS
            rop_expected = result.lead_time_demand + result.safety_stock
            checks.append(VerificationCheck(
                name="Reorder Point Formula Consistency",
                passed=abs(result.reorder_point - rop_expected) < 1e-4,
                details=f"ROP ({result.reorder_point:.2f}) matches Lead Time Demand ({result.lead_time_demand:.2f}) + Safety Stock ({result.safety_stock:.2f})."
            ))

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Check 1: Protection interval covers review period + lead time
            expected_pi = result.review_period_days + result.lead_time_days
            checks.append(VerificationCheck(
                name="Protection Interval Validity",
                passed=abs(result.protection_interval_days - expected_pi) < 1e-4,
                details=f"Protection interval is {result.protection_interval_days:.1f} days (P={result.review_period_days:.1f}d + L={result.lead_time_days:.1f}d)."
            ))
            # Check 2: Non-negative order quantity
            checks.append(VerificationCheck(
                name="Non-Negative Replenishment Order",
                passed=result.recommended_order_quantity >= 0,
                details=f"Order quantity calculated: {result.recommended_order_quantity:.2f} units."
            ))

        elif model_type == InventoryModelType.NEWSVENDOR:
            # Check 1: Critical ratio in (0, 1)
            checks.append(VerificationCheck(
                name="Critical Fractile Domain Validity",
                passed=0.0 < result.critical_ratio < 1.0,
                details=f"Critical Ratio CR = {result.critical_ratio:.4f} (Cu=${result.cost_underage:.2f}, Co=${result.cost_overage:.2f})."
            ))
            # Check 2: Expected profit positivity
            checks.append(VerificationCheck(
                name="Positive Expected Economic Profit",
                passed=result.expected_profit > 0,
                details=f"Expected net seasonal profit: ${result.expected_profit:,.2f}."
            ))

        all_passed = all(c.passed for c in checks if c.severity != "WARNING")
        return all_passed, checks

    # =========================================================================
    # Step 7: Explain and Formulate Business Recommendations
    # =========================================================================

    def explain(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any,
        checks: List[VerificationCheck],
    ) -> Tuple[str, str, str, List[str]]:
        """
        Synthesizes results into:
        1. Executive Summary
        2. Mathematical Derivation
        3. Cost & Trade-off Analysis
        4. Actionable Strategic Recommendations
        """
        if model_type == InventoryModelType.EOQ:
            summary = (
                f"**Optimal Order Quantity (EOQ): {result.optimal_order_quantity:.0f} units per replenishment order.**\n"
                f"To fulfill annual demand of {result.annual_demand:,.0f} units, place **{result.orders_per_year:.1f} orders per year** "
                f"(approximately once every **{result.cycle_time_days:.1f} working days**). "
                f"This minimizes total annual inventory holding and ordering costs at **${result.total_annual_inventory_cost:,.2f}**."
            )
            derivation = (
                f"- **EOQ Formula:** $Q^* = \\sqrt{{\\frac{{2DS}}{{H}}}}$\n"
                f"- **Substitution:** $Q^* = \\sqrt{{\\frac{{2 \\times {result.annual_demand:,.0f} \\times \\${result.order_cost:.2f}}}{{\\${result.holding_cost_per_unit:.2f}}}}} = \\mathbf{{{result.optimal_order_quantity:.2f}\\text{{ units}}}}$\n"
                f"- **Annual Order Frequency:** $N = \\frac{{D}}{{Q^*}} = \\frac{{{result.annual_demand:,.0f}}}{{{result.optimal_order_quantity:.2f}}} = \\mathbf{{{result.orders_per_year:.2f}\\text{{ orders/yr}}}}$\n"
                f"- **Order Interval / Cycle Time:** $T = \\frac{{Q^*}}{{D}} \\times {result.working_days_per_year} = \\mathbf{{{result.cycle_time_days:.1f}\\text{{ working days}}}}$"
            )
            cost_analysis = (
                f"| Cost Component | Annual Amount | % of Inventory Cost |\n"
                f"|---|---|---|\n"
                f"| Annual Ordering Cost | ${result.annual_ordering_cost:,.2f} | 50.0% |\n"
                f"| Annual Holding Cost | ${result.annual_holding_cost:,.2f} | 50.0% |\n"
                f"| **Total Relevant Inventory Cost** | **${result.total_annual_inventory_cost:,.2f}** | **100.0%** |\n"
            )
            if result.total_annual_cost_with_purchase is not None:
                cost_analysis += f"| Annual Direct Material Cost | ${result.annual_purchase_cost:,.2f} | — |\n"
                cost_analysis += f"| **Total Annual Procurement Cost** | **${result.total_annual_cost_with_purchase:,.2f}** | — |\n"

            recommendations = [
                f"Establish standard purchase order batch size of {result.optimal_order_quantity:.0f} units with suppliers.",
                f"Schedule replenishment orders roughly every {result.cycle_time_days:.0f} working days ({result.orders_per_year:.1f} times annually).",
                "Invest in electronic purchase ordering (EDI/API) to reduce setup cost ($S), which will naturally lower optimal batch sizes and free working capital.",
            ]

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            summary = (
                f"**Recommended Discount Tier: Tier {result.optimal_tier_index} at ${result.optimal_unit_price:.2f}/unit with Order Quantity {result.optimal_quantity:.0f} units.**\n"
                f"Taking advantage of this volume tier generates **${result.savings_vs_base_tier:,.2f}** in net annual savings "
                f"(a **{result.percentage_savings:.2f}% total cost reduction**) compared to ordering at baseline single-tier pricing."
            )
            derivation = (
                f"- **All-Units Total Cost Equation:** $TC(Q) = D \\cdot C + \\frac{{D}}{{Q}} S + \\frac{{Q}}{{2}} (i \\cdot C)$\n"
                f"- **Optimal Quantity Selected:** Tier {result.optimal_tier_index} breakpoint $Q = \\mathbf{{{result.optimal_quantity:.0f}\\text{{ units}}}}$\n"
                f"- **Purchase Price:** $\\mathbf{{\\${result.optimal_unit_price:.2f}/unit}}$ vs Base Tier $\\${result.base_tier_cost / result.annual_demand:.2f}/unit$\n"
                f"- **Net Annual Financial Advantage:** $\\mathbf{{\\${result.savings_vs_base_tier:,.2f}}}$"
            )
            cost_analysis = (
                f"| Tier | Price/Unit | Order Qty | Purchase Cost | Order Cost | Holding Cost | Total Annual Cost | Status |\n"
                f"|---|---|---|---|---|---|---|---|\n"
            )
            for t in result.tier_evaluations:
                prefix = "**" if t.tier_index == result.optimal_tier_index else ""
                suffix = " (BEST)**" if t.tier_index == result.optimal_tier_index else ""
                cost_analysis += (
                    f"| {prefix}Tier {t.tier_index}{suffix} | ${t.unit_price:.2f} | {t.realized_order_quantity:.0f} | "
                    f"${t.annual_purchase_cost:,.0f} | ${t.annual_ordering_cost:,.0f} | ${t.annual_holding_cost:,.0f} | "
                    f"${t.total_annual_cost:,.2f} | {'Feasible' if t.is_feasible else 'Dominated'} |\n"
                )
            recommendations = [
                f"Increase order size to {result.optimal_quantity:.0f} units to lock in the ${result.optimal_unit_price:.2f}/unit discounted price bracket.",
                f"Warehouse capacity check: ordering {result.optimal_quantity:.0f} units increases cycle inventory holding costs by ${result.annual_holding_cost - result.tier_evaluations[0].annual_holding_cost:,.2f}, which is heavily compensated by ${result.tier_evaluations[0].annual_purchase_cost - result.annual_purchase_cost:,.2f} in purchase discounts.",
                "Ensure sufficient warehouse physical footprint exists before issuing larger purchase orders.",
            ]

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            summary = (
                f"**Reorder Point (ROP): {result.reorder_point:.0f} units | Safety Stock: {result.safety_stock:.0f} units.**\n"
                f"To maintain a **{result.service_level * 100:.1f}% Cycle Service Level** across a {result.lead_time_days:.0f}-day lead time, "
                f"place a replenishment order whenever inventory position drops to **{result.reorder_point:.0f} units**. "
                f"Stockout risk per replenishment cycle is strictly bounded to **{result.stockout_probability_per_cycle * 100:.1f}%**."
            )
            derivation = (
                f"- **Mean Lead Time Demand:** $DL = d \\times L = {result.demand_mean_daily:.1f} \\times {result.lead_time_days:.1f} = \\mathbf{{{result.lead_time_demand:.2f}\\text{{ units}}}}$\n"
                f"- **Lead Time Standard Deviation:** $\\sigma_{{DL}} = \\sigma_d \\sqrt{{L}} = {result.demand_std_daily:.1f} \\times \\sqrt{{{result.lead_time_days:.1f}}} = \\mathbf{{{result.lead_time_std:.2f}\\text{{ units}}}}$\n"
                f"- **Safety Factor (Z for {result.service_level * 100:.1f}% CSL):** $Z = \\Phi^{{-1}}({result.service_level:.4f}) = \\mathbf{{{result.z_factor:.4f}}}$\n"
                f"- **Safety Stock:** $SS = Z \\times \\sigma_{{DL}} = {result.z_factor:.4f} \\times {result.lead_time_std:.2f} = \\mathbf{{{result.safety_stock:.2f}\\text{{ units}}}}$\n"
                f"- **Reorder Point:** $ROP = DL + SS = {result.lead_time_demand:.2f} + {result.safety_stock:.2f} = \\mathbf{{{result.reorder_point:.2f}\\text{{ units}}}}$"
            )
            cost_analysis = (
                f"| Metric | Value | Interpretation |\n"
                f"|---|---|---|\n"
                f"| Mean Lead Time Demand | {result.lead_time_demand:.1f} units | Expected consumption during supplier transit |\n"
                f"| Buffer / Safety Stock | {result.safety_stock:.1f} units | Statistical protection against demand spikes |\n"
                f"| Reorder Point Trigger | **{result.reorder_point:.1f} units** | Inventory threshold to issue replenishment PO |\n"
                f"| Expected Cycle Shortage | {result.expected_units_short_per_cycle:.3f} units | Average unmet units per replenishment cycle |\n"
                f"| Cycle Service Level | {result.service_level * 100:.1f}% | Probability of zero stockouts during lead time |\n"
            )
            recommendations = [
                f"Configure ERP/WMS automated trigger: issue PO immediately when Inventory Position <= {result.reorder_point:.0f} units.",
                f"If supplier lead time could be shortened from {result.lead_time_days:.0f} to {max(1.0, result.lead_time_days / 2):.0f} days, safety stock could be reduced by ~29%, unlocking working capital.",
                "Review demand variability monthly; if demand standard deviation shifts, adjust Z and safety stock accordingly.",
            ]

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            summary = (
                f"**Target Order-Up-To Level (S): {result.order_up_to_level:.0f} units | Recommended Order: {result.recommended_order_quantity:.0f} units.**\n"
                f"For a periodic review cadence of **{result.review_period_days:.0f} days** and lead time of **{result.lead_time_days:.0f} days**, "
                f"the protection interval is **{result.protection_interval_days:.0f} days**. "
                f"With current inventory position of **{result.effective_inventory_position:.0f} units**, place an order for **{result.recommended_order_quantity:.0f} units** today."
            )
            derivation = (
                f"- **Protection Interval:** $T = P + L = {result.review_period_days:.0f} + {result.lead_time_days:.0f} = \\mathbf{{{result.protection_interval_days:.0f}\\text{{ days}}}}$\n"
                f"- **Expected Protection Demand:** $D_{{P+L}} = \\mu_d \\times (P + L) = {result.demand_mean_daily:.1f} \\times {result.protection_interval_days:.0f} = \\mathbf{{{result.protection_interval_demand:.2f}\\text{{ units}}}}$\n"
                f"- **Protection Std Dev:** $\\sigma_{{P+L}} = \\sigma_d \\sqrt{{P + L}} = {result.demand_std_daily:.1f} \\times \\sqrt{{{result.protection_interval_days:.0f}}} = \\mathbf{{{result.protection_interval_std:.2f}\\text{{ units}}}}$\n"
                f"- **Safety Stock:** $SS = Z \\times \\sigma_{{P+L}} = {result.z_factor:.4f} \\times {result.protection_interval_std:.2f} = \\mathbf{{{result.safety_stock:.2f}\\text{{ units}}}}$\n"
                f"- **Order-Up-To Level (S):** $S = D_{{P+L}} + SS = \\mathbf{{{result.order_up_to_level:.2f}\\text{{ units}}}}$\n"
                f"- **Effective Inventory Position (IP):** $\\text{{On-Hand}} ({result.on_hand:.0f}) + \\text{{On-Order}} ({result.on_order:.0f}) - \\text{{Backorders}} ({result.backorders:.0f}) = \\mathbf{{{result.effective_inventory_position:.0f}\\text{{ units}}}}$\n"
                f"- **Replenishment Order:** $q = \\max(0, S - IP) = \\mathbf{{{result.recommended_order_quantity:.2f}\\text{{ units}}}}$"
            )
            cost_analysis = (
                f"| Inventory Position Factor | Value (Units) |\n"
                f"|---|---|\n"
                f"| Target Order-Up-To Level ($S$) | {result.order_up_to_level:.1f} |\n"
                f"| Current On-Hand Inventory | {result.on_hand:.1f} |\n"
                f"| Current In-Transit On-Order | +{result.on_order:.1f} |\n"
                f"| Unfulfilled Backorders | -{result.backorders:.1f} |\n"
                f"| Effective Inventory Position ($IP$) | = {result.effective_inventory_position:.1f} |\n"
                f"| **Net Recommended Order Quantity ($q$)** | **{result.recommended_order_quantity:.1f}** |\n"
            )
            recommendations = [
                f"Place an order immediately for {result.recommended_order_quantity:.0f} units to restore inventory position to target {result.order_up_to_level:.0f} units.",
                f"Maintain strict review rhythm every {result.review_period_days:.0f} calendar days.",
                "Note that periodic review systems require higher safety stock than continuous review (ROP) because protection must cover the review period PLUS lead time.",
            ]

        elif model_type == InventoryModelType.NEWSVENDOR:
            summary = (
                f"**Optimal Stocking Quantity: {result.optimal_order_quantity:.0f} units | Critical Ratio (CR): {result.critical_ratio * 100:.1f}%.**\n"
                f"With a retail price of **${result.selling_price:.2f}**, unit cost of **${result.unit_cost:.2f}**, and salvage liquidation value of **${result.salvage_value:.2f}**, "
                f"the marginal profit of selling an item ($Cu = ${result.cost_underage:.2f}) outweighs overage risk ($Co = ${result.cost_overage:.2f}). "
                f"Ordering **{result.optimal_order_quantity:.0f} units** delivers an expected seasonal net profit of **${result.expected_profit:,.2f}**."
            )
            derivation = (
                f"- **Underage Cost (Lost Margin):** $C_u = p - c = \\${result.selling_price:.2f} - \\${result.unit_cost:.2f} = \\mathbf{{\\${result.cost_underage:.2f}}}$\n"
                f"- **Overage Cost (Depreciation Loss):** $C_o = c - s = \\${result.unit_cost:.2f} - \\${result.salvage_value:.2f} = \\mathbf{{\\${result.cost_overage:.2f}}}$\n"
                f"- **Critical Fractile Ratio:** $CR = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{result.cost_underage:.2f}}}{{{result.cost_underage + result.cost_overage:.2f}}} = \\mathbf{{{result.critical_ratio:.4f}}}$\n"
                f"- **Optimal Standard Normal Z:** $Z = \\Phi^{{-1}}({result.critical_ratio:.4f}) = \\mathbf{{{result.z_score:.4f}}}$\n"
                f"- **Optimal Order Quantity:** $Q^* = \\mu + Z \\cdot \\sigma = {result.demand_mean:.1f} + ({result.z_score:.4f} \\times {result.demand_std:.1f}) = \\mathbf{{{result.optimal_order_quantity:.2f}\\text{{ units}}}}$"
            )
            cost_analysis = (
                f"| Financial & Operational Expectation | Value |\n"
                f"|---|---|\n"
                f"| Optimal Commitment Quantity | **{result.optimal_order_quantity:.0f} units** |\n"
                f"| Expected Full-Price Sales | {result.expected_sales:.1f} units |\n"
                f"| Expected Unsold Salvage Units | {result.expected_leftover:.1f} units |\n"
                f"| Expected Unmet Demand (Lost Sales) | {result.expected_lost_sales:.1f} units |\n"
                f"| Expected Gross Revenue | ${result.expected_revenue:,.2f} |\n"
                f"| Expected Salvage Revenue | ${result.expected_salvage_revenue:,.2f} |\n"
                f"| Expected Procurement Expense | ${result.expected_purchase_cost:,.2f} |\n"
                f"| **Expected Net Seasonal Profit** | **${result.expected_profit:,.2f}** |\n"
            )
            recommendations = [
                f"Commit to purchasing {result.optimal_order_quantity:.0f} units for the single selling season.",
                f"Because Critical Ratio is {result.critical_ratio * 100:.1f}%, ordering above the mean demand ({result.demand_mean:.0f} units) by {result.optimal_order_quantity - result.demand_mean:.0f} buffer units maximizes expected profit.",
                f"Negotiate higher buy-back / salvage terms with liquidators: every $5 increase in salvage value shifts optimal order quantity upward and increases expected profit.",
            ]

        else:
            summary = "Analysis complete."
            derivation = "N/A"
            cost_analysis = "N/A"
            recommendations = []

        return summary, derivation, cost_analysis, recommendations

    # =========================================================================
    # Full End-to-End Orchestrator Pipeline
    # =========================================================================

    def process_question(self, question: str, override_inputs: Optional[Dict[str, Any]] = None) -> DecisionTrace:
        """
        Executes the full 7-stage architecture:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        # Step 1 & 2: Identify Model
        model_type, confidence, rationale = self.identify_model(question)

        if model_type == InventoryModelType.UNKNOWN:
            return DecisionTrace(
                customer_question=question,
                identified_model=model_type,
                model_confidence=confidence,
                model_rationale=rationale,
                extracted_inputs={},
                missing_inputs=["query_not_recognized"],
                selected_tool_name="None",
                error_message="Could not classify question into any of the 5 supported inventory models.",
            )

        # Step 3: Extract Inputs
        extracted_inputs, missing = self.extract_inputs(question, model_type)
        if override_inputs:
            extracted_inputs.update(override_inputs)
            # Recompute missing
            missing = [m for m in missing if m.split()[0] not in override_inputs]

        # Step 4: Select Python Tool
        tool_func, tool_name = self.select_tool(model_type)

        if missing:
            return DecisionTrace(
                customer_question=question,
                identified_model=model_type,
                model_confidence=confidence,
                model_rationale=rationale,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing,
                selected_tool_name=tool_name,
                error_message=f"Missing required parameters for {model_type.value}: {', '.join(missing)}.",
            )

        # Step 5: Calculate
        try:
            calc_result = self.calculate(tool_func, extracted_inputs)
        except Exception as e:
            return DecisionTrace(
                customer_question=question,
                identified_model=model_type,
                model_confidence=confidence,
                model_rationale=rationale,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool_name=tool_name,
                error_message=f"Mathematical calculation failed: {str(e)}",
            )

        # Step 6: Verify
        verified, checks = self.verify(model_type, extracted_inputs, calc_result)

        # Step 7: Explain
        summary, derivation, cost_analysis, recommendations = self.explain(
            model_type, extracted_inputs, calc_result, checks
        )

        return DecisionTrace(
            customer_question=question,
            identified_model=model_type,
            model_confidence=confidence,
            model_rationale=rationale,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            selected_tool_name=tool_name,
            calculation_result=calc_result,
            verification_passed=verified,
            verification_checks=checks,
            executive_summary=summary,
            mathematical_derivation=derivation,
            cost_and_tradeoff_analysis=cost_analysis,
            managerial_recommendations=recommendations,
        )


# =============================================================================
# Curated Benchmark Scenarios
# =============================================================================

BENCHMARK_SCENARIOS = {
    "1. EOQ (Economic Order Quantity)": {
        "question": (
            "Our company sells 12,000 units per year. Each order incurs a fixed setup cost of $50, "
            "and holding inventory costs $2.00 per unit per year. Our facility operates 250 working days a year. "
            "What is our optimal order quantity, how many orders will we place per year, and what is our cycle time in days?"
        ),
        "expected_model": InventoryModelType.EOQ,
    },
    "2. Quantity Discount Model": {
        "question": (
            "Annual demand for industrial valves is 5,000 units. Ordering cost is $40 per order, "
            "and annual inventory holding cost rate is 20% of the unit purchase price. The supplier provides "
            "the following pricing tiers: Tier 1: 1 to 499 units at $10.00; Tier 2: 500 to 999 units at $9.50; "
            "Tier 3: 1,000 units or more at $9.00. Which order quantity maximizes our total cost savings?"
        ),
        "expected_model": InventoryModelType.QUANTITY_DISCOUNT,
    },
    "3. Safety Stock & Reorder Point": {
        "question": (
            "A retail distribution center has an average daily demand of 50 units with a standard deviation of 8 units. "
            "The supplier lead time is 6 days. The company targets a 95% cycle service level. "
            "What safety stock should be held and at what inventory level should a reorder point be triggered?"
        ),
        "expected_model": InventoryModelType.SAFETY_STOCK_ROP,
    },
    "4. Fixed-Period Review Model": {
        "question": (
            "A distribution warehouse reviews inventory every 14 days with a supplier lead time of 7 days. "
            "Average daily demand is 25 units with a standard deviation of 6 units. The target cycle service level is 98%. "
            "Currently, on-hand inventory is 210 units, 80 units are on order, and there are 10 backorders. "
            "What is the target order-up-to level and how many units should be ordered now?"
        ),
        "expected_model": InventoryModelType.FIXED_PERIOD_REVIEW,
    },
    "5. Newsvendor (Seasonal / Perishable)": {
        "question": (
            "A boutique retailer is ordering a seasonal summer dress. Retail selling price is $120, "
            "wholesale acquisition cost is $50, and any unsold dresses at the end of the season can be liquidated "
            "at a salvage value of $20. Market research estimates seasonal demand to be normally distributed "
            "with a mean of 400 units and standard deviation of 80 units. "
            "What is the optimal number of dresses to order, the critical ratio, and expected profit?"
        ),
        "expected_model": InventoryModelType.NEWSVENDOR,
    },
}
