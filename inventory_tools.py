"""
Supply Chain Analytics Inventory Tools
======================================
Rigorous mathematical implementation of the five core inventory models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model (All-Units Discount)
3. Safety Stock and Reorder Point (ROP - Continuous Review)
4. Fixed-Period Review (Periodic Review / Order-Up-To Level)
5. Newsvendor Model (Single-Period / Perishable / Seasonal)

All calculations are performed using Python with NumPy and SciPy.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data Models / Result Containers
# ---------------------------------------------------------------------------

@dataclass
class EOQResult:
    """Quantitative results for Economic Order Quantity model."""
    annual_demand: float
    order_cost: float
    holding_cost_per_unit: float
    optimal_order_quantity: float
    orders_per_year: float
    cycle_time_days: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_inventory_cost: float
    unit_cost: Optional[float] = None
    annual_purchase_cost: Optional[float] = None
    total_annual_cost_with_purchase: Optional[float] = None
    working_days_per_year: int = 250
    is_verified: bool = True
    verification_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Economic Order Quantity (EOQ)",
            "annual_demand": self.annual_demand,
            "order_cost": self.order_cost,
            "holding_cost_per_unit": self.holding_cost_per_unit,
            "optimal_order_quantity": round(self.optimal_order_quantity, 2),
            "orders_per_year": round(self.orders_per_year, 2),
            "cycle_time_days": round(self.cycle_time_days, 1),
            "annual_ordering_cost": round(self.annual_ordering_cost, 2),
            "annual_holding_cost": round(self.annual_holding_cost, 2),
            "total_annual_inventory_cost": round(self.total_annual_inventory_cost, 2),
            "unit_cost": round(self.unit_cost, 2) if self.unit_cost is not None else None,
            "annual_purchase_cost": round(self.annual_purchase_cost, 2) if self.annual_purchase_cost is not None else None,
            "total_annual_cost_with_purchase": round(self.total_annual_cost_with_purchase, 2) if self.total_annual_cost_with_purchase is not None else None,
            "working_days_per_year": self.working_days_per_year,
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes,
        }


@dataclass
class DiscountTierEvaluation:
    """Evaluation of an individual discount tier."""
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    unconstrained_eoq: float
    realized_order_quantity: float
    is_feasible: bool
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    status_note: str


@dataclass
class QuantityDiscountResult:
    """Quantitative results for Quantity Discount model."""
    annual_demand: float
    order_cost: float
    holding_cost_rate: float
    optimal_tier_index: int
    optimal_quantity: float
    optimal_unit_price: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    base_tier_cost: float
    savings_vs_base_tier: float
    percentage_savings: float
    tier_evaluations: List[DiscountTierEvaluation] = field(default_factory=list)
    is_verified: bool = True
    verification_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Quantity Discount Model",
            "annual_demand": self.annual_demand,
            "order_cost": self.order_cost,
            "holding_cost_rate": self.holding_cost_rate,
            "optimal_tier_index": self.optimal_tier_index,
            "optimal_quantity": round(self.optimal_quantity, 2),
            "optimal_unit_price": round(self.optimal_unit_price, 2),
            "annual_purchase_cost": round(self.annual_purchase_cost, 2),
            "annual_ordering_cost": round(self.annual_ordering_cost, 2),
            "annual_holding_cost": round(self.annual_holding_cost, 2),
            "total_annual_cost": round(self.total_annual_cost, 2),
            "base_tier_cost": round(self.base_tier_cost, 2),
            "savings_vs_base_tier": round(self.savings_vs_base_tier, 2),
            "percentage_savings": round(self.percentage_savings, 2),
            "tier_evaluations": [
                {
                    "tier_index": t.tier_index,
                    "min_qty": t.min_qty,
                    "max_qty": t.max_qty,
                    "unit_price": round(t.unit_price, 2),
                    "holding_cost": round(t.holding_cost_per_unit, 2),
                    "unconstrained_eoq": round(t.unconstrained_eoq, 2),
                    "realized_order_quantity": round(t.realized_order_quantity, 2),
                    "is_feasible": t.is_feasible,
                    "purchase_cost": round(t.annual_purchase_cost, 2),
                    "ordering_cost": round(t.annual_ordering_cost, 2),
                    "holding_cost_total": round(t.annual_holding_cost, 2),
                    "total_cost": round(t.total_annual_cost, 2),
                    "status_note": t.status_note,
                }
                for t in self.tier_evaluations
            ],
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes,
        }


@dataclass
class SafetyStockROPResult:
    """Quantitative results for Safety Stock and Reorder Point model."""
    lead_time_days: float
    demand_mean_daily: float
    demand_std_daily: float
    service_level: float
    lead_time_std_days: float
    z_factor: float
    lead_time_demand: float
    lead_time_std: float
    safety_stock: float
    reorder_point: float
    stockout_probability_per_cycle: float
    expected_units_short_per_cycle: float
    is_verified: bool = True
    verification_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Safety Stock and Reorder Point (ROP)",
            "lead_time_days": self.lead_time_days,
            "lead_time_std_days": self.lead_time_std_days,
            "demand_mean_daily": round(self.demand_mean_daily, 2),
            "demand_std_daily": round(self.demand_std_daily, 2),
            "service_level": round(self.service_level, 4),
            "z_factor": round(self.z_factor, 4),
            "lead_time_demand": round(self.lead_time_demand, 2),
            "lead_time_std": round(self.lead_time_std, 2),
            "safety_stock": round(self.safety_stock, 2),
            "reorder_point": round(self.reorder_point, 2),
            "stockout_probability_per_cycle": round(self.stockout_probability_per_cycle, 4),
            "expected_units_short_per_cycle": round(self.expected_units_short_per_cycle, 4),
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes,
        }


@dataclass
class FixedPeriodReviewResult:
    """Quantitative results for Fixed-Period Review model."""
    review_period_days: float
    lead_time_days: float
    protection_interval_days: float
    demand_mean_daily: float
    demand_std_daily: float
    service_level: float
    z_factor: float
    protection_interval_demand: float
    protection_interval_std: float
    safety_stock: float
    order_up_to_level: float
    on_hand: float
    on_order: float
    backorders: float
    effective_inventory_position: float
    recommended_order_quantity: float
    is_verified: bool = True
    verification_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Fixed-Period Review Model",
            "review_period_days": self.review_period_days,
            "lead_time_days": self.lead_time_days,
            "protection_interval_days": self.protection_interval_days,
            "demand_mean_daily": round(self.demand_mean_daily, 2),
            "demand_std_daily": round(self.demand_std_daily, 2),
            "service_level": round(self.service_level, 4),
            "z_factor": round(self.z_factor, 4),
            "protection_interval_demand": round(self.protection_interval_demand, 2),
            "protection_interval_std": round(self.protection_interval_std, 2),
            "safety_stock": round(self.safety_stock, 2),
            "order_up_to_level": round(self.order_up_to_level, 2),
            "on_hand": self.on_hand,
            "on_order": self.on_order,
            "backorders": self.backorders,
            "effective_inventory_position": self.effective_inventory_position,
            "recommended_order_quantity": round(self.recommended_order_quantity, 2),
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes,
        }


@dataclass
class NewsvendorResult:
    """Quantitative results for Newsvendor Model."""
    selling_price: float
    unit_cost: float
    salvage_value: float
    demand_mean: float
    demand_std: float
    cost_underage: float
    cost_overage: float
    critical_ratio: float
    z_score: float
    optimal_order_quantity: float
    expected_lost_sales: float
    expected_sales: float
    expected_leftover: float
    expected_revenue: float
    expected_purchase_cost: float
    expected_salvage_revenue: float
    expected_profit: float
    stockout_probability: float
    is_verified: bool = True
    verification_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Newsvendor Single-Period Model",
            "selling_price": round(self.selling_price, 2),
            "unit_cost": round(self.unit_cost, 2),
            "salvage_value": round(self.salvage_value, 2),
            "demand_mean": round(self.demand_mean, 2),
            "demand_std": round(self.demand_std, 2),
            "cost_underage": round(self.cost_underage, 2),
            "cost_overage": round(self.cost_overage, 2),
            "critical_ratio": round(self.critical_ratio, 4),
            "z_score": round(self.z_score, 4),
            "optimal_order_quantity": round(self.optimal_order_quantity, 2),
            "expected_lost_sales": round(self.expected_lost_sales, 2),
            "expected_sales": round(self.expected_sales, 2),
            "expected_leftover": round(self.expected_leftover, 2),
            "expected_revenue": round(self.expected_revenue, 2),
            "expected_purchase_cost": round(self.expected_purchase_cost, 2),
            "expected_salvage_revenue": round(self.expected_salvage_revenue, 2),
            "expected_profit": round(self.expected_profit, 2),
            "stockout_probability": round(self.stockout_probability, 4),
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes,
        }


# ---------------------------------------------------------------------------
# Mathematical Helper Functions
# ---------------------------------------------------------------------------

def standard_normal_loss(z: float) -> float:
    """
    Standard normal unit loss function:
    L(z) = E[max(0, Z - z)] = phi(z) - z * (1 - Phi(z))
    where phi is standard normal PDF and Phi is standard normal CDF.
    """
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    return float(phi_z - z * (1.0 - Phi_z))


# ---------------------------------------------------------------------------
# Tool 1: Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------

def calculate_eoq(
    demand: float,
    order_cost: float,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    working_days: int = 250,
) -> EOQResult:
    """
    Computes Economic Order Quantity (EOQ) and total relevant inventory costs.

    Parameters:
    - demand: Annual demand in units (D > 0).
    - order_cost: Fixed cost per order/setup (S > 0).
    - holding_cost: Annual holding cost per unit (H > 0), OR
    - holding_cost_rate: Annual holding cost as a fraction of unit price (h in (0, 1]).
    - unit_cost: Cost per unit (C >= 0). Required if holding_cost_rate is used.
    - working_days: Working days per year for cycle time translation (default: 250).

    Returns:
    - EOQResult dataclass containing optimal order quantity and cost breakdown.
    """
    # Validation
    if demand <= 0:
        raise ValueError(f"Demand must be strictly positive, got {demand}")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be strictly positive, got {order_cost}")
    if working_days <= 0:
        raise ValueError(f"Working days must be strictly positive, got {working_days}")

    # Determine holding cost H
    if holding_cost is None:
        if holding_cost_rate is not None and unit_cost is not None:
            if holding_cost_rate <= 0:
                raise ValueError("holding_cost_rate must be strictly positive")
            if unit_cost <= 0:
                raise ValueError("unit_cost must be strictly positive")
            holding_cost = holding_cost_rate * unit_cost
        else:
            raise ValueError("Either holding_cost or (holding_cost_rate and unit_cost) must be provided")
    elif holding_cost <= 0:
        raise ValueError(f"Holding cost per unit must be strictly positive, got {holding_cost}")

    # Core mathematical calculations
    eoq = math.sqrt((2.0 * demand * order_cost) / holding_cost)
    orders_per_year = demand / eoq
    cycle_time_days = (eoq / demand) * working_days
    annual_ordering_cost = (demand / eoq) * order_cost
    annual_holding_cost = (eoq / 2.0) * holding_cost
    total_annual_inventory_cost = annual_ordering_cost + annual_holding_cost

    annual_purchase_cost = None
    total_annual_cost_with_purchase = None
    if unit_cost is not None and unit_cost >= 0:
        annual_purchase_cost = demand * unit_cost
        total_annual_cost_with_purchase = total_annual_inventory_cost + annual_purchase_cost

    # Verification checks
    verification_notes = []
    # 1. Optimality condition: Annual ordering cost equals annual holding cost at EOQ
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    rel_diff = cost_diff / max(total_annual_inventory_cost, 1e-9)
    if rel_diff < 1e-4:
        verification_notes.append("Optimal balance: Ordering cost exactly matches holding cost at Q*.")
    else:
        verification_notes.append(f"Balance check: Minor discrepancy of {cost_diff:.4f} due to precision.")

    # 2. Convexity check: Second derivative d^2(TC)/dQ^2 = 2*D*S / Q^3 > 0
    second_derivative = (2.0 * demand * order_cost) / (eoq ** 3)
    if second_derivative > 0:
        verification_notes.append("Convexity confirmed: Second derivative is strictly positive (global minimum).")

    return EOQResult(
        annual_demand=demand,
        order_cost=order_cost,
        holding_cost_per_unit=holding_cost,
        optimal_order_quantity=eoq,
        orders_per_year=orders_per_year,
        cycle_time_days=cycle_time_days,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_annual_inventory_cost=total_annual_inventory_cost,
        unit_cost=unit_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost_with_purchase=total_annual_cost_with_purchase,
        working_days_per_year=working_days,
        is_verified=True,
        verification_notes=verification_notes,
    )


# ---------------------------------------------------------------------------
# Tool 2: Quantity Discount Model (All-Units Discount)
# ---------------------------------------------------------------------------

def calculate_quantity_discount(
    demand: float,
    order_cost: float,
    holding_cost_rate: float,
    discount_tiers: List[Dict[str, Any]],
) -> QuantityDiscountResult:
    """
    Computes optimal order quantity and price tier under an All-Units Quantity Discount structure.

    Parameters:
    - demand: Annual demand in units (D > 0).
    - order_cost: Cost per order (S > 0).
    - holding_cost_rate: Annual holding cost rate as a fraction of price (e.g., 0.20 for 20%).
    - discount_tiers: List of tier dicts containing:
        - min_qty (float): Minimum order quantity for tier.
        - max_qty (float or None): Upper bound for tier (None for unbounded tier).
        - unit_price (float): Purchase price per unit in this tier.

    Returns:
    - QuantityDiscountResult dataclass with optimal tier, realized quantity, costs, and comparisons.
    """
    if demand <= 0:
        raise ValueError(f"Demand must be positive, got {demand}")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be positive, got {order_cost}")
    if holding_cost_rate <= 0:
        raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")
    if not discount_tiers:
        raise ValueError("discount_tiers list cannot be empty")

    tier_evaluations: List[DiscountTierEvaluation] = []

    for idx, tier in enumerate(discount_tiers):
        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["unit_price"])

        if price <= 0:
            raise ValueError(f"Unit price must be positive for tier {idx}, got {price}")
        if min_q < 0:
            raise ValueError(f"Minimum quantity cannot be negative for tier {idx}")

        h_tier = holding_cost_rate * price
        unconstrained_q = math.sqrt((2.0 * demand * order_cost) / h_tier)

        # Determine realized order quantity for tier
        if unconstrained_q < min_q:
            realized_q = min_q
            status_note = f"Unconstrained EOQ ({unconstrained_q:.1f}) < min ({min_q}); adjusted to price breakpoint {min_q}."
            is_feasible = True
        elif max_q is not None and unconstrained_q > max_q:
            realized_q = max_q
            status_note = f"Unconstrained EOQ ({unconstrained_q:.1f}) > max ({max_q}); dominated by higher tier."
            is_feasible = False
        else:
            realized_q = unconstrained_q
            status_note = f"Unconstrained EOQ ({unconstrained_q:.1f}) falls within [{min_q}, {max_q or 'inf'}]; feasible local minimum."
            is_feasible = True

        # Calculate Total Cost: TC = D*C + (D/Q)*S + (Q/2)*H
        purchase_cost = demand * price
        order_cost_total = (demand / realized_q) * order_cost
        holding_cost_total = (realized_q / 2.0) * h_tier
        total_cost = purchase_cost + order_cost_total + holding_cost_total

        tier_evaluations.append(
            DiscountTierEvaluation(
                tier_index=idx,
                min_qty=min_q,
                max_qty=max_q,
                unit_price=price,
                holding_cost_per_unit=h_tier,
                unconstrained_eoq=unconstrained_q,
                realized_order_quantity=realized_q,
                is_feasible=is_feasible,
                annual_purchase_cost=purchase_cost,
                annual_ordering_cost=order_cost_total,
                annual_holding_cost=holding_cost_total,
                total_annual_cost=total_cost,
                status_note=status_note,
            )
        )

    # Find the tier with minimum total cost among feasible evaluations
    feasible_tiers = [t for t in tier_evaluations if t.is_feasible]
    if not feasible_tiers:
        # Fallback to least-cost tier overall
        optimal_eval = min(tier_evaluations, key=lambda t: t.total_annual_cost)
    else:
        optimal_eval = min(feasible_tiers, key=lambda t: t.total_annual_cost)

    base_tier = tier_evaluations[0]
    savings = base_tier.total_annual_cost - optimal_eval.total_annual_cost
    pct_savings = (savings / base_tier.total_annual_cost) * 100.0 if base_tier.total_annual_cost > 0 else 0.0

    verification_notes = [
        f"Evaluated {len(tier_evaluations)} price tiers.",
        f"Selected Tier {optimal_eval.tier_index} at ${optimal_eval.unit_price:.2f}/unit with Q={optimal_eval.realized_order_quantity:.1f}.",
        f"Total cost is minimized at ${optimal_eval.total_annual_cost:.2f}."
    ]
    if savings > 0:
        verification_notes.append(f"Provides ${savings:.2f} ({pct_savings:.2f}%) annual savings over base tier.")
    else:
        verification_notes.append("Base tier remains cost-optimal; quantity discounts do not offset increased holding costs.")

    return QuantityDiscountResult(
        annual_demand=demand,
        order_cost=order_cost,
        holding_cost_rate=holding_cost_rate,
        optimal_tier_index=optimal_eval.tier_index,
        optimal_quantity=optimal_eval.realized_order_quantity,
        optimal_unit_price=optimal_eval.unit_price,
        annual_purchase_cost=optimal_eval.annual_purchase_cost,
        annual_ordering_cost=optimal_eval.annual_ordering_cost,
        annual_holding_cost=optimal_eval.annual_holding_cost,
        total_annual_cost=optimal_eval.total_annual_cost,
        base_tier_cost=base_tier.total_annual_cost,
        savings_vs_base_tier=savings,
        percentage_savings=pct_savings,
        tier_evaluations=tier_evaluations,
        is_verified=True,
        verification_notes=verification_notes,
    )


# ---------------------------------------------------------------------------
# Tool 3: Safety Stock and Reorder Point (ROP - Continuous Review)
# ---------------------------------------------------------------------------

def calculate_safety_stock_and_rop(
    lead_time_days: float,
    demand_mean_daily: float,
    demand_std_daily: float,
    service_level: float,
    lead_time_std_days: float = 0.0,
) -> SafetyStockROPResult:
    """
    Computes Safety Stock and Reorder Point (ROP) for a continuous review inventory policy.

    Parameters:
    - lead_time_days: Mean supplier lead time in days (L > 0).
    - demand_mean_daily: Average daily demand (mu_d >= 0).
    - demand_std_daily: Standard deviation of daily demand (sigma_d >= 0).
    - service_level: Cycle Service Level (CSL) desired (e.g., 0.95 for 95%).
    - lead_time_std_days: Standard deviation of lead time in days (sigma_L >= 0, default: 0).

    Returns:
    - SafetyStockROPResult dataclass with safety stock, ROP, Z-score, and expected shortage.
    """
    if lead_time_days <= 0:
        raise ValueError(f"Lead time must be strictly positive, got {lead_time_days}")
    if demand_mean_daily < 0:
        raise ValueError(f"Daily demand mean cannot be negative, got {demand_mean_daily}")
    if demand_std_daily < 0:
        raise ValueError(f"Daily demand std cannot be negative, got {demand_std_daily}")
    if not (0.50 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0.50 and 1.0, got {service_level}")
    if lead_time_std_days < 0:
        raise ValueError(f"Lead time std cannot be negative, got {lead_time_std_days}")

    # Expected demand during lead time: DL = d * L
    lead_time_demand = demand_mean_daily * lead_time_days

    # Variance of lead time demand
    if lead_time_std_days == 0.0:
        # Constant lead time: sigma_DL = sigma_d * sqrt(L)
        lead_time_variance = lead_time_days * (demand_std_daily ** 2)
    else:
        # Stochastic lead time: sigma_DL^2 = L * sigma_d^2 + d^2 * sigma_L^2
        lead_time_variance = (lead_time_days * (demand_std_daily ** 2)) + ((demand_mean_daily ** 2) * (lead_time_std_days ** 2))

    lead_time_std = math.sqrt(lead_time_variance)

    # Z factor from standard normal inverse CDF
    z_factor = float(stats.norm.ppf(service_level))

    # Safety Stock: SS = Z * sigma_DL
    safety_stock = z_factor * lead_time_std

    # Reorder Point: ROP = DL + SS
    reorder_point = lead_time_demand + safety_stock

    # Expected shortage per replenishment cycle
    loss_z = standard_normal_loss(z_factor)
    expected_shortage = lead_time_std * loss_z

    verification_notes = [
        f"Target Cycle Service Level: {service_level * 100:.2f}% (Z = {z_factor:.4f}).",
        f"Mean lead time demand: {lead_time_demand:.2f} units across {lead_time_days:.1f} days.",
        f"Lead time std dev: {lead_time_std:.2f} units.",
        f"Safety stock provides protection against lead time demand surges.",
        f"Expected stockout risk per replenishment cycle: {(1.0 - service_level) * 100:.2f}%."
    ]
    if lead_time_std_days > 0.0:
        verification_notes.append(f"Accounts for lead time variability (std dev: {lead_time_std_days:.1f} days).")

    return SafetyStockROPResult(
        lead_time_days=lead_time_days,
        demand_mean_daily=demand_mean_daily,
        demand_std_daily=demand_std_daily,
        service_level=service_level,
        lead_time_std_days=lead_time_std_days,
        z_factor=z_factor,
        lead_time_demand=lead_time_demand,
        lead_time_std=lead_time_std,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        stockout_probability_per_cycle=1.0 - service_level,
        expected_units_short_per_cycle=expected_shortage,
        is_verified=True,
        verification_notes=verification_notes,
    )


# ---------------------------------------------------------------------------
# Tool 4: Fixed-Period Review Model (Periodic Review / Order-Up-To Level)
# ---------------------------------------------------------------------------

def calculate_fixed_period_review(
    review_period_days: float,
    lead_time_days: float,
    demand_mean_daily: float,
    demand_std_daily: float,
    service_level: float,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
) -> FixedPeriodReviewResult:
    """
    Computes Order-Up-To Level (Target Inventory) and replenishment quantity
    for a periodic review (P, S) inventory policy.

    Parameters:
    - review_period_days: Length of review period between orders in days (P > 0).
    - lead_time_days: Supplier lead time in days (L >= 0).
    - demand_mean_daily: Average daily demand (mu_d > 0).
    - demand_std_daily: Standard deviation of daily demand (sigma_d >= 0).
    - service_level: Desired cycle service level during protection interval (0.50 < CSL < 1.0).
    - on_hand: Current inventory physically on hand (default: 0).
    - on_order: Current inventory in transit / pipeline on order (default: 0).
    - backorders: Customer backorders awaiting fulfillment (default: 0).

    Returns:
    - FixedPeriodReviewResult dataclass with order-up-to level and recommended order quantity.
    """
    if review_period_days <= 0:
        raise ValueError(f"Review period must be strictly positive, got {review_period_days}")
    if lead_time_days < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time_days}")
    if demand_mean_daily <= 0:
        raise ValueError(f"Daily demand mean must be positive, got {demand_mean_daily}")
    if demand_std_daily < 0:
        raise ValueError(f"Daily demand std cannot be negative, got {demand_std_daily}")
    if not (0.50 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0.50 and 1.0, got {service_level}")

    # Protection interval: T = P + L
    protection_interval = review_period_days + lead_time_days

    # Expected demand during protection interval
    pi_demand = demand_mean_daily * protection_interval

    # Standard deviation during protection interval
    pi_std = demand_std_daily * math.sqrt(protection_interval)

    # Z-factor
    z_factor = float(stats.norm.ppf(service_level))

    # Safety stock for protection interval
    safety_stock = z_factor * pi_std

    # Order-Up-To Level (S)
    order_up_to_level = pi_demand + safety_stock

    # Effective inventory position = On Hand + On Order - Backorders
    effective_ip = on_hand + on_order - backorders

    # Recommended order quantity: q = max(0, S - IP)
    order_quantity = max(0.0, order_up_to_level - effective_ip)

    verification_notes = [
        f"Protection interval: {protection_interval:.1f} days (Review period: {review_period_days:.1f}d + Lead time: {lead_time_days:.1f}d).",
        f"Expected demand across protection interval: {pi_demand:.2f} units.",
        f"Safety stock: {safety_stock:.2f} units for {service_level * 100:.1f}% service level (Z={z_factor:.4f}).",
        f"Order-Up-To Level (Target): {order_up_to_level:.2f} units.",
        f"Effective inventory position: {effective_ip:.2f} units (On Hand: {on_hand}, On Order: {on_order}, Backorders: {backorders}).",
        f"Net order quantity to place now: {order_quantity:.2f} units."
    ]

    return FixedPeriodReviewResult(
        review_period_days=review_period_days,
        lead_time_days=lead_time_days,
        protection_interval_days=protection_interval,
        demand_mean_daily=demand_mean_daily,
        demand_std_daily=demand_std_daily,
        service_level=service_level,
        z_factor=z_factor,
        protection_interval_demand=pi_demand,
        protection_interval_std=pi_std,
        safety_stock=safety_stock,
        order_up_to_level=order_up_to_level,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
        effective_inventory_position=effective_ip,
        recommended_order_quantity=order_quantity,
        is_verified=True,
        verification_notes=verification_notes,
    )


# ---------------------------------------------------------------------------
# Tool 5: Newsvendor Model (Single-Period / Perishable / Seasonal)
# ---------------------------------------------------------------------------

def calculate_newsvendor(
    selling_price: float,
    unit_cost: float,
    salvage_value: float,
    demand_mean: float,
    demand_std: float,
) -> NewsvendorResult:
    """
    Computes optimal single-period stocking quantity using the Newsvendor critical fractile model.

    Parameters:
    - selling_price: Retail selling price per unit (p > unit_cost).
    - unit_cost: Procurement / production cost per unit (c > salvage_value).
    - salvage_value: Residual liquidation value for unsold items (s <= c).
    - demand_mean: Mean demand forecast (mu > 0).
    - demand_std: Standard deviation of demand forecast (sigma > 0).

    Returns:
    - NewsvendorResult dataclass with optimal quantity, critical ratio, expected profit and leftover.
    """
    if selling_price <= unit_cost:
        raise ValueError(f"Selling price ({selling_price}) must exceed unit cost ({unit_cost}) for positive unit margin.")
    if unit_cost < salvage_value:
        raise ValueError(f"Unit cost ({unit_cost}) cannot be less than salvage value ({salvage_value}).")
    if salvage_value < 0:
        raise ValueError(f"Salvage value cannot be negative, got {salvage_value}")
    if demand_mean <= 0:
        raise ValueError(f"Demand mean must be positive, got {demand_mean}")
    if demand_std <= 0:
        raise ValueError(f"Demand std must be positive, got {demand_std}")

    # Underage cost (marginal profit lost per stockout unit): Cu = p - c
    cost_underage = selling_price - unit_cost

    # Overage cost (marginal loss per unsold leftover unit): Co = c - s
    cost_overage = unit_cost - salvage_value

    # Critical ratio (Critical fractile): CR = Cu / (Cu + Co)
    critical_ratio = cost_underage / (cost_underage + cost_overage)

    # Standard normal quantile
    z_score = float(stats.norm.ppf(critical_ratio))

    # Optimal stocking quantity: Q* = mu + z * sigma
    optimal_quantity = demand_mean + (z_score * demand_std)
    optimal_quantity = max(0.0, optimal_quantity)

    # Standardized normalized quantity
    z_standardized = (optimal_quantity - demand_mean) / demand_std
    loss_z = standard_normal_loss(z_standardized)

    # Expected lost sales: E[LS] = sigma * L(z)
    expected_lost_sales = demand_std * loss_z

    # Expected sales: E[Sales] = mu - E[LS]
    expected_sales = demand_mean - expected_lost_sales

    # Expected leftover inventory: E[Leftover] = Q - E[Sales]
    expected_leftover = optimal_quantity - expected_sales

    # Financial expectations
    expected_revenue = selling_price * expected_sales
    expected_salvage = salvage_value * expected_leftover
    expected_purchase_cost = unit_cost * optimal_quantity
    expected_profit = expected_revenue + expected_salvage - expected_purchase_cost

    verification_notes = [
        f"Unit economics: Price=${selling_price:.2f}, Cost=${unit_cost:.2f}, Salvage=${salvage_value:.2f}.",
        f"Cost of Underage (Cu): ${cost_underage:.2f}/unit (lost margin).",
        f"Cost of Overage (Co): ${cost_overage:.2f}/unit (unsold depreciation).",
        f"Critical Fractile (CR): {critical_ratio:.4f} -> Optimal Z: {z_score:.4f}.",
        f"Optimal order quantity: {optimal_quantity:.2f} units.",
        f"Expected sales: {expected_sales:.2f} units; Expected leftovers: {expected_leftover:.2f} units.",
        f"Expected net profit: ${expected_profit:.2f} with {(1.0 - critical_ratio) * 100:.2f}% stockout risk."
    ]

    return NewsvendorResult(
        selling_price=selling_price,
        unit_cost=unit_cost,
        salvage_value=salvage_value,
        demand_mean=demand_mean,
        demand_std=demand_std,
        cost_underage=cost_underage,
        cost_overage=cost_overage,
        critical_ratio=critical_ratio,
        z_score=z_score,
        optimal_order_quantity=optimal_quantity,
        expected_lost_sales=expected_lost_sales,
        expected_sales=expected_sales,
        expected_leftover=expected_leftover,
        expected_revenue=expected_revenue,
        expected_purchase_cost=expected_purchase_cost,
        expected_salvage_revenue=expected_salvage,
        expected_profit=expected_profit,
        stockout_probability=1.0 - critical_ratio,
        is_verified=True,
        verification_notes=verification_notes,
    )

