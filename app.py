"""
Supply Chain Analytics Inventory Decision Agent Application
============================================================
Dual-Interface Application:
1. Web Dashboard (Streamlit + Plotly): run with `streamlit run app.py`
2. Interactive Terminal CLI (Rich): run with `python app.py`

Implements the 7-stage Architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
"""

import sys
import os
import argparse
import json

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
    DecisionTrace,
    BENCHMARK_SCENARIOS,
)
from inventory_tools import (
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
)


def is_running_under_streamlit() -> bool:
    """Detects whether script is running inside Streamlit."""
    return "streamlit" in os.path.basename(sys.argv[0]).lower() or os.environ.get("STREAMLIT_SERVER_PORT") is not None


# =============================================================================
# Plotly Visualization Helpers
# =============================================================================

def build_eoq_chart(result: EOQResult) -> go.Figure:
    """Generates interactive EOQ Total Cost curve with Ordering and Holding components."""
    q_opt = result.optimal_order_quantity
    q_range = np.linspace(max(10, q_opt * 0.2), q_opt * 2.2, 200)

    order_costs = (result.annual_demand / q_range) * result.order_cost
    holding_costs = (q_range / 2.0) * result.holding_cost_per_unit
    total_costs = order_costs + holding_costs

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_range, y=order_costs, mode="lines", name="Annual Ordering Cost", line=dict(color="#EF553B", dash="dash")))
    fig.add_trace(go.Scatter(x=q_range, y=holding_costs, mode="lines", name="Annual Holding Cost", line=dict(color="#00CC96", dash="dash")))
    fig.add_trace(go.Scatter(x=q_range, y=total_costs, mode="lines", name="Total Inventory Cost", line=dict(color="#636EFA", width=3)))

    # Optimal Point Marker
    fig.add_trace(go.Scatter(
        x=[q_opt],
        y=[result.total_annual_inventory_cost],
        mode="markers+text",
        name=f"Optimal EOQ (Q* = {q_opt:.0f})",
        marker=dict(color="#FFA15A", size=14, symbol="star"),
        text=[f"Q*={q_opt:.0f} (${result.total_annual_inventory_cost:,.0f})"],
        textposition="top center",
    ))

    fig.update_layout(
        title="Economic Order Quantity (EOQ) Cost Trade-off Curve",
        xaxis_title="Order Quantity (Units per Order)",
        yaxis_title="Annual Cost ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=420,
    )
    return fig


def build_quantity_discount_chart(result: QuantityDiscountResult) -> go.Figure:
    """Generates comparison bar chart of total annual costs across discount tiers."""
    tier_labels = [f"Tier {t.tier_index} (${t.unit_price:.2f})" for t in result.tier_evaluations]
    purchase_costs = [t.annual_purchase_cost for t in result.tier_evaluations]
    order_costs = [t.annual_ordering_cost for t in result.tier_evaluations]
    holding_costs = [t.annual_holding_cost for t in result.tier_evaluations]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Purchase Cost", x=tier_labels, y=purchase_costs, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="Ordering Cost", x=tier_labels, y=order_costs, marker_color="#EF553B"))
    fig.add_trace(go.Bar(name="Holding Cost", x=tier_labels, y=holding_costs, marker_color="#00CC96"))

    fig.update_layout(
        barmode="stack",
        title="Total Annual Cost Breakdown Across Quantity Discount Tiers",
        xaxis_title="Pricing Tier",
        yaxis_title="Total Annual Cost ($)",
        template="plotly_white",
        height=420,
    )
    return fig


def build_safety_stock_chart(result: SafetyStockROPResult) -> go.Figure:
    """Visualizes Lead Time Demand normal distribution, Safety Stock, and ROP."""
    mu = result.lead_time_demand
    sigma = result.lead_time_std
    x = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 300)
    y = stats.norm.pdf(x, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Lead Time Demand Distribution", line=dict(color="#1f77b4", width=2)))

    # Shaded Stockout Region
    x_stockout = x[x >= result.reorder_point]
    y_stockout = stats.norm.pdf(x_stockout, mu, sigma)
    if len(x_stockout) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([[result.reorder_point], x_stockout, [x_stockout[-1]]]),
            y=np.concatenate([[0], y_stockout, [0]]),
            fill="toself",
            fillcolor="rgba(239, 85, 59, 0.4)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"Stockout Risk ({(1 - result.service_level)*100:.1f}%)",
        ))

    # Markers for Mean and ROP
    fig.add_vline(x=mu, line_width=2, line_dash="dash", line_color="#2ca02c", annotation_text=f"Mean Demand ({mu:.1f})")
    fig.add_vline(x=result.reorder_point, line_width=2, line_dash="solid", line_color="#d62728", annotation_text=f"ROP ({result.reorder_point:.1f})")

    fig.update_layout(
        title=f"Lead Time Demand Distribution & Buffer (Safety Stock = {result.safety_stock:.1f} units)",
        xaxis_title="Demand During Lead Time (Units)",
        yaxis_title="Probability Density",
        template="plotly_white",
        height=420,
    )
    return fig


def build_fixed_period_review_chart(result: FixedPeriodReviewResult) -> go.Figure:
    """Visualizes periodic review Order-Up-To Level (S) and Inventory Positions."""
    categories = ["Current On-Hand", "In-Transit On-Order", "Backorders", "Inventory Position", "Order to Place", "Order-Up-To Level"]
    values = [
        result.on_hand,
        result.on_order,
        -result.backorders,
        result.effective_inventory_position,
        result.recommended_order_quantity,
        result.order_up_to_level,
    ]
    colors = ["#2b5c8f", "#3c8dbc", "#e74c3c", "#f39c12", "#00a65a", "#8e44ad"]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        title="Periodic Review Inventory Position & Replenishment Order",
        yaxis_title="Units",
        template="plotly_white",
        height=420,
    )
    return fig


def build_newsvendor_chart(result: NewsvendorResult) -> go.Figure:
    """Visualizes Newsvendor demand distribution, stocking quantity, and critical fractile."""
    mu = result.demand_mean
    sigma = result.demand_std
    x = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 300)
    y = stats.norm.pdf(x, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Demand Forecast PDF", line=dict(color="#1f77b4", width=2)))

    # Shaded fulfilled demand
    x_fill = x[x <= result.optimal_order_quantity]
    y_fill = stats.norm.pdf(x_fill, mu, sigma)
    if len(x_fill) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([[x_fill[0]], x_fill, [result.optimal_order_quantity]]),
            y=np.concatenate([[0], y_fill, [0]]),
            fill="toself",
            fillcolor="rgba(0, 204, 150, 0.35)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"Service Level (CR = {result.critical_ratio*100:.1f}%)",
        ))

    fig.add_vline(x=mu, line_width=2, line_dash="dash", line_color="#7f7f7f", annotation_text=f"Mean ({mu:.0f})")
    fig.add_vline(x=result.optimal_order_quantity, line_width=2, line_dash="solid", line_color="#00cc96", annotation_text=f"Q* ({result.optimal_order_quantity:.0f})")

    fig.update_layout(
        title=f"Newsvendor Single-Period Stocking Optimization (CR = {result.critical_ratio:.3f})",
        xaxis_title="Demand Quantity (Units)",
        yaxis_title="Probability Density",
        template="plotly_white",
        height=420,
    )
    return fig


# =============================================================================
# Streamlit Dashboard UI
# =============================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    agent = InventoryDecisionAgent()

    # Custom Header Styling
    st.markdown(
        """
        <style>
        .main-header { font-size: 2.1rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.2rem; }
        .arch-badge { background-color: #EEF2F6; color: #1E40AF; padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; display: inline-block; margin-bottom: 1rem; border: 1px solid #CBD5E1; }
        .exec-card { background-color: #F8FAFC; border-left: 5px solid #2563EB; padding: 16px 20px; border-radius: 4px; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">📦 Supply Chain Analytics Inventory Decision Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Decision Architecture for Optimal Lot Sizing, Safety Buffers, and Single-Period Stocking</div>', unsafe_allow_html=True)

    # Architectural Pipeline Badge
    st.markdown(
        '<div class="arch-badge">Pipeline: Customer Question ➔ Identify Model ➔ Extract Inputs ➔ Select Python Tool ➔ Calculate ➔ Verify ➔ Explain</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Benchmark Scenarios & Custom Input
    st.sidebar.header("🎯 Scenario Presets")
    selected_scenario_name = st.sidebar.selectbox(
        "Load Benchmark Question:",
        ["Custom Question"] + list(BENCHMARK_SCENARIOS.keys()),
        index=0,
    )

    default_text = ""
    if selected_scenario_name != "Custom Question":
        default_text = BENCHMARK_SCENARIOS[selected_scenario_name]["question"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Supported Models")
    st.sidebar.markdown(
        """
        1. **EOQ**: Economic Order Quantity
        2. **Quantity Discount**: Price tier optimization
        3. **Safety Stock & ROP**: Continuous review
        4. **Fixed-Period Review**: Order-up-to policy
        5. **Newsvendor**: Seasonal / perishable critical fractile
        """
    )

    # Main Question Box
    st.markdown("### 💬 Natural Language Inventory Question")
    question_input = st.text_area(
        "Enter your inventory or supply chain problem in plain English:",
        value=default_text,
        height=110,
        placeholder="e.g. Our company sells 12,000 units per year. Each order incurs a fixed setup cost of $50, and holding inventory costs $2.00 per unit per year. Facility operates 250 days a year. What is our optimal order quantity and how often should we order?",
    )

    col_btn, _ = st.columns([2, 8])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze & Decide", type="primary", use_container_width=True)

    if analyze_clicked or default_text:
        if not question_input.strip():
            st.warning("Please enter a question or select a preset scenario.")
            return

        with st.spinner("Executing 7-stage Decision Pipeline..."):
            trace = agent.process_question(question_input)

        if trace.error_message:
            st.error(f"❌ **Analysis Stopped:** {trace.error_message}")
            if trace.missing_inputs:
                st.info(f"**Missing Parameters Required:** {', '.join(trace.missing_inputs)}")
            return

        # Top Executive Decision Card
        st.markdown(
            f"""
            <div class="exec-card">
                <h4 style="margin-top:0; color:#1E40AF;">📋 Executive Summary & Decision</h4>
                <p style="font-size: 1.05rem; margin-bottom: 0;">{trace.executive_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tabbed 7-Stage Architectural Inspection
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 1 & 2. Model & Extraction",
            "📊 3 & 4. Tool & Calculation",
            "📈 Visual Analytics",
            "✅ 5. Verification",
            "💡 6. Strategic Takeaways",
        ])

        with tab1:
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                st.markdown("#### Step 1: Model Identification")
                st.success(f"**Identified Model:** {trace.identified_model.value}")
                st.metric("Identification Confidence", f"{trace.model_confidence * 100:.1f}%")
                st.caption(f"**Rationale:** {trace.model_rationale}")

            with col_m2:
                st.markdown("#### Step 2: Extracted Numerical Inputs")
                st.json(trace.extracted_inputs)

        with tab2:
            st.markdown(f"#### Step 3: Tool Invocation (`{trace.selected_tool_name}`)")
            st.markdown(f"**Selected Python Tool:** `{trace.selected_tool_name}`")
            st.markdown("#### Step 4: Quantitative Execution Result")
            st.json(trace.calculation_result.to_dict())

            st.markdown("#### Step-by-Step Mathematical Derivation")
            st.markdown(trace.mathematical_derivation)

            st.markdown("#### Cost & Operational Trade-off Analysis")
            st.markdown(trace.cost_and_tradeoff_analysis)

        with tab3:
            st.markdown("#### Interactive Model Visualization")
            if isinstance(trace.calculation_result, EOQResult):
                st.plotly_chart(build_eoq_chart(trace.calculation_result), use_container_width=True)
            elif isinstance(trace.calculation_result, QuantityDiscountResult):
                st.plotly_chart(build_quantity_discount_chart(trace.calculation_result), use_container_width=True)
            elif isinstance(trace.calculation_result, SafetyStockROPResult):
                st.plotly_chart(build_safety_stock_chart(trace.calculation_result), use_container_width=True)
            elif isinstance(trace.calculation_result, FixedPeriodReviewResult):
                st.plotly_chart(build_fixed_period_review_chart(trace.calculation_result), use_container_width=True)
            elif isinstance(trace.calculation_result, NewsvendorResult):
                st.plotly_chart(build_newsvendor_chart(trace.calculation_result), use_container_width=True)

        with tab4:
            st.markdown("#### Step 5: Engineering & Mathematical Verification")
            if trace.verification_passed:
                st.success("All mathematical sanity and boundary constraints PASSED.")
            else:
                st.warning("Some verification checks generated warnings.")

            for chk in trace.verification_checks:
                if chk.passed:
                    st.markdown(f"- ✅ **{chk.name}**: {chk.details}")
                else:
                    st.markdown(f"- ⚠️ **{chk.name}**: {chk.details}")

        with tab5:
            st.markdown("#### Strategic Recommendations for Supply Chain Managers")
            for idx, rec in enumerate(trace.managerial_recommendations, 1):
                st.markdown(f"{idx}. {rec}")

            st.markdown("---")
            st.download_button(
                label="📥 Export Full Decision Trace (JSON)",
                data=json.dumps(trace.to_dict(), indent=2),
                file_name="inventory_decision_trace.json",
                mime="application/json",
            )


# =============================================================================
# Terminal / CLI Interface (Rich)
# =============================================================================

def run_cli_app():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown

    console = Console()

    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent CLI")
    parser.add_argument("--question", "-q", type=str, help="Natural language inventory question to solve")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run all 5 benchmark scenarios")
    parser.add_argument("--web", "-w", action="store_true", help="Launch the Streamlit web dashboard")
    args = parser.parse_args()

    if args.web:
        import subprocess
        console.print("[bold green]Starting Streamlit Web Dashboard...[/bold green]")
        subprocess.run(["streamlit", "run", __file__])
        return

    agent = InventoryDecisionAgent()

    console.print(
        Panel.fit(
            "[bold cyan]📦 Supply Chain Analytics Inventory Decision Agent[/bold cyan]\n"
            "[yellow]Architecture Pipeline:[/yellow] Customer Question ➔ Identify Model ➔ Extract Inputs ➔ Select Python Tool ➔ Calculate ➔ Verify ➔ Explain",
            border_style="cyan",
        )
    )

    def print_trace(trace: DecisionTrace):
        if trace.error_message:
            console.print(f"[bold red]❌ Error:[/bold red] {trace.error_message}")
            if trace.missing_inputs:
                console.print(f"[red]Missing fields: {', '.join(trace.missing_inputs)}[/red]")
            return

        # 1. Executive Summary Panel
        console.print(Panel(trace.executive_summary, title="[bold green]📋 Executive Summary & Decision[/bold green]", border_style="green"))

        # 2. Pipeline Table
        table = Table(title="🔍 7-Stage Architectural Execution Trace", border_style="blue")
        table.add_column("Stage", style="bold cyan", width=22)
        table.add_column("Details", style="white")

        table.add_row("1. Model Identified", f"{trace.identified_model.value} (Confidence: {trace.model_confidence*100:.1f}%)")
        table.add_row("2. Selection Rationale", trace.model_rationale)
        table.add_row("3. Extracted Inputs", json.dumps(trace.extracted_inputs, indent=2))
        table.add_row("4. Python Tool Selected", f"`{trace.selected_tool_name}`")
        table.add_row("5. Verification Status", "[green]PASSED[/green]" if trace.verification_passed else "[yellow]WARNING[/yellow]")

        console.print(table)

        # 3. Verification Details
        v_table = Table(title="✅ Engineering Verification Checklist", border_style="green")
        v_table.add_column("Check", style="cyan")
        v_table.add_column("Status", style="bold")
        v_table.add_column("Details", style="white")

        for c in trace.verification_checks:
            status_str = "[green]PASS[/green]" if c.passed else "[yellow]WARN[/yellow]"
            v_table.add_row(c.name, status_str, c.details)

        console.print(v_table)

        # 4. Derivation and Recommendations
        console.print("\n[bold yellow]📐 Mathematical Derivation & Trade-offs:[/bold yellow]")
        console.print(Markdown(trace.mathematical_derivation))
        console.print(Markdown(trace.cost_and_tradeoff_analysis))

        console.print("\n[bold cyan]💡 Strategic Recommendations:[/bold cyan]")
        for i, rec in enumerate(trace.managerial_recommendations, 1):
            console.print(f"  [bold]{i}.[/bold] {rec}")
        console.print("-" * 75)

    if args.benchmark:
        console.print("[bold yellow]Running all 5 Benchmark Scenarios...[/bold yellow]\n")
        for name, data in BENCHMARK_SCENARIOS.items():
            console.print(f"[bold magenta]▶ Scenario: {name}[/bold magenta]")
            console.print(f"[italic]Question:[/italic] {data['question']}\n")
            trace = agent.process_question(data["question"])
            print_trace(trace)
        return

    if args.question:
        trace = agent.process_question(args.question)
        print_trace(trace)
        return

    # Interactive prompt mode
    console.print("\n[bold]Select an action:[/bold]")
    console.print("  [1-5] Run a benchmark scenario")
    console.print("  [C]   Enter a custom question")
    console.print("  [W]   Launch Web Dashboard in browser")
    console.print("  [Q]   Quit")

    scenarios_list = list(BENCHMARK_SCENARIOS.items())
    for idx, (sname, _) in enumerate(scenarios_list, 1):
        console.print(f"    [{idx}] {sname}")

    choice = input("\nYour choice: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= 5:
        sname, sdata = scenarios_list[int(choice) - 1]
        console.print(f"\n[bold magenta]Processing: {sname}[/bold magenta]")
        trace = agent.process_question(sdata["question"])
        print_trace(trace)
    elif choice.lower() == "c":
        q = input("\nEnter your inventory question: ").strip()
        if q:
            trace = agent.process_question(q)
            print_trace(trace)
    elif choice.lower() == "w":
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    else:
        console.print("[green]Exiting Supply Chain Inventory Agent. Goodbye![/green]")


# =============================================================================
# Entry Point Dispatcher
# =============================================================================

if __name__ == "__main__":
    if is_running_under_streamlit():
        run_streamlit_app()
    else:
        run_cli_app()
