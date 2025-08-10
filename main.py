"""
LangGraph Agentic Workflow: Trading Strategy Designer
----------------------------------------------------
A production-ready scaffold for an agentic workflow that turns user parameters
into backtested trading strategies. Includes a Supervisor that routes work to
specialized agents/teams via subgraphs.

Notes
- Replace OPENAI_API_KEY and model names as needed.
- Tool implementations use placeholders; wire to your infra (e.g., data lake, Arrow/Parquet, FMP, CCXT, Quantstats) as desired.
- This file is intentionally self-contained to ease iteration. Split into
  modules once stable.
"""
from __future__ import annotations
from typing import Literal, TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# --- LangGraph / LLMs ---
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from pydantic import BaseModel, Field, field_validator

# If you prefer Anthropic/Azure/OpenAI wrappers, swap these easily.
from langchain_openai import ChatOpenAI
from langchain.tools import tool

###############################################################
# 1) USER REQUEST SCHEMA
###############################################################

class StrategyParams(BaseModel):
    objective: Literal[
        "absolute_return", "risk_adjusted_return", "market_neutral", "income"
    ] = "risk_adjusted_return"
    universe: List[str] = Field(
        default_factory=lambda: ["SPY", "QQQ"],
        description="Tickers or symbols (equities, futures, crypto, fx).",
    )
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d", "1w"] = "1d"
    horizon_days: int = 365
    max_leverage: float = 1.0
    risk_budget_var_95: float = 0.05
    max_drawdown: float = 0.2
    turnover_limit_annual: float = 10.0
    constraints: List[str] = Field(default_factory=list)
    signals_preference: List[str] = Field(
        default_factory=lambda: ["trend", "momentum", "mean_reversion", "carry", "value"]
    )
    execution_preference: Literal["close", "vwap", "twap", "limit"] = "vwap"
    capital_usd: float = 100000.0
    slippage_bps: float = 2.0
    fees_bps: float = 1.0
    notes: Optional[str] = None

    @field_validator("horizon_days")
    @classmethod
    def _horizon_positive(cls, v):
        assert v > 0, "horizon_days must be > 0"
        return v


###############################################################
# 2) GRAPH STATE
###############################################################

class GraphState(TypedDict):
    user_params: StrategyParams
    tasks: List[str]
    artifacts: Dict[str, Any]   # intermediate outputs
    route: Optional[str]        # supervisor decision
    report: Optional[str]       # final report
    errors: List[str]


###############################################################
# 3) LLM + TOOLS
###############################################################

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Example tools — replace with real implementations
@tool
def fetch_market_data(universe: List[str], timeframe: str, lookback_days: int) -> Dict[str, Any]:
    """Fetch OHLCV for universe/timeframe/lookback_days. Placeholder returns schema only."""
    return {
        "meta": {"universe": universe, "timeframe": timeframe, "lookback_days": lookback_days},
        "data": {sym: f"{sym}_OHLCV_PLACEHOLDER" for sym in universe},
    }

@tool
def compute_features(dataset: Dict[str, Any], features: List[str]) -> Dict[str, Any]:
    """Compute features (e.g., RSI, MACD, volatility). Placeholder only."""
    return {"features": features, "derived_from": list(dataset.get("data", {}).keys())}

@tool
def backtest(strategy_spec: Dict[str, Any], market_data: Dict[str, Any], costs: Dict[str, float]) -> Dict[str, Any]:
    """Run a simple vectorized backtest. Returns placeholder metrics and equity curve."""
    return {
        "metrics": {
            "CAGR": 0.18,
            "Sharpe": 1.35,
            "MaxDrawdown": 0.14,
            "WinRate": 0.54,
            "Sortino": 1.9,
        },
        "equity_curve": [100000 + i * 150 for i in range(250)],
        "constraints_respected": True,
    }

@tool
def risk_checks(strategy_spec: Dict[str, Any], limits: Dict[str, Any]) -> Dict[str, Any]:
    """Validate leverage, concentration, VaR, exposure. Placeholder only."""
    return {
        "leverage_ok": strategy_spec.get("max_leverage", 1.0) <= limits.get("max_leverage", 1.0),
        "dd_limit": limits.get("max_drawdown", 0.2),
        "comment": "All checks passed in placeholder.",
    }

@tool
def cost_model(slippage_bps: float, fees_bps: float, turnover_limit_annual: float) -> Dict[str, float]:
    """Assemble trading cost assumptions."""
    return {
        "slippage_bps": slippage_bps,
        "fees_bps": fees_bps,
        "turnover_limit_annual": turnover_limit_annual,
    }


###############################################################
# 4) AGENTS (Nodes)
###############################################################

class Route(str, Enum):
    DATA = "data_team"
    RESEARCH = "alpha_research_team"
    STRAT = "strategy_team"
    RISK = "risk_team"
    BACKTEST = "backtest_team"
    COST = "cost_team"
    REPORT = "report_team"


def supervisor(state: GraphState) -> GraphState:
    """LLM routes to the right subgraph based on current tasks & artifacts."""
    prompt = (
        "You are a routing supervisor for a trading strategy design workflow.\n"
        "State what team should act next (one of: data_team, alpha_research_team, "
        "strategy_team, risk_team, backtest_team, cost_team, report_team).\n"
        "Reason briefly in 1 line and output ONLY the team id.\n\n"
        f"User params: {state['user_params'].model_dump()}\n"
        f"Tasks: {state.get('tasks')}\n"
        f"Artifacts keys: {list(state.get('artifacts', {}).keys())}\n"
    )
    choice = llm.invoke(prompt).content.strip().split("\n")[0]
    state["route"] = choice
    return state


# --- Subgraph: Data Team ---

def data_team(state: GraphState) -> GraphState:
    params = state["user_params"]
    # Decide lookback based on horizon, add buffer
    lookback = min(max(params.horizon_days * 2, 365), 3650)
    md = fetch_market_data.run(universe=params.universe, timeframe=params.timeframe, lookback_days=lookback)  # type: ignore
    feats = compute_features.run(dataset=md, features=["rsi", "macd", "volatility", "atr"])  # type: ignore
    state["artifacts"]["market_data"] = md
    state["artifacts"]["features"] = feats
    # Update task stack
    state["tasks"].append("alpha ideation")
    return state


# --- Subgraph: Alpha Research Team ---

def alpha_research_team(state: GraphState) -> GraphState:
    params = state["user_params"]
    # Create a few candidate alpha hypotheses based on preferences
    candidates = []
    for s in params.signals_preference:
        candidates.append({
            "name": f"{s}_signal",
            "logic": f"Generate {s} signal using features; score each asset; form long/short list.",
            "params": {"lookback": 50, "zscore": 1.0},
        })
    state["artifacts"]["alpha_candidates"] = candidates
    state["tasks"].append("strategy synthesis")
    return state


# --- Subgraph: Strategy Synthesis Team ---

def strategy_team(state: GraphState) -> GraphState:
    params = state["user_params"]
    alphas = state["artifacts"].get("alpha_candidates", [])
    if not alphas:
        state["errors"].append("No alpha candidates available.")
        return state
    # Simple ensemble
    strategy_spec = {
        "name": "EnsembleMultiSignal",
        "universe": params.universe,
        "timeframe": params.timeframe,
        "objective": params.objective,
        "max_leverage": params.max_leverage,
        "position_bounds": (-1.0 if params.objective == "market_neutral" else 0.0, 1.0),
        "weighting": "risk_parity",
        "signals": alphas[:3],
        "risk_controls": {"max_drawdown": params.max_drawdown, "var_95": params.risk_budget_var_95},
        "execution": {"style": params.execution_preference},
        "constraints": params.constraints,
    }
    state["artifacts"]["strategy_spec"] = strategy_spec
    state["tasks"].append("costing & risk")
    return state


# --- Subgraph: Cost Team ---

def cost_team(state: GraphState) -> GraphState:
    params = state["user_params"]
    costs = cost_model.run(
        slippage_bps=params.slippage_bps,
        fees_bps=params.fees_bps,
        turnover_limit_annual=params.turnover_limit_annual,
    )  # type: ignore
    state["artifacts"]["costs"] = costs
    state["tasks"].append("risk checks")
    return state


# --- Subgraph: Risk Team ---

def risk_team(state: GraphState) -> GraphState:
    params = state["user_params"]
    strat = state["artifacts"].get("strategy_spec")
    if not strat:
        state["errors"].append("No strategy_spec for risk.")
        return state
    limits = {"max_leverage": params.max_leverage, "max_drawdown": params.max_drawdown}
    rc = risk_checks.run(strategy_spec=strat, limits=limits)  # type: ignore
    state["artifacts"]["risk_report"] = rc
    state["tasks"].append("backtest")
    return state


# --- Subgraph: Backtest Team ---

def backtest_team(state: GraphState) -> GraphState:
    strat = state["artifacts"].get("strategy_spec")
    data = state["artifacts"].get("market_data")
    costs = state["artifacts"].get("costs")
    if not (strat and data and costs):
        state["errors"].append("Missing inputs for backtest.")
        return state
    bt = backtest.run(strategy_spec=strat, market_data=data, costs=costs)  # type: ignore
    state["artifacts"]["backtest"] = bt
    state["tasks"].append("report")
    return state


# --- Subgraph: Report Team ---

def report_team(state: GraphState) -> GraphState:
    params = state["user_params"]
    strat = state["artifacts"].get("strategy_spec", {})
    metrics = state["artifacts"].get("backtest", {}).get("metrics", {})
    rc = state["artifacts"].get("risk_report", {})

    summary = f"""
    # Strategy Design Report

    ## Summary
    Objective: {params.objective}\nUniverse: {params.universe}\nTimeframe: {params.timeframe}\nCapital: ${params.capital_usd:,.0f}

    ## Strategy
    Name: {strat.get('name')}\nSignals: {[s['name'] for s in strat.get('signals', [])]}\nWeighting: {strat.get('weighting')}\nExecution: {strat.get('execution')}

    ## Risk
    Limits: {strat.get('risk_controls')}\nChecks: {rc}

    ## Backtest (placeholder)
    Metrics: {metrics}

    ## Next Steps
    - Swap placeholder tools with production data, feature, and backtest engines.\n    - Add walk-forward and cross-validation.\n    - Connect CCXT/OMS for paper/live trading.\n    - Attach reporting to your BI stack.
    """
    state["report"] = summary
    return state


###############################################################
# 5) GRAPH WIRING
###############################################################

def build_graph() -> StateGraph:
    sg = StateGraph(GraphState)

    # Register nodes
    sg.add_node("supervisor", supervisor)
    sg.add_node(Route.DATA.value, data_team)
    sg.add_node(Route.RESEARCH.value, alpha_research_team)
    sg.add_node(Route.STRAT.value, strategy_team)
    sg.add_node(Route.COST.value, cost_team)
    sg.add_node(Route.RISK.value, risk_team)
    sg.add_node(Route.BACKTEST.value, backtest_team)
    sg.add_node(Route.REPORT.value, report_team)

    # Edges
    def router_cond(state: GraphState):
        return state.get("route", Route.DATA.value)

    sg.add_edge(START, "supervisor")
    sg.add_conditional_edges("supervisor", router_cond, {
        Route.DATA.value: Route.DATA.value,
        Route.RESEARCH.value: Route.RESEARCH.value,
        Route.STRAT.value: Route.STRAT.value,
        Route.COST.value: Route.COST.value,
        Route.RISK.value: Route.RISK.value,
        Route.BACKTEST.value: Route.BACKTEST.value,
        Route.REPORT.value: Route.REPORT.value,
    })

    # Local transitions: after each team, go back to supervisor (except report => END)
    for r in [Route.DATA, Route.RESEARCH, Route.STRAT, Route.COST, Route.RISK, Route.BACKTEST]:
        sg.add_edge(r.value, "supervisor")
    sg.add_edge(Route.REPORT.value, END)

    return sg


###############################################################
# 6) ENTRY POINT
###############################################################

def run_once(user_params: Dict[str, Any]) -> Dict[str, Any]:
    params = StrategyParams(**user_params)

    initial: GraphState = {
        "user_params": params,
        "tasks": ["data"],
        "artifacts": {},
        "route": None,
        "report": None,
        "errors": [],
    }

    graph = build_graph().compile(checkpointer=MemorySaver())

    # Simple driver: iterate until END; in production, stream events
    out = None
    for event in graph.stream(initial):
        # event is a dict of {node_name: state}
        out = event
    # last event contains END state
    final_state = list(out.values())[0] if out else initial
    return final_state


if __name__ == "__main__":
    example = {
        "objective": "risk_adjusted_return",
        "universe": ["SPY", "TLT", "GLD"],
        "timeframe": "1d",
        "horizon_days": 730,
        "max_leverage": 1.0,
        "risk_budget_var_95": 0.05,
        "max_drawdown": 0.15,
        "turnover_limit_annual": 8.0,
        "constraints": ["no_short_single_name", "long_only_ETFs"],
        "signals_preference": ["momentum", "trend"],
        "execution_preference": "vwap",
        "capital_usd": 250000,
        "slippage_bps": 2.0,
        "fees_bps": 1.0,
        "notes": "ETF-only pilot",
    }
    final = run_once(example)
    print("--- FINAL REPORT ---\n", final.get("report"))
    if final.get("errors"):
        print("Errors:", final["errors"])
