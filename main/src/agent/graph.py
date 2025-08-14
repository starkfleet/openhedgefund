"""LangGraph multi-node pipeline: generate strategy → run in sandbox → backtest with EasyBT."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
from typing import Any, Dict, Optional, TypedDict
import os
import tempfile
import importlib.util

import pandas as pd
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

try:
    from .agent_strategy_creator import StrategyCreatorAgent
    from .script_runner import run_sandboxed_python_script
except ImportError:  # When loaded as a module without package context
    from agent.agent_strategy_creator import StrategyCreatorAgent
    from agent.script_runner import run_sandboxed_python_script


class Context(TypedDict):
    """Runtime context for configuration."""

    my_configurable_param: str
    backtest_defaults: Dict[str, Any]


@dataclass
class State:
    """Agent state across nodes."""

    # Required strategy inputs
    buy_condition: str
    sell_condition: str
    symbol: str
    from_date: str
    to_date: str
    timeframe: str
    api_url: str

    # Optional execution/backtest config
    backtest_config: Dict[str, Any] = field(default_factory=dict)

    # Produced by nodes
    script_code: Optional[str] = None
    df_csv: Optional[str] = None
    report: Optional[Dict[str, Any]] = None

    # Error channel
    error: Optional[str] = None
    retry_count: int = 0


def _load_easybt_module() -> Any:
    """Dynamically load EasyBT/main.py as a module without packaging changes."""
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "../../.."))
    easybt_path = os.path.join(repo_root, "EasyBT", "main.py")
    spec = importlib.util.spec_from_file_location("easybt_main", easybt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load EasyBT from {easybt_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def generate_strategy_code(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Use LLM to generate Python strategy script and extract the code without blocking the event loop."""
    try:
        agent = StrategyCreatorAgent()
        # Prefer non-blocking OpenAI call
        script_str = await agent.agenerate_strategy_creator_script(
            buy_condition=state.buy_condition,
            sell_condition=state.sell_condition,
            symbol=state.symbol,
            from_date=state.from_date,
            to_date=state.to_date,
            timeframe=state.timeframe,
            api_url=state.api_url,
        )
        code = StrategyCreatorAgent._extract_python_code(script_str)
        if not code:
            return {"error": "No code extracted from model output.", "retry_count": state.retry_count + 1}
        return {"script_code": code, "error": None}
    except Exception as e:
        return {"error": f"generate_strategy_code failed: {e}", "retry_count": state.retry_count + 1}


async def run_script_in_sandbox(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute generated script inside Docker sandbox and capture CSV output."""
    if not state.script_code:
        return {"error": "No script_code present.", "retry_count": state.retry_count + 1}
    try:
        # Offload blocking sandbox execution to a worker thread
        result, stderr, exit_code = await asyncio.to_thread(
            run_sandboxed_python_script, state.script_code
        )
        if exit_code != 0:
            return {"error": f"Sandbox exit_code={exit_code}. STDERR={stderr}", "retry_count": state.retry_count + 1}

        if isinstance(result, pd.DataFrame):
            csv_text = result.to_csv(index=False)
        else:
            csv_text = str(result).strip()

        if not csv_text:
            return {"error": "Sandbox produced empty output."}

        return {"df_csv": csv_text, "error": None}
    except Exception as e:
        return {"error": f"run_script_in_sandbox failed: {e}", "retry_count": state.retry_count + 1}


async def backtest_with_easybt(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Write signals CSV to temp file and backtest with EasyBT, return summary only."""
    if not state.df_csv:
        return {"error": "No df_csv present.", "retry_count": state.retry_count + 1}
    try:
        # Persist CSV to a temp file for EasyBT API (offload blocking I/O)
        def _write_tmp_csv(content: str) -> str:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write(content)
                return f.name

        csv_path = await asyncio.to_thread(_write_tmp_csv, state.df_csv)

        # Safely construct final config outside thread and pass it down
        context_obj = runtime.context if isinstance(runtime.context, dict) else {}
        try:
            defaults = context_obj.get("backtest_defaults", {}) or {}
        except Exception:
            defaults = {}

        merged_cfg = {**defaults, **(state.backtest_config or {})}
        merged_cfg["data"] = csv_path

        # Execute EasyBT backtest in a thread (imports and CPU work can block)
        def _run_backtest(final_cfg_local: Dict[str, Any]) -> Dict[str, Any] | Any:
            easybt = _load_easybt_module()
            bt = easybt.bt
            return bt.backtester_signal_csv_report(final_cfg_local)

        report = await asyncio.to_thread(_run_backtest, merged_cfg)
        # We keep just the summary to keep state lightweight/persistable
        summary = report["summary"] if isinstance(report, dict) and "summary" in report else report

        return {"report": {"summary": summary}, "error": None}
    except Exception as e:
        return {"error": f"backtest_with_easybt failed: {e}", "retry_count": state.retry_count + 1}
    finally:
        try:
            if 'csv_path' in locals() and os.path.exists(csv_path):
                await asyncio.to_thread(os.remove, csv_path)
        except Exception:
            pass


MAX_RETRIES = 2


def branch_after_generate(state: State) -> str:
    if state.error:
        return "retry_self" if state.retry_count < MAX_RETRIES else "end"
    return "next"


def branch_after_run(state: State) -> str:
    if state.error:
        return "retry" if state.retry_count < MAX_RETRIES else "end"
    if state.df_csv:
        return "backtest"
    return "end"


def branch_after_backtest(state: State) -> str:
    if state.error:
        return "retry" if state.retry_count < MAX_RETRIES else "end"
    return "end"


# Build graph
graph = (
    StateGraph(State, context_schema=Context)
        .add_node(generate_strategy_code)
        .add_node(run_script_in_sandbox)
        .add_node(backtest_with_easybt)
        .add_edge(START, "generate_strategy_code")
        .add_conditional_edges(
            "generate_strategy_code",
            branch_after_generate,
            {"retry_self": "generate_strategy_code", "next": "run_script_in_sandbox", "end": END},
        )
        .add_conditional_edges(
            "run_script_in_sandbox",
            branch_after_run,
            {"retry": "generate_strategy_code", "backtest": "backtest_with_easybt", "end": END},
        )
        .add_conditional_edges(
            "backtest_with_easybt",
            branch_after_backtest,
            {"retry": "generate_strategy_code", "end": END},
        )
        .compile(name="Strategy Backtester")
)
