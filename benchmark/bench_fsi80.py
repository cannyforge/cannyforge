#!/usr/bin/env python3
"""
FSI-Bench-80: Asset Management Tool-Selection Benchmark

Four-condition ablation across 80 tasks (40 learning / 40 held-out):
  baseline   — no correction, raw model output
  static     — hand-crafted system prompt injected at agent creation
  cannyforge — corrections learned from Set A failures only (no static prompt)
  static+cf  — static prompt + learned corrections (production scenario)

Two-model design (recommended):
  --model        : the agent model being evaluated (e.g. qwen3.5:4b, llama3.1:8b)
  --learning-model: model used to synthesize corrections from failures (e.g. deepseek-chat)
                    defaults to template-based correction if omitted

Thinking-mode models (Qwen3, QwQ, DeepSeek-R1):
  --no-think     : prepend /no_think to system prompts, disabling chain-of-thought

Usage:
    python bench_fsi80.py --ollama --model qwen3.5:4b --no-think
    python bench_fsi80.py --ollama --model qwen3.5:4b --no-think \\
                          --learning-model deepseek-chat
    python bench_fsi80.py --nvidia --model qwen/qwen3.5-122b-a10b

Environment variables:
    LLM_API_KEY  / OPENAI_API_KEY
    NVIDIA_API_KEY  (when --nvidia)
    LLM_BASE_URL  (optional, for non-OpenAI endpoints)
    MODEL_FAST    (default: deepseek-chat)
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from cannyforge import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware

try:
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import SystemMessage
except ImportError:
    print("Install required packages: pip install langgraph langchain-openai")
    raise SystemExit(1)

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    _NVIDIA_AVAILABLE = True
except ImportError:
    _NVIDIA_AVAILABLE = False

for _name in ("httpx", "httpcore", "openai", "langgraph", "langchain",
              "CannyForge", "Knowledge", "Learning",
              "Skills", "Tools", "MockCalendarMCP", "WebSearchAPI"):
    logging.getLogger(_name).setLevel(logging.ERROR)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

CONFUSION_PAIR_LABELS = {
    1: "client portfolio vs market data",
    2: "compliance check vs risk metrics",
    3: "execute trade vs internal alert",
    4: "client report vs regulatory filing",
}

ACCOUNT_DB = {
    "alderman_trust": {
        "id": "ACC-001",
        "name": "Alderman Trust",
        "aum": 1_450_000,
        "risk": "moderate",
        "advisor": "chen",
        "holdings": ["AAPL", "MSFT", "BND"],
    },
    "wu_family": {
        "id": "ACC-002",
        "name": "Wu Family",
        "aum": 780_000,
        "risk": "aggressive",
        "advisor": "patel",
        "holdings": ["NVDA", "QQQ", "BTC-USD"],
    },
    "castellano_account": {
        "id": "ACC-003",
        "name": "Castellano Account",
        "aum": 2_100_000,
        "risk": "conservative",
        "advisor": "jones",
        "holdings": ["BND", "IEF", "MSFT"],
    },
    "kowalski_account": {
        "id": "ACC-004",
        "name": "Kowalski Account",
        "aum": 890_000,
        "risk": "moderate",
        "advisor": "ng",
        "holdings": ["SPY", "AGG", "VEA"],
    },
    "pvt_2209": {
        "id": "ACC-005",
        "name": "PVT-2209",
        "aum": 1_120_000,
        "risk": "moderate",
        "advisor": "patel",
        "holdings": ["GLD", "BND", "AAPL"],
    },
    "greenfield_capital": {
        "id": "ACC-006",
        "name": "Greenfield Capital",
        "aum": 1_980_000,
        "risk": "moderate",
        "advisor": "chen",
        "holdings": ["MSFT", "AMZN", "IEF"],
    },
}

ACCOUNT_ALIASES = {
    "alderman trust": "alderman_trust",
    "wu family": "wu_family",
    "wu family account": "wu_family",
    "castellano account": "castellano_account",
    "castellano portfolio": "castellano_account",
    "kowalski account": "kowalski_account",
    "kowalski retirement account": "kowalski_account",
    "pvt 2209": "pvt_2209",
    "account pvt 2209": "pvt_2209",
    "greenfield capital": "greenfield_capital",
    "greenfield capital account": "greenfield_capital",
    "client 4821": "greenfield_capital",
    "account 4821": "greenfield_capital",
    "tru 7701": "greenfield_capital",
    "account tru 7701": "greenfield_capital",
}

MARKET_DB = {
    "AAPL": {"price": 182.50, "52w_high": 199.62, "52w_low": 143.90, "pe": 28.4},
    "MSFT": {"price": 415.20, "52w_high": 430.82, "52w_low": 309.45, "pe": 35.1},
    "NVDA": {"price": 924.10, "52w_high": 974.00, "52w_low": 435.20, "pe": 61.8},
    "BND": {"price": 72.11, "yield": 4.2},
    "VIX": {"level": 17.4},
    "WTI": {"price": 79.30},
    "EUR/USD": {"price": 1.0874},
    "10Y_TSRY": {"yield": 4.18},
    "FED_FUNDS": {"rate": 4.75},
    "MSCI_EM": {"return": 0.084},
    "BTC-USD": {"price": 68_400, "market_cap": "1.3T"},
}

MARKET_ALIASES = {
    "aapl": "AAPL",
    "msft": "MSFT",
    "nvda": "NVDA",
    "bnd": "BND",
    "vix": "VIX",
    "wti": "WTI",
    "eur usd": "EUR/USD",
    "eur/usd": "EUR/USD",
    "10 year treasury": "10Y_TSRY",
    "10-year treasury": "10Y_TSRY",
    "federal funds rate": "FED_FUNDS",
    "fed funds": "FED_FUNDS",
    "msci em": "MSCI_EM",
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
}

VALID_CLIENT_REPORT_TYPES = {
    "performance_statement",
    "trade_confirmation",
    "portfolio_summary",
    "investment_review",
    "tax_lot_summary",
    "benchmark_comparison",
}

VALID_REGULATORY_REPORT_TYPES = {
    "13f",
    "13g",
    "adv",
    "sar",
    "trace",
    "aml",
    "4511",
}

VALID_METRIC_TYPES = {
    "var",
    "cvar",
    "sharpe",
    "beta",
    "duration",
    "volatility",
    "information_ratio",
    "stress_test",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _stable_suffix(value: str) -> str:
    return f"{sum(ord(ch) for ch in value) % 10000:04d}"


def _ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "data": data}


def _error(code: str, hint: str) -> Dict[str, Any]:
    return {"status": "error", "code": code, "hint": hint}


def _resolve_account(query: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_key(query)
    key = ACCOUNT_ALIASES.get(normalized)
    if key:
        return dict(ACCOUNT_DB[key])

    for alias, account_key in ACCOUNT_ALIASES.items():
        if alias in normalized:
            return dict(ACCOUNT_DB[account_key])

    if normalized and any(token in normalized for token in ["account", "portfolio", "trust", "fund", "family", "client"]):
        return {
            "id": f"ACC-{_stable_suffix(normalized)}",
            "name": query,
            "aum": 950_000,
            "risk": "moderate",
            "advisor": "ops",
            "holdings": ["SPY", "BND"],
        }
    return None


def _resolve_market(query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    normalized = _normalize_key(query)
    for alias, symbol in MARKET_ALIASES.items():
        if alias in normalized:
            return symbol, dict(MARKET_DB[symbol])
    upper = query.strip().upper()
    if upper in MARKET_DB:
        return upper, dict(MARKET_DB[upper])
    if normalized and any(token in normalized for token in ["price", "yield", "rate", "spread", "index", "spot", "vix"]):
        return "SYNTH", {"query": query, "price": 100.0}
    return None


def _infer_metric_type(text: str) -> Optional[str]:
    normalized = _normalize_key(text)
    mapping = {
        "cvar": "cvar",
        "sharpe": "sharpe",
        "beta": "beta",
        "duration": "duration",
        "volatility": "volatility",
        "information ratio": "information_ratio",
        "stress test": "stress_test",
        "var": "var",
    }
    for phrase, metric_type in mapping.items():
        if phrase in normalized:
            return metric_type
    return None


def _infer_client_report_type(text: str) -> Optional[str]:
    normalized = _normalize_key(text)
    mapping = {
        "performance": "performance_statement",
        "trade confirmation": "trade_confirmation",
        "portfolio summary": "portfolio_summary",
        "summary": "portfolio_summary",
        "investment review": "investment_review",
        "review": "investment_review",
        "tax lot": "tax_lot_summary",
        "benchmark comparison": "benchmark_comparison",
    }
    for phrase, report_type in mapping.items():
        if phrase in normalized:
            return report_type
    return None


def _infer_regulatory_report_type(text: str) -> Optional[str]:
    normalized = _normalize_key(text)
    mapping = {
        "13f": "13f",
        "13g": "13g",
        "adv": "adv",
        "sar": "sar",
        "suspicious activity": "sar",
        "trace": "trace",
        "anti money laundering": "aml",
        "aml": "aml",
        "4511": "4511",
    }
    for phrase, report_type in mapping.items():
        if phrase in normalized:
            return report_type
    return None


def _contains_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _extract_period(text: str) -> str:
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if date_match:
        return date_match.group(0)
    quarter_match = re.search(r"\b\d{4}-Q[1-4]\b", text, re.IGNORECASE)
    if quarter_match:
        return quarter_match.group(0).upper()
    return "current"


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def fetch_client_portfolio(client_id_or_name: str) -> Dict[str, Any]:
    """Retrieve a specific client's current holdings, account balances, position weights,
    or transaction history from the internal portfolio management system (PMS).
    Use for client-specific data: their positions, allocations, cash flows, trade history."""
    account = _resolve_account(client_id_or_name)
    if not account:
        return _error("NOT_FOUND", "valid keys: Alderman Trust, Wu Family, Castellano Account, Kowalski Account, PVT-2209, Greenfield Capital")
    return _ok(account)

@tool
def query_market_data(instrument_or_query: str) -> Dict[str, Any]:
    """Query current or historical market prices, yields, benchmark indices, FX rates,
    spreads, or macroeconomic rates from the external market data feed.
    Use for security prices, benchmark returns, macro rates — not client-specific data."""
    resolved = _resolve_market(instrument_or_query)
    if not resolved:
        return _error("NOT_FOUND", "valid keys: AAPL, MSFT, NVDA, BND, VIX, WTI, EUR/USD, 10Y Treasury, Fed Funds")
    symbol, data = resolved
    return _ok({"symbol": symbol, **data})

@tool
def run_compliance_check(check_request: str, rule_scope: str = "") -> Dict[str, Any]:
    """Run a pre-trade or position compliance check against investment policy statements (IPS),
    regulatory concentration limits, restricted/watch lists, and mandate constraints.
    Use when verifying whether a trade or position is permitted — not for calculating numbers."""
    if not check_request.strip():
        return _error("INVALID_REQUEST", "check_request must be non-empty")
    normalized = _normalize_key(f"{check_request} {rule_scope}")
    flagged = any(token in normalized for token in ["restricted", "watch list", "violate", "exceed"])
    return _ok({
        "allowed": not flagged,
        "scope": rule_scope or "ips",
        "request": check_request,
    })

@tool
def calculate_risk_metrics(portfolio_or_position: str, metric_type: str = "") -> Dict[str, Any]:
    """Compute quantitative risk analytics: VaR, CVaR, Sharpe ratio, beta, duration,
    tracking error, max drawdown, volatility, or information ratio for a portfolio or position.
    Use for computing a number — not for verifying whether something is allowed."""
    resolved_metric = (metric_type or _infer_metric_type(portfolio_or_position) or "").lower()
    if resolved_metric not in VALID_METRIC_TYPES:
        return _error("INVALID_METRIC", f"valid metric_type values: {sorted(VALID_METRIC_TYPES)}")

    account = _resolve_account(portfolio_or_position) or {
        "id": f"ACC-{_stable_suffix(_normalize_key(portfolio_or_position))}",
        "name": portfolio_or_position,
        "aum": 1_000_000,
        "risk": "moderate",
    }
    base_values = {
        "var": 0.082,
        "cvar": 0.114,
        "sharpe": 1.18,
        "beta": 0.93,
        "duration": 5.4,
        "volatility": 0.167,
        "information_ratio": 0.61,
        "stress_test": -0.124,
    }
    return _ok({
        "account_id": account["id"],
        "metric_type": resolved_metric,
        "value": base_values[resolved_metric],
    })

@tool
def execute_trade(order_details: str, account_id: str = "", symbol: str = "", action: str = "") -> Dict[str, Any]:
    """Place, modify, or cancel a trade order in the order management system (OMS).
    Use for buy/sell/rebalance instructions that result in an actual order being submitted."""
    if not order_details.strip():
        return _error("INVALID_ORDER", "order_details must be non-empty")
    resolved_account = _resolve_account(account_id or order_details)
    resolved_action = action or ("sell" if "sell" in _normalize_key(order_details) else "buy")
    return _ok({
        "order_id": f"ORD-{_stable_suffix(order_details)}",
        "account_id": resolved_account["id"] if resolved_account else "ACC-UNKNOWN",
        "action": resolved_action,
        "symbol": symbol or "TBD",
    })

@tool
def send_internal_alert(message: str, recipient: str = "") -> Dict[str, Any]:
    """Send an internal notification or alert to an advisor, risk manager, compliance officer,
    or back-office team via the firm's internal messaging system.
    Use for flagging issues, escalating to colleagues, or notifying internal teams — not regulators."""
    if not message.strip():
        return _error("INVALID_ALERT", "message must be non-empty")
    return _ok({
        "alert_id": f"ALT-{_stable_suffix(message)}",
        "recipient": recipient or "operations",
        "message": message,
    })

@tool
def generate_client_report(client_id_or_name: str, report_type: str = "", period: str = "") -> Dict[str, Any]:
    """Generate a client-facing document: portfolio statement, performance report,
    trade confirmation, or investment summary for delivery to the client or their advisor.
    Use for client communications and reporting — not regulatory submissions."""
    account = _resolve_account(client_id_or_name)
    if not account:
        return _error("NOT_FOUND", "valid keys: Alderman Trust, Wu Family, Castellano Account, Kowalski Account, PVT-2209, Greenfield Capital")

    resolved_type = (report_type or _infer_client_report_type(client_id_or_name) or "").lower()
    if resolved_type not in VALID_CLIENT_REPORT_TYPES:
        return _error("INVALID_REPORT_TYPE", f"valid report_type values: {sorted(VALID_CLIENT_REPORT_TYPES)}")
    return _ok({
        "report_id": f"RPT-{_stable_suffix(account['id'] + resolved_type + (period or 'current'))}",
        "client_id": account["id"],
        "report_type": resolved_type,
        "period": period or _extract_period(client_id_or_name),
    })

@tool
def file_regulatory_report(report_type: str, filing_spec: str = "", start_date: str = "", end_date: str = "") -> Dict[str, Any]:
    """Submit filings and reports to external regulators: SEC 13F, Form ADV, FINRA trade
    reporting, CFTC position reports, FinCEN SARs, or TRACE OTC bond reporting.
    Use for mandatory external regulatory obligations — not internal alerts or client reports."""
    spec_text = filing_spec or report_type
    resolved_type = report_type.lower()
    if resolved_type not in VALID_REGULATORY_REPORT_TYPES:
        inferred = _infer_regulatory_report_type(spec_text)
        if not inferred:
            return _error("INVALID_REPORT_TYPE", f"valid report_type values: {sorted(VALID_REGULATORY_REPORT_TYPES)}")
        resolved_type = inferred

    if start_date and not _contains_iso_date(start_date):
        return _error("INVALID_DATE", "start_date must be YYYY-MM-DD")
    if end_date and not _contains_iso_date(end_date):
        return _error("INVALID_DATE", "end_date must be YYYY-MM-DD")

    return _ok({
        "filing_id": f"FIL-{_stable_suffix(resolved_type + spec_text + start_date + end_date)}",
        "report_type": resolved_type,
        "filing_spec": spec_text,
        "start_date": start_date,
        "end_date": end_date,
    })

TOOLS = [
    fetch_client_portfolio, query_market_data, run_compliance_check,
    calculate_risk_metrics, execute_trade, send_internal_alert,
    generate_client_report, file_regulatory_report,
]


# ── Static system prompt (ablation condition 2) ────────────────────────────────

STATIC_SYSTEM_PROMPT = """\
You are a financial operations assistant. Select the correct tool using these enterprise rules:

- fetch_client_portfolio  : client-specific account data — balances, holdings, allocations, \
transaction history for a named client or account number
- query_market_data       : prices, yields, FX rates, benchmark index returns, macro rates — \
for securities and market instruments, not client accounts
- run_compliance_check    : verify whether a trade or position is *permitted* by IPS, mandate, \
concentration limits, or restricted list — use when asking "is this allowed?"
- calculate_risk_metrics  : compute quantitative risk numbers — VaR, CVaR, Sharpe, beta, \
duration, drawdown, volatility, tracking error, information ratio
- execute_trade           : place, modify, or cancel orders in the OMS — use when an actual \
order should be submitted
- send_internal_alert     : notify internal colleagues — advisors, risk managers, compliance, \
back-office — via the firm's internal messaging system
- generate_client_report  : create client-facing documents — statements, confirmations, \
performance summaries, investment reviews
- file_regulatory_report  : submit external regulatory filings — SEC, FINRA, CFTC, FinCEN, \
Form ADV, 13F, TRACE, SAR, whistleblower tips

Argument rules:
- generate_client_report expects report_type values like performance_statement, trade_confirmation, \
portfolio_summary, investment_review, tax_lot_summary, benchmark_comparison
- file_regulatory_report expects report_type values like 13f, 13g, adv, sar, trace, aml, 4511 and \
ISO dates in YYYY-MM-DD when date fields are provided
- calculate_risk_metrics expects metric_type values like var, cvar, sharpe, beta, duration, \
volatility, information_ratio, stress_test
"""


# ── LLM builder ───────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Prefix injected into every system message when --no-think is set.
# Qwen3/QwQ honour /no_think in system context to skip chain-of-thought.
NO_THINK_PREFIX = "/no_think\n\n"


def build_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    nvidia: bool = False,
) -> Optional[Any]:
    if nvidia:
        if not _NVIDIA_AVAILABLE:
            print("Install langchain-nvidia-ai-endpoints: pip install langchain-nvidia-ai-endpoints")
            return None
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        model = model or "meta/llama-3.1-8b-instruct"
        if not api_key:
            print("Set NVIDIA_API_KEY or pass --api-key")
            return None
        return ChatNVIDIA(model=model, api_key=api_key, temperature=0,
                          max_tokens=1024, timeout=timeout)

    api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("LLM_BASE_URL")
    model = model or os.environ.get("MODEL_FAST", "deepseek-chat")
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {
        "model": model, "api_key": api_key, "temperature": 0, "timeout": timeout,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_learning_provider(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Build a CannyForge LLMProvider for correction generation."""
    from cannyforge.llm import OpenAIProvider
    return OpenAIProvider(
        model=model,
        api_key=api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("LLM_BASE_URL"),
    )


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    id: str
    set: str
    task: str
    expected: str
    actual: str
    correct: bool
    difficulty: str
    confusion_pair: int
    confusable_with: str
    correction_injected: bool
    condition: str
    param_score: float = -1.0
    sequence_correct: Optional[bool] = None
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Agent helpers ─────────────────────────────────────────────────────────────

def _prepend_no_think(content: str, no_think: bool) -> str:
    if no_think and not content.startswith("/no_think"):
        return NO_THINK_PREFIX + content
    return content


def _make_static_pre_hook(no_think: bool = False):
    prompt = _prepend_no_think(STATIC_SYSTEM_PROMPT, no_think)

    def hook(state: Dict) -> Dict:
        msgs = list(state.get("messages", []))
        if msgs and hasattr(msgs[0], "type") and msgs[0].type == "system":
            return state
        return {"messages": [SystemMessage(content=prompt)] + msgs}
    return hook


def _make_combined_pre_hook(middleware: CannyForgeMiddleware, no_think: bool = False):
    """Static guidelines first, then CannyForge learned corrections on top.

    Message order the model sees:
      [CANNYFORGE] corrections  ← innermost, most specific
      static system prompt      ← general domain guidelines
      user message
    """
    prompt = _prepend_no_think(STATIC_SYSTEM_PROMPT, no_think)

    def hook(state: Dict) -> Dict:
        result = middleware.before_model(state)
        msgs = list(result.get("messages", []))
        # If CF injected a [CANNYFORGE] system message, also prepend /no_think to it
        if no_think and msgs and hasattr(msgs[0], "type") and msgs[0].type == "system":
            cf_content = getattr(msgs[0], "content", "")
            if "[CANNYFORGE]" in cf_content and not cf_content.startswith("/no_think"):
                msgs[0] = SystemMessage(content=NO_THINK_PREFIX + cf_content)
        return {"messages": [SystemMessage(content=prompt)] + msgs}
    return hook


def make_baseline_agent(llm: ChatOpenAI, no_think: bool = False):
    if not no_think:
        return create_react_agent(llm, TOOLS)
    # For thinking models with no static prompt: inject /no_think via a minimal hook
    def _no_think_hook(state: Dict) -> Dict:
        msgs = list(state.get("messages", []))
        if msgs and hasattr(msgs[0], "type") and msgs[0].type == "system":
            return state
        return {"messages": [SystemMessage(content=NO_THINK_PREFIX.strip())] + msgs}
    return create_react_agent(llm, TOOLS, pre_model_hook=_no_think_hook)


def make_static_agent(llm: ChatOpenAI, no_think: bool = False):
    return create_react_agent(llm, TOOLS, pre_model_hook=_make_static_pre_hook(no_think))


def make_cannyforge_agent(llm: ChatOpenAI, middleware: CannyForgeMiddleware, no_think: bool = False):
    """Baseline + CannyForge corrections only (no static prompt)."""
    if no_think:
        # Wrap middleware.before_model to also prepend /no_think on CF messages
        original = middleware.before_model
        def _cf_no_think_hook(state: Dict) -> Dict:
            result = original(state)
            msgs = list(result.get("messages", []))
            if msgs and hasattr(msgs[0], "type") and msgs[0].type == "system":
                content = getattr(msgs[0], "content", "")
                if not content.startswith("/no_think"):
                    msgs[0] = SystemMessage(content=NO_THINK_PREFIX + content)
            return {"messages": msgs}
        return create_react_agent(llm, TOOLS,
                                  pre_model_hook=_cf_no_think_hook,
                                  post_model_hook=middleware.after_model)
    return create_react_agent(llm, TOOLS,
                              pre_model_hook=middleware.before_model,
                              post_model_hook=middleware.after_model)


def make_static_cf_agent(llm: ChatOpenAI, middleware: CannyForgeMiddleware, no_think: bool = False):
    """Static prompt + CannyForge corrections (production deployment scenario)."""
    return create_react_agent(
        llm, TOOLS,
        pre_model_hook=_make_combined_pre_hook(middleware, no_think),
        post_model_hook=middleware.after_model,
    )


# ── Extraction ────────────────────────────────────────────────────────────────

def _message_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    if isinstance(msg, dict):
        return msg.get("tool_calls", []) or []
    return getattr(msg, "tool_calls", []) or []


def _message_status(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("status", "")
    return getattr(msg, "status", "")


def extract_tool_info(result: Any) -> Tuple[str, dict]:
    """Return (tool_name, tool_args) from first tool call in result."""
    for msg in result.get("messages", []):
        tool_calls = _message_tool_calls(msg)
        if tool_calls:
            first = tool_calls[0]
            return first.get("name", "none"), first.get("args", {}) or {}
    return "none", {}


def extract_tool_sequence(result: Any) -> List[str]:
    sequence: List[str] = []
    for msg in result.get("messages", []):
        for tool_call in _message_tool_calls(msg):
            name = tool_call.get("name")
            if name:
                sequence.append(name)
    return sequence


def score_param(actual_args: dict, expected_contains: dict) -> float:
    """Score argument match against expected regex fragments."""
    if not expected_contains:
        return 1.0
    hits = 0
    for key, pattern in expected_contains.items():
        value = str(actual_args.get(key, ""))
        if re.search(pattern, value, re.IGNORECASE):
            hits += 1
    return hits / len(expected_contains)


def _sequence_matches(actual: List[str], expected: List[str]) -> bool:
    if not expected:
        return True
    return actual[:len(expected)] == expected


def _analyze_recovery(result: Any) -> Tuple[bool, bool]:
    messages = result.get("messages", [])
    had_error = any(_message_status(msg) == "error" for msg in messages)
    sequence = extract_tool_sequence(result)
    recovery_attempted = had_error and len(sequence) > 1
    recovery_succeeded = recovery_attempted and _message_status(messages[-1]) != "error"
    return recovery_attempted, recovery_succeeded


# ── Phase runner ──────────────────────────────────────────────────────────────

def run_phase(
    agent,
    tasks: List[Dict],
    condition: str,
    phase_label: str,
    middleware: Optional[CannyForgeMiddleware] = None,
    delay: float = 0.0,
) -> List[TaskResult]:
    print(f"\n{phase_label}")
    print("-" * len(phase_label))
    results: List[TaskResult] = []

    for t in tasks:
        actual = "none"
        actual_args: Dict[str, Any] = {}
        actual_sequence: List[str] = []
        param_score = -1.0
        sequence_correct: Optional[bool] = None
        recovery_attempted = False
        recovery_succeeded = False
        err = None
        injected = False
        try:
            out = agent.invoke({"messages": [("user", t["task"])]})
            actual, actual_args = extract_tool_info(out)
            actual_sequence = extract_tool_sequence(out)
            recovery_attempted, recovery_succeeded = _analyze_recovery(out)
            if middleware:
                injected = bool(middleware._corrections_injected)
        except Exception as e:
            err = str(e)[:120]
            actual = "error"

        expected_args = t.get("expected_arg_contains")
        if expected_args is not None:
            param_score = score_param(actual_args, expected_args) if actual != "error" else 0.0

        expected_sequence = t.get("expected_sequence")
        if expected_sequence:
            sequence_correct = _sequence_matches(actual_sequence, expected_sequence)

        params_ok = expected_args is None or param_score == 1.0
        sequence_ok = sequence_correct is not False
        correct = actual == t["expected"] and params_ok and sequence_ok
        marker = "OK  " if correct else "MISS"
        diff_tag = t["difficulty"][0].upper()
        inj_note = " +INJ" if injected else ""
        print(f"  [{t['id']}] [{marker}][{diff_tag}] {t['task'][:54]:<54} → {actual}{inj_note}")

        results.append(TaskResult(
            id=t["id"], set=t["set"], task=t["task"],
            expected=t["expected"], actual=actual, correct=correct,
            difficulty=t["difficulty"], confusion_pair=t["confusion_pair"],
            confusable_with=t["confusable_with"],
            correction_injected=injected, condition=condition,
            param_score=param_score, sequence_correct=sequence_correct,
            recovery_attempted=recovery_attempted,
            recovery_succeeded=recovery_succeeded,
            error=err,
        ))

        if delay:
            time.sleep(delay)

    correct_count = sum(r.correct for r in results)
    print(f"\n  → {correct_count}/{len(results)} correct ({correct_count/len(results):.0%})")
    return results


# ── Learning ──────────────────────────────────────────────────────────────────

def build_fresh_forge() -> CannyForge:
    forge = CannyForge(llm_provider=None)
    forge.reset()
    for attr in ("rules_by_skill", "corrections_by_skill", "rule_index", "correction_index"):
        getattr(forge.knowledge_base, attr).clear()
    return forge


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_phase_checkpoint(run_dir: Path, phase_name: str,
                           results: List[TaskResult]) -> None:
    """Save phase results to a checkpoint JSONL file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cp = run_dir / f"ckpt_{phase_name}.jsonl"
    with cp.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def _load_phase_checkpoint(run_dir: Path, phase_name: str) -> Optional[List[TaskResult]]:
    """Load phase results from checkpoint.  Returns None if checkpoint missing."""
    cp = run_dir / f"ckpt_{phase_name}.jsonl"
    if not cp.exists():
        return None
    results = []
    with cp.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            results.append(TaskResult(
                id=d["id"], set=d["set"], task=d["task"],
                expected=d["expected"], actual=d["actual"],
                correct=d["correct"], difficulty=d["difficulty"],
                confusion_pair=d["confusion_pair"],
                confusable_with=d.get("confusable_with", ""),
                correction_injected=d.get("correction_injected", False),
                condition=d["condition"],
                param_score=d.get("param_score", -1.0),
                sequence_correct=d.get("sequence_correct"),
                recovery_attempted=d.get("recovery_attempted", False),
                recovery_succeeded=d.get("recovery_succeeded", False),
                error=d.get("error"),
            ))
    return results


def _save_learning_checkpoint(run_dir: Path, forge: CannyForge) -> None:
    """Save forge corrections so they survive a resume."""
    run_dir.mkdir(parents=True, exist_ok=True)
    corrections = forge.knowledge_base.get_corrections("tool_use_fsi")
    with (run_dir / "ckpt_corrections.json").open("w") as f:
        json.dump([c.to_dict() for c in corrections], f, indent=2)


def _load_learning_checkpoint(run_dir: Path, forge: CannyForge) -> bool:
    """Restore corrections into forge from checkpoint.  Returns True if loaded."""
    cp = run_dir / "ckpt_corrections.json"
    if not cp.exists():
        return False
    from cannyforge.corrections import Correction
    corrections_data = json.loads(cp.read_text())
    for cdata in corrections_data:
        forge.knowledge_base.add_correction("tool_use_fsi", Correction.from_dict(cdata))
    return True


def learn_from_failures(forge: CannyForge, results: List[TaskResult], llm_provider=None):
    failures = [r for r in results if not r.correct and r.actual != "error"]
    for r in failures:
        forge.learning_engine.record_error(
            skill_name="tool_use_fsi",
            task_description=r.task,
            error_type="WrongToolError",
            error_message=f"Called {r.actual} instead of {r.expected}",
            context_snapshot={
                "task": {"description": r.task},
                "context": {"selected_tool": r.actual, "expected_tool": r.expected},
            },
        )
    return forge.run_learning_cycle(
        min_frequency=1, min_confidence=0.15, llm_provider=llm_provider
    )


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def accuracy(results: List[TaskResult]):
    if not results:
        return 0, 0, 0.0
    c = sum(r.correct for r in results)
    t = len(results)
    return c, t, c / t


def acc_by_difficulty(results: List[TaskResult]) -> Dict:
    out = {}
    for d in ("easy", "medium", "hard"):
        sub = [r for r in results if r.difficulty == d]
        c, t, p = accuracy(sub)
        out[d] = {"correct": c, "total": t, "pct": round(p, 3)}
    return out


def acc_by_pair(results: List[TaskResult]) -> Dict:
    out = {}
    for p in (1, 2, 3, 4):
        sub = [r for r in results if r.confusion_pair == p]
        c, t, pct = accuracy(sub)
        out[str(p)] = {"correct": c, "total": t, "pct": round(pct, 3),
                       "label": CONFUSION_PAIR_LABELS[p]}
    return out


# ── Artifact saving ───────────────────────────────────────────────────────────

def save_artifacts(
    run_dir: Path,
    phases: Dict[str, List[TaskResult]],
    corrections: List,
    model_name: str,
) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-phase JSONL
    for label, results in phases.items():
        with open(run_dir / f"{label}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")

    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "set", "condition", "difficulty", "expected", "actual", "correct",
                "param_score", "sequence_correct", "recovery_attempted", "recovery_succeeded",
                "correction_injected", "error",
            ],
        )
        writer.writeheader()
        for results in phases.values():
            for result in results:
                writer.writerow({
                    "id": result.id,
                    "set": result.set,
                    "condition": result.condition,
                    "difficulty": result.difficulty,
                    "expected": result.expected,
                    "actual": result.actual,
                    "correct": result.correct,
                    "param_score": result.param_score,
                    "sequence_correct": result.sequence_correct,
                    "recovery_attempted": result.recovery_attempted,
                    "recovery_succeeded": result.recovery_succeeded,
                    "correction_injected": result.correction_injected,
                    "error": result.error,
                })

    # Corrections
    with open(run_dir / "corrections.json", "w") as f:
        json.dump(
            [{"content": c.content,
              "skill": getattr(c, "skill_name", "tool_use_fsi")}
             for c in corrections],
            f, indent=2,
        )

    # Summary by condition across both sets
    conditions = {
        "baseline":   phases.get("a_baseline", [])  + phases.get("b_baseline", []),
        "static":     phases.get("a_static", [])    + phases.get("b_static", []),
        "cannyforge": phases.get("a_cannyforge", []) + phases.get("b_cannyforge", []),
        "static+cf":  phases.get("a_static_cf", []) + phases.get("b_static_cf", []),
    }

    summary: Dict[str, Any] = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "corrections_count": len(corrections),
        "conditions": {},
    }

    for cond_label, all_results in conditions.items():
        set_a = [r for r in all_results if r.set == "A"]
        set_b = [r for r in all_results if r.set == "B"]
        ca, ta, pa = accuracy(set_a)
        cb, tb, pb = accuracy(set_b)
        summary["conditions"][cond_label] = {
            "set_a": {"correct": ca, "total": ta, "accuracy": round(pa, 3)},
            "set_b": {"correct": cb, "total": tb, "accuracy": round(pb, 3)},
            "set_b_by_difficulty": acc_by_difficulty(set_b),
            "set_b_by_confusion_pair": acc_by_pair(set_b),
        }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nArtifacts saved → {run_dir}/")
    return summary


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(summary: Dict):
    print("\n" + "=" * 72)
    print("FSI-Bench-80  Ablation Summary")
    print(f"Model: {summary['model']}  |  {summary['timestamp'][:19]}")
    print("=" * 72)

    conds = summary["conditions"]
    b_base = conds.get("baseline", {}).get("set_b", {}).get("accuracy", 0)

    header = f"  {'Condition':<14} {'Set A':>8} {'Set B':>8} {'Δ vs base':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, data in conds.items():
        sa = data["set_a"]["accuracy"]
        sb = data["set_b"]["accuracy"]
        delta = f"+{sb - b_base:.1%}" if label != "baseline" else "  —"
        print(f"  {label:<14} {sa:>7.1%} {sb:>8.1%} {delta:>10}")

    print()
    print("  Set B accuracy by difficulty tier:")
    for label, data in conds.items():
        diffs = data.get("set_b_by_difficulty", {})
        parts = "  ".join(
            f"{d[0].upper()}={v['correct']}/{v['total']}"
            for d, v in diffs.items()
        )
        print(f"  {label:<14} {parts}")

    print()
    print("  Set B accuracy by confusion pair:")
    pair_header = f"  {'Pair / description':<40}"
    for label in conds:
        pair_header += f"  {label[:8]:>8}"
    print(pair_header)
    print("  " + "-" * (len(pair_header) - 2))
    for p in ("1", "2", "3", "4"):
        row_label = f"  P{p}: {CONFUSION_PAIR_LABELS[int(p)][:36]:<36}"
        row = row_label
        for label, data in conds.items():
            pct = data.get("set_b_by_confusion_pair", {}).get(p, {}).get("pct", 0)
            row += f"  {pct:>8.1%}"
        print(row)

    print()
    print(f"  Corrections generated: {summary['corrections_count']}")
    print("=" * 72)


def print_stratified_table(all_results: List[TaskResult], tasks_meta: Dict[str, Dict[str, Any]]):
    """Print accuracy broken down by error_mode for Set B tasks."""
    error_modes = ["format", "ambiguity", "context_miss", None]
    conditions = ["baseline", "static", "cannyforge", "static+cf"]

    print("\n--- Set B: Accuracy by Error Mode ---")
    header = f"{'error_mode':<15}" + "".join(f"{condition:>15}" for condition in conditions)
    print(header)

    for mode in error_modes:
        label = str(mode) if mode else "tool_routing"
        row = f"{label:<15}"
        for condition in conditions:
            subset = [
                result for result in all_results
                if result.set == "B"
                and result.condition == condition
                and tasks_meta.get(result.id, {}).get("error_mode") == mode
            ]
            if subset:
                acc = sum(1 for result in subset if result.correct) / len(subset)
                row += f"{acc:>14.1%}"
            else:
                row += f"{'n/a':>15}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FSI-Bench-80 ablation runner")
    parser.add_argument("--sets", choices=["A", "B", "AB"], default="AB",
                        help="Which task sets to run (default: AB)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait between LLM calls (rate limiting)")
    # Agent model (the model being evaluated)
    parser.add_argument("--model", default=None,
                        help="Agent model name, e.g. llama3.1:8b, qwen3.5:4b, deepseek-chat")
    parser.add_argument("--ollama", action="store_true",
                        help=f"Use local Ollama endpoint ({OLLAMA_BASE_URL}); implies --api-key ollama")
    parser.add_argument("--nvidia", action="store_true",
                        help="Use NVIDIA NIM endpoint via ChatNVIDIA (requires NVIDIA_API_KEY)")
    parser.add_argument("--base-url", default=None, dest="base_url",
                        help="LLM API base URL (overrides LLM_BASE_URL env var)")
    parser.add_argument("--api-key", default=None, dest="api_key",
                        help="API key (overrides LLM_API_KEY; default 'ollama' when --ollama)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Seconds per agent LLM call (default: 120)")
    # Thinking-model control
    parser.add_argument("--no-think", action="store_true", dest="no_think",
                        help="Prepend /no_think to system prompts (disables CoT in Qwen3/QwQ/DeepSeek-R1)")
    # Learning model (separate, better model for correction generation)
    parser.add_argument("--learning-model", default=None, dest="learning_model",
                        help="Model for LLM-based correction generation (e.g. deepseek-chat). "
                             "Uses same endpoint as agent model unless --learning-base-url is set. "
                             "If omitted, falls back to template-based corrections.")
    parser.add_argument("--learning-base-url", default=None, dest="learning_base_url",
                        help="API base URL for the learning model (if different from agent endpoint)")
    parser.add_argument("--learning-api-key", default=None, dest="learning_api_key",
                        help="API key for the learning model (if different from agent key)")
    parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                        help="Resume from a previous run directory (skips completed phases)")
    args = parser.parse_args()

    print("=" * 72)
    print("FSI-Bench-80: Financial Operations Tool-Selection Ablation")
    print("Conditions: baseline | static | CannyForge-only | static+CF")
    print("=" * 72)

    # Resolve Ollama shorthand
    base_url = args.base_url
    api_key = args.api_key
    if args.ollama:
        base_url = base_url or OLLAMA_BASE_URL
        api_key = api_key or "ollama"

    llm = build_llm(model=args.model, base_url=base_url, api_key=api_key,
                    timeout=args.timeout, nvidia=args.nvidia)
    if not llm:
        print("No API key. Set LLM_API_KEY / OPENAI_API_KEY / NVIDIA_API_KEY, or pass --ollama / --api-key.")
        raise SystemExit(1)

    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    model_slug = model_name.replace(":", "_").replace(".", "_").replace("/", "_")
    print(f"Agent model   : {model_name}")
    if args.nvidia:
        print("Agent provider: NVIDIA NIM")
    elif args.ollama or base_url:
        print(f"Agent URL     : {base_url or '(env)'}")
    print(f"Timeout       : {args.timeout}s per call")
    if args.no_think:
        print("Thinking mode : OFF (/no_think)")

    # Build learning provider (optional)
    learning_provider = None
    if args.learning_model:
        learn_base_url = args.learning_base_url or base_url
        learn_api_key  = args.learning_api_key  or api_key
        learning_provider = build_learning_provider(args.learning_model, learn_base_url, learn_api_key)
        print(f"Learning model: {args.learning_model} (LLM-based corrections)")
    else:
        print("Learning model: none (template-based corrections)")

    all_tasks = json.loads((DATA_DIR / "fsi80_tasks.json").read_text())
    tasks_meta = {task["id"]: task for task in all_tasks}
    tasks_a = [t for t in all_tasks if t["set"] == "A"]
    tasks_b = [t for t in all_tasks if t["set"] == "B"]
    print(f"Tasks : Set A={len(tasks_a)}, Set B={len(tasks_b)}")
    if args.delay:
        print(f"Delay : {args.delay}s between calls")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.resume) if args.resume else RESULTS_DIR / f"run_{model_slug}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print(f"Resuming from: {run_dir}")

    phases: Dict[str, List[TaskResult]] = {}
    corrections: List = []
    forge = build_fresh_forge()
    middleware: Optional[CannyForgeMiddleware] = None

    run_a = args.sets in ("A", "AB")
    run_b = args.sets in ("B", "AB")

    # Build agents once (static_cf built after learning cycle)
    no_think = args.no_think
    agent_baseline = make_baseline_agent(llm, no_think)
    agent_static = make_static_agent(llm, no_think)

    # Helper to run or load a phase from checkpoint
    def run_or_load(phase_name, agent, tasks, condition, label, mw=None):
        cached = _load_phase_checkpoint(run_dir, phase_name)
        if cached is not None:
            c = sum(r.correct for r in cached)
            print(f"\n{label}")
            print(f"  ↳ resumed from checkpoint ({c}/{len(cached)} correct)")
            return cached
        results = run_phase(agent, tasks, condition, label,
                            middleware=mw, delay=args.delay)
        _save_phase_checkpoint(run_dir, phase_name, results)
        return results

    # ── Set A ──────────────────────────────────────────────────────────────
    if run_a:
        phases["a_baseline"] = run_or_load(
            "a_baseline", agent_baseline, tasks_a, "baseline",
            "Phase 1/8 — Set A  baseline")

        # Learn from Set A failures (or restore from checkpoint)
        if _load_learning_checkpoint(run_dir, forge):
            corrections = forge.knowledge_base.get_corrections("tool_use_fsi")
            print(f"\nLearning cycle\n  ↳ resumed from checkpoint ({len(corrections)} corrections)")
        else:
            print("\nLearning cycle")
            print("-" * 14)
            if learning_provider:
                print(f"  Using LLM-based corrections ({args.learning_model})")
            metrics = learn_from_failures(forge, phases["a_baseline"], llm_provider=learning_provider)
            corrections = forge.knowledge_base.get_corrections("tool_use_fsi")
            _save_learning_checkpoint(run_dir, forge)

            print(f"  Errors analyzed     : {metrics.errors_analyzed}")
            print(f"  Patterns detected   : {metrics.patterns_detected}")
            print(f"  Corrections generated: {metrics.corrections_generated}")
            for i, c in enumerate(corrections, 1):
                print(f"  {i}. {c.content}")

        middleware = CannyForgeMiddleware(forge, skill_name="tool_use_fsi")
        agent_cf = make_cannyforge_agent(llm, middleware, no_think)
        agent_static_cf = make_static_cf_agent(llm, middleware, no_think)

        phases["a_static"] = run_or_load(
            "a_static", agent_static, tasks_a, "static",
            "Phase 2/8 — Set A  static prompt only")

        phases["a_cannyforge"] = run_or_load(
            "a_cannyforge", agent_cf, tasks_a, "cannyforge",
            "Phase 3/8 — Set A  CannyForge only (baseline + corrections)",
            mw=middleware)

        phases["a_static_cf"] = run_or_load(
            "a_static_cf", agent_static_cf, tasks_a, "static+cf",
            "Phase 4/8 — Set A  static + CannyForge (production scenario)",
            mw=middleware)

    # ── Set B ──────────────────────────────────────────────────────────────
    if run_b:
        if middleware is None:
            middleware = CannyForgeMiddleware(forge, skill_name="tool_use_fsi")
            agent_cf = make_cannyforge_agent(llm, middleware, no_think)
            agent_static_cf = make_static_cf_agent(llm, middleware, no_think)

        phases["b_baseline"] = run_or_load(
            "b_baseline", agent_baseline, tasks_b, "baseline",
            "Phase 5/8 — Set B  baseline (held-out, no correction)")

        phases["b_static"] = run_or_load(
            "b_static", agent_static, tasks_b, "static",
            "Phase 6/8 — Set B  static prompt only")

        phases["b_cannyforge"] = run_or_load(
            "b_cannyforge", agent_cf, tasks_b, "cannyforge",
            "Phase 7/8 — Set B  CannyForge only (baseline + corrections)",
            mw=middleware)

        phases["b_static_cf"] = run_or_load(
            "b_static_cf", agent_static_cf, tasks_b, "static+cf",
            "Phase 8/8 — Set B  static + CannyForge (key result: production scenario)",
            mw=middleware)

    summary = save_artifacts(run_dir, phases, corrections, model_name)
    print_summary(summary)
    print_stratified_table([result for phase in phases.values() for result in phase], tasks_meta)

    print(f"\nRun ablation_report.py {run_dir}  for full per-task analysis.")


if __name__ == "__main__":
    main()
