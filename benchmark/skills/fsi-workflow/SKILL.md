---
name: fsi-workflow
description: >-
  Skill for financial services operations workflow. Monitors tool selection
  across 8 core FSI tools (portfolio management, market data, compliance,
  risk, trading, alerting, reporting), detects systematic tool confusions,
  and applies learned corrections to prevent recurring mis-routing.
license: BSL-1.1
compatibility: Python 3.10+
metadata:
  author: cannyforge
  version: "1.0"
  category: financial-services
  output_type: tool_call
  triggers:
    # Portfolio / account
    - portfolio
    - holdings
    - positions
    - account
    - client account
    - allocation
    - trust account
    - securities held
    # Market data
    - price
    - quote
    - market data
    - current price
    - bid
    - ask
    - last trade
    - intraday
    - market feed
    # Compliance
    - compliance
    - regulatory check
    - violation
    - restricted list
    - watchlist
    - suitability
    - know your customer
    - kyc
    - aml
    # Risk
    - risk
    - var
    - value at risk
    - exposure
    - drawdown
    - stress test
    - portfolio risk
    - concentration risk
    # Trading
    - execute
    - trade
    - order
    - buy
    - sell
    - place order
    - fill
    - execution
    # Alerting (internal)
    - alert
    - notify operations
    - flag
    - escalate
    - internal notification
    - ops team
    - desk alert
    # Client report
    - client report
    - statement
    - performance report
    - generate report
    - send to client
    - client-facing
    # Regulatory filing
    - regulatory filing
    - file with
    - submit to
    - sec filing
    - finra
    - form adv
    - regulatory submission
  tools:
    - fetch_client_portfolio
    - get_market_data
    - run_compliance_check
    - get_risk_metrics
    - execute_trade_order
    - send_internal_alert
    - generate_client_report
    - file_regulatory_report
  context_fields:
    selected_tool:         { type: str,   default: "" }
    tool_match_confidence: { type: float, default: 1.0 }
    confusion_pair:        { type: int,   default: 0 }
    client_id_present:     { type: bool,  default: false }
    market_symbol_present: { type: bool,  default: false }
    is_real_time_request:  { type: bool,  default: false }
    is_client_facing:      { type: bool,  default: false }
    is_regulatory:         { type: bool,  default: false }
  confusion_pairs:
    - id: 1
      tools: [fetch_client_portfolio, get_market_data]
      description: >-
        client portfolio vs market data — same financial instruments,
        different data sources. Portfolio = what a specific client OWNS
        (positions, cost basis, allocation). Market data = current PRICES
        from the exchange feed (bid/ask, last trade, intraday).
      disambiguation_rule: >-
        If the task names a specific client, account, or trust — use
        fetch_client_portfolio. If it asks for current price, quote, or
        market-wide data without a specific client — use get_market_data.
    - id: 2
      tools: [run_compliance_check, get_risk_metrics]
      description: >-
        compliance check vs risk metrics — both quantitative assessments,
        but compliance validates against rules/regulations (restricted
        lists, suitability, AML/KYC), while risk metrics measure financial
        exposure (VaR, drawdown, concentration).
      disambiguation_rule: >-
        If the task mentions regulations, rules, suitability, watchlists,
        or restricted assets — use run_compliance_check. If it asks for
        numerical risk exposure, VaR, or stress testing — use get_risk_metrics.
    - id: 3
      tools: [execute_trade_order, send_internal_alert]
      description: >-
        execute trade vs internal alert — both can follow a decision, but
        execution results in a market transaction while alerts are
        internal-only communications to operations/desk teams.
      disambiguation_rule: >-
        If the task results in a TRADE in the market — use execute_trade_order.
        If the task notifies an internal team without touching the market
        (e.g., flag for review, escalate to ops) — use send_internal_alert.
    - id: 4
      tools: [generate_client_report, file_regulatory_report]
      description: >-
        client report vs regulatory filing — both are formal documents,
        but client reports are delivered to the client (statements,
        performance summaries), while regulatory filings are submitted
        to regulators (SEC, FINRA, ADV forms).
      disambiguation_rule: >-
        If the recipient is the CLIENT — use generate_client_report.
        If the recipient is a REGULATOR (SEC, FINRA, etc.) — use
        file_regulatory_report.
---

# FSI Workflow Skill

Closed-loop learning skill for financial services operations. Encodes
knowledge of 8 core FSI tools and 4 systematic confusion pairs so
CannyForge can auto-detect mis-routing and generate targeted corrections
without requiring external ground truth labels.

## The 8 Tools

| Tool | Domain | Key Signals |
|------|--------|-------------|
| `fetch_client_portfolio` | Portfolio Mgmt | client id, holdings, positions, allocation |
| `get_market_data` | Market Data | price, quote, bid/ask, symbol, intraday |
| `run_compliance_check` | Compliance | violations, restricted list, suitability, KYC/AML |
| `get_risk_metrics` | Risk | VaR, exposure, drawdown, concentration, stress |
| `execute_trade_order` | Trading / OMS | execute, buy, sell, order, fill |
| `send_internal_alert` | Ops Alerting | flag, escalate, notify ops, internal |
| `generate_client_report` | Client Reporting | statement, performance report, send to client |
| `file_regulatory_report` | Regulatory | SEC, FINRA, ADV form, regulatory submission |

## Why These Confusions Are Hard

The 4 confusion pairs share surface vocabulary that tricks models without
domain grounding:

1. **Portfolio vs Market Data**: both involve "securities", "prices", and
   "holdings" — the difference is *whose* data and *which system* it comes from.
2. **Compliance vs Risk**: both produce numbers and both block trades — the
   difference is *rules-based* (compliance) vs *statistical* (risk).
3. **Trade Execution vs Alert**: both follow a decision — the difference is
   *market-facing* (execution) vs *internal-only* (alert).
4. **Client Report vs Regulatory Filing**: both are formal documents — the
   difference is the *recipient* (client vs regulator).

## How CannyForge Closes the Loop

1. Agent is given the 8 tools and FSI task descriptions.
2. If it picks the wrong tool from a confusion pair, `record_error()` is called.
3. The learning engine detects the pair-specific pattern.
4. `CorrectionGenerator` produces a correction rule targeting the pair's
   disambiguation signal (see `disambiguation_rule` above).
5. On the next run, the correction is injected as a `SystemMessage` warning
   before the model sees the task — no human needed.

## Usage

```python
from cannyforge import CannyForge
from cannyforge.adapters.langgraph import CannyForgeMiddleware

forge = CannyForge()
# Skill auto-loaded when scan_skills() finds this SKILL.md
middleware = CannyForgeMiddleware(forge, skill_name="fsi_workflow")
agent = create_react_agent(llm, tools=fsi_tools,
                           pre_model_hook=middleware.before_model,
                           post_model_hook=middleware.after_model)
```

Or point CannyForge at the skill directory:

```python
forge = CannyForge(skills_dir="benchmark/skills/")
```
