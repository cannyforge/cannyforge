---
name: fsi-workflow
description: >-
  Portable corrections for asset management agents operating the FSI-Bench-80 tool suite.
  Covers four confusion-pair disambiguations (client portfolio vs market data,
  compliance check vs risk metrics, execute trade vs internal alert, client report vs
  regulatory filing) plus ISO-8601 date and enum-value format enforcement.
license: BSL-1.1
compatibility: Python 3.10+
metadata:
  author: cannyforge
  version: "0.3.0"
  category: tool-use
  output_type: tool_call
  domain: financial-services
  triggers:
    - portfolio
    - compliance
    - trade
    - report
    - filing
    - client
    - market data
    - risk
---

# fsi-workflow

Portable CannyForge correction bundle for the **FSI-Bench-80 financial operations benchmark**.
Targets eight-tool asset management agents operating against a portfolio management system (PMS),
market data feed, compliance engine, OMS, and client reporting platform.

## Corrections included

| Type | Description |
|---|---|
| `tool_selection` | `fetch_client_portfolio` vs `query_market_data` — client account vs market instrument disambiguation |
| `tool_selection` | `run_compliance_check` vs `calculate_risk_metrics` — permitted-check vs numeric computation |
| `tool_selection` | `execute_trade` vs `send_internal_alert` — OMS order vs internal notification |
| `tool_selection` | `generate_client_report` vs `file_regulatory_report` — client document vs regulatory filing |
| `arg_format` | ISO 8601 date enforcement for `file_regulatory_report`; lowercase enum values for `calculate_risk_metrics` |

## Tools covered

`fetch_client_portfolio`, `query_market_data`, `run_compliance_check`, `calculate_risk_metrics`,
`execute_trade`, `send_internal_alert`, `generate_client_report`, `file_regulatory_report`

## Usage

```python
from cannyforge import CannyForge

forge = CannyForge()
forge.import_skill("cannyforge/bundled_skills/fsi-workflow/bundle.cannyforge")
```

Imported corrections start with reset usage counters so they must earn their effectiveness in your
environment before they reach full Thompson-sampling weight.

## Benchmark

FSI-Bench-80 — 100 tasks across 4 confusion pairs, 3 difficulty tiers, 3 error modes
(tool_routing, format, context_miss). Run with:

```
python benchmark/bench_fsi80.py --ollama --model qwen2.5:3b --no-think
```
