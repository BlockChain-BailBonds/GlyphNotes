# Standalone AI Bail Bonds & Recovery App (Prototype)

This directory contains a standalone, deterministic prototype that models the end-to-end bail-bonding lifecycle described in the product brief. The implementation is intentionally **backend-first** and focuses on the single-action trigger that kicks off case creation, asynchronous court/jail enrichment, compliance checks, advisory risk assessment, and contract generation.

## Quick start

```bash
python -m bailbonds_app.app
```

The script simulates a user pressing the **"I'm going to jail"** button and prints the immediate UI response. Background tasks then enrich the case asynchronously and write normalized data into the in-memory store.

## Key behaviors

- **Single explicit user action:** `press_im_going_to_jail_button` creates a provisional case and returns immediately.
- **Asynchronous enrichment:** court and jail adapters fetch public data and append immutable events.
- **Deterministic advisory AI:** a deterministic risk model provides advisory scores only.
- **Compliance-first:** jurisdictional checks gate the next steps without auto-approving or denying bonds.
- **Transparent contracts:** contract packet generation produces human-readable and on-chain hash records.
- **Recovery gating:** recovery actions are logged only when a court order is supplied.

## Modules

- `models.py`: dataclasses for cases, events, custody status, and contracts.
- `audit.py`: immutable audit log entries.
- `store.py`: normalized data store (append-only events, custody status, cases).
- `services.py`: case creation, adapters, compliance, risk, and contract services.
- `mcp_server.py`: deterministic MCP-like functions for compliance, risk, and data retrieval.
- `app.py`: orchestration and demo run.

## Tulsa data sources (referenced, not scraped)

The Tulsa adapter in this prototype tracks the public sources that would be queried in a production integration:

- OSCN case search: https://www.oscn.net/dockets/search.aspx
- Tulsa County inmate info: https://www2.tulsacounty.org/community/inmate-information/
- Tulsa Municipal Jail info: https://www.tulsapolice.org/tulsamunicipaljail
- Tulsa County Court Clerk: https://courtclerk.tulsacounty.org/

> Note: This prototype intentionally avoids external dependencies so it can run in any standard Python 3.11 environment.
