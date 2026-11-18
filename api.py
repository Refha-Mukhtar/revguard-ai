import os
import sys
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_generator import generate_dataset
from extractor import load_all_billing_data
from detector import detect_leakages, evaluate_and_save
from agents import investigate_leak
from report_generator import generate_audit_pdf
from database import init_db, save_audit_run, get_all_leaks, get_leak_by_id, mark_recovered, get_audit_runs, get_kpi_summary
from contract_vector_search import search_contract_clauses, index_contracts

# ── Bootstrap DB on startup ──────────────────────────────────────────────────
init_db()

app = FastAPI(
    title="RevGuard AI — Revenue Assurance & Agentic RCA API",
    description=(
        "Enterprise REST microservice for multi-modal contract/invoice reconciliation, "
        "Isolation Forest anomaly detection, CrewAI root-cause ticket generation, "
        "and SQLite-persisted audit ledger with recovery workflow."
    ),
    version="3.0.0"
)


# ── Request / Response Models ────────────────────────────────────────────────
class AuditRequest(BaseModel):
    num_customers: int = 20


class RecoverRequest(BaseModel):
    jira_ticket: str = ""


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "service": "RevGuard AI Engine",
        "version": "3.0.0"
    }


# ── Audit ────────────────────────────────────────────────────────────────────
@app.post("/audit/run", tags=["Audit"])
def run_full_audit(req: AuditRequest):
    """
    Generate synthetic billing data, run Hybrid ML + Deterministic detection,
    persist results to SQLite, and return evaluation metrics + top leaks.
    """
    generate_dataset(num_customers=req.num_customers)
    records = load_all_billing_data()
    leaks = detect_leakages(records)
    metrics = evaluate_and_save(leaks)
    run_id = save_audit_run(metrics, leaks)       # ← persist to SQLite
    return {
        "message": f"Audited {req.num_customers} enterprise accounts successfully.",
        "run_id": run_id,
        "evaluation_metrics": metrics,
        "leaks_detected": len(leaks),
        "top_leaks": leaks[:5]
    }


@app.get("/audit/runs", tags=["Audit"])
def list_audit_runs():
    """Return all historical audit runs from SQLite, newest first."""
    return get_audit_runs()


@app.get("/audit/kpi", tags=["Audit"])
def kpi_summary():
    """Return live KPI summary: open vs. recovered leakage USD and counts."""
    return get_kpi_summary()


# ── Leaks ────────────────────────────────────────────────────────────────────
@app.get("/leaks", tags=["Leaks"])
def get_detected_leaks(resolution: str = None):
    """
    Return all detected leaks from SQLite.
    Optional query param: ?resolution=OPEN or ?resolution=RECOVERED
    Falls back to flat JSON file if DB is empty (backward-compat).
    """
    db_leaks = get_all_leaks(resolution_filter=resolution)
    if db_leaks:
        return db_leaks

    # Backward-compat fallback for pre-DB data
    if not os.path.exists("data/logs/detected_leaks.json"):
        raise HTTPException(
            status_code=404,
            detail="No audit results found. Run POST /audit/run first."
        )
    with open("data/logs/detected_leaks.json", "r") as f:
        return json.load(f)


@app.get("/leaks/{leak_id}", tags=["Leaks"])
def get_single_leak(leak_id: int):
    """Return a single leak record by its SQLite ID."""
    leak = get_leak_by_id(leak_id)
    if not leak:
        raise HTTPException(status_code=404, detail=f"Leak ID {leak_id} not found.")
    return leak


# ── Recovery ─────────────────────────────────────────────────────────────────
@app.patch("/leaks/{leak_id}/recover", tags=["Recovery"])
def recover_leak(leak_id: int, req: RecoverRequest):
    """
    Mark a detected leak as RECOVERED in SQLite and optionally attach a Jira ticket reference.
    Returns updated KPI summary after resolution.
    """
    leak = get_leak_by_id(leak_id)
    if not leak:
        raise HTTPException(status_code=404, detail=f"Leak ID {leak_id} not found.")
    if leak["resolution"] == "RECOVERED":
        raise HTTPException(status_code=409, detail=f"Leak ID {leak_id} is already RECOVERED.")

    success = mark_recovered(leak_id, jira_ticket=req.jira_ticket)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update resolution status.")

    return {
        "message": f"Leak ID {leak_id} marked as RECOVERED.",
        "jira_ticket": req.jira_ticket or "N/A",
        "updated_kpi": get_kpi_summary()
    }


# ── Investigation ─────────────────────────────────────────────────────────────
@app.post("/investigate/{customer_id}", tags=["Investigation"])
def investigate_customer_leak(customer_id: str):
    """
    Run CrewAI multi-agent RCA on the highest-value OPEN leak for a given customer.
    Generates and returns an audit certificate PDF path.
    """
    # Try DB first
    db_leaks = get_all_leaks(resolution_filter="OPEN")
    target = next(
        (item for item in db_leaks if item["customer_id"] == customer_id),
        None
    )

    # Fallback to flat JSON
    if not target:
        if not os.path.exists("data/logs/detected_leaks.json"):
            raise HTTPException(
                status_code=404,
                detail="No audit results found. Run POST /audit/run first."
            )
        with open("data/logs/detected_leaks.json", "r") as f:
            leaks = json.load(f)
        target = next((item for item in leaks if item["customer_id"] == customer_id), None)

    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"No OPEN revenue leak recorded for {customer_id}."
        )

    ai_report = investigate_leak(target)
    pdf_path = generate_audit_pdf(target, ai_report)
    return {
        "customer_id": customer_id,
        "leak_summary": target,
        "ai_investigation_report": ai_report,
        "audit_certificate_pdf": pdf_path
    }


# ── Vector Search ────────────────────────────────────────────────────────────
@app.get("/contracts/search", tags=["Vector Search"])
def search_contracts(q: str, limit: int = 4):
    """Semantic vector search across enterprise Master Service Agreements (MSAs) using ChromaDB."""
    return {
        "query": q,
        "results": search_contract_clauses(query=q, n_results=limit)
    }
