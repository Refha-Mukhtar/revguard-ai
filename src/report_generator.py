import os
import re
from datetime import datetime
from fpdf import FPDF

def sanitize_text(text):
    """Replaces non-latin1 characters so FPDF core fonts never throw encoding errors."""
    cleaned = re.sub(r"[#*`]", "", str(text))
    return cleaned.encode("latin-1", "replace").decode("latin-1")

def generate_audit_pdf(leak_record, ai_report):
    os.makedirs("data/reports", exist_ok=True)
    cust_id = leak_record["customer_id"]
    filepath = f"data/reports/{cust_id}_Audit_Recovery_Report.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header Banner
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "REVGUARD AI - REVENUE ASSURANCE AUDIT CERTIFICATE", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | Status: ACTION REQUIRED", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(12)

    # Discrepancy Summary Box
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Financial Discrepancy & Anomaly Telemetry", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    details = [
        f"Customer ID: {cust_id}  |  Invoice Number: {leak_record.get('invoice_number', 'N/A')}",
        f"Affected Service: {leak_record['service']}  |  Leak Category: {leak_record['leak_type']}",
        f"Expected Billed Amount: ${leak_record['expected_usd']:,.2f}  |  Actual Invoiced: ${leak_record['billed_usd']:,.2f}",
        f"Net Revenue Leakage Exposure: ${leak_record['leaked_usd']:,.2f}  |  Priority: {leak_record['priority']}",
        f"Detection Engine: {leak_record.get('detection_engine', 'Hybrid')} (ML Anomaly Score: {leak_record.get('ml_anomaly_score', 0)}%)",
        f"Audit Evidence: {leak_record['evidence']}"
    ]
    for line in details:
        pdf.set_x(10)
        pdf.multi_cell(190, 6, sanitize_text(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Multi-Agent RCA & Recovery Plan
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Autonomous Multi-Agent RCA & Recovery Action Plan", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(10)
    pdf.multi_cell(190, 6, sanitize_text(ai_report), new_x="LMARGIN", new_y="NEXT")

    pdf.output(filepath)
    return filepath

if __name__ == "__main__":
    sample_leak = {
        "customer_id": "CUST-1001",
        "invoice_number": "INV-202610-1",
        "service": "Cloud_Compute_Hours",
        "leak_type": "RATE_MISMATCH",
        "expected_usd": 1350.0,
        "billed_usd": 945.0,
        "leaked_usd": 405.0,
        "priority": "CRITICAL",
        "ml_anomaly_score": 92.4,
        "detection_engine": "Hybrid (Deterministic + IsolationForest)",
        "evidence": "Contract rate: $1.35, Billed rate: $0.945 | Actual usage: 1000, Billed qty: 1000"
    }
    out = generate_audit_pdf(sample_leak, "Verified rate mismatch of 30%. Issue credit/debit addendum and patch CRM sync.")
    print(f"Success! Generated PDF report at: {out}")