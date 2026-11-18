import os
import json
import random
import pandas as pd
from fpdf import FPDF

os.makedirs("data/contracts", exist_ok=True)
os.makedirs("data/invoices", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

SERVICES = {
    "Cloud_Compute_Hours": 1.50,
    "API_Gateway_Calls": 0.02,
    "Dedicated_Storage_GB": 0.15,
    "Premium_Support_SLA": 500.00
}

def create_pdf(filepath, title, lines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    for line in lines:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(filepath)

def generate_dataset(num_customers=20):
    usage_records = []
    provisioning_records = []
    ground_truth_leaks = []

    for i in range(1, num_customers + 1):
        cust_id = f"CUST-{1000 + i}"
        cust_name = f"Enterprise_Client_{i}"

        discount = random.choice([0.0, 0.10])
        contract_rates = {k: round(v * (1 - discount), 4) for k, v in SERVICES.items()}

        contract_lines = [
            f"Contract ID: CNT-2026-{i}",
            f"Customer ID: {cust_id} | Client: {cust_name}",
            f"Agreed Discount Tier: {int(discount*100)}%",
            "--- Agreed Unit Rates (USD) ---"
        ] + [f"{srv}: ${rate}" for srv, rate in contract_rates.items()]
        create_pdf(f"data/contracts/{cust_id}_contract.pdf", "MASTER SERVICE AGREEMENT", contract_lines)

        compute_hrs = random.randint(100, 1000)
        api_calls = random.randint(5000, 50000)
        storage_gb = random.randint(200, 2000)
        has_support = random.choice([True, False])

        usage_records.append({
            "customer_id": cust_id,
            "billing_month": "2026-10",
            "Cloud_Compute_Hours": compute_hrs,
            "API_Gateway_Calls": api_calls,
            "Dedicated_Storage_GB": storage_gb
        })
        provisioning_records.append({
            "customer_id": cust_id,
            "Premium_Support_SLA": "ACTIVE" if has_support else "INACTIVE",
            "status": "VERIFIED"
        })

        leak_type = random.choice(["NONE", "NONE", "NONE", "MISSING_CHARGE", "RATE_MISMATCH", "UNBILLED_SLA", "DUPLICATE_ENTRY"])

        billed_compute_rate = contract_rates["Cloud_Compute_Hours"]
        billed_api_calls = api_calls
        billed_support = contract_rates["Premium_Support_SLA"] if has_support else 0.0
        leak_amount = 0.0
        extra_duplicate_item = None

        if leak_type == "RATE_MISMATCH":
            wrong_rate = round(billed_compute_rate * 0.7, 4)
            leak_amount = round((billed_compute_rate - wrong_rate) * compute_hrs, 2)
            billed_compute_rate = wrong_rate
        elif leak_type == "MISSING_CHARGE":
            missed_calls = int(api_calls * 0.4)
            billed_api_calls = api_calls - missed_calls
            leak_amount = round(missed_calls * contract_rates["API_Gateway_Calls"], 2)
        elif leak_type == "UNBILLED_SLA" and has_support:
            leak_amount = billed_support
            billed_support = 0.0
        elif leak_type == "DUPLICATE_ENTRY":
            # Inadvertently billed an extra duplicate line item for Dedicated_Storage_GB
            dup_qty = storage_gb
            dup_rate = contract_rates["Dedicated_Storage_GB"]
            leak_amount = round(dup_qty * dup_rate, 2)
            extra_duplicate_item = f"Dedicated_Storage_GB | Qty: {dup_qty} | Rate: ${dup_rate} | Total: ${leak_amount}"
        else:
            leak_type = "NONE"

        line_items = [
            f"Cloud_Compute_Hours | Qty: {compute_hrs} | Rate: ${billed_compute_rate} | Total: ${round(compute_hrs * billed_compute_rate, 2)}",
            f"API_Gateway_Calls | Qty: {billed_api_calls} | Rate: ${contract_rates['API_Gateway_Calls']} | Total: ${round(billed_api_calls * contract_rates['API_Gateway_Calls'], 2)}",
            f"Dedicated_Storage_GB | Qty: {storage_gb} | Rate: ${contract_rates['Dedicated_Storage_GB']} | Total: ${round(storage_gb * contract_rates['Dedicated_Storage_GB'], 2)}",
        ]
        if extra_duplicate_item:
            line_items.append(extra_duplicate_item)
        if billed_support > 0:
            line_items.append(f"Premium_Support_SLA | Qty: 1 | Rate: ${billed_support} | Total: ${billed_support}")

        total_billed = round(
            compute_hrs * billed_compute_rate +
            billed_api_calls * contract_rates["API_Gateway_Calls"] +
            storage_gb * contract_rates["Dedicated_Storage_GB"] +
            (leak_amount if leak_type == "DUPLICATE_ENTRY" else 0.0) +
            billed_support, 2
        )

        invoice_lines = [
            f"Invoice Number: INV-202610-{i}",
            f"Customer ID: {cust_id} | Billing Period: 2026-10",
            "--- Line Items ---"
        ] + line_items + [f"TOTAL INVOICED AMOUNT: ${total_billed}"]

        create_pdf(f"data/invoices/{cust_id}_invoice.pdf", "TAX INVOICE", invoice_lines)

        if leak_type != "NONE" and leak_amount > 0:
            ground_truth_leaks.append({
                "customer_id": cust_id,
                "leak_type": leak_type,
                "leaked_usd": leak_amount
            })

    with open("data/logs/usage_logs.json", "w") as f:
        json.dump(usage_records, f, indent=2)
    pd.DataFrame(provisioning_records).to_csv("data/logs/provisioning.csv", index=False)
    with open("data/logs/ground_truth_leaks.json", "w") as f:
        json.dump(ground_truth_leaks, f, indent=2)

    print(f"Success! Generated 20 customer records with {len(ground_truth_leaks)} injected revenue leaks.")

if __name__ == "__main__":
    generate_dataset()