import os
import re
import json
import pandas as pd
from pypdf import PdfReader

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def parse_contract(filepath):
    text = extract_pdf_text(filepath)
    cust_match = re.search(r"Customer ID:\s*(CUST-\d+)", text)
    discount_match = re.search(r"Agreed Discount Tier:\s*(\d+)%", text)
    rates = {
        m[0]: float(m[1])
        for m in re.findall(
            r"(Cloud_Compute_Hours|API_Gateway_Calls|Dedicated_Storage_GB|Premium_Support_SLA):\s*\$([\d.]+)",
            text
        )
    }
    return {
        "customer_id": cust_match.group(1) if cust_match else None,
        "discount_pct": int(discount_match.group(1)) if discount_match else 0,
        "contract_rates": rates
    }

def parse_invoice(filepath):
    text = extract_pdf_text(filepath)
    inv_match = re.search(r"Invoice Number:\s*(INV-[\d-]+)", text)
    cust_match = re.search(r"Customer ID:\s*(CUST-\d+)", text)
    total_match = re.search(r"TOTAL INVOICED AMOUNT:\s*\$([\d.]+)", text)

    matches = re.findall(
        r"(Cloud_Compute_Hours|API_Gateway_Calls|Dedicated_Storage_GB|Premium_Support_SLA)\s*\|\s*Qty:\s*(\d+)\s*\|\s*Rate:\s*\$([\d.]+)\s*\|\s*Total:\s*\$([\d.]+)",
        text
    )
    raw_line_items = [
        {"service": m[0], "qty": int(m[1]), "rate": float(m[2]), "total": float(m[3])}
        for m in matches
    ]
    line_items = {
        item["service"]: {"qty": item["qty"], "rate": item["rate"], "total": item["total"]}
        for item in raw_line_items
    }
    return {
        "invoice_number": inv_match.group(1) if inv_match else None,
        "customer_id": cust_match.group(1) if cust_match else None,
        "line_items": line_items,
        "raw_line_items": raw_line_items,
        "total_billed": float(total_match.group(1)) if total_match else 0.0
    }

def load_all_billing_data():
    with open("data/logs/usage_logs.json", "r") as f:
        usage_list = json.load(f)
    usage_map = {item["customer_id"]: item for item in usage_list}

    prov_df = pd.read_csv("data/logs/provisioning.csv")
    prov_map = prov_df.set_index("customer_id").to_dict(orient="index")

    merged_records = []
    for cust_id, usage in usage_map.items():
        contract_path = f"data/contracts/{cust_id}_contract.pdf"
        invoice_path = f"data/invoices/{cust_id}_invoice.pdf"

        if os.path.exists(contract_path) and os.path.exists(invoice_path):
            contract_data = parse_contract(contract_path)
            invoice_data = parse_invoice(invoice_path)
            prov_data = prov_map.get(cust_id, {})

            merged_records.append({
                "customer_id": cust_id,
                "billing_month": usage["billing_month"],
                "contract": contract_data,
                "invoice": invoice_data,
                "usage": usage,
                "provisioning": prov_data
            })
    return merged_records

if __name__ == "__main__":
    records = load_all_billing_data()
    print(f"Success! Extracted & merged {len(records)} complete customer billing profiles.")
    print("Sample Record (CUST-1001):", json.dumps(records[0], indent=2))