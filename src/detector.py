import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extractor import load_all_billing_data

METERED_SERVICES = ["Cloud_Compute_Hours", "API_Gateway_Calls", "Dedicated_Storage_GB"]

def assign_priority(amount):
    if amount >= 200.0:
        return "CRITICAL"
    elif amount >= 50.0:
        return "HIGH"
    return "MEDIUM"

def compute_ml_anomaly_scores(records):
    """Trains an Unsupervised Isolation Forest on usage-to-billing ratios to flag statistical outliers."""
    rows = []
    for rec in records:
        usage = rec["usage"]
        billed = rec["invoice"]["total_billed"]
        rates = rec["contract"]["contract_rates"]
        expected_total = (
            usage.get("Cloud_Compute_Hours", 0) * rates.get("Cloud_Compute_Hours", 0) +
            usage.get("API_Gateway_Calls", 0) * rates.get("API_Gateway_Calls", 0) +
            usage.get("Dedicated_Storage_GB", 0) * rates.get("Dedicated_Storage_GB", 0) +
            (rates.get("Premium_Support_SLA", 0) if rec["provisioning"].get("Premium_Support_SLA") == "ACTIVE" else 0.0)
        )
        billing_ratio = billed / expected_total if expected_total > 0 else 1.0
        rows.append({
            "customer_id": rec["customer_id"],
            "compute": usage.get("Cloud_Compute_Hours", 0),
            "api_calls": usage.get("API_Gateway_Calls", 0),
            "storage": usage.get("Dedicated_Storage_GB", 0),
            "billed": billed,
            "billing_ratio": billing_ratio
        })

    df = pd.DataFrame(rows)
    features = ["compute", "api_calls", "storage", "billed", "billing_ratio"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    iso = IsolationForest(n_estimators=150, contamination=0.35, random_state=42)
    preds = iso.fit_predict(X_scaled)
    raw_scores = -iso.decision_function(X_scaled)  # Higher = more anomalous

    # Normalize anomaly confidence to 0-100%
    min_s, max_s = raw_scores.min(), raw_scores.max()
    norm_scores = ((raw_scores - min_s) / (max_s - min_s + 1e-9)) * 100

    ml_map = {}
    for idx, row in df.iterrows():
        ml_map[row["customer_id"]] = {
            "ml_flagged": bool(preds[idx] == -1),
            "ml_anomaly_score": round(float(norm_scores[idx]), 1),
            "billing_ratio": round(float(row["billing_ratio"]), 3)
        }
    return ml_map

def detect_leakages(records):
    ml_scores = compute_ml_anomaly_scores(records)
    detected = []

    for rec in records:
        cust_id = rec["customer_id"]
        inv_num = rec["invoice"]["invoice_number"]
        rates = rec["contract"]["contract_rates"]
        line_items = rec["invoice"]["line_items"]
        usage = rec["usage"]
        prov = rec["provisioning"]
        ml_info = ml_scores.get(cust_id, {"ml_flagged": False, "ml_anomaly_score": 0.0, "billing_ratio": 1.0})

        # 0. Duplicate Entries Detection
        raw_items = rec["invoice"].get("raw_line_items", [])
        srv_counts = {}
        for it in raw_items:
            s_name = it["service"]
            srv_counts[s_name] = srv_counts.get(s_name, 0) + 1

        duplicate_services = set()
        for s_name, count in srv_counts.items():
            if count > 1:
                duplicate_services.add(s_name)
                dup_entries = [it for it in raw_items if it["service"] == s_name]
                dup_charge = round(sum(it["total"] for it in dup_entries[1:]), 2)
                detected.append({
                    "customer_id": cust_id,
                    "invoice_number": inv_num,
                    "service": s_name,
                    "leak_type": "DUPLICATE_ENTRY",
                    "expected_usd": dup_entries[0]["total"],
                    "billed_usd": round(sum(it["total"] for it in dup_entries), 2),
                    "leaked_usd": dup_charge,
                    "priority": assign_priority(dup_charge),
                    "ml_anomaly_score": ml_info["ml_anomaly_score"],
                    "ml_flagged": ml_info["ml_flagged"],
                    "detection_engine": "Hybrid (Deterministic + IsolationForest)" if ml_info["ml_flagged"] else "Deterministic Rule Engine",
                    "evidence": f"Duplicate line item: service '{s_name}' billed {count} times on invoice {inv_num}. Redundant billing: ${dup_charge}"
                })

        # 1. Metered Services Reconciliation
        for srv in METERED_SERVICES:
            if srv in duplicate_services:
                continue
            actual_qty = usage.get(srv, 0)
            contract_rate = rates.get(srv, 0.0)
            expected_total = round(actual_qty * contract_rate, 2)

            inv_item = line_items.get(srv, {"qty": 0, "rate": 0.0, "total": 0.0})
            billed_qty = inv_item["qty"]
            billed_rate = inv_item["rate"]
            billed_total = inv_item["total"]

            diff = round(expected_total - billed_total, 2)
            if diff > 0.01:
                leak_type = "RATE_MISMATCH" if billed_rate < contract_rate else "MISSING_CHARGE"
                detected.append({
                    "customer_id": cust_id,
                    "invoice_number": inv_num,
                    "service": srv,
                    "leak_type": leak_type,
                    "expected_usd": expected_total,
                    "billed_usd": billed_total,
                    "leaked_usd": diff,
                    "priority": assign_priority(diff),
                    "ml_anomaly_score": ml_info["ml_anomaly_score"],
                    "ml_flagged": ml_info["ml_flagged"],
                    "detection_engine": "Hybrid (Deterministic + IsolationForest)" if ml_info["ml_flagged"] else "Deterministic Rule Engine",
                    "evidence": (
                        f"Contract rate: ${contract_rate}, Billed rate: ${billed_rate} | "
                        f"Actual usage: {actual_qty}, Billed qty: {billed_qty}"
                    )
                })

        # 2. Provisioned-but-Unbilled SLA Check
        if prov.get("Premium_Support_SLA") == "ACTIVE":
            contract_rate = rates.get("Premium_Support_SLA", 0.0)
            inv_item = line_items.get("Premium_Support_SLA")
            billed_total = inv_item["total"] if inv_item else 0.0
            diff = round(contract_rate - billed_total, 2)

            if diff > 0.01:
                detected.append({
                    "customer_id": cust_id,
                    "invoice_number": inv_num,
                    "service": "Premium_Support_SLA",
                    "leak_type": "UNBILLED_SLA",
                    "expected_usd": contract_rate,
                    "billed_usd": billed_total,
                    "leaked_usd": diff,
                    "priority": assign_priority(diff),
                    "ml_anomaly_score": ml_info["ml_anomaly_score"],
                    "ml_flagged": ml_info["ml_flagged"],
                    "detection_engine": "Hybrid (Deterministic + IsolationForest)" if ml_info["ml_flagged"] else "Deterministic Rule Engine",
                    "evidence": "SLA status is ACTIVE in provisioning.csv, but omitted on invoice."
                })

    detected.sort(key=lambda x: x["leaked_usd"], reverse=True)
    return detected

def evaluate_and_save(detected):
    os.makedirs("data/logs", exist_ok=True)
    with open("data/logs/detected_leaks.json", "w") as f:
        json.dump(detected, f, indent=2)

    with open("data/logs/ground_truth_leaks.json", "r") as f:
        gt = json.load(f)

    gt_customers = {item["customer_id"] for item in gt}
    det_customers = {item["customer_id"] for item in detected}
    ml_customers = {item["customer_id"] for item in detected if item.get("ml_flagged")}

    tp = len(gt_customers & det_customers)
    fp = len(det_customers - gt_customers)
    fn = len(gt_customers - det_customers)

    precision = round((tp / (tp + fp)) * 100, 2) if (tp + fp) else 0.0
    recall = round((tp / (tp + fn)) * 100, 2) if (tp + fn) else 0.0
    f1 = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) else 0.0
    total_leaked = round(sum(item["leaked_usd"] for item in detected), 2)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ml_corroborated_leaks": len(ml_customers),
        "total_leaked_usd": total_leaked
    }
    with open("data/logs/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Success! Detected {len(detected)} leaks (${total_leaked}) | ML Corroborated: {len(ml_customers)} accounts")
    print(f"Metrics -> Precision: {precision}% | Recall: {recall}% | F1-Score: {f1}%")
    return metrics

if __name__ == "__main__":
    records = load_all_billing_data()
    leaks = detect_leakages(records)
    evaluate_and_save(leaks)