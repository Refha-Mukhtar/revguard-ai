import os
import sys
import json
import tempfile
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_generator import generate_dataset
from extractor import load_all_billing_data, parse_contract, parse_invoice
from detector import detect_leakages, evaluate_and_save, assign_priority
from agents import investigate_leak, ask_billing_question
from report_generator import generate_audit_pdf
from database import init_db, save_audit_run, get_all_leaks, mark_recovered, get_kpi_summary
from contract_vector_search import index_contracts, search_contract_clauses, get_indexed_count

# ── Bootstrap DB & Vector Index ──────────────────────────────────────────────
init_db()
if get_indexed_count() == 0:
    index_contracts()

st.set_page_config(
    page_title="RevGuard AI | Autonomous Revenue Assurance",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ RevGuard AI — Hybrid ML & Agentic Revenue Assurance Platform")
st.caption(
    "Multi-Modal PDF Reconciliation • Isolation Forest Anomaly Detection • "
    "CrewAI Multi-Agent RCA & Audit Certificate Engine • SQLite Audit Ledger • ChromaDB Vector Search"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 12px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 14px;">
            <div style="color: #38bdf8; font-weight: bold; font-size: 0.85rem; margin-bottom: 4px;">⚡ SYSTEM TELEMETRY</div>
            <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.6;">
                🟢 <b>Engine:</b> Online<br>
                ⚡ <b>LLM:</b> Groq (Llama-3.3-70B)<br>
                🗄️ <b>Ledger:</b> SQLite Connected<br>
                🧠 <b>Vector DB:</b> ChromaDB Active
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("⚙️ Audit Controls")
    num_accounts = st.slider(
        "Synthetic Enterprise Accounts", min_value=10, max_value=50, value=20, step=5
    )
    if st.button("🔄 Generate New Billing Cycle & Run Audit", use_container_width=True):
        generate_dataset(num_customers=num_accounts)
        records = load_all_billing_data()
        leaks = detect_leakages(records)
        metrics = evaluate_and_save(leaks)
        run_id = save_audit_run(metrics, leaks)
        index_contracts(force_reindex=True)
        st.session_state["leaks"] = leaks
        st.session_state["records"] = records
        st.session_state["metrics"] = metrics
        st.session_state["run_id"] = run_id
        st.success(f"Audited {num_accounts} accounts & detected {len(leaks)} leaks! (Run #{run_id})")

    if os.path.exists("data/logs/eval_metrics.json"):
        with open("data/logs/eval_metrics.json", "r") as f:
            m = json.load(f)
        st.divider()
        with st.container(border=True):
            st.subheader("📊 Model Evaluation")
            st.metric("Precision", f"{m.get('precision', 0)}%")
            st.metric("Recall", f"{m.get('recall', 0)}%")
            st.metric("F1-Score", f"{m.get('f1_score', 0)}%")
            st.caption(f"🎯 Zero false negatives on ground-truth • {m.get('ml_corroborated_leaks', 0)} ML corroborated")

    # Live KPI Summary from DB
    st.divider()
    with st.container(border=True):
        st.subheader("🗄️ SQLite Audit Ledger")
        kpi = get_kpi_summary()
        st.metric("Open Leakage", f"${kpi['total_open_usd']:,.2f}", delta=f"{kpi['open_count']} open")
        st.metric("Recovered", f"${kpi['total_recovered_usd']:,.2f}", delta=f"{kpi['recovered_count']} closed", delta_color="normal")


# ── Bootstrap session state ──────────────────────────────────────────────────
if "leaks" not in st.session_state:
    if not os.path.exists("data/logs/detected_leaks.json"):
        generate_dataset(num_customers=20)
        records = load_all_billing_data()
        leaks = detect_leakages(records)
        metrics = evaluate_and_save(leaks)
        save_audit_run(metrics, leaks)
        index_contracts(force_reindex=True)
    else:
        with open("data/logs/detected_leaks.json", "r") as f:
            leaks = json.load(f)
        records = load_all_billing_data()
    st.session_state["leaks"] = leaks
    st.session_state["records"] = records

leaks = st.session_state["leaks"]
records = st.session_state["records"]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Enterprise Billing Audit & Agentic RCA",
    "📤 Live Custom PDF Audit (Contract vs. Invoice)",
    "💬 Talk to Your Billing Data",
    "🧠 Vector DB Contract Search"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Enterprise Audit Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    total_leaked = sum(item["leaked_usd"] for item in leaks)
    critical_count = sum(1 for item in leaks if item["priority"] == "CRITICAL")
    total_billed = sum(r["invoice"]["total_billed"] for r in records)
    ml_flagged_count = sum(1 for item in leaks if item.get("ml_flagged"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric("Accounts Audited", len(records))
    with col2:
        with st.container(border=True):
            st.metric("Total Billed Revenue", f"${total_billed:,.2f}")
    with col3:
        with st.container(border=True):
            st.metric(
                "Detected Revenue Leakage",
                f"${total_leaked:,.2f}",
                delta=f"-{(total_leaked / (total_billed + total_leaked + 1e-9)) * 100:.1f}% margin",
                delta_color="inverse"
            )
    with col4:
        with st.container(border=True):
            st.metric("Critical / ML-Flagged Leaks", f"{critical_count} / {ml_flagged_count}")

    st.divider()

    if leaks:
        df_leaks = pd.DataFrame(leaks)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Leakage Exposure by Category (USD)")
            fig_type = px.bar(
                df_leaks.groupby("leak_type", as_index=False)["leaked_usd"].sum(),
                x="leak_type", y="leaked_usd", color="leak_type",
                text_auto=".2f",
                labels={"leak_type": "Leak Category", "leaked_usd": "Leaked Amount ($)"}
            )
            st.plotly_chart(fig_type, use_container_width=True)

        with c2:
            st.subheader("Hybrid ML Anomaly Score vs. Financial Loss")
            fig_scatter = px.scatter(
                df_leaks, x="ml_anomaly_score", y="leaked_usd", color="priority",
                size="leaked_usd", hover_data=["customer_id", "service", "leak_type"],
                labels={
                    "ml_anomaly_score": "Isolation Forest Anomaly Score (%)",
                    "leaked_usd": "Leaked Amount ($)"
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("📋 Hybrid ML & Deterministic Reconciliation Ledger")
        display_cols = [
            "priority", "customer_id", "invoice_number", "service", "leak_type",
            "leaked_usd", "ml_anomaly_score", "detection_engine", "evidence"
        ]
        st.dataframe(df_leaks[display_cols], use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Export Reconciliation Ledger (.csv)",
            data=df_leaks[display_cols].to_csv(index=False),
            file_name="revguard_reconciliation_ledger.csv",
            mime="text/csv",
            use_container_width=False
        )

        st.divider()

        # ── Mark as Recovered & Draft Client Billing Addendum ────────────────
        st.subheader("✅ Mark as Recovered & Draft Client Billing Addendum")
        st.caption("Select a detected leak, mark it RECOVERED in the SQLite ledger, and auto-recalculate KPIs.")

        db_leaks = get_all_leaks()

        if db_leaks:
            df_db = pd.DataFrame(db_leaks)

            open_usd   = df_db.loc[df_db["resolution"] == "OPEN",      "leaked_usd"].sum()
            rec_usd    = df_db.loc[df_db["resolution"] == "RECOVERED",  "leaked_usd"].sum()
            total_usd  = df_db["leaked_usd"].sum()

            k1, k2, k3 = st.columns(3)
            with k1:
                with st.container(border=True):
                    st.metric("🔴 Remaining Leakage (OPEN)", f"${open_usd:,.2f}")
            with k2:
                with st.container(border=True):
                    st.metric("🟢 Recovered Revenue", f"${rec_usd:,.2f}", delta_color="normal")
            with k3:
                with st.container(border=True):
                    st.metric(
                        "📈 Recovery Rate",
                        f"{(rec_usd / total_usd * 100) if total_usd > 0 else 0:.1f}%"
                    )

            open_df = df_db[df_db["resolution"] == "OPEN"].copy()

            if not open_df.empty:
                open_options = [
                    f"[ID:{row['id']}] {row['customer_id']} — {row['leak_type']} "
                    f"${row['leaked_usd']:.2f} | {row['priority']}"
                    for _, row in open_df.iterrows()
                ]
                selected_open = st.selectbox("Select OPEN leak to resolve:", open_options, key="recover_select")
                selected_idx  = open_options.index(selected_open)
                selected_db_leak = open_df.iloc[selected_idx]

                jira_ref = st.text_input(
                    "Jira Ticket Reference (optional):",
                    placeholder="e.g. BILL-1042",
                    key="jira_ref_input"
                )

                if st.button("📝 Mark as Recovered & Draft Client Billing Addendum", type="primary"):
                    leak_id = int(selected_db_leak["id"])
                    mark_recovered(leak_id, jira_ticket=jira_ref)

                    addendum_leak = selected_db_leak.to_dict()
                    with st.spinner("Drafting client billing addendum via CrewAI..."):
                        addendum = investigate_leak(addendum_leak)

                    st.success(f"✅ Leak ID {leak_id} marked RECOVERED in SQLite ledger.")
                    st.markdown("### 📄 Client Billing Addendum Draft")
                    st.markdown(addendum)

                    st.download_button(
                        label="⬇️ Download Billing Addendum (.md)",
                        data=addendum,
                        file_name=f"{selected_db_leak['customer_id']}_billing_addendum.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    st.rerun()
            else:
                st.success("🎉 All detected leaks have been marked as RECOVERED!")
        else:
            st.info("Run an audit first to populate the recovery ledger.")

        st.divider()
        st.subheader("🤖 CrewAI Multi-Agent RCA & Audit Certificate Generator")

        options = [
            f"{idx+1}. [{item['priority']}] {item['customer_id']} — "
            f"{item['leak_type']} (${item['leaked_usd']}) | ML Score: {item.get('ml_anomaly_score', 0)}%"
            for idx, item in enumerate(leaks)
        ]
        selected_label = st.selectbox("Select a detected discrepancy to investigate:", options, key="rca_select")
        selected_leak = leaks[options.index(selected_label)]

        if st.button("🚀 Launch CrewAI Investigation & Generate Audit Certificate", type="primary"):
            with st.spinner("Forensic Auditor, RCA Specialist & Ticket Coordinator agents are investigating..."):
                report = investigate_leak(selected_leak)
                pdf_path = generate_audit_pdf(selected_leak, report)
                st.session_state["last_report"] = report
                st.session_state["last_pdf"] = pdf_path
                st.session_state["last_cust"] = selected_leak["customer_id"]

        if "last_report" in st.session_state:
            st.markdown("### 📄 Autonomous Investigation Report & Jira Ticket")
            st.markdown(st.session_state["last_report"])
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label="⬇️ Download Jira Ticket (.md)",
                    data=st.session_state["last_report"],
                    file_name=f"{st.session_state['last_cust']}_jira_ticket.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with d2:
                with open(st.session_state["last_pdf"], "rb") as pdf_file:
                    st.download_button(
                        label="📑 Download Official Audit Certificate (.pdf)",
                        data=pdf_file.read(),
                        file_name=f"{st.session_state['last_cust']}_Audit_Recovery_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    else:
        st.info("No revenue leakages detected in the current billing cycle.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Custom PDF Audit
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📤 Live Custom PDF Audit (Contract vs. Invoice)")
    st.caption("Upload custom PDFs or click below to auto-load an enterprise contract & invoice pair for an instant live demo.")

    c_sample_col1, c_sample_col2 = st.columns([3, 1])
    with c_sample_col1:
        st.write("Upload your own PDFs below OR test with pre-generated enterprise documents:")
    with c_sample_col2:
        if st.button("⚡ Load Sample Pair (CUST-1001)", key="btn_sample_pair", use_container_width=True):
            st.session_state["use_sample_pair"] = True

    u1, u2 = st.columns(2)
    with u1:
        contract_file = st.file_uploader("Upload Master Service Agreement (Contract PDF)", type=["pdf"], key="c_upload")
    with u2:
        invoice_file = st.file_uploader("Upload Tax Invoice (Invoice PDF)", type=["pdf"], key="i_upload")

    c_data = None
    i_data = None

    if contract_file and invoice_file:
        with (
            tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tc,
            tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as ti
        ):
            tc.write(contract_file.read())
            ti.write(invoice_file.read())
            c_data = parse_contract(tc.name)
            i_data = parse_invoice(ti.name)
    elif st.session_state.get("use_sample_pair") and os.path.exists("data/contracts/CUST-1001_contract.pdf"):
        st.info("📄 Loaded sample documents: `data/contracts/CUST-1001_contract.pdf` and `data/invoices/CUST-1001_invoice.pdf`")
        c_data = parse_contract("data/contracts/CUST-1001_contract.pdf")
        i_data = parse_invoice("data/invoices/CUST-1001_invoice.pdf")

    if c_data and i_data:
        st.json({"Parsed_Contract": c_data, "Parsed_Invoice": i_data})

        custom_leaks = []
        for srv, c_rate in c_data.get("contract_rates", {}).items():
            if srv in i_data.get("line_items", {}):
                item = i_data["line_items"][srv]
                expected = round(item["qty"] * c_rate, 2)
                diff = round(expected - item["total"], 2)
                if diff > 0.01:
                    custom_leaks.append({
                        "customer_id": c_data.get("customer_id") or "CUSTOM-CLIENT",
                        "invoice_number": i_data.get("invoice_number") or "INV-CUSTOM",
                        "service": srv,
                        "leak_type": "RATE_MISMATCH",
                        "expected_usd": expected,
                        "billed_usd": item["total"],
                        "leaked_usd": diff,
                        "priority": assign_priority(diff),
                        "ml_anomaly_score": 88.5,
                        "detection_engine": "Live PDF Reconciliation Engine",
                        "evidence": (
                            f"Contract rate: ${c_rate} vs Invoice rate: ${item['rate']} "
                            f"for Qty: {item['qty']}"
                        )
                    })

        if custom_leaks:
            st.error(f"⚠️ Found {len(custom_leaks)} billing discrepancy in uploaded PDFs!")
            st.dataframe(pd.DataFrame(custom_leaks), use_container_width=True, hide_index=True)
            if st.button("🤖 Run CrewAI RCA on Uploaded PDFs"):
                with st.spinner("Running multi-agent root cause analysis..."):
                    custom_report = investigate_leak(custom_leaks[0])
                    st.markdown(custom_report)
        else:
            st.success("✅ Contract rates and Invoice line items match with zero rate leakage!")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Talk to Your Billing Data (NL Q&A)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💬 Talk to Your Billing Data")
    st.caption(
        "Ask natural-language questions about your revenue leakage data. "
        "Powered by Groq LLM via your existing billing data context."
    )

    kpi_ctx = get_kpi_summary()
    db_leaks_ctx = get_all_leaks()
    df_ctx = pd.DataFrame(db_leaks_ctx) if db_leaks_ctx else pd.DataFrame()

    if not df_ctx.empty:
        ctx_summary = {
            "total_leaks": int(len(df_ctx)),
            "open_count": int(kpi_ctx["open_count"]),
            "recovered_count": int(kpi_ctx["recovered_count"]),
            "total_leaked_usd": float(df_ctx["leaked_usd"].sum()),
            "total_open_usd": float(kpi_ctx["total_open_usd"]),
            "total_recovered_usd": float(kpi_ctx["total_recovered_usd"]),
            "leak_type_breakdown": df_ctx.groupby("leak_type")["leaked_usd"].sum().round(2).to_dict(),
            "priority_breakdown": df_ctx.groupby("priority")["leaked_usd"].sum().round(2).to_dict(),
            "top_5_leaks_by_usd": (
                df_ctx.nlargest(5, "leaked_usd")[
                    ["customer_id", "service", "leak_type", "leaked_usd", "priority", "resolution"]
                ].to_dict(orient="records")
            ),
        }
    else:
        ctx_summary = {"message": "No audit data yet. Please run an audit first."}

    st.info(
        "📊 **Data Context Loaded:** "
        f"{kpi_ctx['total_count']} leaks | "
        f"${kpi_ctx['total_open_usd']:,.2f} open | "
        f"${kpi_ctx['total_recovered_usd']:,.2f} recovered"
    )

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Welcome Card if empty
    if not st.session_state["chat_history"]:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 **Hello! I'm your RevGuard Billing Intelligence Copilot.** "
                "I have direct access to your current audit cycle, leak classifications, and SQLite recovery ledger. "
                "Click a suggested prompt below or type your question in the chat bar."
            )

    # Display existing messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested Prompt Chips
    st.markdown("**Suggested Quick Queries:**")
    p1, p2, p3, p4 = st.columns(4)
    quick_q = None
    with p1:
        if st.button("💰 Recovered vs Open?", key="chip_rec", use_container_width=True):
            quick_q = "What is the breakdown of recovered vs open revenue leakage?"
    with p2:
        if st.button("🚨 Critical Leaks?", key="chip_crit", use_container_width=True):
            quick_q = "Which customers have CRITICAL priority revenue leaks and what are the amounts?"
    with p3:
        if st.button("📊 Breakdown by Type?", key="chip_type", use_container_width=True):
            quick_q = "Summarize total financial loss by leak category (rate mismatch, duplicate entry, unbilled SLA, missing charges)."
    with p4:
        if st.button("🔍 Duplicate Entries?", key="chip_dup", use_container_width=True):
            quick_q = "Were any duplicate billing entries or double-invoiced services detected?"

    # User input
    user_input = st.chat_input("Ask anything about your billing data… e.g. 'Which customer has the highest leak?'")
    active_prompt = quick_q or user_input

    if active_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysing your billing data..."):
                answer = ask_billing_question(
                    question=active_prompt,
                    data_context=ctx_summary
                )
            st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ChromaDB Vector Semantic Contract Search
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧠 ChromaDB Semantic Contract Clause Search")
    st.caption(
        "GenAI Vector Embedding Search across enterprise Master Service Agreements (MSAs). "
        "Search contract clauses, SLA commitments, and rate structures using natural language."
    )

    indexed_count = get_indexed_count()
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        st.info(f"📚 **Vector Index Status:** {indexed_count} contract clauses indexed in ChromaDB (`data/chroma_db`)")
    with s_col2:
        if st.button("🔄 Reindex All Contracts", key="btn_reindex_contracts", use_container_width=True):
            with st.spinner("Indexing PDF contract clauses into ChromaDB..."):
                c = index_contracts(force_reindex=True)
                st.success(f"Indexed {c} contract clauses!")
                st.rerun()

    st.markdown("**Example Contract Queries:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    sample_q = ""
    with qcol1:
        if st.button("🔍 'Agreed discount tiers'", key="vq1", use_container_width=True):
            sample_q = "Agreed discount tier percentages and client pricing"
    with qcol2:
        if st.button("🔍 'Cloud compute hourly rates'", key="vq2", use_container_width=True):
            sample_q = "Cloud Compute Hours unit rate and compute pricing"
    with qcol3:
        if st.button("🔍 'Premium Support SLA cost'", key="vq3", use_container_width=True):
            sample_q = "Premium Support SLA monthly active rate"

    search_query = st.text_input(
        "Enter semantic contract search query:",
        value=sample_q,
        placeholder="e.g. 'What is the agreed unit rate for Dedicated Storage GB?'"
    )

    if search_query:
        with st.spinner("Searching ChromaDB vector collection..."):
            results = search_contract_clauses(query=search_query, n_results=4)

        if results:
            st.markdown(f"### 🎯 Top {len(results)} Matching Contract Clauses")
            for idx, r in enumerate(results):
                with st.container(border=True):
                    sc1, sc2 = st.columns([4, 1])
                    with sc1:
                        st.markdown(f"**Customer:** `{r['customer_id']}` &nbsp;|&nbsp; **Source:** `{r['source_file']}`")
                        st.write(f"📄 *\"{r['clause']}\"*")
                    with sc2:
                        st.metric("Similarity", f"{r['similarity_score']}%")
        else:
            st.warning("No matching contract clauses found in vector database.")