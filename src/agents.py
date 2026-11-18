import os
import json
import urllib.request
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

def get_active_groq_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in .env file!")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as resp:
        models = [m["id"] for m in json.loads(resp.read().decode())["data"]]
    for preferred in ["llama-3.3-70b", "gpt-oss-120b", "qwen", "llama-3.1-8b", "llama"]:
        for m in models:
            if preferred in m and "guard" not in m and "whisper" not in m:
                return f"groq/{m}"
    return f"groq/{models[0]}"

active_model = get_active_groq_model()
print(f"Using active Groq model: {active_model}")

agent_llm = LLM(
    model=active_model,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

def investigate_leak(leak_record):
    auditor = Agent(
        role="Senior Revenue Forensic Auditor",
        goal="Verify detected billing discrepancies and quantify exact financial exposure.",
        backstory="Expert in SaaS/Telecom billing reconciliation, MSA contracts, and usage metering audits.",
        llm=agent_llm,
        verbose=False
    )

    rca_specialist = Agent(
        role="Billing Systems Root Cause Analyst",
        goal="Identify the exact technical or operational failure causing the revenue leak.",
        backstory="Specialist in CRM-to-ERP sync pipelines, API metering webhooks, and provisioning workflows.",
        llm=agent_llm,
        verbose=False
    )

    ticket_router = Agent(
        role="Automated Recovery & Ticketing Coordinator",
        goal="Draft an actionable Jira investigation ticket and customer billing addendum.",
        backstory="Automates revenue recovery workflows and assigns tickets to Finance or Engineering teams.",
        llm=agent_llm,
        verbose=False
    )

    audit_task = Task(
        description=f"Audit this detected revenue leak record and confirm financial impact:\n{json.dumps(leak_record, indent=2)}",
        expected_output="2-3 bullet points verifying the contract vs invoice discrepancy and exact USD loss.",
        agent=auditor
    )

    rca_task = Task(
        description="Based on the audit, pinpoint the likely root cause (e.g., CRM discount sync bug, API metering drop, or provisioning webhook failure) and a preventive fix.",
        expected_output="Concise Root Cause and Technical Fix.",
        agent=rca_specialist,
        context=[audit_task]
    )

    ticket_task = Task(
        description="Create a structured Jira Ticket (Ticket ID, Priority, Assigned Team: Billing-Ops or Data-Eng, Summary, Action Items) to recover the leaked revenue.",
        expected_output="Clean Markdown Jira ticket ready for engineering/finance action.",
        agent=ticket_router,
        context=[audit_task, rca_task]
    )

    crew = Crew(
        agents=[auditor, rca_specialist, ticket_router],
        tasks=[audit_task, rca_task, ticket_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return str(result)

def ask_billing_question(question: str, data_context: dict) -> str:
    """
    Natural-language Q&A over billing data.
    Uses the same Groq LLM (via direct API call, not CrewAI) for fast single-turn answers.
    """
    import urllib.parse

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY not configured. Please check your .env file."

    system_prompt = (
        "You are RevGuard AI's billing intelligence assistant. "
        "You have access to the current audit data context provided in JSON. "
        "Answer the user's question concisely, accurately, and in a business-friendly tone. "
        "Format numbers as currency (USD) where appropriate. "
        "If the data does not contain enough information to answer, say so clearly."
    )

    user_prompt = (
        f"Current audit data context:\n{json.dumps(data_context, indent=2)}\n\n"
        f"User question: {question}"
    )

    payload = json.dumps({
        "model": active_model.replace("groq/", ""),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RevGuard-AI/3.0"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM call failed: {str(e)}"


if __name__ == "__main__":
    with open("data/logs/detected_leaks.json", "r") as f:
        leaks = json.load(f)

    if leaks:
        print(f"Running CrewAI investigation on top leak: {leaks[0]['customer_id']} (${leaks[0]['leaked_usd']})...")
        report = investigate_leak(leaks[0])
        print("\n=== GENERATED AI INVESTIGATION & JIRA TICKET ===\n")
        print(report)