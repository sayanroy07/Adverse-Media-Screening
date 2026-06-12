"""
Adverse Media Screening Copilot — Streamlit UI
Hosted on Streamlit Cloud, calls FastAPI backend on AMD VM
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────
BACKEND_URL = st.secrets.get("https://1c2b35f46b2411.lhr.life", "http://localhost:8080")
# Set BACKEND_URL in Streamlit Cloud secrets (your AMD VM's public IP)

# ── Page Setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adverse Media Copilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .risk-HIGH    { color: #dc2626; font-weight: bold; font-size: 1.4rem; }
  .risk-MEDIUM  { color: #d97706; font-weight: bold; font-size: 1.4rem; }
  .risk-LOW     { color: #2563eb; font-weight: bold; font-size: 1.4rem; }
  .risk-CLEAR   { color: #16a34a; font-weight: bold; font-size: 1.4rem; }
  .finding-card { border-left: 4px solid #6366f1; padding: 12px 16px;
                  background: #f8f9ff; border-radius: 4px; margin: 8px 0; }
  .source-badge { background: #e0e7ff; color: #3730a3; padding: 2px 8px;
                  border-radius: 12px; font-size: 0.75rem; margin: 2px; }
  .rec-badge    { padding: 6px 14px; border-radius: 20px;
                  font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield-warning.png", width=64)
    st.title("Adverse Media\nCopilot")
    st.caption("UK Banking Compliance | Powered by Open-Source LLM")
    st.divider()

    st.subheader("⚙️ Settings")
    entity_type = st.selectbox(
        "Entity Type",
        ["auto", "Person", "Company", "Organisation"],
        help="Auto-detect or specify entity type"
    )
    context_hint = st.text_input(
        "Context Hint (optional)",
        placeholder="e.g. UK banker, hedge fund manager",
        help="Helps disambiguate common names"
    )
    show_all_articles = st.checkbox("Show all articles (incl. non-adverse)", False)

    st.divider()
    st.subheader("🔗 Backend Status")
    if st.button("Check Connection"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ Connected\nModel: {data.get('model')}")
            else:
                st.error(f"Backend error: {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")

# ── Main UI ───────────────────────────────────────────────────────
st.title("🔍 Adverse Media Screening Copilot")
st.caption("Explainable | Source-Traceable | FCA-Aligned Risk Categorisation")

# Search Bar
col1, col2 = st.columns([4, 1])
with col1:
    entity_name = st.text_input(
        "Entity Name",
        placeholder="e.g. Jes Staley, Wirecard AG, Ruja Ignatova",
        label_visibility="collapsed"
    )
with col2:
    search_btn = st.button("🔍 Screen", type="primary", use_container_width=True)

# ── Screening Logic ───────────────────────────────────────────────
if search_btn and entity_name:
    with st.spinner(f"Screening **{entity_name}** — fetching news, scoring relevance, analysing risk..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/screen",
                json={
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "context_hint": context_hint or None
                },
                timeout=120  # LLM analysis can take up to 2 min
            )

            if response.status_code == 200:
                report = response.json()
                st.session_state["report"] = report
                st.session_state["entity_name"] = entity_name
            else:
                st.error(f"Backend error: {response.status_code} — {response.text}")

        except requests.exceptions.Timeout:
            st.error("⏳ Request timed out. The LLM may be loading — try again in 30s.")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at `{BACKEND_URL}`. Check VM is running.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Report Display ────────────────────────────────────────────────
if "report" in st.session_state:
    report = st.session_state["report"]
    risk = report.get("overall_risk", "CLEAR")
    score = report.get("risk_score", 0)

    # ── Risk Header ──────────────────────────────────────────────
    st.divider()
    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1.5])

    with col1:
        st.subheader("📋 Entity Profile")
        entity = report.get("resolved_entity", {})
        st.markdown(f"**{entity.get('name', entity_name)}**")
        st.caption(f"{entity.get('entity_type', '')} — {entity.get('description', '')}")
        conf = entity.get("confidence", 0)
        conf_color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
        st.markdown(f"Entity confidence: :{conf_color}[{conf:.0%}]")

    with col2:
        st.subheader("⚠️ Risk Level")
        risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "CLEAR": "🟢"}
        st.markdown(f"<span class='risk-{risk}'>{risk_colors.get(risk, '⚪')} {risk}</span>",
                    unsafe_allow_html=True)

    with col3:
        st.subheader("📊 Risk Score")
        score_color = "red" if score >= 70 else "orange" if score >= 40 else "blue"
        st.markdown(f"<span class='risk-{risk}'>{score}/100</span>", unsafe_allow_html=True)
        st.progress(score / 100)

    with col4:
        st.subheader("📌 Recommendation")
        rec = report.get("recommendation", "")
        rec_map = {
            "CLEAR_TO_PROCEED": ("🟢 Clear to Proceed", "green"),
            "ENHANCED_DUE_DILIGENCE": ("🟡 Enhanced DD", "orange"),
            "ESCALATE_TO_MLRO": ("🔴 Escalate to MLRO", "red"),
            "REJECT": ("🚫 Reject", "red")
        }
        label, color = rec_map.get(rec, (rec, "grey"))
        st.markdown(f":{color}[**{label}**]")

    # ── Summary ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Analyst Summary")
    st.info(report.get("summary", "No summary available."))

    # ── Risk Findings ────────────────────────────────────────────
    findings = report.get("findings", [])
    if findings:
        st.divider()
        st.subheader(f"🔎 Risk Findings ({len(findings)} identified)")

        for finding in findings:
            sev = finding.get("severity", "LOW")
            sev_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(sev, "⚪")

            with st.expander(
                f"{sev_icon} [{finding.get('category')}] {finding.get('headline')}",
                expanded=(sev == "HIGH")
            ):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown("**Explanation**")
                    st.write(finding.get("explanation", ""))

                    st.markdown("**Evidence**")
                    for snippet in finding.get("evidence_snippets", []):
                        st.markdown(f"> {snippet}")

                with col_b:
                    st.markdown("**Severity**")
                    st.markdown(f"{sev_icon} **{sev}**")

                    if finding.get("date_range"):
                        st.markdown(f"**Period:** {finding['date_range']}")

                    st.markdown("**Sources**")
                    for url in finding.get("source_urls", []):
                        st.markdown(f"[🔗 {url[:50]}...]({url})")

    else:
        st.success("✅ No material adverse findings identified.")

    # ── Metadata ─────────────────────────────────────────────────
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Articles Reviewed", report.get("articles_reviewed", 0))
    col2.metric("Adverse Findings", len(findings))
    col3.metric("Model", report.get("model_used", "unknown"))

    # Sources list
    sources = report.get("sources", [])
    if sources:
        st.markdown("**News Sources Reviewed:**  " +
                    "  ".join([f"`{s}`" for s in sources]))

    # ── Reviewer Workflow ────────────────────────────────────────
    st.divider()
    st.subheader("👤 Reviewer Decision")
    st.caption("Your decision will be logged for audit purposes.")

    reviewer_id = st.text_input("Reviewer ID / Badge Number", placeholder="e.g. COMP-0042")
    reviewer_notes = st.text_area(
        "Reviewer Notes",
        placeholder="Add context, rationale for decision, or escalation notes..."
    )

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        if st.button("✅ Accept Finding", use_container_width=True):
            if reviewer_id:
                st.success("Decision recorded: ACCEPTED")
                # In production: POST to /review/{report_id}
            else:
                st.warning("Please enter your Reviewer ID")

    with col_r2:
        if st.button("📤 Escalate to MLRO", use_container_width=True, type="primary"):
            if reviewer_id:
                st.warning("⚠️ Escalated to MLRO — case created")
            else:
                st.warning("Please enter your Reviewer ID")

    with col_r3:
        if st.button("✏️ Override Risk Level", use_container_width=True):
            override_risk = st.selectbox("Override to:", ["HIGH", "MEDIUM", "LOW", "CLEAR"])
            if reviewer_id:
                st.info(f"Risk overridden to {override_risk}")
            else:
                st.warning("Please enter your Reviewer ID")

    # ── Export ───────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "⬇️ Export Report (JSON)",
        data=json.dumps(report, indent=2, default=str),
        file_name=f"adverse_media_{entity_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

elif not search_btn:
    # Empty state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info("🏦 **UK Banking Compliant**\nFCA-aligned risk categories: Financial Crime, Sanctions, Fraud, Bribery, Regulatory Action")
    col2.info("🔍 **Explainable AI**\nEvery risk finding traces back to dated, sourced news articles")
    col3.info("⚡ **Powered by vLLM**\nOpen-source LLM running on AMD VM — no data leaves your infrastructure")