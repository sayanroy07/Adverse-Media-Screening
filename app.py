import streamlit as st
import requests
import json
from datetime import datetime

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Adverse Media Copilot", page_icon="🔍", layout="wide")
st.title("🔍 Adverse Media Screening Copilot")
st.caption("UK Banking Compliance | FCA-Aligned | Powered by Qwen + vLLM")

with st.sidebar:
    st.subheader("⚙️ Settings")
    if st.button("Check Backend"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            st.success(f"✅ Connected: {r.json().get('model')}")
        except:
            st.error("❌ Cannot reach backend")

entity_name = st.text_input("Enter entity name", placeholder="e.g. HSBC, Wirecard, Jes Staley")
search_btn = st.button("🔍 Screen", type="primary")

if search_btn and entity_name:
    with st.spinner(f"Screening {entity_name}..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/screen",
                json={"entity_name": entity_name},
                timeout=120
            )
            if response.status_code == 200:
                st.session_state["report"] = response.json()
            else:
                st.error(f"Error: {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("Timed out. Try again.")
        except Exception as e:
            st.error(f"Error: {e}")

if "report" in st.session_state:
    report = st.session_state["report"]

    st.divider()

    # ── LLM Risk Report ──────────────────────────────────────────
    st.subheader("📋 Risk Report")
    st.info(report.get("report", "No report generated"))

    # ── Articles ─────────────────────────────────────────────────
    articles = report.get("articles", [])
    st.divider()
    st.subheader(f"📰 News Articles Found ({len(articles)})")

    if articles:
        for i, article in enumerate(articles, 1):
            with st.expander(f"{i}. {article.get('title', 'No title')} — {article.get('source', '')}"):
                st.caption(f"📅 {article.get('published', 'Unknown date')}")
                st.write(article.get("snippet", ""))
                link = article.get("link", "")
                if link:
                    st.markdown(f"[🔗 Read full article]({link})")
    else:
        st.warning("No articles fetched — report based on LLM knowledge only.")

    # ── Export ───────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "⬇️ Export Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"screening_{entity_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )