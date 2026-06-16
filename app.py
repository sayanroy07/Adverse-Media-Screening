import streamlit as st
import requests
import json, time
from datetime import datetime

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="Adverse Media Copilot", page_icon="🏦", layout="wide")
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

with st.sidebar:
    st.subheader("🤖  AI Agent Workflow")
    def bullet_stream_generator():
        items = [
            "Accepts an entity name (person or company).",
            "Fetches adverse live news from multiple sources (Google/Bing/etc).",
            "Disambiguates the entity (e.g., 'John Smith' the banker vs. the footballer).",
            "Risk scores & ranks across FCA-aligned categories (Financial Crime, Fraud, Bribery).",
            "Produces an explainable, traceable risk report with a structured workflow.",
            "Runs on AMD VM (vLLM inference with Qwen 2.5-7B Instructs model) with a Streamlit front-end."]

        for item in items:
            # Yield the markdown bullet indicator first
            yield "- "
            # Stream the text chunk letter by letter or word by word
            for word in item.split(" "):
                yield word + " "
                time.sleep(0.1)
                # Yield a newline to start the next bullet point properly
            yield "\n"

    if st.button("Workflow Details:"):
        st.write_stream(bullet_stream_generator())

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
