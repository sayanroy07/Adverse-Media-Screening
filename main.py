from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, feedparser, json, re

app = Flask(__name__)
CORS(app)

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen"


def ask_llm(prompt):
    r = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1
    }, timeout=60)
    return r.json()["choices"][0]["message"]["content"]


def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = text.replace('&nbsp;', ' ')  # fix &nbsp;
    text = text.replace('&amp;', '&')  # fix &amp;
    text = text.replace('&#39;', "'")  # fix apostrophes
    text = text.replace('&quot;', '"')  # fix quotes
    return text.strip()


def fetch_news(entity):
    articles = []
    queries = [
        f"{entity} fraud UK financial crime",
        f"{entity} money laundering scandal",
        f"{entity} FCA investigation fine"
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for query in queries:
        try:
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-GB&gl=GB&ceid=GB:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                articles.append({
                    "title": clean_html(entry.get("title", "No title")),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", "Unknown date"),
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "snippet": clean_html(entry.get("summary", ""))[:300]
                })
        except:
            pass

        try:
            url = f"https://www.bing.com/news/search?q={query.replace(' ', '+')}&format=RSS"
            r = requests.get(url, timeout=10, headers=headers)
            feed = feedparser.parse(r.text)
            for entry in feed.entries[:5]:
                articles.append({
                    "title": clean_html(entry.get("title", "No title")),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", "Unknown date"),
                    "source": entry.get("source", {}).get("title", "Bing News"),
                    "snippet": clean_html(entry.get("summary", ""))[:300]
                })
        except:
            pass
    # Deduplicate by title
    seen = set()
    unique = []
    for a in articles:
        if a["title"] not in seen and a["title"]:
            seen.add(a["title"])
            unique.append(a)

    return unique


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL})


@app.route("/screen", methods=["POST"])
def screen():
    data = request.json
    entity = data.get("entity_name", "")

    # Fetch news
    articles = fetch_news(entity)
    news_text = "\n".join([
        f"- {a['title']} ({a['source']}, {a['published']}): {a['snippet']}"
        for a in articles
    ]) if articles else "No news found. Use your knowledge."

    # Ask LLM
    prompt = f"""You are a Senior UK Banking Compliance Analyst specializing in Anti-Money Laundering (AML),
        Know Your Customer (KYC), Financial Crime Compliance, Sanctions Screening, and Adverse Media Screening.
        Your responsibility is to review adverse media articles & disambiguation associated with a customer, 
        business, beneficial owner, director, or related entity, and provide an explainable risk assessment.
        You must act conservatively, objectively, and based only on evidence present in the article.
        Do not make assumptions. Do not fabricate facts. If evidence is insufficient, indicate uncertainty.. 
                Entity to screen: {entity}

                News found:
                {news_text}

                Give a brief adverse media report with:
                1. Possible Event Categories like Money Laundering, Fraud, Terrorist Financing, Sanctions Violation, Corruption,
                    Bribery, Tax Evasion, Financial Crime, Organized Crime, Regulatory Action, Litigation, Cybercrime,
                    Human Rights Violation, Environmental Crime, Reputational Risk, No Adverse Information 
                2. Risk level:
                        HIGH - Convictions, criminal charges, sanctions breaches, proven fraud, money laundering, 
                                terrorism financing, significant regulatory  penalties.
                        MEDIUM - Credible allegations, investigations, regulatory findings, moderate impact.
                        LOW - Minor allegations, weak evidence, isolated incidents, low impact.
                        CLEAR - All Good
                3. Risk Score:
                        0-20 : No adverse information.
                        21-40: Minor reputational concerns.
                        41-60: Regulatory concerns or ongoing investigations.
                        61-80: Serious allegations or repeated adverse findings.
                        81-100:Confirmed criminal activity, sanctions breaches, terrorism financing, money laundering,
                                major fraud, convictions, or significant regulatory penalties.
                4. Compliance Recommendation: based on Risk Scores
                        0-20  = APPROVE
                        21-40 = MONITOR
                        41-60 = ENHANCED_DUE_DILIGENCE
                        61-80 = ESCALATE
                        81-100 = REJECT

                5. Provide a concise explanation covering What happened, Why it matters, Key evidence found, 
                        Compliance concerns, Potential banking risk (2-3 sentences).
                6. Key findings if any

                Be concise and factual."""

    response = ask_llm(prompt)

    return jsonify({
        "entity_name": entity,
        "articles_found": len(articles),
        "articles": articles,
        "report": response
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)