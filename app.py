import os
import re
import html
import json
import types
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from datetime import datetime

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load the ML model code from ml_model.ipynb
# ---------------------------------------------------------------------------
def _load_notebook_module(path: str, module_name: str = "ml_model"):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    source = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    module = types.ModuleType(module_name)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


ml_model = _load_notebook_module("ml_model.ipynb")


def classify_text(text: str):
    """Thin UI-layer wrapper around ml_model.classify_text: runs the actual
    TF-IDF + Logistic Regression pipeline (defined in ml_model.ipynb), then
    applies the manual override checkbox from the sidebar (a UI concern
    that has no business living inside the ML notebook)."""
    label, confidence = ml_model.classify_text(text)

    if st.session_state.get("flip_labels"):
        label = "Fake" if label == "Real" else "Real"

    return label, confidence


# ---------------------------------------------------------------------------
# Load CSS from style.css
# ---------------------------------------------------------------------------
def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


load_css("style.css")

# ---------------------------------------------------------------------------
# Real-time news fetching — API key is fully automated, no typing required
# ---------------------------------------------------------------------------
GOOGLE_NEWS_TOPICS = {
    "general": None,
    "business": "BUSINESS",
    "technology": "TECHNOLOGY",
    "sports": "SPORTS",
    "health": "HEALTH",
    "entertainment": "ENTERTAINMENT",
    "science": "SCIENCE",
}
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    return html.unescape(_TAG_RE.sub("", raw)).strip()


def get_auto_api_key() -> str:
    """Resolve a NewsAPI key automatically - no manual entry anywhere."""
    try:
        if "NEWS_API_KEY" in st.secrets:
            return st.secrets["NEWS_API_KEY"]
    except Exception:
        pass
    return os.environ.get("NEWS_API_KEY", "")


def _build_rss_url(country_code: str, category: str) -> str:
    cc = country_code.upper()
    topic = GOOGLE_NEWS_TOPICS.get(category)
    if topic:
        return (
            f"https://news.google.com/rss/headlines/section/topic/{topic}"
            f"?hl=en-{cc}&gl={cc}&ceid={cc}:en"
        )
    return f"https://news.google.com/rss?hl=en-{cc}&gl={cc}&ceid={cc}:en"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_headlines_rss(country_code: str, category: str, page_size: int = 10):
    """No-key source: Google News RSS."""
    url = _build_rss_url(country_code, category)
    response = requests.get(
        url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    items = root.findall(".//item")[:page_size]

    articles = []
    for item in items:
        title = _strip_html(item.findtext("title") or "Untitled")
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        description = _strip_html(item.findtext("description") or "")

        source_el = item.find("source")
        source_name = source_el.text.strip() if source_el is not None and source_el.text else "Google News"

        if description and description.lower().startswith(title.lower()[:20]):
            description = ""

        articles.append({
            "title": title, "source": source_name, "publishedAt": pub_date,
            "description": description, "url": link,
        })
    return articles


@st.cache_data(ttl=300, show_spinner=False)
def fetch_headlines_newsapi(api_key: str, country_code: str, category: str, page_size: int = 10):
    """Key-based source: NewsAPI.org. Key is auto-detected, never typed."""
    params = {"apiKey": api_key, "country": country_code, "category": category, "pageSize": page_size}
    response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    articles = []
    for a in payload.get("articles", []):
        articles.append({
            "title": a.get("title") or "Untitled",
            "source": (a.get("source") or {}).get("name", "Unknown source"),
            "publishedAt": a.get("publishedAt", ""),
            "description": a.get("description") or "",
            "url": a.get("url", ""),
            "content": a.get("content") or "",
        })
    return articles


# ---------------------------------------------------------------------------
# Sidebar: live news feed controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Live News Feed")

    auto_key = get_auto_api_key()
    if auto_key:
        st.markdown(
            '<div class="status-box">🔑 NewsAPI key auto-detected — richer results enabled</div>',
            unsafe_allow_html=True
        )
        st.caption("Pulls current headlines from NewsAPI.org (key loaded automatically from secrets/env - nothing to type).")
    else:
        st.markdown(
            '<div class="status-box">🌐 No API key needed — powered by free RSS feeds</div>',
            unsafe_allow_html=True
        )
        st.caption("Pulls current headlines straight from Google News RSS. Add a NEWS_API_KEY to secrets/env for NewsAPI results instead - no code changes needed.")

    country_options = {
        "United States": "us", "India": "in", "United Kingdom": "gb",
        "Canada": "ca", "Australia": "au",
    }
    category_options = ["general", "business", "technology", "sports", "health", "entertainment", "science"]

    country_label = st.selectbox("Country", list(country_options.keys()))
    category = st.selectbox("Category", category_options)

    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=True)
    fetch_clicked = st.button("🔄 Fetch Latest Headlines")

    with st.expander("⚙️ Advanced"):
        st.checkbox(
            "Flip Real/Fake labels",
            key="flip_labels",
            help="Enable this if the model's predictions look backwards (e.g. it always "
                 "says Fake for articles you know are real). This swaps the output label "
                 "without needing to retrain or re-save the model.",
        )
        st.caption(f"Model classes (raw, as saved in the pickle): `{ml_model.get_model_classes()}`")

    if "live_articles" not in st.session_state:
        st.session_state.live_articles = []
    if "live_fetched_at" not in st.session_state:
        st.session_state.live_fetched_at = None
    if "live_last_params" not in st.session_state:
        st.session_state.live_last_params = None

    current_params = (country_options[country_label], category, bool(auto_key))

    should_auto_fetch = (
        st.session_state.live_last_params != current_params
        or (auto_refresh and not st.session_state.live_articles)
    )

    if fetch_clicked or should_auto_fetch:
        try:
            with st.spinner("Fetching latest headlines..."):
                cc, cat, _ = current_params
                if auto_key:
                    articles = fetch_headlines_newsapi(auto_key, cc, cat)
                else:
                    articles = fetch_headlines_rss(cc, cat)
            st.session_state.live_articles = articles
            st.session_state.live_fetched_at = datetime.now().strftime("%H:%M:%S")
            st.session_state.live_last_params = current_params
            if not articles:
                st.info("No articles returned for this country/category combination.")
        except requests.exceptions.HTTPError:
            st.error("The news source rejected the request (e.g. invalid auto-detected key). Falling back to RSS on next refresh.")
        except requests.exceptions.RequestException:
            st.error("Could not reach the news source. Check your internet connection and try again.")
        except ET.ParseError:
            st.error("Received an unexpected response from the news feed. Try again in a moment.")

# ---------------------------------------------------------------------------
# Main card: manual paste-and-check (original functionality, unchanged)
# ---------------------------------------------------------------------------
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste a news article below to check whether it is <b>Fake</b> or <b>Real</b>.</div>', unsafe_allow_html=True)

inputn = st.text_area("News Article", "", height=250, placeholder="Paste your news content here...")

if st.button("Check News"):
    if inputn.strip():
        label, confidence = classify_text(inputn)
        conf_text = f" ({confidence:.0%} confidence)" if confidence is not None else ""
        if label == "Real":
            st.markdown(
                f'<div class="result-box real-news">✅ The News is Real!{conf_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box fake-news">❌ The News is Fake!{conf_text}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="warning-box">⚠ Please enter some text to analyze.</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live headlines card: real-time fetched articles, each checked on demand
# ---------------------------------------------------------------------------
if st.session_state.live_articles:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="live-header"><span class="live-dot"></span>'
        '<span class="title" style="font-size:26px;margin-bottom:0;">Live Headlines</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">Last refreshed at {st.session_state.live_fetched_at} '
        f'&middot; refreshes automatically every 5 minutes</div>',
        unsafe_allow_html=True
    )

    for i, article in enumerate(st.session_state.live_articles):
        title = article.get("title") or "Untitled"
        source = article.get("source") or "Unknown source"
        published = article.get("publishedAt", "")
        description = article.get("description") or ""
        url = article.get("url", "")
        combined_text = " ".join(filter(None, [title, description, article.get("content") or ""]))

        st.markdown('<div class="news-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="news-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="news-meta">{source} &middot; {published}</div>', unsafe_allow_html=True)
        if description:
            st.markdown(f'<div class="news-desc">{description}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Analyze this article", key=f"analyze_{i}"):
                st.session_state[f"result_{i}"] = classify_text(combined_text)
        with col2:
            if url:
                st.markdown(f"[Read full article ↗]({url})")

        if f"result_{i}" in st.session_state:
            result_label, result_conf = st.session_state[f"result_{i}"]
            conf_text = f" ({result_conf:.0%})" if result_conf is not None else ""
            badge_class = "badge-real" if result_label == "Real" else "badge-fake"
            badge_text = f"✅ Predicted Real{conf_text}" if result_label == "Real" else f"❌ Predicted Fake{conf_text}"
            st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
