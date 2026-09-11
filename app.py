"""Streamlit interface for the Spam Shield text classifier."""

from __future__ import annotations

import re
import string
from pathlib import Path

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Spam Shield | Message intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --ink:#101828; --muted:#667085; --surface:#ffffff; --line:#e7eaf0; --navy:#15254a; --blue:#3366ff; --mint:#1fc59a; --bg:#f7f8fc; }
        .stApp { background: var(--bg); color: var(--ink); font-family: 'Manrope', sans-serif; }
        .block-container { max-width: 1180px; padding: 1.35rem 2rem 3rem; }
        #MainMenu, footer, header { visibility: hidden; }
        .brandbar { display:flex; align-items:center; justify-content:space-between; padding: 0.4rem 0 2rem; }
        .brand { display:flex; align-items:center; gap:.7rem; font-weight:800; font-size:1.1rem; letter-spacing:-.04em; }
        .logo { width:34px; height:34px; display:grid; place-items:center; background:var(--navy); color:white; border-radius:10px; font-size:18px; }
        .live { color:#067647; background:#ecfdf3; border:1px solid #abefc6; border-radius:999px; padding:.35rem .7rem; font-size:.72rem; font-weight:700; }
        .eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.13em; font-family:'DM Mono',monospace; font-size:.72rem; font-weight:500; }
        h1 { font-size:clamp(2.25rem,5vw,4rem); letter-spacing:-.065em; line-height:1.04; margin:.5rem 0 .85rem; max-width:850px; }
        .hero-copy { color:var(--muted); font-size:1.02rem; max-width:650px; line-height:1.65; margin-bottom:1.7rem; }
        .panel { background:var(--surface); border:1px solid var(--line); border-radius:20px; padding:1.35rem; box-shadow:0 12px 35px rgba(23,38,74,.06); }
        .panel-title { font-size:.82rem; font-weight:800; letter-spacing:.01em; margin-bottom:.2rem; }
        .panel-subtitle { font-size:.78rem; color:var(--muted); margin-bottom:.8rem; }
        div[data-testid="stTextArea"] textarea { background:#fbfcff; border:1px solid #d7ddea; border-radius:12px; font-family:'Manrope', sans-serif; padding: .85rem; font-size:.95rem; }
        div[data-testid="stTextArea"] textarea:focus { border-color:#3366ff; box-shadow:0 0 0 3px rgba(51,102,255,.12); }
        .stButton>button { background:var(--blue); border:0; color:#fff; border-radius:10px; font-family:'Manrope',sans-serif; font-weight:800; padding:.66rem 1rem; width:100%; transition:transform .15s ease, background .15s ease; }
        .stButton>button:hover { background:#244fda; transform:translateY(-1px); color:#fff; }
        .sample-label { color:var(--muted); font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; margin:.75rem 0 .35rem; }
        .result-label { font-family:'DM Mono',monospace; color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }
        .risk-spam { background:#fff4ed; color:#b54708; border:1px solid #fed7aa; }
        .risk-safe { background:#ecfdf3; color:#067647; border:1px solid #abefc6; }
        .risk-pill { display:inline-block; margin:.45rem 0 .8rem; padding:.38rem .7rem; border-radius:999px; font-size:.75rem; font-weight:800; }
        .risk-number { font-size:2.25rem; font-weight:800; letter-spacing:-.06em; margin-bottom:.25rem; }
        .risk-copy { color:var(--muted); font-size:.83rem; line-height:1.55; }
        .empty-icon { font-size:1.75rem; margin:1rem 0 .3rem; }
        .empty-title { font-weight:800; margin-bottom:.35rem; }
        .empty-copy { color:var(--muted); font-size:.84rem; line-height:1.55; }
        .feature { background:rgba(255,255,255,.75); border:1px solid var(--line); border-radius:14px; padding:1rem; min-height:118px; }
        .feature-icon { font-size:1.15rem; }.feature-title { font-size:.82rem; font-weight:800; margin:.45rem 0 .2rem; }.feature-copy { color:var(--muted); font-size:.74rem; line-height:1.5; }
        .footer-note { color:#98a2b3; font-size:.72rem; text-align:center; padding-top:1.8rem; }
        @media(max-width:650px) { .block-container { padding:1rem 1rem 2rem; } h1 { font-size:2.45rem; } .brandbar { padding-bottom:1.5rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_assets():
    return (
        joblib.load(BASE_DIR / "spam_mnb_model.pkl"),
        joblib.load(BASE_DIR / "bow_vectorizer.pkl"),
    )


STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}


def clean_message(message: str) -> str:
    """Match the preprocessing used to train the saved model."""
    text = message.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d", "", text)
    # The model was trained after punctuation and digits were stripped. Regex
    # tokenization reproduces the word tokens without requiring NLTK data files.
    return " ".join(token for token in re.findall(r"[a-z_]+", text) if token not in STOP_WORDS)


def result_card(probability: float) -> str:
    is_spam = probability >= 0.5
    status = "Likely spam" if is_spam else "Looks safe"
    description = (
        "This message has patterns commonly associated with unsolicited or deceptive content."
        if is_spam
        else "This message does not show the common language patterns found in spam."
    )
    tone = "risk-spam" if is_spam else "risk-safe"
    return f"""
        <div class="result-label">Assessment</div>
        <div class="risk-pill {tone}">{status}</div>
        <div class="risk-number">{probability:.0%}</div>
        <div class="risk-copy">spam probability<br><br>{description}</div>
    """


inject_style()
model, vectorizer = load_assets()

st.markdown("""<div class="brandbar"><div class="brand"><div class="logo">◈</div>Spam Shield</div><div class="live">● Model online</div></div>""", unsafe_allow_html=True)
st.markdown("<div class='eyebrow'>Message intelligence</div><h1>Know what’s worth<br>your attention.</h1><div class='hero-copy'>Paste a message below for a fast, private spam assessment powered by your trained classification model.</div>", unsafe_allow_html=True)

input_col, output_col = st.columns([1.55, 0.8], gap="large")
with input_col:
    st.markdown("<div class='panel'><div class='panel-title'>Analyse a message</div><div class='panel-subtitle'>Email, SMS, or any text you want to check.</div>", unsafe_allow_html=True)
    message = st.text_area("Message", placeholder="Paste the message here…", height=185, label_visibility="collapsed")
    run = st.button("Check message  →", use_container_width=True)
    st.markdown("<div class='sample-label'>Try a sample</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        try_spam = st.button("You’ve won a £1,000 prize!", key="spam_sample")
    with s2:
        try_safe = st.button("Can we meet at 3 PM?", key="safe_sample")
    st.markdown("</div>", unsafe_allow_html=True)

if try_spam:
    message = "Congratulations! You have won a £1000 cash prize. Reply now to claim it."
    run = True
elif try_safe:
    message = "Hi, can we meet at 3 PM today to review the project?"
    run = True

with output_col:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if run and message.strip():
        cleaned = clean_message(message)
        probability = float(model.predict_proba(vectorizer.transform([cleaned]))[0][1])
        st.markdown(result_card(probability), unsafe_allow_html=True)
    elif run:
        st.warning("Add a message before checking it.")
    else:
        st.markdown("<div class='empty-icon'>✦</div><div class='empty-title'>Your result will appear here</div><div class='empty-copy'>We’ll estimate the likelihood of spam and give you a clear recommendation.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3, gap="medium")
for col, icon, title, copy in (
    (f1, "⚡", "Fast assessment", "Get a model prediction in seconds."),
    (f2, "◌", "Private by design", "Messages are processed only for this check."),
    (f3, "⌁", "Built for clarity", "A simple signal, without the guesswork."),
):
    with col:
        st.markdown(f"<div class='feature'><div class='feature-icon'>{icon}</div><div class='feature-title'>{title}</div><div class='feature-copy'>{copy}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='footer-note'>Spam Shield · Classification results are guidance, not a guarantee.</div>", unsafe_allow_html=True)
