"""RAG Knowledge Assistant - Modern Frontend"""

import os
import html
import json
import streamlit as st
import requests
import uuid
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
BACKEND_TIMEOUT = int(os.getenv("BACKEND_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knowledge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

# ---------------------------------------------------------------------------
# Helper: format file size
# ---------------------------------------------------------------------------
def _fmt_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}" if b != int(b) else f"{int(b)} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _fmt_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M")
    except Exception:
        return ts[:16] if len(ts) > 16 else ts


def _file_icon(fname: str) -> str:
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    return {"pdf": "📕", "docx": "📘", "doc": "📘", "txt": "📄", "md": "📝"}.get(ext, "📎")


# ---------------------------------------------------------------------------
# Backend connectivity check (cached 10s)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def _check_backend():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


# ---------------------------------------------------------------------------
# CSS: Modern, professional design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Imports ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ─────────────────────────────────────── */
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d27;
    --bg-card: #1e2130;
    --bg-card-hover: #252840;
    --accent: #6c63ff;
    --accent-soft: rgba(108, 99, 255, 0.12);
    --accent-glow: rgba(108, 99, 255, 0.25);
    --success: #10b981;
    --success-soft: rgba(16, 185, 129, 0.12);
    --warning: #f59e0b;
    --warning-soft: rgba(245, 158, 11, 0.12);
    --danger: #ef4444;
    --danger-soft: rgba(239, 68, 68, 0.12);
    --text-primary: #f0f0f5;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --border: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(108, 99, 255, 0.3);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-lg: 16px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --shadow-accent: 0 4px 20px rgba(108, 99, 255, 0.15);
}

/* ── Global ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.main .block-container {
    padding: 1.5rem 2rem 4rem 2rem;
    max-width: 1200px;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13152a 0%, #0f1117 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ── Brand header ───────────────────────────────────────── */
.brand-header {
    padding: 0.2rem 0 1.2rem 0;
    text-align: center;
}
.brand-header h1 {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.3;
}
.brand-header p {
    color: var(--text-muted);
    font-size: 0.72rem;
    margin: 0.15rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
}

/* ── Stat cards in sidebar ──────────────────────────────── */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 0.6rem 0;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.2;
}
.stat-card .stat-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Section label ──────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-weight: 600;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
}

/* ── Hero header ────────────────────────────────────────── */
.hero-wrap {
    padding: 1.8rem 0 0.6rem 0;
}
.hero-wrap h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem 0;
    color: var(--text-primary);
    line-height: 1.15;
}
.hero-wrap h1 .gradient-text {
    background: linear-gradient(135deg, #6c63ff, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-wrap .hero-sub {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}

/* ── Status badge ───────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-online {
    background: var(--success-soft);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.status-offline {
    background: var(--danger-soft);
    color: var(--danger);
    border: 1px solid rgba(239, 68, 68, 0.2);
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.status-online .status-dot { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-offline .status-dot { background: var(--danger); }

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.5rem;
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: var(--shadow-accent);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Document card ──────────────────────────────────────── */
.doc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 10px;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 14px;
}
.doc-card:hover {
    border-color: var(--border-accent);
    background: var(--bg-card-hover);
    box-shadow: var(--shadow-accent);
}
.doc-icon {
    font-size: 1.8rem;
    flex-shrink: 0;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-soft);
    border-radius: var(--radius-sm);
}
.doc-info { flex-grow: 1; min-width: 0; }
.doc-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin: 2px 0 0 0;
    display: flex;
    gap: 12px;
}
.doc-meta span { display: inline-flex; align-items: center; gap: 3px; }

/* ── Upload zone ────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed rgba(108, 99, 255, 0.25);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem;
    text-align: center;
    background: rgba(108, 99, 255, 0.04);
    transition: all 0.3s ease;
    margin: 1rem 0;
}
.upload-zone:hover {
    border-color: var(--accent);
    background: rgba(108, 99, 255, 0.08);
}
.upload-zone .upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.upload-zone h3 {
    margin: 0.3rem 0 0.2rem 0;
    color: var(--text-primary);
    font-size: 1rem;
    font-weight: 600;
}
.upload-zone p {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin: 0;
}

/* ── Empty state ────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
}
.empty-state .empty-icon {
    font-size: 3.5rem;
    margin-bottom: 0.8rem;
    display: block;
}
.empty-state h3 {
    color: var(--text-primary);
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
}
.empty-state p {
    color: var(--text-muted);
    font-size: 0.88rem;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Chat welcome ───────────────────────────────────────── */
.chat-welcome {
    text-align: center;
    padding: 3rem 2rem 2rem 2rem;
}
.chat-welcome .welcome-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    display: block;
}
.chat-welcome h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.4rem 0;
}
.chat-welcome p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin: 0 0 1.5rem 0;
}
.tip-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    max-width: 660px;
    margin: 0 auto;
}
.tip-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}
.tip-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-1px);
}
.tip-card .tip-icon {
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
    display: block;
}
.tip-card .tip-title {
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--text-primary);
    margin: 0 0 0.2rem 0;
}
.tip-card .tip-desc {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin: 0;
    line-height: 1.45;
}

/* ── Citation pill ──────────────────────────────────────── */
.citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-soft);
    border: 1px solid var(--border-accent);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #a78bfa;
    margin: 3px 4px 3px 0;
}
.citation-detail {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    margin: 6px 0;
}
.citation-detail .cd-source {
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--text-primary);
    margin: 0 0 4px 0;
}
.citation-detail .cd-snippet {
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.5;
    margin: 0;
    font-style: italic;
}

/* ── Chat message overrides ─────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 8px !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6c63ff, #7c6cff) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(108, 99, 255, 0.25) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
    transform: translateY(-1px);
}

/* ── Chat input ─────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* ── Slider ─────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important;
}

/* ── File uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] > div > div {
    border-radius: var(--radius) !important;
    border: 2px dashed var(--border-accent) !important;
    background: rgba(108, 99, 255, 0.03) !important;
}

/* ── Expander ───────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Divider ────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Progress bar ───────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6c63ff, #a78bfa) !important;
}

/* ── Hide Streamlit branding ────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backend status
# ---------------------------------------------------------------------------
backend_online, health_data = _check_backend()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
        <h1>⚡ Knowledge AI</h1>
        <p>Intelligent Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Status
    if backend_online:
        chunks_count = health_data.get("vector_store_chunks", 0)
        st.markdown(
            '<span class="status-badge status-online">'
            '<span class="status-dot"></span>System Online</span>',
            unsafe_allow_html=True,
        )
    else:
        chunks_count = 0
        st.markdown(
            '<span class="status-badge status-offline">'
            '<span class="status-dot"></span>Backend Offline</span>',
            unsafe_allow_html=True,
        )

    # Fetch docs count
    doc_count = 0
    if backend_online:
        try:
            r = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=5)
            if r.status_code == 200:
                doc_count = len(r.json())
        except Exception:
            pass

    # Stat cards
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{doc_count}</div>
            <div class="stat-label">Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{chunks_count}</div>
            <div class="stat-label">Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.chat_messages)}</div>
            <div class="stat-label">Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.chat_messages) // 2}</div>
            <div class="stat-label">Q&A Pairs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Retrieval settings
    st.markdown('<div class="section-label">Retrieval Settings</div>', unsafe_allow_html=True)

    top_k = st.slider(
        "Context chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to retrieve for context",
    )

    # Actions
    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ New Chat", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.rerun()
    with col2:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    # Session
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
    st.caption(f"ID: `{st.session_state.session_id[:12]}...`")

# ---------------------------------------------------------------------------
# MAIN AREA - Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <h1>Your <span class="gradient-text">AI Knowledge</span> Assistant</h1>
    <p class="hero-sub">Upload documents and get intelligent answers with source citations</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_docs, tab_chat = st.tabs(["  📂  Knowledge Base  ", "  💬  AI Chat  "])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("")

    # Upload area
    upload_col, action_col = st.columns([4, 1])

    with upload_col:
        uploaded_files = st.file_uploader(
            "Drop files here or click to browse",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, Markdown",
            label_visibility="collapsed",
        )

    with action_col:
        st.markdown("<br>", unsafe_allow_html=True)
        upload_clicked = st.button(
            "⬆ Upload",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        )

    if upload_clicked and uploaded_files:
        progress = st.progress(0, text="Preparing...")
        for idx, file in enumerate(uploaded_files):
            progress.progress(
                (idx) / len(uploaded_files),
                text=f"Processing **{file.name}**...",
            )
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/documents/upload",
                    files={"file": (file.name, file, file.type)},
                    timeout=BACKEND_TIMEOUT,
                )
                if resp.status_code == 200:
                    st.session_state.documents.append(resp.json())
                    st.toast(f"✅ **{file.name}** uploaded!", icon="📄")
                else:
                    st.toast(f"Failed: {file.name}", icon="❌")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is the API server running?")
                break
            except Exception as e:
                st.toast(f"Error: {e}", icon="❌")

            progress.progress((idx + 1) / len(uploaded_files))

        progress.empty()
        st.rerun()

    st.markdown("")

    # Document list
    if not backend_online:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">🔌</span>
            <h3>Backend Unavailable</h3>
            <p>Cannot connect to the API server. Start the backend with <code>python main.py</code> and refresh.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            resp = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=BACKEND_TIMEOUT)
            if resp.status_code == 200:
                docs = resp.json()
                if docs:
                    st.markdown(
                        f'<div class="section-label" style="margin-top:0">'
                        f'{len(docs)} document{"s" if len(docs) != 1 else ""} indexed</div>',
                        unsafe_allow_html=True,
                    )
                    for doc in docs:
                        icon = _file_icon(doc["filename"])
                        name = html.escape(doc["filename"])
                        size = _fmt_size(doc["size_bytes"])
                        chunks = doc["chunks_count"]
                        time = _fmt_time(doc.get("uploaded_at", ""))

                        c1, c2 = st.columns([6, 1])
                        with c1:
                            st.markdown(
                                f'<div class="doc-card">'
                                f'<div class="doc-icon">{icon}</div>'
                                f'<div class="doc-info">'
                                f'<p class="doc-name">{name}</p>'
                                f'<div class="doc-meta">'
                                f'<span>{size}</span>'
                                f'<span>{chunks} chunks</span>'
                                f'<span>{time}</span>'
                                f'</div></div></div>',
                                unsafe_allow_html=True,
                            )
                        with c2:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(
                                "Remove",
                                key=f"del_{doc['doc_id']}",
                                use_container_width=True,
                            ):
                                try:
                                    requests.delete(
                                        f"{API_BASE_URL}/api/v1/documents/{doc['doc_id']}",
                                        timeout=BACKEND_TIMEOUT,
                                    )
                                    st.toast(f"Removed **{doc['filename']}**", icon="🗑")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.markdown("""
                    <div class="empty-state">
                        <span class="empty-icon">📭</span>
                        <h3>No Documents Yet</h3>
                        <p>Upload PDF, DOCX, TXT, or Markdown files above to build your knowledge base.
                           The AI will use these documents to answer your questions.</p>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading documents: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: AI CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Chat history
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        avatar = "🧑‍💻" if role == "user" else "⚡"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

            # Citations
            if role == "assistant" and msg.get("citations"):
                citations = msg["citations"]
                # Show citation pills
                pills_html = ""
                for c in citations:
                    src = html.escape(c.get("source", "Unknown"))
                    pg = f" p.{c['page']}" if c.get("page") else ""
                    pills_html += f'<span class="citation-pill">📎 {src}{pg}</span>'
                st.markdown(pills_html, unsafe_allow_html=True)

                with st.expander(f"📚 View {len(citations)} source{'s' if len(citations) != 1 else ''}"):
                    for c in citations:
                        src = html.escape(c.get("source", "Unknown"))
                        pg = f" &middot; Page {c['page']}" if c.get("page") else ""
                        snippet = html.escape(c.get("content_snippet", "")[:200])
                        st.markdown(
                            f'<div class="citation-detail">'
                            f'<p class="cd-source">📄 {src}{pg}</p>'
                            f'<p class="cd-snippet">"{snippet}..."</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    user_input = st.chat_input(
        "Ask anything — say hi, or ask about your documents..."
    )

    if user_input:
        # Show user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Stream assistant response — smooth word-by-word typing
        with st.chat_message("assistant", avatar="⚡"):
            message_placeholder = st.empty()
            citations_container = st.container()

            full_response = ""
            citations = []

            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/chat/stream",
                    json={
                        "query": user_input,
                        "session_id": st.session_state.session_id,
                        "top_k": top_k,
                    },
                    stream=True,
                    timeout=BACKEND_TIMEOUT,
                )

                if resp.status_code == 200:
                    for chunk in resp.iter_content(
                        chunk_size=None, decode_unicode=True
                    ):
                        if chunk:
                            full_response += chunk
                            # Show only text before citation marker
                            display = full_response.split("\n\n__CITATIONS__")[0]
                            message_placeholder.markdown(display + " ▌")

                    # Parse citations from the end of stream
                    if "\n\n__CITATIONS__" in full_response:
                        text_part, cite_json = full_response.split(
                            "\n\n__CITATIONS__", 1
                        )
                        full_response = text_part
                        try:
                            citations = json.loads(cite_json)
                        except (json.JSONDecodeError, ValueError):
                            citations = []

                    # Final render (remove cursor)
                    message_placeholder.markdown(full_response)

                    # Render citations
                    if citations:
                        with citations_container:
                            pills_html = ""
                            for c in citations:
                                src = html.escape(c.get("source", "Unknown"))
                                pg = (
                                    f" p.{c['page']}" if c.get("page") else ""
                                )
                                pills_html += (
                                    f'<span class="citation-pill">'
                                    f"📎 {src}{pg}</span>"
                                )
                            st.markdown(pills_html, unsafe_allow_html=True)

                            with st.expander(
                                f"📚 View {len(citations)} "
                                f"source{'s' if len(citations) != 1 else ''}"
                            ):
                                for c in citations:
                                    src = html.escape(
                                        c.get("source", "Unknown")
                                    )
                                    pg = (
                                        f" &middot; Page {c['page']}"
                                        if c.get("page")
                                        else ""
                                    )
                                    snippet = html.escape(
                                        c.get("content_snippet", "")[:200]
                                    )
                                    st.markdown(
                                        f'<div class="citation-detail">'
                                        f'<p class="cd-source">📄 {src}{pg}</p>'
                                        f'<p class="cd-snippet">"{snippet}..."</p>'
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                    # Save to session
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "citations": citations,
                        }
                    )
                else:
                    message_placeholder.empty()
                    st.error(f"Error: {resp.text}")

            except requests.exceptions.ConnectionError:
                message_placeholder.empty()
                st.error(
                    "Cannot connect to backend. "
                    "Make sure the API server is running."
                )
            except requests.exceptions.Timeout:
                message_placeholder.empty()
                st.error(f"Request timed out after {BACKEND_TIMEOUT}s")
            except Exception as e:
                message_placeholder.empty()
                st.error(f"Error: {e}")

    # Welcome screen
    if not st.session_state.chat_messages:
        st.markdown("""
        <div class="chat-welcome">
            <span class="welcome-icon">⚡</span>
            <h2>Ready to explore your knowledge</h2>
            <p>Ask questions and get instant AI-powered answers with source citations</p>
            <div class="tip-grid">
                <div class="tip-card">
                    <span class="tip-icon">📂</span>
                    <p class="tip-title">Upload Docs</p>
                    <p class="tip-desc">Add PDF, Word, Text, or Markdown files to your knowledge base</p>
                </div>
                <div class="tip-card">
                    <span class="tip-icon">💬</span>
                    <p class="tip-title">Ask Questions</p>
                    <p class="tip-desc">Query your documents in natural language and get precise answers</p>
                </div>
                <div class="tip-card">
                    <span class="tip-icon">📎</span>
                    <p class="tip-title">Cited Sources</p>
                    <p class="tip-desc">Every answer includes references to the exact source documents</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
