import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.cluster.hierarchy import dendrogram, linkage
import pathlib
import warnings
warnings.filterwarnings('ignore')
import hashlib
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CustomerIQ | Segmentation Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  AUTH SYSTEM
#  USERS_DB lives in session_state so newly
#  registered accounts survive st.rerun()
# ─────────────────────────────────────────────

_DEFAULT_USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Admin User",
        "role": "Administrator",
        "avatar": "👑"
    },
    "analyst": {
        "password": hashlib.sha256("analyst123".encode()).hexdigest(),
        "name": "Data Analyst",
        "role": "Analyst",
        "avatar": "📊"
    },
    "demo": {
        "password": hashlib.sha256("demo".encode()).hexdigest(),
        "name": "Demo User",
        "role": "Viewer",
        "avatar": "👤"
    }
}

# Initialise exactly once per browser session
if "users_db" not in st.session_state:
    st.session_state["users_db"] = {k: dict(v) for k, v in _DEFAULT_USERS.items()}

def check_password(username: str, password: str) -> bool:
    db = st.session_state["users_db"]
    if username in db:
        return hashlib.sha256(password.encode()).hexdigest() == db[username]["password"]
    return False

def register_user(username: str, password: str, name: str):
    db = st.session_state["users_db"]
    if username in db:
        return False, "Username already exists."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    # Deep-copy the dict so Streamlit reliably detects the state mutation
    new_db = {k: dict(v) for k, v in db.items()}
    new_db[username] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "name": name,
        "role": "Analyst",
        "avatar": "🧑‍💻",
    }
    st.session_state["users_db"] = new_db
    return True, "Account created! You can now sign in."

# ─────────────────────────────────────────────
#  CUSTOM CSS — Dark Luxury Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #080c14;
    --bg-secondary: #0d1520;
    --bg-card: #111b28;
    --bg-card-hover: #162030;
    --accent-blue: #3b9eff;
    --accent-cyan: #00d4ff;
    --accent-purple: #a855f7;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --accent-pink: #ec4899;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: rgba(59,158,255,0.12);
    --border-hover: rgba(59,158,255,0.35);
    --glow: rgba(59,158,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-primary);
}

/* Animated background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(59,158,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(168,85,247,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0,212,255,0.02) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stSidebar"] {
    min-width: 280px !important;
    max-width: 320px !important;
    background: linear-gradient(180deg, #090e18 0%, #0c1420 100%) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: rgba(59,158,255,0.15) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]),
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-secondary);
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem !important;
    padding: 0.3rem 0 !important;
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #e2e8f0 !important;
    background-color: rgba(13,21,32,0.9) !important;
}

[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"] {
    background-color: #0d1520 !important;
    color: #e2e8f0 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: rgba(59,158,255,0.15) !important;
    color: #3b9eff !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    color: #e2e8f0 !important;
    border-color: rgba(59,158,255,0.3) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] span {
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label {
    color: var(--text-secondary) !important;
    font-size: 0.83rem !important;
}

/* AUTH PAGE */
.auth-container {
    max-width: 440px;
    margin: 0 auto;
    padding: 2.5rem;
    background: linear-gradient(135deg, #0f1c2e 0%, #111827 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,158,255,0.05), inset 0 1px 0 rgba(255,255,255,0.05);
}
.auth-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #3b9eff, #00d4ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.auth-tagline {
    text-align: center;
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.auth-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}
.demo-creds {
    background: rgba(59,158,255,0.06);
    border: 1px solid rgba(59,158,255,0.15);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: 'DM Mono', monospace;
    margin-top: 1rem;
}
.demo-creds strong { color: var(--accent-blue); }

/* HERO */
.hero-banner {
    background: linear-gradient(135deg, #0f1e35 0%, #0d1a2d 50%, #0f1530 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60%; right: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(59,158,255,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40%; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(168,85,247,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b9eff 0%, #00d4ff 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
}
.hero-sub { font-size: 0.9rem; color: var(--text-muted); margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(59,158,255,0.1);
    border: 1px solid rgba(59,158,255,0.3);
    color: var(--accent-blue);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

/* SECTION HEADERS */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #90cdf4;
    border-left: 3px solid var(--accent-blue);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* KPI CARDS */
.kpi-card {
    background: linear-gradient(135deg, #111b28 0%, #0e1720 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 0.75rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
    min-width: 0;
}
.kpi-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(59,158,255,0.03), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}
.kpi-card:hover { border-color: var(--border-hover); transform: translateY(-3px); box-shadow: 0 8px 25px rgba(59,158,255,0.1); }
.kpi-card:hover::before { opacity: 1; }
.kpi-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--accent-blue);
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-label { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.35rem; }

/* INFO BOXES */
.info-box {
    background: rgba(59,158,255,0.05);
    border: 1px solid rgba(59,158,255,0.18);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.87rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.info-box strong { color: #90cdf4; }
.success-box {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.87rem;
    color: #86efac;
}
.warning-box {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.87rem;
    color: #fcd34d;
}
.danger-box {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.87rem;
    color: #fca5a5;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,21,32,0.9);
    border-radius: 10px;
    padding: 4px;
    gap: 3px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    border-radius: 7px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,158,255,0.12) !important;
    color: var(--accent-blue) !important;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #1e4a8a, #1a3a6a);
    border: 1px solid rgba(59,158,255,0.3);
    color: #e2e8f0;
    border-radius: 9px;
    font-weight: 600;
    font-size: 0.87rem;
    padding: 0.55rem 1.3rem;
    transition: all 0.2s ease;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1e4a8a);
    border-color: rgba(59,158,255,0.6);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59,158,255,0.25);
}

/* METRIC OVERRIDES */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent-blue) !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* INPUT FIELDS */
.stTextInput > div > div > input,
.stPasswordInput > div > div > input {
    background: rgba(13,21,32,0.8) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 9px !important;
}
.stTextInput > div > div > input:focus,
.stPasswordInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,158,255,0.15) !important;
}

[data-baseweb="select"] > div:first-child {
    background: rgba(13,21,32,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
}
[data-baseweb="select"] > div:first-child:hover {
    border-color: var(--border-hover) !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span {
    color: var(--text-primary) !important;
}

[data-baseweb="popover"] {
    background: #0d1520 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-baseweb="menu"] {
    background: #0d1520 !important;
}
[data-baseweb="menu"] li {
    color: #e2e8f0 !important;
    background: transparent !important;
}
[data-baseweb="menu"] li:hover {
    background: rgba(59,158,255,0.12) !important;
    color: #3b9eff !important;
}

/* EXPANDERS */
.streamlit-expanderHeader {
    background: rgba(17,27,40,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: #90cdf4 !important;
    font-weight: 600 !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* SLIDERS */
.stSlider label, .stSelectbox label, .stMultiSelect label, .stTextInput label {
    color: var(--text-secondary) !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}

/* DIVIDER */
hr { border-color: var(--border) !important; }

/* USER BADGE */
.user-badge {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1rem;
    background: rgba(59,158,255,0.06);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 1rem;
}
.user-badge .avatar { font-size: 1.4rem; }
.user-badge .info { flex: 1; }
.user-badge .name { font-weight: 600; font-size: 0.88rem; color: var(--text-primary) !important; }
.user-badge .role { font-size: 0.72rem; color: var(--text-muted) !important; }

/* INSIGHT CARDS */
.insight-card {
    background: linear-gradient(135deg, #0f1e35, #0d1929);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: 0.5rem 0;
    transition: all 0.2s;
}
.insight-card:hover { border-color: var(--border-hover); transform: translateX(3px); }
.insight-card .title { font-family: 'Syne', sans-serif; font-size: 0.92rem; font-weight: 700; color: #90cdf4; margin-bottom: 0.3rem; }
.insight-card .body { font-size: 0.83rem; color: var(--text-secondary); line-height: 1.55; }

/* POTENTIAL BADGES */
.risk-high { color: #4ade80; background: rgba(34,197,94,0.1);  border: 1px solid rgba(34,197,94,0.25);  padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.risk-med  { color: #fbbf24; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25); padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.risk-low  { color: #f87171; background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.25);  padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* CLV Score ring */
.clv-ring { text-align: center; padding: 1.5rem; }
.clv-score { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: var(--accent-blue); }
.clv-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }

/* NOTIFICATION BADGE */
.notif-badge {
    display: inline-block;
    background: #ef4444;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 20px;
    margin-left: 6px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'auth_mode' not in st.session_state:
    st.session_state.auth_mode = 'login'
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0


# ─────────────────────────────────────────────
#  LOGIN / SIGNUP PAGE
# ─────────────────────────────────────────────
def render_auth_page():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; margin-bottom: 2rem;'>
            <div style='font-size:3rem; margin-bottom:0.5rem;'>🎯</div>
            <div style='font-family:Syne,sans-serif; font-size:2.5rem; font-weight:800;
                        background:linear-gradient(135deg,#3b9eff,#00d4ff,#a855f7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        line-height:1; margin-bottom:0.4rem;'>CustomerIQ</div>
            <div style='font-size:0.75rem; color:#475569; letter-spacing:0.15em; text-transform:uppercase;'>
                Segmentation Intelligence Studio
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐  Sign In", "✨  Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

            col_btn, _ = st.columns([1, 1])
            with col_btn:
                login_btn = st.button("Sign In →", use_container_width=True)

            if login_btn:
                if st.session_state.login_attempts >= 5:
                    st.error("🔒 Too many failed attempts. Please try again later.")
                elif check_password(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.login_attempts = 0
                    st.success(f"✅ Welcome back, {st.session_state['users_db'][username]['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    remaining = 5 - st.session_state.login_attempts
                    st.error(f"❌ Invalid credentials. {remaining} attempts remaining.")

            st.markdown("""
            <div class='demo-creds'>
            <strong>Demo Credentials:</strong><br>
            👑 admin / admin123 &nbsp;·&nbsp; Admin role<br>
            📊 analyst / analyst123 &nbsp;·&nbsp; Analyst role<br>
            👤 demo / demo &nbsp;·&nbsp; Viewer role
            </div>
            """, unsafe_allow_html=True)

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            new_name = st.text_input("Full Name", placeholder="Your full name", key="reg_name")
            new_user = st.text_input("Username", placeholder="Choose a username (min 3 chars)", key="reg_user")
            new_pass = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_confirm")

            signup_btn = st.button("Create Account →", use_container_width=True)

            if signup_btn:
                if not new_name or not new_user or not new_pass:
                    st.error("⚠️ All fields are required.")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = register_user(new_user, new_pass, new_name)
                    if success:
                        st.success(f"✅ {msg} Switch to the Sign In tab to log in.")
                    else:
                        st.error(f"❌ {msg}")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; font-size:0.7rem; color:#334155; padding-top:1rem;
                    border-top: 1px solid rgba(59,158,255,0.08);'>
            CustomerIQ v2.0 · ML Segmentation Platform · Undergraduate Project
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING
#  Default: ../data/newdata.csv
#  Fallback: synthetic data if default file is missing
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = pathlib.Path(__file__).parent.parent / 'data' / 'newdata.csv'
    if data_path.exists():
        return pd.read_csv(data_path), False

    # Fallback synthetic mall-customer data
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        'CustomerID':          range(1000, 1000 + n),
        'Gender':              np.random.choice(['M', 'F'], n),
        'Age':                 np.random.randint(18, 70, n).astype(float),
        'Annual Income (k$)':  np.round(np.random.uniform(15, 130, n), 1),
        'Spending Score (1-100)': np.round(np.random.uniform(1, 100, n), 1),
    }), True


@st.cache_data
def preprocess(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names regardless of exact spelling / units suffix
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(' ', '').replace('(', '').replace(')', '').replace('$', '').replace('-', '')
        if 'customerid' in cl or ('customer' in cl and 'id' in cl):
            col_map[c] = 'CustomerID'
        elif 'gender' in cl or 'sex' in cl:
            col_map[c] = 'Gender'
        elif cl == 'age':
            col_map[c] = 'Age'
        elif 'annualincome' in cl or ('income' in cl and 'annual' in cl):
            col_map[c] = 'Annual Income (k$)'
        elif 'spendingscore' in cl or 'spending' in cl:
            col_map[c] = 'Spending Score'
    df.rename(columns=col_map, inplace=True)

    # Add any missing required columns with sensible defaults
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = range(1000, 1000 + len(df))
    if 'Gender' not in df.columns:
        df['Gender'] = 'Unknown'

    # Numeric coercion
    for col in ['Age', 'Annual Income (k$)', 'Spending Score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    df['CustomerID'] = (pd.to_numeric(df['CustomerID'], errors='coerce')
                        .fillna(0).astype(int))

    # Impute missing numerics with column median
    for col in ['Age', 'Annual Income (k$)', 'Spending Score']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df['Gender'] = df['Gender'].fillna('Unknown').astype(str).str.strip()

    # Derived features
    df['Gender_Encoded'] = df['Gender'].str.upper().str.startswith('M').astype(int)

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 25, 35, 45, 55, 200],
        labels=['18–25', '26–35', '36–45', '46–55', '55+']
    )

    df['Income_Tier'] = pd.qcut(
        df['Annual Income (k$)'].rank(method='first'), q=3,
        labels=['Low Income', 'Mid Income', 'High Income']
    )

    return df


@st.cache_data
def build_scores(df):
    """
    Income-Spending-Age scoring (analogous to RFM).
    I_Score  1-5 : Annual Income percentile (higher = richer)
    S_Score  1-5 : Spending Score percentile (higher = more active spender)
    A_Score  1-5 : Youth score — younger customers get higher score
    Segment labels are assigned from combined rule logic.
    """
    s = df.copy()

    s['I_Score'] = pd.qcut(
        s['Annual Income (k$)'].rank(method='first'), 5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    s['S_Score'] = pd.qcut(
        s['Spending Score'].rank(method='first'), 5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    s['A_Score'] = pd.qcut(
        s['Age'].rank(method='first'), 5,
        labels=[5, 4, 3, 2, 1]   # inverted: younger → higher score
    ).astype(int)

    s['Total_Score'] = s['I_Score'] + s['S_Score'] + s['A_Score']

    def label_segment(row):
        i, sp, a = row['I_Score'], row['S_Score'], row['A_Score']
        if i >= 4 and sp >= 4:
            return 'Premium Shoppers'
        elif i >= 3 and sp >= 3:
            return 'Engaged Customers'
        elif i >= 4 and sp <= 2:
            return 'Wealthy Resistors'
        elif a >= 4 and sp >= 4:
            return 'Young Enthusiasts'
        elif i <= 2 and sp >= 4:
            return 'Aspirational Spenders'
        elif i <= 2 and sp <= 2:
            return 'Budget Conscious'
        elif i >= 3 and sp <= 2:
            return 'Selective Buyers'
        else:
            return 'Moderate Consumers'

    s['Segment'] = s.apply(label_segment, axis=1)
    return s


@st.cache_data
def compute_value(scored_df):
    """
    Customer Value Score = Annual Income × (Spending Score / 100).
    Tier: Bronze / Silver / Gold / Platinum via quartile.
    """
    val = scored_df.copy()
    val['Value_Raw'] = val['Annual Income (k$)'] * (val['Spending Score'] / 100)
    mn, mx = val['Value_Raw'].min(), val['Value_Raw'].max()
    val['Value_Score'] = ((val['Value_Raw'] - mn) / (mx - mn) * 100).round(1)
    val['Est_Annual_Value'] = (val['Value_Raw'] * 1.2).round(2)   # k$ proxy
    val['Est_2yr_Value']    = (val['Est_Annual_Value'] * 2).round(2)
    val['Value_Tier'] = pd.qcut(
        val['Value_Score'], 4,
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    return val


@st.cache_data
def predict_spending(scored_df):
    """
    Gradient Boosting classifier:
    Target  → High Spender (Spending Score ≥ median)
    Features → Age, Annual Income, Gender_Encoded
    """
    pred = scored_df.copy()
    median_s = pred['Spending Score'].median()
    pred['High_Spender'] = (pred['Spending Score'] >= median_s).astype(int)

    features = ['Age', 'Annual Income (k$)', 'Gender_Encoded']
    X = pred[features].copy()
    y = pred['High_Spender']

    # Edge-case guard
    if len(y.unique()) < 2:
        pred['Spend_Prob'] = 0.5
        pred['Potential'] = 'Medium'
        dummy_fi = pd.DataFrame({'Feature': features, 'Importance': [1/3]*3})
        return pred, np.array([0, 1]), np.array([0, 1]), 0.5, dummy_fi

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_tr_s, y_train)

    pred['Spend_Prob'] = model.predict_proba(scaler.transform(X))[:, 1]
    pred['Potential'] = pd.cut(
        pred['Spend_Prob'],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    y_prob_te = model.predict_proba(X_te_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_te)
    auc = roc_auc_score(y_test, y_prob_te)

    feat_imp = pd.DataFrame({
        'Feature':    features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    return pred, fpr, tpr, auc, feat_imp


@st.cache_data
def compute_personas(scored_df):
    """Demographic breakdown tables & heatmap data."""
    df = scored_df.copy()

    age_income_heatmap = (
        df.groupby(['Age_Group', 'Income_Tier'], observed=True)['Spending Score']
        .mean().round(1).unstack(fill_value=0)
    )

    gender_stats = df.groupby('Gender').agg(
        Count        = ('CustomerID',           'count'),
        Avg_Age      = ('Age',                   'mean'),
        Avg_Income   = ('Annual Income (k$)',    'mean'),
        Avg_Spending = ('Spending Score',        'mean')
    ).round(1).reset_index()

    age_stats = df.groupby('Age_Group', observed=True).agg(
        Count        = ('CustomerID',            'count'),
        Avg_Income   = ('Annual Income (k$)',    'mean'),
        Avg_Spending = ('Spending Score',        'mean')
    ).round(1).reset_index()

    return age_income_heatmap, gender_stats, age_stats


@st.cache_data
def run_kmeans(features_scaled, k):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(features_scaled)
    return labels, silhouette_score(features_scaled, labels), davies_bouldin_score(features_scaled, labels), km


@st.cache_data
def run_hierarchical(features_scaled, k, linkage_method='ward'):
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = hc.fit_predict(features_scaled)
    return labels, silhouette_score(features_scaled, labels), davies_bouldin_score(features_scaled, labels)


@st.cache_data
def run_pca(features_scaled, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(features_scaled)
    return components, pca.explained_variance_ratio_


# ─────────────────────────────────────────────
#  PLOT LAYOUT HELPER
# ─────────────────────────────────────────────
def dark_layout(fig, title='', height=None):
    updates = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        title_font=dict(family='Syne', size=14, color='#90cdf4'),
        title_text=title,
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        margin=dict(t=45, b=5, l=5, r=5),
    )
    if height:
        updates['height'] = height
    fig.update_layout(**updates)
    return fig


# ─────────────────────────────────────────────
#  COLOR PALETTES
# ─────────────────────────────────────────────
SEG_COLORS = {
    'Premium Shoppers':      '#f59e0b',
    'Engaged Customers':     '#22c55e',
    'Wealthy Resistors':     '#3b9eff',
    'Young Enthusiasts':     '#a855f7',
    'Aspirational Spenders': '#ef4444',
    'Budget Conscious':      '#64748b',
    'Selective Buyers':      '#ec4899',
    'Moderate Consumers':    '#06b6d4',
}
CLUSTER_PALETTE = px.colors.qualitative.Bold
VALUE_COLORS    = {'Bronze': '#a16207', 'Silver': '#94a3b8', 'Gold': '#ca8a04', 'Platinum': '#3b9eff'}
POTENTIAL_COLORS = {'Low': '#ef4444',   'Medium': '#f59e0b', 'High': '#22c55e'}
GENDER_COLORS   = {'M': '#3b9eff', 'F': '#ec4899', 'Male': '#3b9eff', 'Female': '#ec4899', 'Unknown': '#64748b'}


# ─────────────────────────────────────────────
#  MAIN APP ROUTER
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    render_auth_page()
else:
    user_info = st.session_state["users_db"].get(st.session_state.username, {})

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1.2rem 0 1rem 0;'>
            <div style='font-size:1.8rem;'>🎯</div>
            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                        background:linear-gradient(90deg,#3b9eff,#a855f7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                CustomerIQ
            </div>
            <div style='font-size:0.65rem; color:#334155; letter-spacing:0.12em; margin-top:2px;'>
                SEGMENTATION STUDIO v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='user-badge'>
            <div class='avatar'>{user_info.get('avatar', '👤')}</div>
            <div class='info'>
                <div class='name'>{user_info.get('name', 'User')}</div>
                <div class='role'>{user_info.get('role', 'Viewer')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📁 Data Source")
        st.caption("Using default dataset: `data/newdata.csv`")

        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        k_clusters = st.slider("K-Means Clusters",      2, 10, 5)
        hc_k       = st.slider("Hierarchical Clusters", 2, 8,  5)
        hc_link    = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'])
        pca_comp   = st.slider("PCA Components",         2, 3,  3)

        st.markdown("---")
        st.markdown("### 🗺️ Navigation")
        page = st.radio("", [
            "🏠 Overview",
            "🔍 EDA & Data Quality",
            "💡 Customer Scoring",
            "🔵 K-Means Clustering",
            "🌿 Hierarchical Clustering",
            "🔮 PCA & Dimensionality",
            "💎 Value Analysis",
            "⚠️ Spending Prediction",
            "👥 Customer Personas",
            "🤖 AI Segment Insights",
            "📊 Model Comparison",
            "🎯 Segment Profiler",
        ], label_visibility='collapsed')

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

        st.markdown("""
        <div style='font-size:0.68rem; color:#1e293b; text-align:center; margin-top:1rem; line-height:1.7;'>
            Undergraduate ML Project<br>CustomerIQ v2.0 · 2024
        </div>
        """, unsafe_allow_html=True)

    # ── LOAD & PROCESS DATA ──────────────────
    raw_df, is_synthetic = load_data()
    df      = preprocess(raw_df)
    scored  = build_scores(df)

    FEAT_COLS      = ['Age', 'Annual Income (k$)', 'Spending Score']
    scaler_main    = StandardScaler()
    features_scaled = scaler_main.fit_transform(scored[FEAT_COLS])

    # ══════════════════════════════════════════
    #  PAGE: OVERVIEW
    # ══════════════════════════════════════════
    if page == "🏠 Overview":
        st.markdown("""
        <div class='hero-banner'>
            <div class='hero-badge'>🎓 ML Course Project — Mall Customer Segmentation · v2.0</div>
            <h1 class='hero-title'>Customer Segmentation<br>Intelligence Studio</h1>
            <p class='hero-sub'>Income · Spending · Age Scoring · K-Means · Hierarchical · PCA · Value Analysis · Spending Prediction · AI Insights</p>
        </div>
        """, unsafe_allow_html=True)

        if is_synthetic:
            st.info("⚡ **Demo Mode** — Synthetic data loaded because `data/newdata.csv` was not found.")

        # KPIs
        female_pct   = (df['Gender'].str.upper().str.startswith('F')).mean() * 100
        high_spenders = (df['Spending Score'] >= 60).sum()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpis = [
            ("👥", "Customers",         f"{len(df):,}"),
            ("📅", "Avg Age",           f"{df['Age'].mean():.1f} yrs"),
            ("💰", "Avg Income",        f"${df['Annual Income (k$)'].mean():.1f}k"),
            ("🛍️", "Avg Spend Score",  f"{df['Spending Score'].mean():.1f}/100"),
            ("⚧",  "Female %",          f"{female_pct:.1f}%"),
            ("⭐", "High Spenders",     f"{high_spenders:,}"),
        ]
        for col, (icon, label, value) in zip([c1, c2, c3, c4, c5, c6], kpis):
            col.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>{icon}</div>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📌 Platform Architecture</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class='insight-card'>
                <div class='title'>🔵 Unsupervised Learning</div>
                <div class='body'>K-Means & Hierarchical Clustering on Age, Annual Income and Spending Score. PCA for dimensionality reduction & visualisation.</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='insight-card'>
                <div class='title'>🟠 Supervised Learning</div>
                <div class='body'>Rule-based Income-Spending-Age scoring → segment labels. Gradient Boosting predicts High vs Low Spending Potential with ROC/AUC evaluation.</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='insight-card'>
                <div class='title'>💡 Business Intelligence</div>
                <div class='body'>Customer Value Score (income × spending proxy), demographic persona heatmaps, and AI-generated strategic recommendations per segment.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Segment Distribution</div>", unsafe_allow_html=True)
        seg_counts = scored['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']

        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = px.bar(seg_counts, x='Segment', y='Count',
                         color='Segment', color_discrete_map=SEG_COLORS)
            dark_layout(fig, 'Customer Count by Segment')
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(seg_counts, names='Segment', values='Count',
                          color='Segment', color_discrete_map=SEG_COLORS, hole=0.5)
            dark_layout(fig2, 'Segment Share')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-header'>💰 Income vs Spending Overview</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            avg_sp = scored.groupby('Segment')['Spending Score'].mean().reset_index().sort_values('Spending Score')
            fig3 = px.bar(avg_sp, x='Spending Score', y='Segment', orientation='h',
                          color='Segment', color_discrete_map=SEG_COLORS)
            dark_layout(fig3, 'Avg Spending Score by Segment', height=310)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            pivot = scored.groupby(['I_Score', 'S_Score'])['CustomerID'].count().reset_index()
            pivot_wide = pivot.pivot(index='I_Score', columns='S_Score', values='CustomerID').fillna(0)
            fig4 = px.imshow(pivot_wide, color_continuous_scale='Blues',
                             labels=dict(x='Spending Score Quintile', y='Income Score Quintile', color='Customers'))
            dark_layout(fig4, 'Score Heatmap — Customer Density', height=310)
            st.plotly_chart(fig4, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: EDA
    # ══════════════════════════════════════════
    elif page == "🔍 EDA & Data Quality":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Exploratory Data Analysis</h1><p class='hero-sub'>Distributions, demographic insights, and data quality assessment</p></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Overview", "📈 Distributions", "⚧ Gender Analysis", "🔧 Data Quality"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw Rows",       f"{raw_df.shape[0]:,}")
            c2.metric("Columns",        raw_df.shape[1])
            c3.metric("After Cleaning", f"{len(df):,}")
            st.dataframe(df.head(500), use_container_width=True, height=260)
            st.markdown("**Statistical Summary**")
            st.dataframe(df[['Age', 'Annual Income (k$)', 'Spending Score']].describe().round(2), use_container_width=True)

        with tab2:
            c1, c2, c3 = st.columns(3)
            for col, metric, color in zip(
                [c1, c2, c3],
                ['Age', 'Annual Income (k$)', 'Spending Score'],
                ['#3b9eff', '#22c55e', '#f59e0b']
            ):
                fig = px.histogram(df, x=metric, nbins=30, color_discrete_sequence=[color])
                dark_layout(fig, f'{metric} Distribution')
                fig.update_layout(showlegend=False)
                col.plotly_chart(fig, use_container_width=True)

            # 2-D scatter: Income vs Spending Score coloured by Age Group
            fig_s = px.scatter(df, x='Annual Income (k$)', y='Spending Score',
                               color='Age_Group',
                               color_discrete_sequence=px.colors.qualitative.Bold,
                               opacity=0.7, hover_data=['CustomerID', 'Gender', 'Age'])
            dark_layout(fig_s, 'Annual Income vs Spending Score (coloured by Age Group)')
            st.plotly_chart(fig_s, use_container_width=True)

        with tab3:
            gender_data = df.groupby('Gender').agg(
                Count        = ('CustomerID',            'count'),
                Avg_Age      = ('Age',                    'mean'),
                Avg_Income   = ('Annual Income (k$)',    'mean'),
                Avg_Spending = ('Spending Score',        'mean')
            ).round(1).reset_index()
            st.dataframe(gender_data, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(gender_data, names='Gender', values='Count',
                             color='Gender', color_discrete_map=GENDER_COLORS, hole=0.45)
                dark_layout(fig, 'Gender Split')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(gender_data, x='Gender', y=['Avg_Income', 'Avg_Spending'],
                             barmode='group',
                             color_discrete_sequence=['#3b9eff', '#f59e0b'])
                dark_layout(fig, 'Avg Income & Spending Score by Gender')
                st.plotly_chart(fig, use_container_width=True)

            # Box plots by gender
            c1, c2 = st.columns(2)
            with c1:
                fig = px.box(df, x='Gender', y='Annual Income (k$)',
                             color='Gender', color_discrete_map=GENDER_COLORS)
                dark_layout(fig, 'Income Distribution by Gender')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, x='Gender', y='Spending Score',
                             color='Gender', color_discrete_map=GENDER_COLORS)
                dark_layout(fig, 'Spending Score by Gender')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            null_df = pd.DataFrame({
                'Column':    raw_df.columns,
                'Missing':   raw_df.isnull().sum().values,
                'Missing %': (raw_df.isnull().sum().values / len(raw_df) * 100).round(2)
            })
            c1, c2 = st.columns(2)
            with c1:
                missing_only = null_df[null_df['Missing'] > 0]
                if not missing_only.empty:
                    fig = px.bar(missing_only, x='Column', y='Missing %',
                                 color='Missing %', color_continuous_scale='Reds')
                    dark_layout(fig, 'Missing Data %')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values in the raw dataset!")
            with c2:
                dtype_df = pd.DataFrame({'Column': raw_df.columns, 'Type': raw_df.dtypes.astype(str).values})
                st.dataframe(dtype_df, use_container_width=True)

            removed = len(raw_df) - len(df)
            st.markdown(f"""
            <div class='info-box'>
            <strong>Cleaning Summary</strong><br>
            • Imputed missing Age, Income, Spending Score values with column median<br>
            • Filled missing Gender with 'Unknown'<br>
            • Derived Age_Group, Income_Tier, Gender_Encoded features<br>
            • <strong>{removed:,}</strong> rows adjusted → <strong>{len(df):,}</strong> clean records remain
            </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════
    #  PAGE: CUSTOMER SCORING  (replaces RFM)
    # ══════════════════════════════════════════
    elif page == "💡 Customer Scoring":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Customer Scoring</h1><p class='hero-sub'>Rule-Based Supervised Segmentation · Income · Spending · Age</p></div>", unsafe_allow_html=True)

        with st.expander("📖 What is Customer Scoring?", expanded=False):
            st.markdown("""
| Dimension | Definition | Insight |
|---|---|---|
| **Income Score (I)** | Annual income percentile, ranked 1–5 | Higher = greater purchasing power |
| **Spending Score (S)** | Spending behaviour percentile, ranked 1–5 | Higher = more actively spending |
| **Age Score (A)** | Youth score (inverted age), ranked 1–5 | Higher = younger, more growth potential |

Each customer receives three scores (1–5), which are combined to assign a **business segment label**.
            """)

        tab1, tab2, tab3 = st.tabs(["📊 Distributions & 3D", "🗺️ Heatmap & Summary", "📋 Full Table"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            for col, metric, color in zip(
                [c1, c2, c3],
                ['Annual Income (k$)', 'Spending Score', 'Age'],
                ['#3b9eff', '#22c55e', '#f59e0b']
            ):
                fig = px.histogram(scored, x=metric, nbins=35, color_discrete_sequence=[color])
                dark_layout(fig, f'{metric} Distribution')
                fig.update_layout(showlegend=False)
                col.plotly_chart(fig, use_container_width=True)

            sample = scored.sample(min(1000, len(scored)), random_state=42)
            fig3d = px.scatter_3d(
                sample, x='Age', y='Annual Income (k$)', z='Spending Score',
                color='Segment', color_discrete_map=SEG_COLORS, opacity=0.75,
                hover_data=['CustomerID', 'Gender']
            )
            fig3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                title='3D Feature Space — Coloured by Segment',
                title_font=dict(family='Syne', size=14, color='#90cdf4'),
                scene=dict(
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                ), height=500, margin=dict(t=45, b=5, l=5, r=5))
            st.plotly_chart(fig3d, use_container_width=True)

        with tab2:
            pivot = scored.groupby(['I_Score', 'S_Score'])['CustomerID'].count().reset_index()
            pivot_wide = pivot.pivot(index='I_Score', columns='S_Score', values='CustomerID').fillna(0)
            fig = px.imshow(pivot_wide, color_continuous_scale='Blues',
                            labels=dict(x='Spending Score Quintile', y='Income Score Quintile', color='Customers'))
            dark_layout(fig, 'Score Heatmap: Customer Count (Income vs Spending)')
            st.plotly_chart(fig, use_container_width=True)

            seg_summary = scored.groupby('Segment').agg(
                Count        = ('CustomerID',            'count'),
                Avg_Income   = ('Annual Income (k$)',    'mean'),
                Avg_Spending = ('Spending Score',        'mean'),
                Avg_Age      = ('Age',                   'mean'),
                Avg_I_Score  = ('I_Score',               'mean'),
                Avg_S_Score  = ('S_Score',               'mean'),
            ).round(1).reset_index()
            st.dataframe(seg_summary, use_container_width=True)

        with tab3:
            segs = ['All'] + list(scored['Segment'].unique())
            sel = st.selectbox("Filter by Segment", segs)
            disp = scored if sel == 'All' else scored[scored['Segment'] == sel]
            cols_show = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score',
                         'I_Score', 'S_Score', 'A_Score', 'Total_Score', 'Segment']
            st.dataframe(disp[cols_show].sort_values('Total_Score', ascending=False).reset_index(drop=True),
                         use_container_width=True, height=400)
            st.download_button("⬇️ Download Scored Table", disp[cols_show].to_csv(index=False).encode(),
                               "customer_scores.csv", "text/csv")


    # ══════════════════════════════════════════
    #  PAGE: K-MEANS
    # ══════════════════════════════════════════
    elif page == "🔵 K-Means Clustering":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>K-Means Clustering</h1><p class='hero-sub'>Unsupervised partition-based clustering on Age · Income · Spending Score</p></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📐 Elbow & Silhouette", "📊 Cluster Results", "🔬 Deep Dive"])

        with tab1:
            with st.spinner("Computing metrics..."):
                inertias, sils, dbs = [], [], []
                k_range = range(2, 11)
                for k_ in k_range:
                    km_ = KMeans(n_clusters=k_, init='k-means++', n_init=10, random_state=42)
                    lbl_ = km_.fit_predict(features_scaled)
                    inertias.append(km_.inertia_)
                    sils.append(silhouette_score(features_scaled, lbl_))
                    dbs.append(davies_bouldin_score(features_scaled, lbl_))

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                         marker=dict(color='#3b9eff', size=8), line=dict(color='#3b9eff', width=2)))
                fig.add_vline(x=k_clusters, line_dash='dash', line_color='#f59e0b',
                              annotation_text=f'k={k_clusters}', annotation_font_color='#f59e0b')
                dark_layout(fig, 'Elbow Method (WCSS Inertia)')
                fig.update_layout(xaxis_title='k', yaxis_title='Inertia')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=list(k_range), y=sils, mode='lines+markers', name='Silhouette',
                                          line=dict(color='#22c55e', width=2), marker=dict(size=8)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=list(k_range), y=dbs, mode='lines+markers', name='Davies-Bouldin',
                                          line=dict(color='#ef4444', width=2), marker=dict(size=8, symbol='square')), secondary_y=True)
                fig2.add_vline(x=k_clusters, line_dash='dash', line_color='#f59e0b')
                dark_layout(fig2, 'Silhouette & Davies-Bouldin')
                st.plotly_chart(fig2, use_container_width=True)

            best_k = list(k_range)[np.argmax(sils)]
            st.markdown(f"""
            <div class='info-box'>
            📌 Optimal k by Silhouette = <strong>{best_k}</strong> |
            Selected k = <strong>{k_clusters}</strong> |
            Silhouette @ k: <strong>{sils[k_clusters-2]:.3f}</strong> |
            Davies-Bouldin: <strong>{dbs[k_clusters-2]:.3f}</strong>
            </div>""", unsafe_allow_html=True)

        with tab2:
            km_labels, km_sil, km_db, km_model = run_kmeans(features_scaled, k_clusters)
            scored['KM_Cluster'] = km_labels
            scored['KM_Label']   = scored['KM_Cluster'].apply(lambda x: f'Cluster {x}')

            # ── Summary table + pie ────────────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                cs = scored.groupby('KM_Label').agg(
                    Count        = ('CustomerID',         'count'),
                    Avg_Age      = ('Age',                'mean'),
                    Avg_Income   = ('Annual Income (k$)', 'mean'),
                    Avg_Spending = ('Spending Score',     'mean'),
                ).round(1).reset_index()
                st.dataframe(cs, use_container_width=True)
            with c2:
                fig = px.pie(cs, names='KM_Label', values='Count',
                             color_discrete_sequence=CLUSTER_PALETTE, hole=0.4)
                dark_layout(fig, 'Cluster Sizes')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""<div class='info-box'>
            Silhouette: <strong>{km_sil:.3f}</strong> &nbsp;|&nbsp;
            Davies-Bouldin: <strong>{km_db:.3f}</strong> &nbsp;|&nbsp;
            k = <strong>{k_clusters}</strong>
            </div>""", unsafe_allow_html=True)

            # ── Two 2-D feature-axis scatter plots ─────────────────
            st.markdown("<div class='section-header'>📍 Cluster Visualizations</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig_inc_sp = px.scatter(
                    scored, x='Annual Income (k$)', y='Spending Score',
                    color='KM_Label',
                    color_discrete_sequence=CLUSTER_PALETTE,
                    opacity=0.82,
                    labels={'KM_Label': 'Cluster',
                            'Spending Score': 'Spending Score (1-100)'},
                    hover_data=['CustomerID', 'Gender', 'Age']
                )
                dark_layout(fig_inc_sp, 'Annual Income vs Spending Score', height=400)
                fig_inc_sp.update_traces(
                    marker=dict(size=7, line=dict(width=0.4,
                                color='rgba(0,0,0,0.3)')))
                st.plotly_chart(fig_inc_sp, use_container_width=True)

            with c2:
                fig_age_inc = px.scatter(
                    scored, x='Age', y='Annual Income (k$)',
                    color='KM_Label',
                    color_discrete_sequence=CLUSTER_PALETTE,
                    opacity=0.82,
                    labels={'KM_Label': 'Cluster'},
                    hover_data=['CustomerID', 'Gender', 'Spending Score']
                )
                dark_layout(fig_age_inc, 'Age vs Annual Income', height=400)
                fig_age_inc.update_traces(
                    marker=dict(size=7, line=dict(width=0.4,
                                color='rgba(0,0,0,0.3)')))
                st.plotly_chart(fig_age_inc, use_container_width=True)

            # ── Full 3-D cluster view ──────────────────────────────
            fig_3d = px.scatter_3d(
                scored, x='Age', y='Annual Income (k$)', z='Spending Score',
                color='KM_Label',
                color_discrete_sequence=CLUSTER_PALETTE,
                opacity=0.78,
                labels={'KM_Label': 'Cluster'},
                hover_data=['CustomerID', 'Gender']
            )
            fig_3d.update_traces(marker=dict(size=4))
            fig_3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                title='3D Cluster View — Age / Annual Income / Spending Score',
                title_font=dict(family='Syne', size=14, color='#90cdf4'),
                scene=dict(
                    xaxis=dict(title='Age',
                               backgroundcolor='rgba(0,0,0,0)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True),
                    yaxis=dict(title='Annual Income (k$)',
                               backgroundcolor='rgba(0,0,0,0)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True),
                    zaxis=dict(title='Spending Score',
                               backgroundcolor='rgba(0,0,0,0)',
                               gridcolor='rgba(255,255,255,0.08)',
                               showbackground=True),
                    bgcolor='rgba(8,12,20,0.95)',
                ),
                height=570,
                margin=dict(t=50, b=5, l=5, r=5),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11))
            )
            st.plotly_chart(fig_3d, use_container_width=True)


        with tab3:
            km_labels2, _, _, _ = run_kmeans(features_scaled, k_clusters)
            scored['KM_Cluster'] = km_labels2
            metrics = ['Age', 'Annual Income (k$)', 'Spending Score']
            fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
            for i, m in enumerate(metrics, 1):
                for c_id in sorted(scored['KM_Cluster'].unique()):
                    vals = scored[scored['KM_Cluster'] == c_id][m]
                    fig.add_trace(go.Box(y=vals, name=f'C{c_id}', marker_color=CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)],
                                         showlegend=(i == 1)), row=1, col=i)
            dark_layout(fig, 'Feature Distribution per Cluster', height=400)
            st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: HIERARCHICAL
    # ══════════════════════════════════════════
    elif page == "🌿 Hierarchical Clustering":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Hierarchical Clustering</h1><p class='hero-sub'>Agglomerative bottom-up clustering with dendrogram</p></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🌳 Dendrogram", "📊 Results", "📈 vs K-Means"])

        with tab1:
            with st.spinner("Building dendrogram..."):
                idx = np.random.choice(len(features_scaled), min(300, len(features_scaled)), replace=False)
                Z   = linkage(features_scaled[idx], method=hc_link)
            fig_d, ax = plt.subplots(figsize=(14, 5))
            fig_d.patch.set_facecolor('#0d1520')
            ax.set_facecolor('#0d1520')
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
                       color_threshold=0.7 * max(Z[:, 2]),
                       above_threshold_color='#475569', leaf_rotation=90, leaf_font_size=8)
            ax.set_title(f'Dendrogram (linkage={hc_link})', color='#90cdf4', fontsize=13, pad=10)
            ax.set_xlabel('Sample Index', color='#64748b')
            ax.set_ylabel('Distance',     color='#64748b')
            ax.tick_params(colors='#64748b')
            for sp in ax.spines.values():
                sp.set_edgecolor('#1e293b')
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close()

        with tab2:
            hc_labels, hc_sil, hc_db = run_hierarchical(features_scaled, hc_k, hc_link)
            scored['HC_Cluster'] = hc_labels
            scored['HC_Label']   = scored['HC_Cluster'].apply(lambda x: f'Cluster {x}')

            hcs = scored.groupby('HC_Label').agg(
                Count        = ('CustomerID',            'count'),
                Avg_Age      = ('Age',                   'mean'),
                Avg_Income   = ('Annual Income (k$)',    'mean'),
                Avg_Spending = ('Spending Score',        'mean'),
            ).round(1).reset_index()

            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(hcs, use_container_width=True)
                st.markdown(f"""<div class='info-box'>
                Silhouette: <strong>{hc_sil:.3f}</strong> | Davies-Bouldin: <strong>{hc_db:.3f}</strong>
                </div>""", unsafe_allow_html=True)
            with c2:
                fig = px.bar(hcs, x='HC_Label', y='Count',
                             color='HC_Label', color_discrete_sequence=CLUSTER_PALETTE)
                dark_layout(fig, 'Hierarchical Cluster Sizes')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            pca_2d = PCA(n_components=2, random_state=42)
            c2d = pca_2d.fit_transform(features_scaled)
            scored['PCA1'], scored['PCA2'] = c2d[:, 0], c2d[:, 1]
            fig = px.scatter(scored.sample(min(1000, len(scored)), random_state=2),
                             x='PCA1', y='PCA2', color='HC_Label',
                             color_discrete_sequence=CLUSTER_PALETTE, opacity=0.7)
            dark_layout(fig, f'Hierarchical Clusters in PCA Space (k={hc_k})')
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            km_l, _, _, _ = run_kmeans(features_scaled, k_clusters)
            hc_l, _, _    = run_hierarchical(features_scaled, hc_k, hc_link)
            scored['KM_L'] = [f'KM-{x}' for x in km_l]
            scored['HC_L'] = [f'HC-{x}' for x in hc_l]
            overlap = pd.crosstab(scored['HC_L'], scored['KM_L'])
            fig = px.imshow(overlap, color_continuous_scale='Blues',
                            labels=dict(x='K-Means', y='Hierarchical', color='Count'))
            dark_layout(fig, 'K-Means vs Hierarchical Overlap')
            st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: PCA
    # ══════════════════════════════════════════
    elif page == "🔮 PCA & Dimensionality":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>PCA & Dimensionality Reduction</h1><p class='hero-sub'>Principal Component Analysis on Age · Income · Spending Score</p></div>", unsafe_allow_html=True)

        pca_components, pca_variance = run_pca(features_scaled, pca_comp)
        km_labels_pca, _, _, _ = run_kmeans(features_scaled, k_clusters)
        scored['KM_Cluster_PCA'] = km_labels_pca.astype(str)

        tab1, tab2, tab3 = st.tabs(["📊 Variance Explained", "🗺️ 2D & 3D Plots", "🔬 Loadings"])

        with tab1:
            var_df = pd.DataFrame({
                'Component':  [f'PC{i+1}' for i in range(pca_comp)],
                'Variance':   pca_variance,
                'Cumulative': np.cumsum(pca_variance)
            })
            c1, c2 = st.columns(2)
            with c1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=var_df['Component'], y=var_df['Variance'],
                                     name='Individual', marker_color='#3b9eff'), secondary_y=False)
                fig.add_trace(go.Scatter(x=var_df['Component'], y=var_df['Cumulative'],
                                         mode='lines+markers', name='Cumulative',
                                         line=dict(color='#f59e0b', width=2)), secondary_y=True)
                dark_layout(fig, 'Explained Variance')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(var_df.style.format({'Variance': '{:.1%}', 'Cumulative': '{:.1%}'}), use_container_width=True)
                st.markdown(f"""<div class='info-box'>
                First <strong>2 PCs</strong>: <strong>{pca_variance[:2].sum():.1%}</strong> variance explained<br>
                First <strong>3 PCs</strong>: <strong>{pca_variance[:3].sum():.1%}</strong> variance explained
                </div>""", unsafe_allow_html=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(x=pca_components[:, 0], y=pca_components[:, 1],
                                 color=scored['KM_Cluster_PCA'],
                                 color_discrete_sequence=CLUSTER_PALETTE,
                                 opacity=0.7, labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'})
                dark_layout(fig, 'PC1 vs PC2 — K-Means')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(x=pca_components[:, 0], y=pca_components[:, 1],
                                 color=scored['Segment'],
                                 color_discrete_map=SEG_COLORS,
                                 opacity=0.7, labels={'x': 'PC1', 'y': 'PC2', 'color': 'Segment'})
                dark_layout(fig, 'PC1 vs PC2 — Segments')
                st.plotly_chart(fig, use_container_width=True)
            if pca_comp >= 3:
                fig3d = px.scatter_3d(
                    x=pca_components[:, 0], y=pca_components[:, 1], z=pca_components[:, 2],
                    color=scored['KM_Cluster_PCA'], color_discrete_sequence=CLUSTER_PALETTE,
                    opacity=0.7, labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Cluster'}
                )
                fig3d.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                    title='3D PCA Space', title_font=dict(family='Syne', size=14, color='#90cdf4'),
                    scene=dict(
                        xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                        zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                    ), height=480, margin=dict(t=45, b=5, l=5, r=5))
                st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            pca_full = PCA(n_components=pca_comp, random_state=42)
            pca_full.fit(features_scaled)
            loadings = pd.DataFrame(
                pca_full.components_.T,
                columns=[f'PC{i+1}' for i in range(pca_comp)],
                index=FEAT_COLS
            )
            fig = px.imshow(loadings, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, text_auto=True)
            dark_layout(fig, 'PCA Feature Loadings')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='info-box'>
            <strong>Reading Loadings:</strong> Large positive values indicate the feature strongly influences that PC. 
            Large negative values mean it influences the PC in the opposite direction.
            Features with loadings near 0 contribute little to that component.
            </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════
    #  PAGE: VALUE ANALYSIS  (replaces CLV)
    # ══════════════════════════════════════════
    elif page == "💎 Value Analysis":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Customer Value Analysis</h1><p class='hero-sub'>Value Score = Annual Income × Spending Rate · Bronze · Silver · Gold · Platinum tiers</p></div>", unsafe_allow_html=True)

        with st.spinner("Computing value scores..."):
            val_df = compute_value(scored)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Value Score",        f"{val_df['Value_Score'].mean():.1f}/100")
        c2.metric("Avg Est. Annual Value",  f"${val_df['Est_Annual_Value'].mean():.1f}k")
        c3.metric("Top 10% Value Score",    f"{val_df['Value_Score'].quantile(0.9):.1f}")
        c4.metric("Platinum Customers",     f"{(val_df['Value_Tier']=='Platinum').sum():,}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🏅 Tier Analysis", "🔗 Value vs Demographics", "📋 Customer Value Table"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(val_df, x='Value_Score', nbins=40, color_discrete_sequence=['#3b9eff'])
                dark_layout(fig, 'Value Score Distribution (0–100)')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                tier_rev = val_df.groupby('Value_Tier', observed=True).agg(
                    Count         = ('CustomerID',        'count'),
                    Total_Value   = ('Est_Annual_Value',  'sum')
                ).reset_index()
                fig = px.bar(tier_rev, x='Value_Tier', y='Total_Value',
                             color='Value_Tier', color_discrete_map=VALUE_COLORS)
                dark_layout(fig, 'Total Estimated Annual Value by Tier')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Value Lorenz curve
            sorted_v = np.sort(val_df['Value_Score'].values)
            cum_share = np.cumsum(sorted_v) / sorted_v.sum()
            pop_share = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pop_share * 100, y=cum_share * 100, mode='lines',
                                     name='Value Concentration', line=dict(color='#3b9eff', width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Perfect Equality',
                                     line=dict(color='#475569', dash='dash', width=1.5)))
            dark_layout(fig, 'Value Lorenz Curve — Customer Value Concentration')
            fig.update_layout(xaxis_title='% Customers', yaxis_title='% Total Value')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            tier_stats = val_df.groupby('Value_Tier', observed=True).agg(
                Customers        = ('CustomerID',          'count'),
                Avg_Value_Score  = ('Value_Score',         'mean'),
                Avg_Annual_Value = ('Est_Annual_Value',    'mean'),
                Avg_Income       = ('Annual Income (k$)', 'mean'),
                Avg_Spending     = ('Spending Score',     'mean'),
                Avg_Age          = ('Age',                'mean'),
            ).round(2).reset_index()
            st.dataframe(tier_stats, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(tier_stats, names='Value_Tier', values='Customers',
                             color='Value_Tier', color_discrete_map=VALUE_COLORS, hole=0.45)
                dark_layout(fig, 'Customer Count by Value Tier')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(tier_stats, x='Value_Tier', y='Avg_Value_Score',
                             color='Value_Tier', color_discrete_map=VALUE_COLORS)
                dark_layout(fig, 'Average Value Score by Tier')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(val_df.sample(min(800, len(val_df)), random_state=7),
                                 x='Annual Income (k$)', y='Value_Score',
                                 color='Value_Tier', color_discrete_map=VALUE_COLORS,
                                 opacity=0.65, hover_data=['CustomerID', 'Spending Score'])
                dark_layout(fig, 'Income vs Value Score (coloured by Tier)')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(val_df.sample(min(800, len(val_df)), random_state=8),
                                 x='Age', y='Value_Score',
                                 color='Segment', color_discrete_map=SEG_COLORS, opacity=0.65)
                dark_layout(fig, 'Age vs Value Score by Segment')
                st.plotly_chart(fig, use_container_width=True)

            fig = px.box(val_df, x='Segment', y='Value_Score',
                         color='Segment', color_discrete_map=SEG_COLORS)
            dark_layout(fig, 'Value Score Distribution by Segment', height=380)
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            tier_filter = st.selectbox("Filter by Value Tier", ['All', 'Bronze', 'Silver', 'Gold', 'Platinum'])
            disp = val_df if tier_filter == 'All' else val_df[val_df['Value_Tier'] == tier_filter]
            cols_show = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score',
                         'Segment', 'Value_Tier', 'Value_Score', 'Est_Annual_Value', 'Est_2yr_Value']
            st.dataframe(disp[cols_show].sort_values('Value_Score', ascending=False).reset_index(drop=True).head(500),
                         use_container_width=True, height=400)
            st.download_button("⬇️ Export Value Data", disp[cols_show].to_csv(index=False).encode(),
                               "customer_value.csv", "text/csv")


    # ══════════════════════════════════════════
    #  PAGE: SPENDING PREDICTION  (replaces Churn)
    # ══════════════════════════════════════════
    elif page == "⚠️ Spending Prediction":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Spending Potential Prediction</h1><p class='hero-sub'>Gradient Boosting classifier · Predict High vs Low Spending Potential from demographics</p></div>", unsafe_allow_html=True)

        with st.expander("📖 Model Methodology", expanded=False):
            st.markdown(f"""
**Target Definition**: A customer is labelled **High Spender** if their Spending Score ≥ {scored['Spending Score'].median():.0f} (dataset median).

**Features used**: Age, Annual Income, Gender (3 demographic features)

**Model**: Gradient Boosting Classifier (100 estimators, max_depth=3)

**Evaluation**: ROC/AUC on 20% holdout test set

**Business use**: Identify which demographic profiles are most likely to engage with high-spend campaigns.
            """)

        with st.spinner("Training model..."):
            pred_df, fpr, tpr, auc, feat_imp = predict_spending(scored)

        high_rate   = pred_df['High_Spender'].mean()
        high_pot    = (pred_df['Potential'] == 'High').sum()
        med_pot     = (pred_df['Potential'] == 'Medium').sum()
        avg_prob    = pred_df['Spend_Prob'].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Spender Rate",      f"{high_rate:.1%}")
        c2.metric("High Potential Customers", f"{high_pot:,}")
        c3.metric("Medium Potential",         f"{med_pot:,}")
        c4.metric("Avg Spend Probability",    f"{avg_prob:.1%}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Potential Distribution", "📈 ROC Curve", "🔍 Feature Importance", "📋 Customer Potential Table"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                pot_counts = pred_df['Potential'].value_counts().reset_index()
                pot_counts.columns = ['Potential', 'Count']
                fig = px.pie(pot_counts, names='Potential', values='Count',
                             color='Potential', color_discrete_map=POTENTIAL_COLORS, hole=0.45)
                dark_layout(fig, 'Spending Potential Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(pred_df, x='Spend_Prob', nbins=35,
                                   color_discrete_sequence=['#3b9eff'])
                dark_layout(fig, 'Spend Probability Distribution')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Potential rate by segment
            seg_pot = pred_df.groupby('Segment').agg(
                High_Rate = ('High_Spender', 'mean'),
                Count     = ('CustomerID',   'count')
            ).reset_index().sort_values('High_Rate', ascending=True)
            seg_pot['High_Rate_Pct'] = (seg_pot['High_Rate'] * 100).round(1)

            fig = px.bar(seg_pot, x='High_Rate_Pct', y='Segment', orientation='h',
                         color='High_Rate', color_continuous_scale='Blues')
            dark_layout(fig, 'High Spend Rate by Segment', height=320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.3f})',
                                     line=dict(color='#3b9eff', width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                                     line=dict(color='#475569', dash='dash', width=1.5)))
            fig.add_annotation(x=0.7, y=0.3, text=f'AUC = {auc:.3f}',
                                font=dict(size=16, color='#3b9eff', family='Syne'),
                                showarrow=False,
                                bgcolor='rgba(59,158,255,0.1)',
                                bordercolor='rgba(59,158,255,0.3)', borderwidth=1, borderpad=8)
            dark_layout(fig, 'ROC Curve — Gradient Boosting Spending Potential Classifier')
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div class='{"success-box" if auc > 0.65 else "warning-box"}'>
            Model AUC = <strong>{auc:.3f}</strong> —
            {'Good discriminative power — the model meaningfully captures spending patterns from demographics.' if auc > 0.65 else 'Moderate performance — spending behaviour may not be strongly determined by demographics alone. Consider adding more features.'}
            An AUC of 1.0 means perfect prediction; 0.5 means no better than random.
            </div>""", unsafe_allow_html=True)

        with tab3:
            fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues')
            dark_layout(fig, 'Feature Importance — Gradient Boosting')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class='info-box'>
            <strong>Interpretation:</strong> Features with higher importance contribute more to predicting spending potential.
            Annual Income tends to dominate as it directly reflects purchasing power, while Age captures life-stage spending patterns.
            </div>""", unsafe_allow_html=True)

        with tab4:
            pot_filter = st.selectbox("Filter by Potential Level", ['High', 'Medium', 'Low', 'All'])
            disp = pred_df if pot_filter == 'All' else pred_df[pred_df['Potential'] == pot_filter]
            disp_sorted = disp.sort_values('Spend_Prob', ascending=(pot_filter == 'Low')).reset_index(drop=True)
            show_cols = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score',
                         'Segment', 'Potential', 'Spend_Prob']
            st.markdown(f"**{len(disp_sorted):,} customers in '{pot_filter}' potential category**")
            st.dataframe(disp_sorted[show_cols].head(300).style.format({'Spend_Prob': '{:.1%}'}),
                         use_container_width=True, height=380)
            st.download_button("⬇️ Export Potential List", disp_sorted[show_cols].to_csv(index=False).encode(),
                               f"potential_{pot_filter.lower()}.csv", "text/csv")


    # ══════════════════════════════════════════
    #  PAGE: CUSTOMER PERSONAS  (replaces Product Affinity)
    # ══════════════════════════════════════════
    elif page == "👥 Customer Personas":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Customer Persona Analysis</h1><p class='hero-sub'>Demographic deep-dive · Age × Income × Gender spending heatmaps</p></div>", unsafe_allow_html=True)

        with st.spinner("Analysing personas..."):
            age_income_hm, gender_stats, age_stats = compute_personas(scored)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers",   f"{len(scored):,}")
        c2.metric("Age Groups",         len(scored['Age_Group'].unique()))
        c3.metric("Income Tiers",       3)

        tab1, tab2, tab3 = st.tabs(["🔥 Income × Age Heatmap", "⚧ Gender Analysis", "📅 Age Group Breakdown"])

        with tab1:
            fig = px.imshow(
                age_income_hm,
                color_continuous_scale='RdYlGn',
                labels=dict(x='Income Tier', y='Age Group', color='Avg Spending Score'),
                text_auto=True
            )
            dark_layout(fig, 'Average Spending Score by Age Group × Income Tier', height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='info-box'>
            <strong>How to read:</strong> Darker green = higher average spending score in that demographic cell.
            Identify the Age × Income combinations most likely to spend heavily — these are prime targeting groups.
            </div>""", unsafe_allow_html=True)

            # Income Tier × Spending Score scatter
            fig2 = px.box(scored, x='Income_Tier', y='Spending Score',
                          color='Income_Tier',
                          color_discrete_sequence=['#3b9eff', '#22c55e', '#f59e0b'],
                          category_orders={'Income_Tier': ['Low Income', 'Mid Income', 'High Income']})
            dark_layout(fig2, 'Spending Score Distribution by Income Tier')
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.dataframe(gender_stats, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(gender_stats, x='Gender', y='Avg_Spending',
                             color='Gender', color_discrete_map=GENDER_COLORS)
                dark_layout(fig, 'Avg Spending Score by Gender')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(gender_stats, x='Gender', y='Avg_Income',
                             color='Gender', color_discrete_map=GENDER_COLORS)
                dark_layout(fig, 'Avg Annual Income by Gender')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Gender × Age Group spending heatmap
            g_age = scored.groupby(['Gender', 'Age_Group'], observed=True)['Spending Score'].mean().round(1).unstack(fill_value=0)
            fig3 = px.imshow(g_age, color_continuous_scale='Blues', text_auto=True,
                             labels=dict(x='Age Group', y='Gender', color='Avg Spending Score'))
            dark_layout(fig3, 'Spending Score by Gender × Age Group', height=300)
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.dataframe(age_stats, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(age_stats, x='Age_Group', y='Avg_Spending',
                             color='Age_Group', color_discrete_sequence=px.colors.qualitative.Bold)
                dark_layout(fig, 'Avg Spending Score by Age Group')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(age_stats, x='Age_Group', y='Avg_Income',
                             color='Age_Group', color_discrete_sequence=px.colors.qualitative.Bold)
                dark_layout(fig, 'Avg Annual Income by Age Group')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(scored, x='Age', y='Spending Score',
                             color='Income_Tier',
                             color_discrete_sequence=['#3b9eff', '#22c55e', '#f59e0b'],
                             opacity=0.6, trendline='ols',
                             category_orders={'Income_Tier': ['Low Income', 'Mid Income', 'High Income']})
            dark_layout(fig, 'Age vs Spending Score (with OLS trendlines per Income Tier)')
            st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: AI SEGMENT INSIGHTS
    # ══════════════════════════════════════════
    elif page == "🤖 AI Segment Insights":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>AI Segment Insights</h1><p class='hero-sub'>Automated intelligence reports & strategic recommendations per segment</p></div>", unsafe_allow_html=True)

        seg_actions = {
            'Premium Shoppers': {
                'icon': '🥇', 'color': '#f59e0b',
                'desc': 'High-income, high-spending — your most valuable customers. They spend proportional to their means.',
                'actions': ['Launch exclusive VIP loyalty programme with premium rewards', 'Early access to new product lines & limited editions', 'Personalised concierge-level service & dedicated account managers', 'Referral incentives — they become brand ambassadors', 'Premium membership upsell (annual subscription model)'],
                'risks': ['Risk of churn if service quality drops or a competitor offers better perks', 'Price-sensitive to abrupt tier changes'],
                'kpi_targets': 'Target: 90%+ retention | Grow Avg Spend by 15% YoY | NPS > 70'
            },
            'Engaged Customers': {
                'icon': '💚', 'color': '#22c55e',
                'desc': 'Solid mid-to-high income with consistent spending. A reliable revenue base.',
                'actions': ['Tiered loyalty rewards (points per $ spent)', 'Upsell higher-margin product categories', 'Bundle recommendations based on past purchases', 'Quarterly "thank you" discounts (10–15%)', 'Cross-sell to adjacent product lines'],
                'risks': ['May plateau if not incentivised to increase spend', 'Vulnerable to competitor loyalty programmes'],
                'kpi_targets': 'Target: Increase Avg Spending Score by 10 pts | Grow revenue contribution by 20%'
            },
            'Wealthy Resistors': {
                'icon': '💼', 'color': '#3b9eff',
                'desc': 'Affluent customers who spend conservatively. High potential that remains untapped.',
                'actions': ['Targeted premium product showcases — emphasise exclusivity & quality', 'Personalised, non-intrusive outreach (email / direct mail)', 'In-store events or private previews to build brand affinity', 'Survey to understand purchase barriers', 'Limited-time access offers that create urgency without discounting heavily'],
                'risks': ['Difficult to convert — they are selective by nature', 'Aggressive promotions may feel off-brand and backfire'],
                'kpi_targets': 'Target: Move 20% to Engaged Customers tier within 6 months'
            },
            'Young Enthusiasts': {
                'icon': '🌟', 'color': '#a855f7',
                'desc': 'Young customers with high enthusiasm and high spending relative to their income.',
                'actions': ['Social media-first campaigns (TikTok, Instagram)', 'Trend-based and experiential product offerings', 'Student / young professional loyalty programme', 'Gamification of rewards & purchases', 'Influencer partnerships targeting their demographic'],
                'risks': ['Spending may decline as financial responsibilities grow with age', 'High sensitivity to brand perception and social trends'],
                'kpi_targets': 'Target: Retain 70% as they enter higher income brackets | Increase AOV by 12%'
            },
            'Aspirational Spenders': {
                'icon': '🚀', 'color': '#ef4444',
                'desc': 'Low-to-mid income but high spending. Highly engaged — financially stretched.',
                'actions': ['Installment / buy-now-pay-later payment options', 'Budget-friendly product lines & value bundles', 'Loyalty rewards that reduce effective price over time', 'Flash sales and exclusive member discounts', 'Avoid upselling premium tiers that are financially unrealistic'],
                'risks': ['Financial constraints may force abrupt spending cuts', 'High churn risk during economic downturns'],
                'kpi_targets': 'Target: Maintain spend level | Reduce churn risk with loyalty lock-in'
            },
            'Budget Conscious': {
                'icon': '💰', 'color': '#64748b',
                'desc': 'Low income, low spending. Price-sensitive, minimal engagement.',
                'actions': ['Heavy discount campaigns and clearance offers', 'Value product ranges and own-brand alternatives', 'Basic loyalty scheme (punch-card style)', 'Low-cost reactivation emails', 'Use feedback to improve product accessibility'],
                'risks': ['Low ROI from marketing; only respond to significant discounts', 'May inflate promotion costs without proportional revenue uplift'],
                'kpi_targets': 'Target: Reactivate 10% per quarter | Use data to improve product range accessibility'
            },
            'Selective Buyers': {
                'icon': '🎯', 'color': '#ec4899',
                'desc': 'Mid-to-high income but low spending. Have the means — choosing not to spend here.',
                'actions': ['Understand competitor preferences through exit surveys', 'Premium product demonstrations and experience events', 'Exclusivity appeal — limited edition items only available to select customers', 'Personalised "we noticed you like X" campaign', 'Remove friction from purchase journey'],
                'risks': ['Very selective — only respond to perfectly targeted messaging', 'May be loyal to a competitor brand'],
                'kpi_targets': 'Target: Move 25% to Engaged Customers within 90 days'
            },
            'Moderate Consumers': {
                'icon': '📊', 'color': '#06b6d4',
                'desc': 'Average across all dimensions. Reliable baseline customers with growth potential.',
                'actions': ['Consistent engagement via email & app notifications', 'Seasonal promotions aligned with spending cycles', 'Gradual upselling to higher-value categories', 'Personalised product discovery features', 'Segment further by age / income for more targeted campaigns'],
                'risks': ['Easily lost to competitors without active engagement', 'No strong brand loyalty; price comparisons common'],
                'kpi_targets': 'Target: Move 15% to Engaged Customers per quarter | Increase visit frequency by 20%'
            },
        }

        st.markdown("<div class='section-header'>📊 Segment Intelligence Overview</div>", unsafe_allow_html=True)

        seg_stats = scored.groupby('Segment').agg(
            Count        = ('CustomerID',            'count'),
            Avg_Income   = ('Annual Income (k$)',    'mean'),
            Avg_Spending = ('Spending Score',        'mean'),
            Avg_Age      = ('Age',                   'mean'),
        ).round(1).reset_index()

        for _, row in seg_stats.iterrows():
            s    = row['Segment']
            info = seg_actions.get(s, {})
            color = info.get('color', '#3b9eff')
            icon  = info.get('icon',  '📌')

            with st.expander(f"{icon} {s}  —  {row['Count']:,} customers  ·  Avg Spending {row['Avg_Spending']:.0f}/100", expanded=False):
                c1, c2, c3 = st.columns([2, 1.5, 1.5])
                with c1:
                    st.markdown(f"<div style='color:{color}; font-size:0.9rem; margin-bottom:0.8rem;'>{info.get('desc','')}</div>", unsafe_allow_html=True)
                    st.markdown("**🎯 Recommended Actions:**")
                    for act in info.get('actions', []):
                        st.markdown(f"• {act}")
                with c2:
                    st.metric("Avg Age",          f"{row['Avg_Age']:.0f} yrs")
                    st.metric("Avg Income",        f"${row['Avg_Income']:.1f}k")
                    st.metric("Avg Spend Score",   f"{row['Avg_Spending']:.0f}/100")
                with c3:
                    st.markdown("**⚠️ Key Risks:**")
                    for risk in info.get('risks', []):
                        st.markdown(f"• {risk}")
                    st.markdown(f"<div class='info-box' style='margin-top:0.8rem; border-color:{color}44;'><strong>KPI Targets</strong><br>{info.get('kpi_targets','')}</div>", unsafe_allow_html=True)

        # Priority action matrix
        st.markdown("<div class='section-header'>🎯 Priority Action Matrix</div>", unsafe_allow_html=True)
        action_matrix = pd.DataFrame({
            'Segment':       list(seg_actions.keys()),
            'Priority':      ['High', 'High', 'Medium', 'High', 'Critical', 'Low', 'Medium', 'Medium'],
            'Action_Type':   ['Retain', 'Upsell', 'Convert', 'Engage', 'Protect', 'Reactivate', 'Win-Over', 'Nurture'],
            'ROI_Potential': ['Very High', 'High', 'Very High', 'High', 'Medium', 'Low', 'High', 'Medium'],
        })
        seg_count_map = scored['Segment'].value_counts().to_dict()
        action_matrix['Customer_Count'] = action_matrix['Segment'].map(seg_count_map).fillna(0)

        fig = px.scatter(action_matrix, x='Action_Type', y='Priority',
                         size='Customer_Count', color='ROI_Potential', text='Segment',
                         color_discrete_map={'Very High': '#f59e0b', 'High': '#22c55e', 'Medium': '#3b9eff', 'Low': '#64748b'},
                         size_max=60)
        dark_layout(fig, 'Action Priority Matrix (bubble size = customer count)', height=380)
        fig.update_traces(textposition='top center', textfont_size=9)
        st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: MODEL COMPARISON
    # ══════════════════════════════════════════
    elif page == "📊 Model Comparison":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Model Comparison</h1><p class='hero-sub'>Objective evaluation — K-Means vs Hierarchical Clustering</p></div>", unsafe_allow_html=True)

        km_labels, km_sil, km_db, _ = run_kmeans(features_scaled, k_clusters)
        hc_labels, hc_sil, hc_db    = run_hierarchical(features_scaled, hc_k, hc_link)
        scored['KM_Cluster'] = km_labels
        scored['HC_Cluster'] = hc_labels

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("K-Means Silhouette",     f"{km_sil:.3f}")
        c2.metric("K-Means Davies-Bouldin", f"{km_db:.3f}")
        c3.metric("HC Silhouette",          f"{hc_sil:.3f}")
        c4.metric("HC Davies-Bouldin",      f"{hc_db:.3f}")

        comp_df = pd.DataFrame({
            'Method':               ['K-Means', 'Hierarchical'],
            'Silhouette (↑)':       [km_sil, hc_sil],
            'Davies-Bouldin (↓)':   [km_db,  hc_db],
            'k':                    [k_clusters, hc_k]
        })

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(comp_df, x='Method', y='Silhouette (↑)', color='Method',
                         color_discrete_sequence=['#3b9eff', '#22c55e'])
            dark_layout(fig, 'Silhouette Score (higher = better)')
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(comp_df, x='Method', y='Davies-Bouldin (↓)', color='Method',
                          color_discrete_sequence=['#ef4444', '#f59e0b'])
            dark_layout(fig2, 'Davies-Bouldin (lower = better)')
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        winner = 'K-Means' if km_sil >= hc_sil else 'Hierarchical Clustering'
        st.markdown(f"""
        <div class='{"success-box" if winner == "K-Means" else "info-box"}'>
        🏆 <strong>Best: {winner}</strong> (by Silhouette Score)<br><br>
        • <strong>Silhouette</strong>: How well each point fits its own cluster vs neighbours. Range [-1,1]. Higher = better separation.<br>
        • <strong>Davies-Bouldin</strong>: Ratio of within-cluster scatter to between-cluster separation. Lower = better.
        </div>""", unsafe_allow_html=True)

        pca_2d  = PCA(n_components=2, random_state=42)
        coords  = pca_2d.fit_transform(features_scaled)
        c1, c2  = st.columns(2)
        with c1:
            fig = px.scatter(x=coords[:, 0], y=coords[:, 1],
                             color=scored['KM_Cluster'].astype(str),
                             color_discrete_sequence=CLUSTER_PALETTE, opacity=0.7,
                             labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'})
            dark_layout(fig, f'K-Means (k={k_clusters}) in PCA Space')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(x=coords[:, 0], y=coords[:, 1],
                              color=scored['HC_Cluster'].astype(str),
                              color_discrete_sequence=CLUSTER_PALETTE, opacity=0.7,
                              labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'})
            dark_layout(fig2, f'Hierarchical (k={hc_k}) in PCA Space')
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(comp_df.set_index('Method'), use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: SEGMENT PROFILER
    # ══════════════════════════════════════════
    elif page == "🎯 Segment Profiler":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Segment Profiler</h1><p class='hero-sub'>Deep dive into any segment — characteristics, demographics & actions</p></div>", unsafe_allow_html=True)

        selected_seg = st.selectbox("Select Segment to Profile", sorted(scored['Segment'].unique()))
        seg_data     = scored[scored['Segment'] == selected_seg]
        color        = SEG_COLORS.get(selected_seg, '#3b9eff')

        seg_actions_map = {
            'Premium Shoppers':      ('🥇', 'Reward & retain. VIP early access, loyalty perks, referral programs.'),
            'Engaged Customers':     ('💚', 'Upsell higher-value products. Offer tiered loyalty rewards.'),
            'Wealthy Resistors':     ('💼', 'Targeted premium showcases. Understand barriers through surveys.'),
            'Young Enthusiasts':     ('🌟', 'Social media campaigns, trend-based products, gamified loyalty.'),
            'Aspirational Spenders': ('🚀', 'Installment plans, value bundles, loyalty to reduce effective price.'),
            'Budget Conscious':      ('💰', 'Discount campaigns, value ranges, low-cost reactivation.'),
            'Selective Buyers':      ('🎯', 'Experience events, exclusivity appeal, frictionless purchase journey.'),
            'Moderate Consumers':    ('📊', 'Consistent engagement, seasonal promotions, gradual upsell.'),
        }
        icon, action = seg_actions_map.get(selected_seg, ('📌', 'Build targeted campaigns.'))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers",        f"{len(seg_data):,}")
        c2.metric("Avg Age",          f"{seg_data['Age'].mean():.0f} yrs")
        c3.metric("Avg Income",       f"${seg_data['Annual Income (k$)'].mean():.1f}k")
        c4.metric("Avg Spend Score",  f"{seg_data['Spending Score'].mean():.1f}/100")

        st.markdown(f"""
        <div class='insight-card' style='border-color:{color}44;'>
            <div class='title' style='color:{color};'>{icon} {selected_seg} — Marketing Action</div>
            <div class='body'>{action}</div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(seg_data, x='Annual Income (k$)', nbins=25, color_discrete_sequence=[color])
            dark_layout(fig, f'{selected_seg} — Income Distribution')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(seg_data, x='Spending Score', nbins=25, color_discrete_sequence=[color])
            dark_layout(fig, f'{selected_seg} — Spending Score Distribution')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart: normalised [Income, Spending, Youth]
        all_avg  = scored.groupby('Segment')[['Annual Income (k$)', 'Spending Score', 'A_Score']].mean()
        all_norm = (all_avg - all_avg.min()) / (all_avg.max() - all_avg.min())
        cats = ['Income', 'Spending', 'Youth']

        fig = go.Figure()
        for s in all_norm.index:
            vals = all_norm.loc[s].tolist() + [all_norm.loc[s].tolist()[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], name=s,
                line_color=SEG_COLORS.get(s, '#aaa'),
                opacity=0.4 if s != selected_seg else 1.0,
                line_width=1 if s != selected_seg else 3,
                fill='toself' if s == selected_seg else 'none',
                fillcolor=f'rgba{tuple(list(bytes.fromhex(SEG_COLORS.get(selected_seg,"#3b9eff")[1:])) + [30])}' if s == selected_seg else 'rgba(0,0,0,0)'
            ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            title='Radar Chart — Normalised Income · Spending · Youth by Segment',
            title_font=dict(family='Syne', size=14, color='#90cdf4'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            margin=dict(t=50, b=10, l=10, r=10), height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Customer List — {selected_seg} ({len(seg_data):,} customers)**")
        show_cols = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score',
                     'I_Score', 'S_Score', 'A_Score', 'Total_Score']
        st.dataframe(
            seg_data[show_cols].sort_values('Total_Score', ascending=False).reset_index(drop=True),
            use_container_width=True, height=320
        )
        st.download_button(
            f"⬇️ Export {selected_seg}",
            seg_data[show_cols].to_csv(index=False).encode(),
            f"segment_{selected_seg.lower().replace(' ', '_')}.csv",
            "text/csv"
        )