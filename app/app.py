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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import hashlib
import json
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
# ─────────────────────────────────────────────
USERS_DB = {
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

def check_password(username, password):
    if username in USERS_DB:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == USERS_DB[username]["password"]
    return False

def register_user(username, password, name):
    if username in USERS_DB:
        return False, "Username already exists"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    USERS_DB[username] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "name": name,
        "role": "Analyst",
        "avatar": "🧑‍💻"
    }
    return True, "Account created successfully!"

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
header {visibility: hidden;}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090e18 0%, #0c1420 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 0.85rem !important;
    padding: 0.3rem 0 !important;
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
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
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
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.75rem; font-weight: 800; color: var(--accent-blue); line-height: 1; }
.kpi-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.35rem; }

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
.stTextInput > div > div > input, .stSelectbox, .stPasswordInput > div > div > input {
    background: rgba(13,21,32,0.8) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 9px !important;
}
.stTextInput > div > div > input:focus, .stPasswordInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,158,255,0.15) !important;
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

/* CHURN RISK BADGES */
.risk-high { color: #f87171; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.risk-med  { color: #fbbf24; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25); padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.risk-low  { color: #4ade80; background: rgba(34,197,94,0.1);  border: 1px solid rgba(34,197,94,0.25);  padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

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
    # Center the auth form
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Logo area
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

        # Toggle tabs
        tab_login, tab_signup = st.tabs(["🔐  Sign In", "✨  Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

            col_btn, col_hint = st.columns([1, 1])
            with col_btn:
                login_btn = st.button("Sign In →", use_container_width=True)

            if login_btn:
                if st.session_state.login_attempts >= 5:
                    st.error("🔒 Too many failed attempts. Please try again later.")
                elif check_password(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.login_attempts = 0
                    st.success(f"✅ Welcome back, {USERS_DB[username]['name']}!")
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
                        st.success(f"✅ {msg} You can now sign in.")
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
#  DATA LOADING & GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def load_or_generate_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        return df, False

    np.random.seed(42)
    n = 5000
    countries = ['United Kingdom']*2800 + ['Germany']*350 + ['France']*300 + \
                ['EIRE']*220 + ['Netherlands']*160 + ['Belgium']*130 + \
                ['Switzerland']*110 + ['Spain']*100 + ['Portugal']*90 + ['Australia']*70 + \
                ['Norway']*60 + ['Sweden']*50 + ['Denmark']*35 + ['Japan']*30 + ['USA']*95
    stock_codes = [f'SC{str(i).zfill(5)}' for i in range(1, 201)]
    descriptions = [
        'WHITE HANGING HEART T-LIGHT HOLDER','WHITE METAL LANTERN',
        'CREAM CUPID HEARTS COAT HANGER','KNITTED UNION FLAG HOT WATER BOTTLE',
        'RED WOOLLY HOTTIE WHITE HEART','SET 7 BABUSHKA NESTING BOXES',
        'GLASS STAR FROSTED T-LIGHT HOLDER','HAND WARMER UNION JACK',
        'HAND WARMER RED POLKA DOT','ASSORTED COLOUR BIRD ORNAMENT',
        'POPPY\'S PLAYHOUSE BEDROOM','LUNCH BAG RED RETROSPOT',
        'PACK OF 72 RETROSPOT CAKE CASES','STRAWBERRY CERAMIC TRINKET BOX',
        'PARTY BUNTING','JUMBO BAG RED RETROSPOT',
        'ROSES REGENCY TEACUP AND SAUCER','GREEN REGENCY TEACUP AND SAUCER',
        'PINK REGENCY TEACUP AND SAUCER','REGENCY CAKESTAND 3 TIER',
    ]
    start = pd.Timestamp('2010-12-01')
    end   = pd.Timestamp('2011-12-09')
    rows = []
    invoice_num = 536365
    for _ in range(n):
        cid = np.random.randint(12346, 18288)
        date = start + pd.Timedelta(days=int(np.random.beta(2,3)*(end-start).days))
        n_items = np.random.randint(1, 8)
        inv = str(invoice_num)
        invoice_num += np.random.randint(1, 5)
        for _ in range(n_items):
            rows.append({
                'InvoiceNo':   inv,
                'StockCode':   np.random.choice(stock_codes),
                'Description': np.random.choice(descriptions),
                'Quantity':    max(1, int(np.random.lognormal(1.5, 1.0))),
                'InvoiceDate': date.strftime('%m/%d/%Y %H:%M'),
                'UnitPrice':   round(max(0.1, np.random.lognormal(1.0, 0.8)), 2),
                'CustomerID':  cid,
                'Country':     np.random.choice(countries)
            })
    df = pd.DataFrame(rows)
    return df, True


@st.cache_data
def preprocess(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'invoice' in cl and 'date' in cl: col_map[c] = 'InvoiceDate'
        elif 'invoice' in cl and 'no' in cl:  col_map[c] = 'InvoiceNo'
        elif 'customer' in cl:                col_map[c] = 'CustomerID'
        elif 'quantity' in cl:                col_map[c] = 'Quantity'
        elif 'price' in cl or 'unitprice' in cl: col_map[c] = 'UnitPrice'
        elif 'country' in cl:                 col_map[c] = 'Country'
        elif 'stock' in cl:                   col_map[c] = 'StockCode'
        elif 'desc' in cl:                    col_map[c] = 'Description'
    df.rename(columns=col_map, inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)
    df['CustomerID']  = pd.to_numeric(df.get('CustomerID', pd.Series()), errors='coerce')
    df['Quantity']    = pd.to_numeric(df.get('Quantity', pd.Series()), errors='coerce')
    df['UnitPrice']   = pd.to_numeric(df.get('UnitPrice', pd.Series()), errors='coerce')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df


@st.cache_data
def build_rfm(df):
    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate',  lambda x: (snapshot - x.max()).days),
        Frequency = ('InvoiceNo',    'nunique'),
        Monetary  = ('TotalPrice',   'sum')
    ).reset_index()
    rfm['R_Score'] = pd.qcut(rfm['Recency'],   5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    def segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4: return 'Champions'
        elif r >= 3 and f >= 3:          return 'Loyal Customers'
        elif r >= 4 and f <= 2:          return 'New Customers'
        elif r >= 3 and f <= 2:          return 'Potential Loyalists'
        elif r == 2 and f >= 2:          return 'At Risk'
        elif r <= 2 and f >= 3:          return 'Cannot Lose Them'
        elif r <= 2 and f <= 2:          return 'Lost'
        else:                             return 'Hibernating'
    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm


@st.cache_data
def compute_clv(df, rfm):
    """Compute Customer Lifetime Value using BG/NBD-inspired simple model."""
    snapshot = df['InvoiceDate'].max()

    # Average order value per customer
    aov = df.groupby('CustomerID')['TotalPrice'].mean().reset_index()
    aov.columns = ['CustomerID', 'AOV']

    # Purchase frequency (orders per month)
    customer_dates = df.groupby('CustomerID')['InvoiceDate'].agg(['min','max','nunique']).reset_index()
    customer_dates.columns = ['CustomerID','first_purchase','last_purchase','total_orders']
    customer_dates['months_active'] = ((customer_dates['last_purchase'] - customer_dates['first_purchase']).dt.days / 30).clip(lower=1)
    customer_dates['purchase_rate'] = customer_dates['total_orders'] / customer_dates['months_active']

    clv_df = rfm.merge(aov, on='CustomerID').merge(customer_dates[['CustomerID','months_active','purchase_rate','first_purchase','last_purchase']], on='CustomerID')

    # CLV = AOV × purchase_rate × predicted_months (12)
    clv_df['CLV_12mo']  = clv_df['AOV'] * clv_df['purchase_rate'] * 12
    clv_df['CLV_24mo']  = clv_df['AOV'] * clv_df['purchase_rate'] * 24

    # CLV tier
    clv_df['CLV_Tier'] = pd.qcut(clv_df['CLV_12mo'], 4, labels=['Bronze','Silver','Gold','Platinum'])

    # Normalize CLV score 0-100
    mn, mx = clv_df['CLV_12mo'].min(), clv_df['CLV_12mo'].max()
    clv_df['CLV_Score'] = ((clv_df['CLV_12mo'] - mn) / (mx - mn) * 100).round(1)
    return clv_df


@st.cache_data
def compute_churn(rfm):
    """Predict churn probability using a simple ML model."""
    # Define churn: Recency > 180 days = churned
    churn_df = rfm.copy()
    churn_df['Churned'] = (churn_df['Recency'] > 180).astype(int)

    features = ['Recency','Frequency','Monetary','R_Score','F_Score','M_Score','RFM_Score']
    X = churn_df[features]
    y = churn_df['Churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_s, y_train)

    X_all_s = scaler.transform(X)
    churn_df['Churn_Prob'] = model.predict_proba(X_all_s)[:,1]
    churn_df['Churn_Risk'] = pd.cut(churn_df['Churn_Prob'], bins=[0,0.33,0.66,1.0], labels=['Low','Medium','High'])

    # ROC on test set
    y_prob_test = model.predict_proba(X_test_s)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    auc = roc_auc_score(y_test, y_prob_test)

    # Feature importances
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)

    return churn_df, fpr, tpr, auc, feat_imp


@st.cache_data
def compute_product_affinity(df, top_n=10):
    """Market basket analysis — which products are bought together."""
    if 'StockCode' not in df.columns or 'Description' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Get top N products by revenue
    top_products = df.groupby('StockCode')['TotalPrice'].sum().nlargest(top_n).index.tolist()
    basket_df = df[df['StockCode'].isin(top_products)]

    # Create basket matrix
    basket = basket_df.groupby(['InvoiceNo','StockCode'])['Quantity'].sum().unstack(fill_value=0)
    basket = (basket > 0).astype(int)

    # Co-occurrence matrix
    co_matrix = basket.T.dot(basket)
    np.fill_diagonal(co_matrix.values, 0)

    # Product names mapping
    prod_names = df[df['StockCode'].isin(top_products)].groupby('StockCode')['Description'].first()
    co_matrix.index = [prod_names.get(c, c)[:25] for c in co_matrix.index]
    co_matrix.columns = [prod_names.get(c, c)[:25] for c in co_matrix.columns]

    # Top pairs
    pairs = []
    for i, p1 in enumerate(co_matrix.columns):
        for p2 in co_matrix.columns[i+1:]:
            pairs.append({'Product A': p1, 'Product B': p2, 'Co-purchases': co_matrix.loc[p1, p2]})
    pairs_df = pd.DataFrame(pairs).sort_values('Co-purchases', ascending=False).head(20)

    return co_matrix, pairs_df


@st.cache_data
def run_kmeans(rfm_scaled, k):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    sil = silhouette_score(rfm_scaled, labels)
    db  = davies_bouldin_score(rfm_scaled, labels)
    return labels, sil, db, km


@st.cache_data
def run_hierarchical(rfm_scaled, k, linkage_method='ward'):
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = hc.fit_predict(rfm_scaled)
    sil = silhouette_score(rfm_scaled, labels)
    db  = davies_bouldin_score(rfm_scaled, labels)
    return labels, sil, db


@st.cache_data
def run_pca(rfm_scaled, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(rfm_scaled)
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
    'Champions':          '#f59e0b',
    'Loyal Customers':    '#22c55e',
    'New Customers':      '#3b9eff',
    'Potential Loyalists':'#a855f7',
    'At Risk':            '#ef4444',
    'Cannot Lose Them':   '#ec4899',
    'Lost':               '#64748b',
    'Hibernating':        '#06b6d4',
}
CLUSTER_PALETTE = px.colors.qualitative.Bold
CLV_COLORS = {'Bronze':'#a16207','Silver':'#94a3b8','Gold':'#ca8a04','Platinum':'#3b9eff'}
CHURN_COLORS = {'Low':'#22c55e','Medium':'#f59e0b','High':'#ef4444'}


# ─────────────────────────────────────────────
#  MAIN APP ROUTER
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    render_auth_page()
else:
    user_info = USERS_DB.get(st.session_state.username, {})

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

        # User badge
        st.markdown(f"""
        <div class='user-badge'>
            <div class='avatar'>{user_info.get('avatar','👤')}</div>
            <div class='info'>
                <div class='name'>{user_info.get('name','User')}</div>
                <div class='role'>{user_info.get('role','Viewer')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📁 Data Source")
        uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'], label_visibility='collapsed')

        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        k_clusters = st.slider("K-Means Clusters", 2, 10, 4)
        hc_k       = st.slider("Hierarchical Clusters", 2, 8, 4)
        hc_link    = st.selectbox("Linkage Method", ['ward','complete','average','single'])
        pca_comp   = st.slider("PCA Components", 2, 5, 3)

        st.markdown("---")
        st.markdown("### 🗺️ Navigation")
        page = st.radio("", [
            "🏠 Overview",
            "🔍 EDA & Data Quality",
            "💡 RFM Analysis",
            "🔵 K-Means Clustering",
            "🌿 Hierarchical Clustering",
            "🔮 PCA & Dimensionality",
            "💎 CLV Analysis",
            "⚠️ Churn Prediction",
            "🛒 Product Affinity",
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

    # ── LOAD DATA ────────────────────────────
    raw_df, is_synthetic = load_or_generate_data(uploaded)
    df  = preprocess(raw_df)
    rfm = build_rfm(df)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

    # ══════════════════════════════════════════
    #  PAGE: OVERVIEW
    # ══════════════════════════════════════════
    if page == "🏠 Overview":
        st.markdown("""
        <div class='hero-banner'>
            <div class='hero-badge'>🎓 ML Course Project — v2.0 with CLV, Churn & Product Intelligence</div>
            <h1 class='hero-title'>Customer Segmentation<br>Intelligence Studio</h1>
            <p class='hero-sub'>RFM · K-Means · Hierarchical · PCA · CLV · Churn Prediction · Product Affinity · AI Insights</p>
        </div>
        """, unsafe_allow_html=True)

        if is_synthetic:
            st.info("⚡ **Demo Mode** — Synthetic e-commerce data loaded. Upload your own CSV in the sidebar.")

        # KPIs
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpis = [
            ("👥", "Customers",    f"{rfm.shape[0]:,}"),
            ("🧾", "Transactions", f"{df['InvoiceNo'].nunique():,}"),
            ("💷", "Revenue",      f"£{df['TotalPrice'].sum():,.0f}"),
            ("📦", "Avg Order",    f"£{df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}"),
            ("🌍", "Countries",    f"{df['Country'].nunique() if 'Country' in df.columns else 'N/A'}"),
            ("📅", "Date Range",   f"{df['InvoiceDate'].min().strftime('%b %y')}–{df['InvoiceDate'].max().strftime('%b %y')}"),
        ]
        for col, (icon, label, value) in zip([c1,c2,c3,c4,c5,c6], kpis):
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
                <div class='body'>K-Means & Hierarchical Clustering on standardised RFM features. PCA for dimensionality reduction and visualisation.</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='insight-card'>
                <div class='title'>🟠 Supervised Learning</div>
                <div class='body'>Rule-based RFM scoring (domain knowledge labels). Gradient Boosting for churn probability prediction with ROC/AUC evaluation.</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='insight-card'>
                <div class='title'>💡 Business Intelligence</div>
                <div class='body'>CLV 12/24-month projection, product affinity (market basket), AI-generated segment insights and marketing actions.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Segment Distribution</div>", unsafe_allow_html=True)
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment','Count']

        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = px.bar(seg_counts, x='Segment', y='Count',
                         color='Segment', color_discrete_map=SEG_COLORS)
            dark_layout(fig, 'Customer Count by RFM Segment')
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(seg_counts, names='Segment', values='Count',
                          color='Segment', color_discrete_map=SEG_COLORS, hole=0.5)
            dark_layout(fig2, 'Segment Share')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-header'>💰 Revenue & RFM Heatmap</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            rev_seg = rfm.groupby('Segment')['Monetary'].sum().reset_index().sort_values('Monetary')
            fig3 = px.bar(rev_seg, x='Monetary', y='Segment', orientation='h',
                          color='Segment', color_discrete_map=SEG_COLORS)
            dark_layout(fig3, 'Revenue Contribution by Segment', height=300)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            pivot = rfm.groupby(['R_Score','F_Score'])['CustomerID'].count().reset_index()
            pivot_wide = pivot.pivot(index='R_Score', columns='F_Score', values='CustomerID').fillna(0)
            fig4 = px.imshow(pivot_wide, color_continuous_scale='Blues',
                             labels=dict(x='Frequency Score', y='Recency Score', color='Customers'))
            dark_layout(fig4, 'RFM Heatmap — Customer Density', height=300)
            st.plotly_chart(fig4, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: EDA
    # ══════════════════════════════════════════
    elif page == "🔍 EDA & Data Quality":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Exploratory Data Analysis</h1><p class='hero-sub'>Data quality, distributions, and business insights</p></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Overview", "📈 Sales Trends", "🌍 Geographic", "🔧 Data Quality"])

        with tab1:
            c1,c2,c3 = st.columns(3)
            c1.metric("Raw Rows", f"{raw_df.shape[0]:,}")
            c2.metric("Columns", raw_df.shape[1])
            c3.metric("After Cleaning", f"{df.shape[0]:,}")
            st.dataframe(df.head(500), use_container_width=True, height=260)
            st.markdown("**Statistical Summary**")
            st.dataframe(df[['Quantity','UnitPrice','TotalPrice']].describe().round(2), use_container_width=True)

        with tab2:
            df_copy = df.copy()
            df_copy['YearMonth'] = df_copy['InvoiceDate'].dt.to_period('M').astype(str)
            monthly_rev    = df_copy.groupby('YearMonth')['TotalPrice'].sum().reset_index()
            monthly_orders = df_copy.groupby('YearMonth')['InvoiceNo'].nunique().reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly_rev['YearMonth'], y=monthly_rev['TotalPrice'],
                                 name='Revenue', marker_color='#3b9eff', opacity=0.8), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly_orders['YearMonth'], y=monthly_orders['InvoiceNo'],
                                     name='Orders', mode='lines+markers',
                                     line=dict(color='#f59e0b', width=2)), secondary_y=True)
            dark_layout(fig, 'Monthly Revenue & Order Volume')
            fig.update_layout(xaxis_tickangle=-45, legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

            c1,c2 = st.columns(2)
            with c1:
                df_copy['DayOfWeek'] = df_copy['InvoiceDate'].dt.day_name()
                dow = df_copy.groupby('DayOfWeek')['TotalPrice'].sum().reindex(
                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index()
                fig2 = px.bar(dow, x='DayOfWeek', y='TotalPrice', color='TotalPrice',
                              color_continuous_scale='Blues')
                dark_layout(fig2, 'Revenue by Day of Week')
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                df_copy['Hour'] = df_copy['InvoiceDate'].dt.hour
                hourly = df_copy.groupby('Hour')['TotalPrice'].sum().reset_index()
                fig3 = px.area(hourly, x='Hour', y='TotalPrice', color_discrete_sequence=['#a855f7'])
                dark_layout(fig3, 'Revenue by Hour of Day')
                st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            if 'Country' in df.columns:
                country_rev = df.groupby('Country').agg(
                    Revenue=('TotalPrice','sum'),
                    Orders=('InvoiceNo','nunique'),
                    Customers=('CustomerID','nunique')
                ).reset_index().sort_values('Revenue', ascending=False)
                fig = px.bar(country_rev.head(15), x='Country', y='Revenue',
                             color='Revenue', color_continuous_scale='Blues')
                dark_layout(fig, 'Top 15 Countries by Revenue')
                fig.update_layout(showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(country_rev, use_container_width=True, height=250)

        with tab4:
            null_df = pd.DataFrame({
                'Column': raw_df.columns,
                'Missing': raw_df.isnull().sum().values,
                'Missing %': (raw_df.isnull().sum().values / len(raw_df) * 100).round(2)
            })
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(null_df[null_df['Missing']>0], x='Column', y='Missing %',
                             color='Missing %', color_continuous_scale='Reds')
                dark_layout(fig, 'Missing Data %')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                dtype_df = pd.DataFrame({'Column': raw_df.columns, 'Type': raw_df.dtypes.astype(str).values})
                st.dataframe(dtype_df, use_container_width=True)
            removed = len(raw_df) - len(df)
            st.markdown(f"""
            <div class='info-box'>
            <strong>Cleaning Summary</strong><br>
            • Removed rows with missing CustomerID<br>
            • Removed negative Quantity (returns/cancellations)<br>
            • Removed zero-price records<br>
            • <strong>{removed:,}</strong> total rows removed → <strong>{len(df):,}</strong> clean records remain
            </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════
    #  PAGE: RFM
    # ══════════════════════════════════════════
    elif page == "💡 RFM Analysis":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>RFM Analysis</h1><p class='hero-sub'>Rule-Based Supervised Segmentation · Recency · Frequency · Monetary</p></div>", unsafe_allow_html=True)

        with st.expander("📖 What is RFM?", expanded=False):
            st.markdown("""
            | Dimension | Definition | Insight |
            |---|---|---|
            | **Recency (R)** | Days since last purchase | Recent = more likely to buy again |
            | **Frequency (F)** | Distinct invoice count | High = loyal |
            | **Monetary (M)** | Total spend | High = high value |

            Each customer scores 1–5 per dimension → combined to assign a business segment label.
            """)

        tab1, tab2, tab3 = st.tabs(["📊 Distributions & 3D", "🗺️ Heatmap & Summary", "📋 Full Table"])

        with tab1:
            c1,c2,c3 = st.columns(3)
            for col, metric, color in zip([c1,c2,c3],
                                           ['Recency','Frequency','Monetary'],
                                           ['#3b9eff','#22c55e','#f59e0b']):
                data = rfm[metric] if metric != 'Monetary' else rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.98)][metric]
                fig = px.histogram(data, nbins=40, color_discrete_sequence=[color])
                dark_layout(fig, f'{metric} Distribution')
                fig.update_layout(showlegend=False)
                col.plotly_chart(fig, use_container_width=True)

            sample = rfm.sample(min(1000, len(rfm)), random_state=42)
            fig3d = px.scatter_3d(sample, x='Recency', y='Frequency', z='Monetary',
                                  color='Segment', color_discrete_map=SEG_COLORS, opacity=0.75,
                                  hover_data=['CustomerID'])
            fig3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                title='3D RFM Space — Coloured by Segment',
                title_font=dict(family='Syne', size=14, color='#90cdf4'),
                scene=dict(
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                ), height=500, margin=dict(t=45,b=5,l=5,r=5))
            st.plotly_chart(fig3d, use_container_width=True)

        with tab2:
            pivot = rfm.groupby(['R_Score','F_Score'])['CustomerID'].count().reset_index()
            pivot_wide = pivot.pivot(index='R_Score', columns='F_Score', values='CustomerID').fillna(0)
            fig = px.imshow(pivot_wide, color_continuous_scale='Blues',
                            labels=dict(x='Frequency Score', y='Recency Score', color='Customers'))
            dark_layout(fig, 'RFM Heatmap: Customer Count (R vs F Scores)')
            st.plotly_chart(fig, use_container_width=True)

            seg_summary = rfm.groupby('Segment').agg(
                Count=('CustomerID','count'), Avg_Recency=('Recency','mean'),
                Avg_Frequency=('Frequency','mean'), Avg_Monetary=('Monetary','mean'),
                Total_Revenue=('Monetary','sum')
            ).round(1).reset_index()
            st.dataframe(seg_summary, use_container_width=True)

        with tab3:
            segs = ['All'] + list(rfm['Segment'].unique())
            sel = st.selectbox("Filter by Segment", segs)
            disp = rfm if sel == 'All' else rfm[rfm['Segment'] == sel]
            st.dataframe(disp.sort_values('RFM_Score', ascending=False).reset_index(drop=True),
                         use_container_width=True, height=400)
            st.download_button("⬇️ Download RFM Table", disp.to_csv(index=False).encode(), "rfm.csv", "text/csv")


    # ══════════════════════════════════════════
    #  PAGE: K-MEANS
    # ══════════════════════════════════════════
    elif page == "🔵 K-Means Clustering":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>K-Means Clustering</h1><p class='hero-sub'>Unsupervised partition-based clustering on standardised RFM features</p></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📐 Elbow & Silhouette", "📊 Cluster Results", "🔬 Deep Dive"])

        with tab1:
            with st.spinner("Computing metrics..."):
                inertias, sils, dbs = [], [], []
                k_range = range(2, 11)
                for k_ in k_range:
                    km_ = KMeans(n_clusters=k_, init='k-means++', n_init=10, random_state=42)
                    lbl_ = km_.fit_predict(rfm_scaled)
                    inertias.append(km_.inertia_)
                    sils.append(silhouette_score(rfm_scaled, lbl_))
                    dbs.append(davies_bouldin_score(rfm_scaled, lbl_))

            c1,c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                         marker=dict(color='#3b9eff',size=8), line=dict(color='#3b9eff',width=2)))
                fig.add_vline(x=k_clusters, line_dash='dash', line_color='#f59e0b',
                              annotation_text=f'k={k_clusters}', annotation_font_color='#f59e0b')
                dark_layout(fig, 'Elbow Method (WCSS Inertia)')
                fig.update_layout(xaxis_title='k', yaxis_title='Inertia')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=list(k_range), y=sils, mode='lines+markers', name='Silhouette',
                                          line=dict(color='#22c55e',width=2), marker=dict(size=8)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=list(k_range), y=dbs, mode='lines+markers', name='Davies-Bouldin',
                                          line=dict(color='#ef4444',width=2), marker=dict(size=8,symbol='square')), secondary_y=True)
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
            km_labels, km_sil, km_db, km_model = run_kmeans(rfm_scaled, k_clusters)
            rfm['KM_Cluster'] = km_labels
            rfm['KM_Label'] = rfm['KM_Cluster'].apply(lambda x: f'Cluster {x}')

            c1,c2 = st.columns(2)
            with c1:
                cs = rfm.groupby('KM_Label').agg(
                    Count=('CustomerID','count'), Avg_Recency=('Recency','mean'),
                    Avg_Frequency=('Frequency','mean'), Avg_Monetary=('Monetary','mean'),
                ).round(1).reset_index()
                st.dataframe(cs, use_container_width=True)
            with c2:
                fig = px.pie(cs, names='KM_Label', values='Count',
                             color_discrete_sequence=CLUSTER_PALETTE, hole=0.4)
                dark_layout(fig, 'Cluster Sizes')
                st.plotly_chart(fig, use_container_width=True)

            pca_2d = PCA(n_components=2, random_state=42)
            c2d = pca_2d.fit_transform(rfm_scaled)
            rfm['PCA1'], rfm['PCA2'] = c2d[:,0], c2d[:,1]
            fig = px.scatter(rfm.sample(min(1000,len(rfm)),random_state=1),
                             x='PCA1', y='PCA2', color='KM_Label',
                             color_discrete_sequence=CLUSTER_PALETTE,
                             opacity=0.7, hover_data=['CustomerID','Recency','Frequency','Monetary'])
            dark_layout(fig, f'K-Means Clusters in PCA Space (k={k_clusters})')
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            metrics = ['Recency','Frequency','Monetary']
            fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
            for i, m in enumerate(metrics, 1):
                for c_id in sorted(rfm['KM_Cluster'].unique()):
                    vals = rfm[rfm['KM_Cluster']==c_id][m]
                    fig.add_trace(go.Box(y=vals, name=f'C{c_id}', marker_color=CLUSTER_PALETTE[c_id],
                                         showlegend=(i==1)), row=1, col=i)
            dark_layout(fig, 'RFM Distribution per Cluster', height=400)
            st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: HIERARCHICAL
    # ══════════════════════════════════════════
    elif page == "🌿 Hierarchical Clustering":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Hierarchical Clustering</h1><p class='hero-sub'>Agglomerative bottom-up clustering with dendrogram</p></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🌳 Dendrogram", "📊 Results", "📈 vs K-Means"])

        with tab1:
            with st.spinner("Building dendrogram..."):
                idx = np.random.choice(len(rfm_scaled), min(300, len(rfm_scaled)), replace=False)
                Z = linkage(rfm_scaled[idx], method=hc_link)
            fig_d, ax = plt.subplots(figsize=(14,5))
            fig_d.patch.set_facecolor('#0d1520')
            ax.set_facecolor('#0d1520')
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
                       color_threshold=0.7*max(Z[:,2]),
                       above_threshold_color='#475569', leaf_rotation=90, leaf_font_size=8)
            ax.set_title(f'Dendrogram (linkage={hc_link})', color='#90cdf4', fontsize=13, pad=10)
            ax.set_xlabel('Sample Index', color='#64748b')
            ax.set_ylabel('Distance', color='#64748b')
            ax.tick_params(colors='#64748b')
            for sp in ax.spines.values(): sp.set_edgecolor('#1e293b')
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close()

        with tab2:
            hc_labels, hc_sil, hc_db = run_hierarchical(rfm_scaled, hc_k, hc_link)
            rfm['HC_Cluster'] = hc_labels
            rfm['HC_Label'] = rfm['HC_Cluster'].apply(lambda x: f'Cluster {x}')

            hcs = rfm.groupby('HC_Label').agg(
                Count=('CustomerID','count'), Avg_Recency=('Recency','mean'),
                Avg_Frequency=('Frequency','mean'), Avg_Monetary=('Monetary','mean'),
            ).round(1).reset_index()

            c1,c2 = st.columns(2)
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
            c2d = pca_2d.fit_transform(rfm_scaled)
            rfm['PCA1'], rfm['PCA2'] = c2d[:,0], c2d[:,1]
            fig = px.scatter(rfm.sample(min(1000,len(rfm)),random_state=2),
                             x='PCA1', y='PCA2', color='HC_Label',
                             color_discrete_sequence=CLUSTER_PALETTE, opacity=0.7)
            dark_layout(fig, f'Hierarchical Clusters in PCA Space (k={hc_k})')
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if 'KM_Cluster' in rfm.columns:
                rfm['KM_L'] = rfm['KM_Cluster'].apply(lambda x: f'KM-{x}')
                overlap = pd.crosstab(rfm['HC_Label'], rfm['KM_L'])
                fig = px.imshow(overlap, color_continuous_scale='Blues',
                                labels=dict(x='K-Means', y='Hierarchical', color='Count'))
                dark_layout(fig, 'K-Means vs Hierarchical Overlap')
                st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: PCA
    # ══════════════════════════════════════════
    elif page == "🔮 PCA & Dimensionality":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>PCA & Dimensionality Reduction</h1><p class='hero-sub'>Principal Component Analysis on RFM features</p></div>", unsafe_allow_html=True)

        pca_components, pca_variance = run_pca(rfm_scaled, pca_comp)
        km_labels_pca, _, _, _ = run_kmeans(rfm_scaled, k_clusters)
        rfm['KM_Cluster_PCA'] = km_labels_pca.astype(str)

        tab1, tab2, tab3 = st.tabs(["📊 Variance Explained", "🗺️ 2D & 3D Plots", "🔬 Loadings"])

        with tab1:
            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(pca_comp)],
                'Variance': pca_variance,
                'Cumulative': np.cumsum(pca_variance)
            })
            c1,c2 = st.columns(2)
            with c1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=var_df['Component'], y=var_df['Variance'],
                                     name='Individual', marker_color='#3b9eff'), secondary_y=False)
                fig.add_trace(go.Scatter(x=var_df['Component'], y=var_df['Cumulative'],
                                         mode='lines+markers', name='Cumulative',
                                         line=dict(color='#f59e0b',width=2)), secondary_y=True)
                dark_layout(fig, 'Explained Variance')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(var_df.style.format({'Variance':'{:.1%}','Cumulative':'{:.1%}'}), use_container_width=True)
                st.markdown(f"""<div class='info-box'>
                First <strong>2 PCs</strong>: <strong>{pca_variance[:2].sum():.1%}</strong> variance explained<br>
                First <strong>3 PCs</strong>: <strong>{pca_variance[:3].sum():.1%}</strong> variance explained
                </div>""", unsafe_allow_html=True)

        with tab2:
            c1,c2 = st.columns(2)
            with c1:
                fig = px.scatter(x=pca_components[:,0], y=pca_components[:,1],
                                 color=rfm['KM_Cluster_PCA'], color_discrete_sequence=CLUSTER_PALETTE,
                                 opacity=0.7, labels={'x':'PC1','y':'PC2','color':'Cluster'})
                dark_layout(fig, 'PC1 vs PC2 — K-Means')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(x=pca_components[:,0], y=pca_components[:,1],
                                 color=rfm['Segment'], color_discrete_map=SEG_COLORS,
                                 opacity=0.7, labels={'x':'PC1','y':'PC2','color':'Segment'})
                dark_layout(fig, 'PC1 vs PC2 — RFM Segments')
                st.plotly_chart(fig, use_container_width=True)
            if pca_comp >= 3:
                fig3d = px.scatter_3d(x=pca_components[:,0], y=pca_components[:,1], z=pca_components[:,2],
                                      color=rfm['KM_Cluster_PCA'], color_discrete_sequence=CLUSTER_PALETTE,
                                      opacity=0.7, labels={'x':'PC1','y':'PC2','z':'PC3','color':'Cluster'})
                fig3d.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                    title='3D PCA Space', title_font=dict(family='Syne',size=14,color='#90cdf4'),
                    scene=dict(
                        xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                        zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.08)'),
                    ), height=480, margin=dict(t=45,b=5,l=5,r=5))
                st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            pca_full = PCA(n_components=pca_comp, random_state=42)
            pca_full.fit(rfm_scaled)
            loadings = pd.DataFrame(pca_full.components_.T,
                                     columns=[f'PC{i+1}' for i in range(pca_comp)],
                                     index=['Recency','Frequency','Monetary'])
            fig = px.imshow(loadings, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, text_auto=True)
            dark_layout(fig, 'PCA Feature Loadings')
            st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: CLV ANALYSIS  ★ NEW ★
    # ══════════════════════════════════════════
    elif page == "💎 CLV Analysis":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Customer Lifetime Value</h1><p class='hero-sub'>12-month & 24-month CLV projections using purchase rate × AOV modelling</p></div>", unsafe_allow_html=True)

        with st.spinner("Computing CLV..."):
            clv_df = compute_clv(df, rfm)

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg CLV (12mo)", f"£{clv_df['CLV_12mo'].mean():,.0f}")
        c2.metric("Avg CLV (24mo)", f"£{clv_df['CLV_24mo'].mean():,.0f}")
        c3.metric("Top 10% CLV",    f"£{clv_df['CLV_12mo'].quantile(0.9):,.0f}")
        c4.metric("Platinum Customers", f"{(clv_df['CLV_Tier']=='Platinum').sum():,}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 CLV Distributions", "🏅 Tier Analysis", "🔗 CLV vs RFM", "📋 Customer CLV Table"])

        with tab1:
            c1,c2 = st.columns(2)
            with c1:
                fig = px.histogram(clv_df[clv_df['CLV_12mo'] < clv_df['CLV_12mo'].quantile(0.95)],
                                   x='CLV_12mo', nbins=50, color_discrete_sequence=['#3b9eff'])
                dark_layout(fig, '12-Month CLV Distribution (95th pctile)')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                tier_rev = clv_df.groupby('CLV_Tier', observed=True).agg(
                    Count=('CustomerID','count'), Total_CLV=('CLV_12mo','sum')
                ).reset_index()
                fig = px.bar(tier_rev, x='CLV_Tier', y='Total_CLV',
                             color='CLV_Tier',
                             color_discrete_map=CLV_COLORS)
                dark_layout(fig, 'Total CLV by Tier')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # CLV Lorenz curve (top customers drive what % of value)
            sorted_clv = np.sort(clv_df['CLV_12mo'].values)
            cum_share = np.cumsum(sorted_clv) / sorted_clv.sum()
            pop_share = np.arange(1, len(sorted_clv)+1) / len(sorted_clv)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pop_share*100, y=cum_share*100, mode='lines',
                                     name='CLV Concentration', line=dict(color='#3b9eff',width=2.5)))
            fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode='lines', name='Perfect Equality',
                                     line=dict(color='#475569', dash='dash', width=1.5)))
            dark_layout(fig, 'CLV Lorenz Curve — Revenue Concentration')
            fig.update_layout(xaxis_title='% Customers', yaxis_title='% Total CLV')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            tier_stats = clv_df.groupby('CLV_Tier', observed=True).agg(
                Customers=('CustomerID','count'),
                Avg_CLV_12=('CLV_12mo','mean'),
                Avg_CLV_24=('CLV_24mo','mean'),
                Avg_AOV=('AOV','mean'),
                Avg_Purchase_Rate=('purchase_rate','mean'),
                Avg_Recency=('Recency','mean'),
                Avg_Frequency=('Frequency','mean'),
            ).round(2).reset_index()
            st.dataframe(tier_stats, use_container_width=True)

            c1,c2 = st.columns(2)
            with c1:
                fig = px.pie(tier_stats, names='CLV_Tier', values='Customers',
                             color='CLV_Tier', color_discrete_map=CLV_COLORS, hole=0.45)
                dark_layout(fig, 'Customer Count by CLV Tier')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(tier_stats, x='CLV_Tier', y='Avg_CLV_12',
                             color='CLV_Tier', color_discrete_map=CLV_COLORS)
                dark_layout(fig, 'Average 12-Month CLV by Tier')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            c1,c2 = st.columns(2)
            with c1:
                fig = px.scatter(clv_df.sample(min(800,len(clv_df)),random_state=7),
                                 x='Frequency', y='CLV_12mo', color='CLV_Tier',
                                 color_discrete_map=CLV_COLORS, opacity=0.6, size='AOV',
                                 hover_data=['CustomerID'])
                dark_layout(fig, 'Frequency vs CLV (size = AOV)')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(clv_df.sample(min(800,len(clv_df)),random_state=8),
                                 x='Recency', y='CLV_12mo', color='Segment',
                                 color_discrete_map=SEG_COLORS, opacity=0.6)
                dark_layout(fig, 'Recency vs CLV by RFM Segment')
                st.plotly_chart(fig, use_container_width=True)

            # CLV vs Segment box
            fig = px.box(clv_df, x='Segment', y='CLV_12mo',
                         color='Segment', color_discrete_map=SEG_COLORS)
            dark_layout(fig, 'CLV Distribution by RFM Segment', height=380)
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            tier_filter = st.selectbox("Filter by CLV Tier", ['All','Bronze','Silver','Gold','Platinum'])
            disp = clv_df if tier_filter=='All' else clv_df[clv_df['CLV_Tier']==tier_filter]
            cols_show = ['CustomerID','Segment','CLV_Tier','CLV_Score','CLV_12mo','CLV_24mo','AOV','purchase_rate','Recency','Frequency']
            st.dataframe(disp[cols_show].sort_values('CLV_12mo', ascending=False).reset_index(drop=True).head(500),
                         use_container_width=True, height=400)
            st.download_button("⬇️ Export CLV Data", disp.to_csv(index=False).encode(), "clv_data.csv", "text/csv")


    # ══════════════════════════════════════════
    #  PAGE: CHURN PREDICTION  ★ NEW ★
    # ══════════════════════════════════════════
    elif page == "⚠️ Churn Prediction":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Churn Prediction</h1><p class='hero-sub'>Gradient Boosting classifier · Churn probability · ROC-AUC evaluation</p></div>", unsafe_allow_html=True)

        with st.expander("📖 Model Methodology", expanded=False):
            st.markdown("""
            **Churn Definition**: A customer is labelled as churned if their last purchase was **> 180 days** ago.
            
            **Features used**: Recency, Frequency, Monetary, R/F/M Scores, RFM Total Score (7 features)
            
            **Model**: Gradient Boosting Classifier (100 estimators, max_depth=3)
            
            **Evaluation**: Silhouette score on holdout set, ROC curve, AUC
            """)

        with st.spinner("Training churn model..."):
            churn_df, fpr, tpr, auc, feat_imp = compute_churn(rfm)

        churn_rate = churn_df['Churned'].mean()
        high_risk  = (churn_df['Churn_Risk']=='High').sum()
        med_risk   = (churn_df['Churn_Risk']=='Medium').sum()
        avg_prob   = churn_df['Churn_Prob'].mean()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Overall Churn Rate",   f"{churn_rate:.1%}")
        c2.metric("High Risk Customers",  f"{high_risk:,}")
        c3.metric("Medium Risk",          f"{med_risk:,}")
        c4.metric("Avg Churn Probability",f"{avg_prob:.1%}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Churn Distribution", "📈 ROC Curve", "🔍 Feature Importance", "📋 At-Risk Customers"])

        with tab1:
            c1,c2 = st.columns(2)
            with c1:
                risk_counts = churn_df['Churn_Risk'].value_counts().reset_index()
                risk_counts.columns = ['Risk','Count']
                fig = px.pie(risk_counts, names='Risk', values='Count',
                             color='Risk', color_discrete_map=CHURN_COLORS, hole=0.45)
                dark_layout(fig, 'Churn Risk Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(churn_df, x='Churn_Prob', nbins=40,
                                   color_discrete_sequence=['#ef4444'])
                dark_layout(fig, 'Churn Probability Distribution')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Churn rate by segment
            seg_churn = churn_df.groupby('Segment').agg(
                Churn_Rate=('Churned','mean'), Count=('CustomerID','count')
            ).reset_index().sort_values('Churn_Rate', ascending=True)
            seg_churn['Churn_Rate_Pct'] = (seg_churn['Churn_Rate']*100).round(1)

            fig = px.bar(seg_churn, x='Churn_Rate_Pct', y='Segment', orientation='h',
                         color='Churn_Rate', color_continuous_scale='RdYlGn_r')
            dark_layout(fig, 'Churn Rate by RFM Segment', height=320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.3f})',
                                     line=dict(color='#3b9eff', width=2.5)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                     line=dict(color='#475569', dash='dash', width=1.5)))
            fig.add_annotation(x=0.7, y=0.3, text=f'AUC = {auc:.3f}',
                                font=dict(size=16, color='#3b9eff', family='Syne'),
                                showarrow=False,
                                bgcolor='rgba(59,158,255,0.1)',
                                bordercolor='rgba(59,158,255,0.3)', borderwidth=1, borderpad=8)
            dark_layout(fig, 'ROC Curve — Gradient Boosting Churn Classifier')
            fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                              xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.02]))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div class='{"success-box" if auc > 0.85 else "warning-box"}'>
            Model AUC = <strong>{auc:.3f}</strong> — 
            {'Excellent discriminative power (AUC > 0.85)' if auc > 0.85 else 'Good performance — further feature engineering may improve results'}.
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
            <strong>Interpretation:</strong> Features with higher importance contribute more to churn prediction. 
            Recency dominates as customers who haven't bought recently are most likely to have churned.
            </div>""", unsafe_allow_html=True)

        with tab4:
            risk_filter = st.selectbox("Filter Risk Level", ['High','Medium','Low','All'])
            disp = churn_df if risk_filter=='All' else churn_df[churn_df['Churn_Risk']==risk_filter]
            disp_sorted = disp.sort_values('Churn_Prob', ascending=False).reset_index(drop=True)
            show_cols = ['CustomerID','Segment','Churn_Risk','Churn_Prob','Recency','Frequency','Monetary','RFM_Score']

            # Show top at-risk with risk badges
            st.markdown(f"**{len(disp_sorted):,} customers in '{risk_filter}' risk category**")
            st.dataframe(disp_sorted[show_cols].head(300).style.format({'Churn_Prob':'{:.1%}'}),
                         use_container_width=True, height=380)
            st.download_button("⬇️ Export At-Risk List", disp_sorted.to_csv(index=False).encode(),
                               f"churn_{risk_filter.lower()}.csv", "text/csv")


    # ══════════════════════════════════════════
    #  PAGE: PRODUCT AFFINITY  ★ NEW ★
    # ══════════════════════════════════════════
    elif page == "🛒 Product Affinity":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Product Affinity Analysis</h1><p class='hero-sub'>Market basket analysis — which products are bought together</p></div>", unsafe_allow_html=True)

        with st.expander("📖 Methodology", expanded=False):
            st.markdown("""
            **Market Basket Analysis** identifies products frequently purchased together:
            
            - **Co-occurrence Matrix**: Counts how often pairs of products appear in the same invoice
            - **Affinity Score**: Normalized co-purchase frequency  
            - **Applications**: Cross-selling, recommendation engines, bundle pricing, store layout
            """)

        with st.spinner("Analysing product affinity..."):
            n_products = st.slider("Top N Products to Analyse", 5, 20, 12)
            co_matrix, pairs_df = compute_product_affinity(df, top_n=n_products)

        if co_matrix.empty:
            st.warning("Product data not available in the current dataset.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Products Analysed", n_products)
            c2.metric("Product Pairs", len(pairs_df))
            c3.metric("Top Pair Co-purchases", f"{pairs_df['Co-purchases'].max():,}" if len(pairs_df) > 0 else "N/A")

            tab1, tab2, tab3 = st.tabs(["🔥 Affinity Heatmap", "🔗 Top Pairs", "📊 Product Revenue"])

            with tab1:
                fig = px.imshow(co_matrix, color_continuous_scale='Blues',
                                title='Product Co-Purchase Matrix',
                                labels=dict(color='Co-purchases'))
                dark_layout(fig, 'Product Co-Purchase Heatmap', height=520)
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if len(pairs_df) > 0:
                    fig = px.bar(pairs_df.head(15), x='Co-purchases',
                                 y=pairs_df.head(15).apply(lambda r: f"{r['Product A'][:20]} + {r['Product B'][:20]}", axis=1),
                                 orientation='h', color='Co-purchases', color_continuous_scale='Blues')
                    dark_layout(fig, 'Top 15 Product Pairs by Co-purchase Frequency', height=480)
                    fig.update_layout(showlegend=False, yaxis_title='')
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(pairs_df, use_container_width=True)

            with tab3:
                top_prods = df.groupby(['StockCode','Description'])['TotalPrice'].sum().reset_index()
                top_prods = top_prods.sort_values('TotalPrice', ascending=False).head(20)
                top_prods['Description'] = top_prods['Description'].str[:30]
                fig = px.bar(top_prods, x='TotalPrice', y='Description', orientation='h',
                             color='TotalPrice', color_continuous_scale='Blues')
                dark_layout(fig, 'Top 20 Products by Revenue', height=500)
                fig.update_layout(showlegend=False, yaxis_title='')
                st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: AI SEGMENT INSIGHTS  ★ NEW ★
    # ══════════════════════════════════════════
    elif page == "🤖 AI Segment Insights":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>AI Segment Insights</h1><p class='hero-sub'>Automated intelligence reports & strategic recommendations per segment</p></div>", unsafe_allow_html=True)

        seg_actions = {
            'Champions': {
                'icon':'🥇', 'color':'#f59e0b',
                'desc': 'Your highest-value, most engaged customers. Recent, frequent, high spenders.',
                'actions': ['Launch exclusive VIP loyalty programme', 'Early product access & beta testing', 'Referral incentives — they become brand ambassadors', 'Premium membership upsell (annual subscription)', 'Personalised concierge-level support'],
                'risks': ['Risk of churn if service quality drops', 'Price-sensitive to premium tier moves'],
                'kpi_targets': 'Target: Maintain 90%+ retention | Grow AOV by 15% | NPS > 70'
            },
            'Loyal Customers': {
                'icon':'💚', 'color':'#22c55e',
                'desc': 'Consistent buyers with solid frequency. May not be top spenders but are reliable.',
                'actions': ['Tiered loyalty rewards (points per £ spent)', 'Upsell higher-margin product categories', 'Bundle recommendations based on purchase history', 'Quarterly "thank you" discounts (10-15%)', 'Cross-sell to adjacent product lines'],
                'risks': ['May plateau if not incentivised to grow spend', 'Vulnerable to competitor loyalty programmes'],
                'kpi_targets': 'Target: Increase frequency by 20% | Grow Monetary by 25%'
            },
            'New Customers': {
                'icon':'🆕', 'color':'#3b9eff',
                'desc': 'Recent first-time buyers. Critical onboarding window to convert to loyal.',
                'actions': ['Welcome email series (3-5 touchpoints over 30 days)', 'First-purchase follow-up + review request', 'Second purchase incentive (15% off, 2 weeks)', 'Educational content about product range', 'Personalised recommendations based on first order'],
                'risks': ['High drop-off risk if first experience is poor', '60-70% of new customers never return without intervention'],
                'kpi_targets': 'Target: 40% second purchase within 60 days | Reduce churn from 70% to 50%'
            },
            'Potential Loyalists': {
                'icon':'🌱', 'color':'#a855f7',
                'desc': 'Recent buyers with moderate engagement. On the fence between loyal and churning.',
                'actions': ['Membership/subscription programme invitation', 'Personalised product recommendations', 'Limited-time offers to increase purchase frequency', 'Engage via email & push notifications', 'Showcase product breadth they haven\'t explored'],
                'risks': ['Without intervention, likely to drift into Hibernating', 'Competing for attention with other brands'],
                'kpi_targets': 'Target: Move 30% to Loyal Customers within 90 days'
            },
            'At Risk': {
                'icon':'⚠️', 'color':'#ef4444',
                'desc': 'Once-active customers showing declining engagement and recency.',
                'actions': ['Win-back campaign: "We miss you" + 20% discount', 'Personalised outreach from account manager', 'Survey to understand reasons for reduced activity', 'Showcase new products/features since last visit', 'Time-limited "return" incentive (expires in 7 days)'],
                'risks': ['Without action, 60%+ will move to Lost within 90 days', 'Aggressive discounting can erode margin'],
                'kpi_targets': 'Target: Re-engage 25% within 60 days | Restore to Potential Loyalist tier'
            },
            'Cannot Lose Them': {
                'icon':'🚨', 'color':'#ec4899',
                'desc': 'Previously high-value customers now disengaged. Historically important, currently inactive.',
                'actions': ['Personal phone call or email from senior team member', 'Exclusive "Welcome Back" package with steep discount (30%+)', 'Dedicated customer success check-in', 'Priority resolution of any past complaints', 'VIP re-onboarding with white-glove experience'],
                'risks': ['High revenue loss risk if permanently lost', 'May have moved to competitor — competitive intelligence needed'],
                'kpi_targets': 'Target: Recover 20% back to Champions/Loyal tier within 6 months'
            },
            'Lost': {
                'icon':'😴', 'color':'#64748b',
                'desc': 'Long-inactive, low-frequency, low-value. Likely disengaged permanently.',
                'actions': ['Low-cost reactivation email campaign', 'Survey for product/service feedback', 'Heavy discount (30-40%) as last-chance offer', 'Consider removing from primary marketing to reduce costs', 'Analyse why they left for product roadmap insight'],
                'risks': ['ROI on re-engagement campaigns typically low', 'May inflate email bounce rates if stale'],
                'kpi_targets': 'Target: Reactivate 5-10% | Use exit data to improve retention of others'
            },
            'Hibernating': {
                'icon':'🌙', 'color':'#06b6d4',
                'desc': 'Inactive but not permanently lost. Seasonal or lapsed buyers with some past value.',
                'actions': ['Seasonal re-engagement campaigns (holiday, new year)', 'Highlight new product arrivals since last purchase', 'Nostalgia marketing: "Remember when you bought X?"', 'Low-barrier re-engagement (free shipping, small gift)', 'Segment by purchase category for personalised reactivation'],
                'risks': ['Email fatigue if over-contacted', 'May confuse with lost customers without proper segmentation'],
                'kpi_targets': 'Target: Reactivate 15% per quarter | Move 10% to Potential Loyalists'
            }
        }

        # Overall segment intelligence dashboard
        st.markdown("<div class='section-header'>📊 Segment Intelligence Overview</div>", unsafe_allow_html=True)

        seg_stats = rfm.groupby('Segment').agg(
            Count=('CustomerID','count'),
            Avg_Recency=('Recency','mean'),
            Avg_Frequency=('Frequency','mean'),
            Avg_Monetary=('Monetary','mean'),
            Total_Revenue=('Monetary','sum')
        ).round(1).reset_index()

        for _, row in seg_stats.iterrows():
            s = row['Segment']
            info = seg_actions.get(s, {})
            color = info.get('color', '#3b9eff')
            icon  = info.get('icon', '📌')
            revenue_share = row['Total_Revenue'] / seg_stats['Total_Revenue'].sum() * 100

            with st.expander(f"{icon} {s}  —  {row['Count']:,} customers  ·  £{row['Total_Revenue']:,.0f} revenue  ({revenue_share:.1f}%)", expanded=False):
                c1, c2, c3 = st.columns([2, 1.5, 1.5])
                with c1:
                    st.markdown(f"<div style='color:{color}; font-size:0.9rem; margin-bottom:0.8rem;'>{info.get('desc','')}</div>", unsafe_allow_html=True)
                    st.markdown("**🎯 Recommended Actions:**")
                    for act in info.get('actions', []):
                        st.markdown(f"• {act}")
                with c2:
                    st.metric("Avg Recency", f"{row['Avg_Recency']:.0f} days")
                    st.metric("Avg Frequency", f"{row['Avg_Frequency']:.1f} orders")
                    st.metric("Avg Spend", f"£{row['Avg_Monetary']:.0f}")
                with c3:
                    st.markdown("**⚠️ Key Risks:**")
                    for risk in info.get('risks', []):
                        st.markdown(f"• {risk}")
                    st.markdown(f"<div class='info-box' style='margin-top:0.8rem; border-color:{color}44;'><strong>KPI Targets</strong><br>{info.get('kpi_targets','')}</div>", unsafe_allow_html=True)

        # Priority action matrix
        st.markdown("<div class='section-header'>🎯 Priority Action Matrix</div>", unsafe_allow_html=True)

        action_matrix = pd.DataFrame({
            'Segment': list(seg_actions.keys()),
            'Priority': ['High','High','High','Medium','Critical','Critical','Low','Medium'],
            'Action_Type': ['Retain','Upsell','Onboard','Nurture','Win-back','Emergency','Reactivate','Re-engage'],
            'ROI_Potential': ['Very High','High','High','Medium','Very High','High','Low','Medium'],
            'Urgency': ['Medium','Low','High','Medium','High','Critical','Low','Medium']
        })
        seg_count_map = rfm['Segment'].value_counts().to_dict()
        action_matrix['Customer_Count'] = action_matrix['Segment'].map(seg_count_map)

        fig = px.scatter(action_matrix,
                         x='Action_Type', y='Priority',
                         size='Customer_Count', color='ROI_Potential',
                         text='Segment',
                         color_discrete_map={'Very High':'#f59e0b','High':'#22c55e','Medium':'#3b9eff','Low':'#64748b'},
                         size_max=60)
        dark_layout(fig, 'Action Priority Matrix (bubble size = customer count)', height=380)
        fig.update_traces(textposition='top center', textfont_size=9)
        st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: MODEL COMPARISON
    # ══════════════════════════════════════════
    elif page == "📊 Model Comparison":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Model Comparison</h1><p class='hero-sub'>Objective evaluation — K-Means vs Hierarchical Clustering</p></div>", unsafe_allow_html=True)

        km_labels, km_sil, km_db, _ = run_kmeans(rfm_scaled, k_clusters)
        hc_labels, hc_sil, hc_db    = run_hierarchical(rfm_scaled, hc_k, hc_link)
        rfm['KM_Cluster'] = km_labels
        rfm['HC_Cluster'] = hc_labels

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("K-Means Silhouette",    f"{km_sil:.3f}")
        c2.metric("K-Means Davies-Bouldin",f"{km_db:.3f}")
        c3.metric("HC Silhouette",         f"{hc_sil:.3f}")
        c4.metric("HC Davies-Bouldin",     f"{hc_db:.3f}")

        comp_df = pd.DataFrame({
            'Method': ['K-Means','Hierarchical'],
            'Silhouette (↑)': [km_sil, hc_sil],
            'Davies-Bouldin (↓)': [km_db, hc_db],
            'k': [k_clusters, hc_k]
        })

        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(comp_df, x='Method', y='Silhouette (↑)', color='Method',
                         color_discrete_sequence=['#3b9eff','#22c55e'])
            dark_layout(fig, 'Silhouette Score (higher = better)')
            fig.update_layout(showlegend=False, yaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(comp_df, x='Method', y='Davies-Bouldin (↓)', color='Method',
                          color_discrete_sequence=['#ef4444','#f59e0b'])
            dark_layout(fig2, 'Davies-Bouldin (lower = better)')
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        winner = 'K-Means' if km_sil >= hc_sil else 'Hierarchical Clustering'
        st.markdown(f"""
        <div class='{"success-box" if winner=="K-Means" else "info-box"}'>
        🏆 <strong>Best: {winner}</strong> (by Silhouette Score)<br><br>
        • <strong>Silhouette</strong>: How well each point fits its own cluster vs neighbours. Range [-1,1]. Higher = better separation.<br>
        • <strong>Davies-Bouldin</strong>: Ratio of within-cluster scatter to between-cluster separation. Lower = better.
        </div>""", unsafe_allow_html=True)

        pca_2d = PCA(n_components=2, random_state=42)
        coords = pca_2d.fit_transform(rfm_scaled)
        c1,c2 = st.columns(2)
        with c1:
            fig = px.scatter(x=coords[:,0], y=coords[:,1],
                             color=rfm['KM_Cluster'].astype(str),
                             color_discrete_sequence=CLUSTER_PALETTE, opacity=0.7,
                             labels={'x':'PC1','y':'PC2','color':'Cluster'})
            dark_layout(fig, f'K-Means (k={k_clusters}) in PCA Space')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(x=coords[:,0], y=coords[:,1],
                              color=rfm['HC_Cluster'].astype(str),
                              color_discrete_sequence=CLUSTER_PALETTE, opacity=0.7,
                              labels={'x':'PC1','y':'PC2','color':'Cluster'})
            dark_layout(fig2, f'Hierarchical (k={hc_k}) in PCA Space')
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(comp_df.set_index('Method'), use_container_width=True)


    # ══════════════════════════════════════════
    #  PAGE: SEGMENT PROFILER
    # ══════════════════════════════════════════
    elif page == "🎯 Segment Profiler":
        st.markdown("<div class='hero-banner'><h1 class='hero-title'>Segment Profiler</h1><p class='hero-sub'>Deep dive into any RFM segment — characteristics, behaviour & actions</p></div>", unsafe_allow_html=True)

        selected_seg = st.selectbox("Select Segment to Profile", sorted(rfm['Segment'].unique()))
        seg_data = rfm[rfm['Segment'] == selected_seg]
        color = SEG_COLORS.get(selected_seg, '#3b9eff')

        seg_actions_map = {
            'Champions': ('🥇','Reward & retain. Early access, loyalty perks, referral programs.'),
            'Loyal Customers': ('💚','Upsell higher-value products. Offer loyalty rewards.'),
            'New Customers': ('🆕','Onboarding emails, welcome discounts, product education.'),
            'Potential Loyalists': ('🌱','Membership programs, personalised recommendations.'),
            'At Risk': ('⚠️','Win-back campaigns, personalised offers, ask for feedback.'),
            'Cannot Lose Them': ('🚨','Reach out immediately — exclusive deals, personal outreach.'),
            'Lost': ('😴','Reactivation campaigns with steep discounts or surveys.'),
            'Hibernating': ('🌙','Seasonal re-engagement, value reminders, nostalgia marketing.'),
        }
        icon, action = seg_actions_map.get(selected_seg, ('📌','Build targeted campaigns.'))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Customers",    f"{len(seg_data):,}")
        c2.metric("Avg Recency",  f"{seg_data['Recency'].mean():.0f} days")
        c3.metric("Avg Frequency",f"{seg_data['Frequency'].mean():.1f} orders")
        c4.metric("Avg Monetary", f"£{seg_data['Monetary'].mean():,.0f}")

        st.markdown(f"""
        <div class='insight-card' style='border-color:{color}44;'>
            <div class='title' style='color:{color};'>{icon} {selected_seg} — Marketing Action</div>
            <div class='body'>{action}</div>
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(seg_data, x='Recency', nbins=25, color_discrete_sequence=[color])
            dark_layout(fig, f'{selected_seg} — Recency Distribution')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(seg_data, x='Monetary', nbins=25, color_discrete_sequence=[color])
            dark_layout(fig, f'{selected_seg} — Monetary Spend Distribution')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        all_avg = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
        all_norm = (all_avg - all_avg.min()) / (all_avg.max() - all_avg.min())
        all_norm['Recency'] = 1 - all_norm['Recency']
        cats = ['Recency (inv)','Frequency','Monetary']

        fig = go.Figure()
        for s in all_norm.index:
            vals = all_norm.loc[s].tolist() + [all_norm.loc[s].tolist()[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], name=s,
                line_color=SEG_COLORS.get(s,'#aaa'),
                opacity=0.45 if s != selected_seg else 1.0,
                line_width=1 if s != selected_seg else 3,
                fill='toself' if s == selected_seg else 'none',
                fillcolor=f'rgba{tuple(list(bytes.fromhex(SEG_COLORS.get(selected_seg,"#3b9eff")[1:])) + [30])}' if s == selected_seg else 'rgba(0,0,0,0)'
            ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            title='Radar Chart — Normalised RFM by Segment',
            title_font=dict(family='Syne', size=14, color='#90cdf4'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            margin=dict(t=50,b=10,l=10,r=10), height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Customer List — {selected_seg} ({len(seg_data):,} customers)**")
        st.dataframe(
            seg_data[['CustomerID','Recency','Frequency','Monetary','R_Score','F_Score','M_Score','RFM_Score']]
            .sort_values('RFM_Score', ascending=False).reset_index(drop=True),
            use_container_width=True, height=320
        )
        st.download_button(
            f"⬇️ Export {selected_seg}",
            seg_data.to_csv(index=False).encode(),
            f"segment_{selected_seg.lower().replace(' ','_')}.csv",
            "text/csv"
        )