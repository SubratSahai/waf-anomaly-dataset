import streamlit as st
import pandas as pd
import json
import joblib
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="WAF Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("waf_http_anomaly_dataset.csv")

@st.cache_resource
def load_model_artifacts():
    model = joblib.load("model_isolation_forest.pkl")
    config = joblib.load("model_config.pkl")
    return model, config

@st.cache_data
def load_rules():
    with open("rules.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_whitelist():
    try:
        with open("ip_whitelist.txt", "r") as f:
            return [line.strip() for line in f]
    except:
        return []

df = load_data()
model, config = load_model_artifacts()
rules = load_rules()
whitelist_ips = load_whitelist()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🛡️ WAF Control Panel")
st.sidebar.markdown("**Isolation Forest Based ML‑WAF**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📜 Security Rules", "🔥 Attack Insights", "⚙️ System Status"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Naval Hackathon • ML‑Powered WAF")

# -----------------------------
# OVERVIEW PAGE
# -----------------------------
if page == "📊 Overview":
    st.title("📊 WAF Security Overview")

    total_requests = len(df)
    total_attacks = int(df["label"].sum())
    attack_rate = df["label"].mean() * 100
    fpr = config["performance"]["fpr"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests", f"{total_requests:,}")
    c2.metric("Detected Attacks", f"{total_attacks:,}")
    c3.metric("Attack Rate", f"{attack_rate:.2f}%")
    c4.metric("False Positive Rate", f"{fpr:.4f}")

    st.markdown("---")

    st.subheader("📊 Traffic Distribution")
    traffic = df["label"].value_counts().rename({0: "Normal", 1: "Attack"})
    st.bar_chart(traffic)

# -----------------------------
# RULES PAGE
# -----------------------------
elif page == "📜 Security Rules":
    st.title("📜 Auto‑Generated WAF Rules")

    st.info(
        "These rules are **automatically generated** from Isolation Forest detected anomalies.\n\n"
        "They can be directly converted into **WAF signatures / firewall policies**."
    )

    rules_df = pd.DataFrame(rules["rules"])

    st.dataframe(
        rules_df[
            [
                "rule_id",
                "type",
                "condition",
                "severity",
                "trigger_count",
                "confidence"
            ]
        ],
        width='stretch'
    )

# -----------------------------
# ATTACK INSIGHTS PAGE
# -----------------------------
elif page == "🔥 Attack Insights":
    st.title("🔥 Attack Intelligence")

    rules_df = pd.DataFrame(rules["rules"])

    st.subheader("📌 Top Attack Types")
    attack_types = (
        rules_df.groupby("type")["trigger_count"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(attack_types)

    st.markdown("---")
    st.subheader("🚫 Recently Blocked IPs")

    blocked_ips = (
        df[df["label"] == 1]["src_ip"]
        .value_counts()
        .reset_index()
    )
    blocked_ips.columns = ["IP", "Blocked Requests"]

    blocked_ips = blocked_ips[~blocked_ips["IP"].isin(whitelist_ips)].head(10)

    if blocked_ips.empty:
        st.success("No blocked IPs outside whitelist 🎉")
    else:
        st.table(blocked_ips)

# -----------------------------
# SYSTEM STATUS PAGE
# -----------------------------
elif page == "⚙️ System Status":
    st.title("⚙️ System Health & Configuration")

    st.success("🟢 Isolation Forest Model: ACTIVE")
    st.success(f"Contamination Rate: {config['contamination']}")
    st.success(f"Optimal Threshold: {config['optimal_threshold']:.4f}")
    st.info(f"Whitelisted IPs: {len(whitelist_ips)}")

    st.markdown("---")
    st.subheader("📦 Loaded Artifacts")
    st.code(
        """
model_isolation_forest.pkl
model_config.pkl
rules.json
ip_whitelist.txt
feature_columns.txt
        """
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
