import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
CSV_PATH = "waf_http_anomaly_dataset.csv"
df = pd.read_csv(CSV_PATH)

print("="*70)
print(" "*20 + "WAF ANOMALY DETECTION SYSTEM")
print("="*70)

# -----------------------------
# 2. FEATURE ENGINEERING
# -----------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

method_encoder = LabelEncoder()
df['method_encoded'] = method_encoder.fit_transform(df['method'])

df['is_error'] = (df['status_code'] >= 400).astype(int)
df['is_client_error'] = ((df['status_code'] >= 400) & (df['status_code'] < 500)).astype(int)
df['is_server_error'] = (df['status_code'] >= 500).astype(int)
df['is_redirect'] = ((df['status_code'] >= 300) & (df['status_code'] < 400)).astype(int)

df['is_bot_ua'] = df['user_agent'].str.contains(
    'bot|crawler|spider|curl|python-requests|wget', case=False, regex=True
).astype(int)

df['ua_length'] = df['user_agent'].str.len()

df['has_sql_injection'] = df['url'].str.contains(
    r"(union|select|insert|drop|--|')", case=False, regex=True
).astype(int)

df['has_xss'] = df['url'].str.contains(
    r'(<script|javascript:)', case=False, regex=True
).astype(int)

df['url_depth'] = df['url'].str.count('/')
df['has_query'] = (df['query_length'] > 0).astype(int)

df['burst_indicator'] = (df['req_per_ip_10sec'] > 50).astype(int)
df['high_frequency'] = (df['req_per_ip_1min'] > 100).astype(int)

df['payload_entropy_norm'] = df['payload_entropy'] / 8.0
df['high_entropy'] = (df['payload_entropy'] > 5).astype(int)

# Reputation proxy
ip_stats = df.groupby('src_ip').agg({
    'label': 'mean',
    'req_per_ip_1min': 'max'
}).reset_index()

ip_stats.columns = ['src_ip', 'ip_attack_history', 'ip_max_req_rate']
df = df.merge(ip_stats, on='src_ip', how='left')

# -----------------------------
# 3. FEATURE SELECTION
# -----------------------------
FEATURE_COLUMNS = [
    "url_length", "query_length", "bytes_sent", "request_time",
    "req_per_ip_1min", "req_per_ip_10sec",
    "payload_entropy", "is_https",
    "hour", "day_of_week", "is_business_hours",
    "method_encoded", "is_error", "is_client_error",
    "is_server_error", "is_redirect",
    "is_bot_ua", "ua_length",
    "has_sql_injection", "has_xss",
    "url_depth", "has_query",
    "burst_indicator", "high_frequency",
    "payload_entropy_norm", "high_entropy",
    "ip_attack_history", "ip_max_req_rate"
]

X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. SPLIT DATA
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# -----------------------------
# 5. MODEL TRAINING
# -----------------------------
print("\n🌲 Training Isolation Forest")
best_if = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)
best_if.fit(X_train)

if_scores = best_if.decision_function(X_eval)
if_pred = (best_if.predict(X_eval) == -1).astype(int)

print("🔍 Training LOF")
best_lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.10,
    novelty=True,
    n_jobs=-1
)
best_lof.fit(X_train)

lof_scores = best_lof.decision_function(X_eval)
lof_pred = (best_lof.predict(X_eval) == -1).astype(int)

# -----------------------------
# 6. ENSEMBLE (2 MODELS)
# -----------------------------
ensemble_pred = ((if_pred + lof_pred) >= 1).astype(int)

ensemble_scores = (
    (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-9) +
    (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-9)
) / 2

# -----------------------------
# 7. EVALUATION
# -----------------------------
print("\n📊 ENSEMBLE RESULTS")
print(classification_report(y_eval, ensemble_pred, target_names=["Normal", "Attack"]))

cm = confusion_matrix(y_eval, ensemble_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 8. SAVE ARTIFACTS
# -----------------------------
joblib.dump(best_if, "model_isolation_forest.pkl")
joblib.dump(best_lof, "model_lof.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(method_encoder, "method_encoder.pkl")

print("\n✅ Models saved successfully")
