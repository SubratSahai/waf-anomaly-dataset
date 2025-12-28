"""
Complete Production-Ready WAF Anomaly Detection System
Naval Hackathon - Optimized Version (Isolation Forest Only)

Features:
- Enhanced feature engineering with security patterns
- Fast Isolation Forest model with hyperparameter tuning
- SHAP explainability
- Comprehensive visualizations
- Real-time inference with whitelisting
- Adaptive threshold mechanism
"""

# -----------------------------
# 1. IMPORTS
# -----------------------------

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report
)

import joblib
import warnings
warnings.filterwarnings('ignore')

# For SHAP explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# -----------------------------
# 2. LOAD DATASET
# -----------------------------

CSV_PATH = r"waf_http_anomaly_dataset.csv"
df = pd.read_csv(CSV_PATH)

print("="*70)
print(" "*20 + "WAF ANOMALY DETECTION SYSTEM")
print("="*70)
print(f"\n📊 DATASET OVERVIEW")
print(f"{'─'*70}")
print(f"Total requests: {len(df):,}")
print(f"Attack samples: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
print(f"Normal samples: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
print()


# -----------------------------
# 3. COMPREHENSIVE FEATURE ENGINEERING
# -----------------------------

print(f"🔧 FEATURE ENGINEERING")
print(f"{'─'*70}")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')

# TEMPORAL FEATURES
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

# METHOD ENCODING
method_encoder = LabelEncoder()
df['method_encoded'] = method_encoder.fit_transform(df['method'])

# HTTP STATUS FEATURES
df['is_error'] = (df['status_code'] >= 400).astype(int)
df['is_client_error'] = ((df['status_code'] >= 400) & (df['status_code'] < 500)).astype(int)
df['is_server_error'] = (df['status_code'] >= 500).astype(int)
df['is_redirect'] = ((df['status_code'] >= 300) & (df['status_code'] < 400)).astype(int)

# USER AGENT FEATURES
df['is_bot_ua'] = df['user_agent'].str.contains(
    'bot|crawler|spider|curl|python-requests|wget', 
    case=False, regex=True
).astype(int)
df['ua_length'] = df['user_agent'].str.len()

# URL SECURITY PATTERN DETECTION
df['has_sql_injection'] = df['url'].str.contains(
    r"('|--|union|select|insert|update|delete|drop|;|\bor\b.*=)", 
    case=False, regex=True
).astype(int)

df['has_xss'] = df['url'].str.contains(
    r'(<script|<iframe|javascript:|onerror|onload)', 
    case=False, regex=True
).astype(int)

df['has_path_traversal'] = df['url'].str.contains(
    r'(\.\.|/etc/|/var/|/proc/|\\\\|%2e%2e)',
    case=False, regex=True
).astype(int)

df['has_command_injection'] = df['url'].str.contains(
    r'(;|\||&|`|\$\(|%0a|%0d)', 
    case=False, regex=True
).astype(int)

df['has_special_chars'] = df['url'].str.contains(
    r'[<>\'\";\(\)\{\}\[\]]', 
    regex=True
).astype(int)

# URL STRUCTURE FEATURES
df['url_depth'] = df['url'].str.count('/')
df['has_query'] = (df['query_length'] > 0).astype(int)
df['query_param_count'] = df['url'].str.count('&') + df['has_query']

# RATE-BASED FEATURES
df['req_ratio_10s_to_1m'] = df['req_per_ip_10sec'] / (df['req_per_ip_1min'] + 1)
df['burst_indicator'] = (df['req_per_ip_10sec'] > 50).astype(int)
df['high_frequency'] = (df['req_per_ip_1min'] > 100).astype(int)

# PAYLOAD FEATURES
df['payload_entropy_norm'] = df['payload_entropy'] / 8.0
df['high_entropy'] = (df['payload_entropy'] > 5.0).astype(int)

# SIZE-BASED ANOMALIES
df['size_anomaly'] = ((df['bytes_sent'] > 5000) | (df['bytes_sent'] < 100)).astype(int)
df['slow_request'] = (df['request_time'] > 1.5).astype(int)

# REPUTATION FEATURES
ip_stats = df.groupby('src_ip').agg({
    'is_error': 'mean',
    'req_per_ip_1min': 'max'
}).reset_index()
ip_stats.columns = ['src_ip', 'ip_error_rate', 'ip_max_req_rate']
df = df.merge(ip_stats, on='src_ip', how='left')

print(f"✓ Created {df.shape[1] - len(pd.read_csv(CSV_PATH).columns)} new features")
print()


# -----------------------------
# 4. FEATURE SELECTION
# -----------------------------

FEATURE_COLUMNS = [
    # Original numerical features
    "url_length", "query_length", "bytes_sent", "request_time",
    "req_per_ip_1min", "req_per_ip_10sec", "unique_urls_per_ip",
    "time_gap", "payload_entropy", "is_https",
    
    # Temporal features
    "hour", "minute", "day_of_week", "is_business_hours",
    
    # HTTP features
    "method_encoded", "is_error", "is_client_error", 
    "is_server_error", "is_redirect",
    
    # User agent features
    "is_bot_ua", "ua_length",
    
    # Security pattern features
    "has_sql_injection", "has_xss", "has_path_traversal",
    "has_command_injection", "has_special_chars",
    
    # URL structure
    "url_depth", "has_query", "query_param_count",
    
    # Rate-based features
    "req_ratio_10s_to_1m", "burst_indicator", "high_frequency",
    
    # Payload features
    "payload_entropy_norm", "high_entropy",
    
    # Anomaly indicators
    "size_anomaly", "slow_request",
    
    # Reputation features
    "ip_error_rate", "ip_max_req_rate"
]

X = df[FEATURE_COLUMNS].copy()
y = df["label"].copy()

# Store original data for analysis
df_original = df.copy()

print(f"📋 FEATURE SUMMARY")
print(f"{'─'*70}")
print(f"Total features: {len(FEATURE_COLUMNS)}")
print(f"Feature categories:")
print(f"  • Security patterns: 5")
print(f"  • Rate-based: 6")
print(f"  • HTTP/Status: 5")
print(f"  • Temporal: 4")
print(f"  • Reputation: 2")
print(f"  • Other: {len(FEATURE_COLUMNS) - 22}")
print()


# -----------------------------
# 5. FEATURE SCALING & CLEANING
# -----------------------------

print(f"⚙️  PREPROCESSING")
print(f"{'─'*70}")

# Handle Infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Log transform for skewed features
SKEWED_COLUMNS = [
    "bytes_sent", "req_per_ip_1min", "req_per_ip_10sec",
    "time_gap", "url_length", "query_length", "ua_length"
]

for col in SKEWED_COLUMNS:
    if col in X.columns:
        X[col] = np.log1p(X[col].clip(lower=0))

print(f"✓ Applied log transformation to {len([c for c in SKEWED_COLUMNS if c in X.columns])} skewed features")

# Handle Missing Values
if X.isnull().values.any():
    nan_count = X.isnull().sum().sum()
    print(f"⚠️  Found {nan_count} missing values. Filling with 0...")
    X.fillna(0, inplace=True)

# Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Standardized all features (mean=0, std=1)")
print()


# -----------------------------
# 6. TRAIN/TEST/EVAL SPLIT
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"📊 DATA SPLIT")
print(f"{'─'*70}")
print(f"Train: {len(X_train):,} samples ({y_train.sum()} attacks, {y_train.mean()*100:.1f}%)")
print(f"Test:  {len(X_test):,} samples ({y_test.sum()} attacks, {y_test.mean()*100:.1f}%)")
print(f"Eval:  {len(X_eval):,} samples ({y_eval.sum()} attacks, {y_eval.mean()*100:.1f}%)")
print()


# -----------------------------
# 7. MODEL TRAINING - ISOLATION FOREST
# -----------------------------

print("="*70)
print(" "*25 + "MODEL TRAINING")
print("="*70)
print()

print(f"🌲 Training Isolation Forest...")
print(f"{'─'*70}")

# Hyperparameter tuning
best_f1 = 0
best_model = None
best_cont = 0

contamination_levels = [0.08, 0.10, 0.12, 0.15]
results = []

for cont in contamination_levels:
    model_temp = IsolationForest(
        n_estimators=300,
        contamination=cont,
        max_samples='auto',
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model_temp.fit(X_train)
    pred_temp = (model_temp.predict(X_test) == -1).astype(int)
    f1_temp = f1_score(y_test, pred_temp)
    
    results.append({'contamination': cont, 'f1': f1_temp})
    print(f"  contamination={cont:.2f} → F1={f1_temp:.4f}")
    
    if f1_temp > best_f1:
        best_f1 = f1_temp
        best_model = model_temp
        best_cont = cont

print(f"\n✓ Best contamination: {best_cont:.2f} (F1={best_f1:.4f})")

# Evaluate on eval set
if_scores = best_model.decision_function(X_eval)
if_pred = (best_model.predict(X_eval) == -1).astype(int)

if_acc = accuracy_score(y_eval, if_pred)
if_prec = precision_score(y_eval, if_pred)
if_rec = recall_score(y_eval, if_pred)
if_f1 = f1_score(y_eval, if_pred)

# Isolation Forest scores are NEGATIVE for anomalies (lower = more anomalous)
# Invert them for ROC-AUC calculation
if_scores_inverted = -if_scores
if_auc = roc_auc_score(y_eval, if_scores_inverted)

print(f"\n📊 Isolation Forest - Final Evaluation Results:")
print(f"  Accuracy:  {if_acc:.4f}")
print(f"  Precision: {if_prec:.4f}")
print(f"  Recall:    {if_rec:.4f}")
print(f"  F1 Score:  {if_f1:.4f}")
print(f"  ROC-AUC:   {if_auc:.4f}")
print()


# -----------------------------
# 8. DETAILED EVALUATION
# -----------------------------

print(f"📋 DETAILED CLASSIFICATION REPORT")
print(f"{'─'*70}")
print(classification_report(y_eval, if_pred, target_names=['Normal', 'Attack']))

cm = confusion_matrix(y_eval, if_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 CONFUSION MATRIX")
print(f"{'─'*70}")
print(f"                 Predicted")
print(f"              Normal  Attack")
print(f"Actual Normal   {tn:4d}   {fp:4d}")
print(f"       Attack   {fn:4d}   {tp:4d}")
print()

# Key metrics for WAF
fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

print(f"🎯 CRITICAL WAF METRICS")
print(f"{'─'*70}")
print(f"False Positive Rate: {fpr:.4f} ({fp}/{tn+fp} legitimate requests blocked)")
print(f"False Negative Rate: {fnr:.4f} ({fn}/{tp+fn} attacks missed)")
print(f"True Positive Rate:  {if_rec:.4f} (Detection rate)")
print()


# -----------------------------
# 9. VISUALIZATIONS
# -----------------------------

print(f"📈 GENERATING VISUALIZATIONS")
print(f"{'─'*70}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('WAF Anomaly Detection - Isolation Forest Analysis', fontsize=16, fontweight='bold')

# 1. ROC Curve
ax1 = axes[0, 0]
fpr_curve, tpr_curve, _ = roc_curve(y_eval, if_scores_inverted)
ax1.plot(fpr_curve, tpr_curve, label=f'Isolation Forest (AUC={if_auc:.3f})', linewidth=3, color='#3498db')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax2 = axes[0, 1]
precision_curve, recall_curve, _ = precision_recall_curve(y_eval, if_pred)
ax2.plot(recall_curve, precision_curve, linewidth=3, color='#2ecc71')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.grid(True, alpha=0.3)

# 3. Performance Metrics Bar Chart
ax3 = axes[0, 2]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
values = [if_acc, if_prec, if_rec, if_f1, if_auc]
colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
bars = ax3.bar(metrics, values, color=colors_bar)
ax3.set_ylabel('Score')
ax3.set_title('Model Performance Metrics')
ax3.set_ylim([0, 1.1])
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 4. Confusion Matrix Heatmap
ax4 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
ax4.set_title('Confusion Matrix')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# 5. Decision Score Distribution
ax5 = axes[1, 1]
attack_scores = if_scores_inverted[y_eval == 1]
normal_scores = if_scores_inverted[y_eval == 0]
ax5.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='#2ecc71')
ax5.hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='#e74c3c')
ax5.set_xlabel('Anomaly Score (inverted)')
ax5.set_ylabel('Frequency')
ax5.set_title('Score Distribution by Class')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Error Analysis
ax6 = axes[1, 2]
error_types = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
error_values = [tp, tn, fp, fn]
error_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax6.pie(error_values, labels=error_types, autopct='%1.1f%%',
                                     colors=error_colors, startangle=90)
ax6.set_title('Prediction Distribution')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('waf_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: waf_analysis.png")
plt.close()


# -----------------------------
# 10. FEATURE IMPORTANCE (SHAP)
# -----------------------------

if SHAP_AVAILABLE:
    print(f"\n🔍 GENERATING SHAP EXPLANATIONS")
    print(f"{'─'*70}")
    
    try:
        sample_size = min(100, len(X_eval))
        X_eval_sample = X_eval[:sample_size]
        
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_eval_sample)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_eval_sample, 
                         feature_names=FEATURE_COLUMNS,
                         show=False)
        plt.title('SHAP Feature Importance - Isolation Forest', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: shap_analysis.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
else:
    print(f"\n⚠️  Skipping SHAP analysis (not installed)")

print()


# -----------------------------
# 11. ATTACK PATTERN ANALYSIS
# -----------------------------

print(f"🎯 ATTACK PATTERN ANALYSIS")
print(f"{'─'*70}")

detected_idx = np.where(if_pred == 1)[0]
actual_attacks_idx = np.where(y_eval == 1)[0]

correctly_detected = np.intersect1d(detected_idx, actual_attacks_idx)
false_positives_idx = np.setdiff1d(detected_idx, actual_attacks_idx)
missed_attacks_idx = np.setdiff1d(actual_attacks_idx, detected_idx)

print(f"Total attacks in eval set: {len(actual_attacks_idx)}")
print(f"Correctly detected: {len(correctly_detected)}")
print(f"False positives: {len(false_positives_idx)}")
print(f"Missed attacks: {len(missed_attacks_idx)}")
print()

if len(detected_idx) > 0:
    print(f"Top features in detected anomalies:")
    # Get the original indices from the full dataset
    eval_start_idx = len(X_train) + len(X_test)
    eval_end_idx = eval_start_idx + len(X_eval)
    detected_full_idx = detected_idx + eval_start_idx
    
    detected_features = X[FEATURE_COLUMNS].iloc[detected_full_idx]
    
    security_features = ['has_sql_injection', 'has_xss', 'has_path_traversal', 
                         'has_command_injection', 'has_special_chars']
    
    for feat in security_features:
        if feat in detected_features.columns:
            mean_val = detected_features[feat].mean()
            print(f"  • {feat}: {mean_val:.2%}")
    
    print(f"\nHigh-risk indicators:")
    print(f"  • Burst traffic: {detected_features['burst_indicator'].mean():.2%}")
    print(f"  • High entropy: {detected_features['high_entropy'].mean():.2%}")
    print(f"  • Error responses: {detected_features['is_error'].mean():.2%}")
    print(f"  • Bot user agents: {detected_features['is_bot_ua'].mean():.2%}")
print()

# -----------------------------
# 11.5 RULE GENERATION LOGIC
# -----------------------------

print(f"📜 RULE GENERATION ENGINE")
print(f"{'─'*70}")

rules = []
rule_counter = 1

# Work only on correctly detected attacks
attack_samples = detected_features.copy()

rule_templates = [
    ("SQL_INJECTION", "has_sql_injection == 1", "HIGH"),
    ("XSS", "has_xss == 1", "HIGH"),
    ("PATH_TRAVERSAL", "has_path_traversal == 1", "HIGH"),
    ("COMMAND_INJECTION", "has_command_injection == 1", "CRITICAL"),
    ("BURST_TRAFFIC", "burst_indicator == 1", "MEDIUM"),
    ("HIGH_ENTROPY", "high_entropy == 1", "MEDIUM"),
    ("HIGH_FREQUENCY", "high_frequency == 1", "MEDIUM"),
    ("ERROR_ABUSE", "is_error == 1", "LOW")
]

rule_stats = defaultdict(int)

for rule_name, condition, severity in rule_templates:
    matched = attack_samples.query(condition.replace("==", "=="))

    if len(matched) == 0:
        continue

    rule_id = f"RULE_{rule_name}_{rule_counter:03d}"
    rule_counter += 1

    rule = {
        "rule_id": rule_id,
        "type": rule_name,
        "condition": condition,
        "action": "BLOCK",
        "severity": severity,
        "trigger_count": int(len(matched)),
        "confidence": round(len(matched) / len(attack_samples), 3),
        "description": f"Auto-generated rule for detecting {rule_name.lower().replace('_',' ')} attacks"
    }

    rules.append(rule)
    rule_stats[severity] += 1

    # Print rule in terminal
    print(f"🛑 {rule_id}")
    print(f"   Condition : {rule['condition']}")
    print(f"   Severity  : {rule['severity']}")
    print(f"   Triggers  : {rule['trigger_count']}")
    print(f"   Confidence: {rule['confidence']}")
    print()

print(f"✓ Total rules generated: {len(rules)}")
print(f"Rule severity distribution: {dict(rule_stats)}")
print()


# -----------------------------
# 12. ADAPTIVE THRESHOLD MECHANISM
# -----------------------------

print(f"⚙️  ADAPTIVE THRESHOLD CALIBRATION")
print(f"{'─'*70}")

# Calculate optimal threshold based on F1 score
thresholds = np.linspace(if_scores_inverted.min(), if_scores_inverted.max(), 100)
f1_scores_threshold = []

for thresh in thresholds:
    pred_thresh = (if_scores_inverted >= thresh).astype(int)
    if pred_thresh.sum() > 0:
        f1_thresh = f1_score(y_eval, pred_thresh)
        f1_scores_threshold.append(f1_thresh)
    else:
        f1_scores_threshold.append(0)

optimal_threshold_idx = np.argmax(f1_scores_threshold)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"Score range: [{if_scores_inverted.min():.4f}, {if_scores_inverted.max():.4f}]")
print(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
print(f"F1 at optimal threshold: {f1_scores_threshold[optimal_threshold_idx]:.4f}")
print()


# -----------------------------
# 13. WHITELISTING MECHANISM
# -----------------------------

print(f"✅ IP WHITELISTING SYSTEM")
print(f"{'─'*70}")

# Identify IPs with consistent normal behavior
ip_behavior = df_original.groupby('src_ip').agg({
    'label': ['mean', 'count'],
    'is_error': 'mean'
}).reset_index()
ip_behavior.columns = ['src_ip', 'attack_rate', 'request_count', 'error_rate']

# Whitelist criteria: 0 attacks, 10+ requests, low error rate
whitelist_candidates = ip_behavior[
    (ip_behavior['attack_rate'] == 0) & 
    (ip_behavior['request_count'] >= 10) &
    (ip_behavior['error_rate'] < 0.1)
]

print(f"Whitelist candidates: {len(whitelist_candidates)} IPs")
print(f"Sample whitelisted IPs:")
if len(whitelist_candidates) > 0:
    for ip in whitelist_candidates['src_ip'].head(5):
        print(f"  • {ip}")
else:
    print("  (No IPs meet whitelist criteria in sample data)")
print()

# Save whitelist
whitelist_ips = set(whitelist_candidates['src_ip'].tolist())


# -----------------------------
# 14. SAVE ALL ARTIFACTS
# -----------------------------

print(f"💾 SAVING MODEL ARTIFACTS")
print(f"{'─'*70}")

# Save model
joblib.dump(best_model, "model_isolation_forest.pkl")

# Save preprocessing artifacts
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(method_encoder, "method_encoder.pkl")

# Save whitelist
with open("ip_whitelist.txt", "w") as f:
    for ip in whitelist_ips:
        f.write(f"{ip}\n")

# Save feature columns
with open("feature_columns.txt", "w") as f:
    f.write("\n".join(FEATURE_COLUMNS))

# Save model configuration
model_config = {
    'model_type': 'IsolationForest',
    'contamination': best_cont,
    'optimal_threshold': optimal_threshold,
    'feature_columns': FEATURE_COLUMNS,
    'performance': {
        'accuracy': if_acc,
        'precision': if_prec,
        'recall': if_rec,
        'f1': if_f1,
        'roc_auc': if_auc,
        'fpr': fpr,
        'fnr': fnr
    }
}
joblib.dump(model_config, "model_config.pkl")

print(f"✓ Saved: model_isolation_forest.pkl")
print(f"✓ Saved: feature_scaler.pkl")
print(f"✓ Saved: method_encoder.pkl")
print(f"✓ Saved: ip_whitelist.txt ({len(whitelist_ips)} IPs)")
print(f"✓ Saved: feature_columns.txt")
print(f"✓ Saved: model_config.pkl")
print()

# -----------------------------
# 14.5 SAVE GENERATED RULES
# -----------------------------

print(f"💾 SAVING GENERATED RULES")
print(f"{'─'*70}")

rules_metadata = {
    "generated_at": datetime.utcnow().isoformat(),
    "model": "IsolationForest",
    "total_rules": len(rules),
    "rules": rules
}

with open("rules.json", "w") as f:
    json.dump(rules_metadata, f, indent=4)

print(f"✓ Saved: rules.json ({len(rules)} rules)")
print()

# -----------------------------
# 15. REAL-TIME INFERENCE CLASS
# -----------------------------

print(f"🚀 CREATING REAL-TIME INFERENCE ENGINE")
print(f"{'─'*70}")

class WAFAnomalyDetector:
    """
    Production-ready WAF anomaly detection system
    """
    
    def __init__(self, model_path_prefix="./"):
        # Load model
        self.model = joblib.load(f"{model_path_prefix}model_isolation_forest.pkl")
        
        # Load preprocessing
        self.scaler = joblib.load(f"{model_path_prefix}feature_scaler.pkl")
        self.method_encoder = joblib.load(f"{model_path_prefix}method_encoder.pkl")
        
        # Load configuration
        self.config = joblib.load(f"{model_path_prefix}model_config.pkl")
        self.feature_columns = self.config['feature_columns']
        self.optimal_threshold = self.config.get('optimal_threshold', 0)
        
        # Load whitelist
        self.whitelist = set()
        try:
            with open(f"{model_path_prefix}ip_whitelist.txt", "r") as f:
                self.whitelist = set(line.strip() for line in f)
        except:
            pass
    
    def preprocess_request(self, request_data):
        """
        Preprocess a single request for inference
        """
        # Check whitelist first
        if request_data.get('src_ip') in self.whitelist:
            return None, "WHITELISTED"
        
        # Create feature vector
        features = {}
        for feat in self.feature_columns:
            if feat in request_data:
                features[feat] = request_data[feat]
            else:
                features[feat] = 0
        
        # Convert to array
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Apply log transform
        skewed_cols = ["bytes_sent", "req_per_ip_1min", "req_per_ip_10sec",
                       "time_gap", "url_length", "query_length", "ua_length"]
        for i, col in enumerate(self.feature_columns):
            if col in skewed_cols:
                feature_vector[0, i] = np.log1p(feature_vector[0, i])
        
        # Scale
        feature_scaled = self.scaler.transform(feature_vector)
        
        return feature_scaled, "OK"
    
    def predict(self, request_data, return_scores=False):
        """
        Predict if request is anomalous
        
        Returns:
            - decision: 'BLOCK' or 'ALLOW'
            - confidence: 0-1 score
            - explanation: dict with details
        """
        # Preprocess
        features, status = self.preprocess_request(request_data)
        
        if status == "WHITELISTED":
            return 'ALLOW', 1.0, {'reason': 'IP whitelisted'}
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        score = self.model.decision_function(features)[0]
        
        # Invert score (higher = more anomalous)
        score_inverted = -score
        
        is_anomaly = (prediction == -1)
        decision = 'BLOCK' if is_anomaly else 'ALLOW'
        
        # Confidence based on score distance from threshold
        max_score = max(abs(score), 1)
        confidence = min(abs(score) / max_score, 1.0)
        
        explanation = {
            'prediction': 'ANOMALY' if is_anomaly else 'NORMAL',
            'anomaly_score': float(score),
            'inverted_score': float(score_inverted),
            'threshold': float(self.optimal_threshold)
        }
        
        # Add attack indicators
        indicators = []
        if request_data.get('has_sql_injection', 0) == 1:
            indicators.append('SQL Injection pattern')
        if request_data.get('has_xss', 0) == 1:
            indicators.append('XSS pattern')
        if request_data.get('has_path_traversal', 0) == 1:
            indicators.append('Path traversal')
        if request_data.get('burst_indicator', 0) == 1:
            indicators.append('Burst traffic')
        if request_data.get('high_entropy', 0) == 1:
            indicators.append('High payload entropy')
        
        if indicators:
            explanation['attack_indicators'] = indicators
        
        if return_scores:
            return decision, confidence, explanation, score_inverted
        
        return decision, confidence, explanation

# Save the class
joblib.dump(WAFAnomalyDetector, "waf_detector_class.pkl")
print(f"✓ Saved: waf_detector_class.pkl")
print()


# -----------------------------
# 16. REAL-TIME DEMO
# -----------------------------

print(f"="*70)
print(" "*20 + "REAL-TIME DETECTION DEMO")
print(f"="*70)
print()

# Initialize detector
detector = WAFAnomalyDetector()

# Test cases
test_requests = [
    {
        'name': 'Normal Request',
        'data': {
            'src_ip': '192.168.5.225',
            'method': 'GET',
            'url_length': 5,
            'query_length': 0,
            'bytes_sent': 1290,
            'request_time': 0.273,
            'req_per_ip_1min': 10,
            'req_per_ip_10sec': 8,
            'unique_urls_per_ip': 1,
            'time_gap': 5.0,
            'is_https': 0,
            'payload_entropy': 2.68,
            'method_encoded': 0,
            'is_error': 0,
            'is_server_error': 0,
            'is_client_error': 0,
            'is_redirect': 0,
            'is_bot_ua': 1,
            'ua_length': 21,
            'has_sql_injection': 0,
            'has_xss': 0,
            'has_path_traversal': 0,
            'has_command_injection': 0,
            'has_special_chars': 0,
            'url_depth': 1,
            'has_query': 0,
            'query_param_count': 0,
            'req_ratio_10s_to_1m': 0.8,
            'burst_indicator': 0,
            'high_frequency': 0,
            'payload_entropy_norm': 0.335,
            'high_entropy': 0,
            'size_anomaly': 0,
            'slow_request': 0,
            'ip_error_rate': 0.0,
            'ip_max_req_rate': 10,
            'hour': 15,
            'minute': 40,
            'day_of_week': 0,
            'is_business_hours': 1
        }
    },
    {
        'name': 'SQL Injection Attack',
        'data': {
            'src_ip': '192.168.4.5',
            'method': 'PUT',
            'url_length': 24,
            'query_length': 19,
            'bytes_sent': 7935,
            'request_time': 2.192,
            'req_per_ip_1min': 131,
            'req_per_ip_10sec': 74,
            'unique_urls_per_ip': 28,
            'time_gap': 5.0,
            'is_https': 1,
            'payload_entropy': 5.25,
            'method_encoded': 2,
            'is_error': 1,
            'is_server_error': 0,
            'is_client_error': 1,
            'is_redirect': 0,
            'is_bot_ua': 1,
            'ua_length': 21,
            'has_sql_injection': 1,
            'has_xss': 0,
            'has_path_traversal': 0,
            'has_command_injection': 0,
            'has_special_chars': 1,
            'url_depth': 2,
            'has_query': 1,
            'query_param_count': 1,
            'req_ratio_10s_to_1m': 0.565,
            'burst_indicator': 1,
            'high_frequency': 1,
            'payload_entropy_norm': 0.656,
            'high_entropy': 1,
            'size_anomaly': 1,
            'slow_request': 1,
            'ip_error_rate': 0.3,
            'ip_max_req_rate': 131,
            'hour': 15,
            'minute': 40,
            'day_of_week': 0,
            'is_business_hours': 1
        }
    },
    {
        'name': 'XSS Attack',
        'data': {
            'src_ip': '192.168.5.134',
            'method': 'POST',
            'url_length': 35,
            'query_length': 16,
            'bytes_sent': 4548,
            'request_time': 1.748,
            'req_per_ip_1min': 194,
            'req_per_ip_10sec': 44,
            'unique_urls_per_ip': 12,
            'time_gap': 5.0,
            'is_https': 0,
            'payload_entropy': 4.88,
            'method_encoded': 1,
            'is_error': 1,
            'is_server_error': 0,
            'is_client_error': 1,
            'is_redirect': 0,
            'is_bot_ua': 0,
            'ua_length': 11,
            'has_sql_injection': 0,
            'has_xss': 1,
            'has_path_traversal': 0,
            'has_command_injection': 0,
            'has_special_chars': 1,
            'url_depth': 1,
            'has_query': 1,
            'query_param_count': 1,
            'req_ratio_10s_to_1m': 0.227,
            'burst_indicator': 0,
            'high_frequency': 1,
            'payload_entropy_norm': 0.610,
            'high_entropy': 0,
            'size_anomaly': 0,
            'slow_request': 1,
            'ip_error_rate': 0.4,
            'ip_max_req_rate': 194,
            'hour': 15,
            'minute': 41,
            'day_of_week': 0,
            'is_business_hours': 1
        }
    }
]

for test in test_requests:
    print(f"🔍 Testing: {test['name']}")
    print(f"{'─'*70}")
    
    decision, confidence, explanation = detector.predict(test['data'])
    
    if decision == 'BLOCK':
        print(f"🚨 Decision: {decision} (Confidence: {confidence*100:.1f}%)")
    else:
        print(f"✅ Decision: {decision} (Confidence: {confidence*100:.1f}%)")
    
    print(f"\nPrediction: {explanation['prediction']}")
    print(f"Anomaly Score: {explanation['anomaly_score']:.4f}")
    
    if 'attack_indicators' in explanation:
        print(f"\nAttack Indicators Detected:")
        for indicator in explanation['attack_indicators']:
            print(f"  ⚠️  {indicator}")
    
    print()

print(f"="*70)
print(" "*25 + "PIPELINE COMPLETE")
print(f"="*70)
print()
print(f"📁 Generated Files:")
print(f"  • waf_analysis.png - Comprehensive visualizations")
if SHAP_AVAILABLE:
    print(f"  • shap_analysis.png - Feature importance")
print(f"  • model_isolation_forest.pkl - Isolation Forest model")
print(f"  • feature_scaler.pkl - Feature scaler")
print(f"  • method_encoder.pkl - Method encoder")
print(f"  • model_config.pkl - Model configuration")
print(f"  • ip_whitelist.txt - Whitelisted IPs")
print(f"  • feature_columns.txt - Feature list")
print(f"  • waf_detector_class.pkl - Detector class")
print()
print(f"🚀 Ready for deployment!")
print(f"   Use WAFAnomalyDetector class for real-time inference")
print(f"="*70)