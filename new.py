import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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

print("=" * 70)
print(" " * 20 + "WAF ANOMALY DETECTION SYSTEM")
print("=" * 70)
print(f"\n📊 DATASET OVERVIEW")
print(f"{'─' * 70}")
print(f"Total requests: {len(df):,}")
print(f"Attack samples: {df['label'].sum():,} ({df['label'].mean() * 100:.1f}%)")
print(f"Normal samples: {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean() * 100:.1f}%)")
print()

# -----------------------------
# 3. COMPREHENSIVE FEATURE ENGINEERING
# -----------------------------

print(f"🔧 FEATURE ENGINEERING")
print(f"{'─' * 70}")

# Convert timestamp
df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='mixed',
    errors='coerce'
)

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
    'label': 'mean',
    'is_error': 'mean',
    'req_per_ip_1min': 'max'
}).reset_index()
ip_stats.columns = ['src_ip', 'ip_attack_history', 'ip_error_rate', 'ip_max_req_rate']

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
    "ip_attack_history", "ip_error_rate", "ip_max_req_rate"
]

X = df[FEATURE_COLUMNS].copy()
y = df["label"].copy()

# Store original data for analysis
df_original = df.copy()

print(f"📋 FEATURE SUMMARY")
print(f"{'─' * 70}")
print(f"Total features: {len(FEATURE_COLUMNS)}")
print(f"Feature categories:")
print(f"  • Security patterns: 5")
print(f"  • Rate-based: 6")
print(f"  • HTTP/Status: 5")
print(f"  • Temporal: 4")
print(f"  • Reputation: 3")
print(f"  • Other: {len(FEATURE_COLUMNS) - 23}")
print()

# -----------------------------
# 5. FEATURE SCALING & CLEANING
# -----------------------------

print(f"⚙️  PREPROCESSING")
print(f"{'─' * 70}")

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
print(f"{'─' * 70}")
print(f"Train: {len(X_train):,} samples ({y_train.sum()} attacks, {y_train.mean() * 100:.1f}%)")
print(f"Test:  {len(X_test):,} samples ({y_test.sum()} attacks, {y_test.mean() * 100:.1f}%)")
print(f"Eval:  {len(X_eval):,} samples ({y_eval.sum()} attacks, {y_eval.mean() * 100:.1f}%)")
print()

# -----------------------------
# 7. MODEL TRAINING & COMPARISON
# -----------------------------

print("=" * 70)
print(" " * 25 + "MODEL TRAINING")
print("=" * 70)
print()

# Store all models and predictions
models = {}
predictions = {}
scores_dict = {}

# ---------------
# MODEL 1: ISOLATION FOREST
# ---------------
print(f"🌲 Training Isolation Forest...")
print(f"{'─' * 70}")

best_if_f1 = 0
best_if_model = None
best_if_cont = 0

contamination_levels = [0.05, 0.07, 0.10, 0.12, 0.15]
for cont in contamination_levels:
    model_temp = IsolationForest(
        n_estimators=300,
        contamination=cont,
        max_samples='auto',
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model_temp.fit(X_train)
    pred_temp = (model_temp.predict(X_test) == -1).astype(int)
    f1_temp = f1_score(y_test, pred_temp)

    print(f"  contamination={cont:.2f} → F1={f1_temp:.4f}")

    if f1_temp > best_if_f1:
        best_if_f1 = f1_temp
        best_if_model = model_temp
        best_if_cont = cont

print(f"\n✓ Best contamination: {best_if_cont:.2f}")

# Evaluate on eval set
if_scores = best_if_model.decision_function(X_eval)
if_pred = (best_if_model.predict(X_eval) == -1).astype(int)

if_acc = accuracy_score(y_eval, if_pred)
if_prec = precision_score(y_eval, if_pred)
if_rec = recall_score(y_eval, if_pred)
if_f1 = f1_score(y_eval, if_pred)
if_auc = roc_auc_score(y_eval, (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10))

print(f"\n📊 Isolation Forest Results:")
print(f"  Accuracy:  {if_acc:.4f}")
print(f"  Precision: {if_prec:.4f}")
print(f"  Recall:    {if_rec:.4f}")
print(f"  F1 Score:  {if_f1:.4f}")
print(f"  ROC-AUC:   {if_auc:.4f}")
print()

models['IsolationForest'] = best_if_model
predictions['IsolationForest'] = if_pred
scores_dict['IsolationForest'] = if_scores

# ---------------
# MODEL 2: LOCAL OUTLIER FACTOR
# ---------------
print(f"🔍 Training Local Outlier Factor...")
print(f"{'─' * 70}")

best_lof_f1 = 0
best_lof_model = None
best_lof_cont = 0

for cont in contamination_levels:
    model_temp = LocalOutlierFactor(
        n_neighbors=20,
        contamination=cont,
        novelty=True,
        n_jobs=-1
    )
    model_temp.fit(X_train)
    pred_temp = (model_temp.predict(X_test) == -1).astype(int)
    f1_temp = f1_score(y_test, pred_temp)

    print(f"  contamination={cont:.2f} → F1={f1_temp:.4f}")

    if f1_temp > best_lof_f1:
        best_lof_f1 = f1_temp
        best_lof_model = model_temp
        best_lof_cont = cont

print(f"\n✓ Best contamination: {best_lof_cont:.2f}")

# Evaluate on eval set
lof_scores = best_lof_model.decision_function(X_eval)
lof_pred = (best_lof_model.predict(X_eval) == -1).astype(int)

lof_acc = accuracy_score(y_eval, lof_pred)
lof_prec = precision_score(y_eval, lof_pred)
lof_rec = recall_score(y_eval, lof_pred)
lof_f1 = f1_score(y_eval, lof_pred)
lof_auc = roc_auc_score(y_eval, (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10))

print(f"\n📊 Local Outlier Factor Results:")
print(f"  Accuracy:  {lof_acc:.4f}")
print(f"  Precision: {lof_prec:.4f}")
print(f"  Recall:    {lof_rec:.4f}")
print(f"  F1 Score:  {lof_f1:.4f}")
print(f"  ROC-AUC:   {lof_auc:.4f}")
print()

models['LOF'] = best_lof_model
predictions['LOF'] = lof_pred
scores_dict['LOF'] = lof_scores

# -----------------------------
# 8. ENSEMBLE VOTING (2 MODELS)
# -----------------------------

print("=" * 70)
print(" " * 25 + "ENSEMBLE MODEL")
print("=" * 70)
print()

# Majority voting (both models predict anomaly)
ensemble_pred = ((predictions['IsolationForest'] +
                  predictions['LOF']) >= 2).astype(int)

# Ensemble metrics
ens_acc = accuracy_score(y_eval, ensemble_pred)
ens_prec = precision_score(y_eval, ensemble_pred)
ens_rec = recall_score(y_eval, ensemble_pred)
ens_f1 = f1_score(y_eval, ensemble_pred)

# For ROC-AUC, use average scores
ensemble_scores = (
                          (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10) +
                          (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
                  ) / 2
ens_auc = roc_auc_score(y_eval, ensemble_scores)

print(f"🎯 Ensemble Voting Strategy: Both models agree")
print(f"{'─' * 70}")
print(f"📊 Ensemble Results:")
print(f"  Accuracy:  {ens_acc:.4f}")
print(f"  Precision: {ens_prec:.4f}")
print(f"  Recall:    {ens_rec:.4f}")
print(f"  F1 Score:  {ens_f1:.4f}")
print(f"  ROC-AUC:   {ens_auc:.4f}")
print()

# Comparison table
print(f"📊 MODEL COMPARISON")
print(f"{'─' * 70}")
comparison_df = pd.DataFrame({
    'Model': ['Isolation Forest', 'LOF', 'Ensemble'],
    'Accuracy': [if_acc, lof_acc, ens_acc],
    'Precision': [if_prec, lof_prec, ens_prec],
    'Recall': [if_rec, lof_rec, ens_rec],
    'F1 Score': [if_f1, lof_f1, ens_f1],
    'ROC-AUC': [if_auc, lof_auc, ens_auc]
})
print(comparison_df.to_string(index=False))
print()

# Determine best model
best_model_name = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
print(f"🏆 Best performing model: {best_model_name}")
print()

# -----------------------------
# 9. DETAILED EVALUATION
# -----------------------------

print(f"📋 DETAILED CLASSIFICATION REPORT (Best Model: {best_model_name})")
print(f"{'─' * 70}")

# Use the best model's predictions
if best_model_name == 'Isolation Forest':
    best_pred = if_pred
elif best_model_name == 'LOF':
    best_pred = lof_pred
else:
    best_pred = ensemble_pred

print(classification_report(y_eval, best_pred, target_names=['Normal', 'Attack']))

cm = confusion_matrix(y_eval, best_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 CONFUSION MATRIX")
print(f"{'─' * 70}")
print(f"                 Predicted")
print(f"              Normal  Attack")
print(f"Actual Normal   {tn:4d}   {fp:4d}")
print(f"       Attack   {fn:4d}   {tp:4d}")
print()

# Key metrics for WAF
fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

print(f"🎯 CRITICAL WAF METRICS")
print(f"{'─' * 70}")
print(f"False Positive Rate: {fpr:.4f} ({fp}/{tn + fp} legitimate requests blocked)")
print(f"False Negative Rate: {fnr:.4f} ({fn}/{tp + fn} attacks missed)")
print(
    f"True Positive Rate:  {if_rec if best_model_name == 'Isolation Forest' else (lof_rec if best_model_name == 'LOF' else ens_rec):.4f} (Detection rate)")
print()

# -----------------------------
# 10. VISUALIZATIONS
# -----------------------------

print(f"📈 GENERATING VISUALIZATIONS")
print(f"{'─' * 70}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('WAF Anomaly Detection - Comprehensive Analysis', fontsize=16, fontweight='bold')

# 1. ROC Curves
ax1 = axes[0, 0]
for name, scores in scores_dict.items():
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    fpr_curve, tpr_curve, _ = roc_curve(y_eval, scores_norm)
    auc_score = roc_auc_score(y_eval, scores_norm)
    ax1.plot(fpr_curve, tpr_curve, label=f'{name} (AUC={auc_score:.3f})', linewidth=2)

# Ensemble ROC
fpr_ens, tpr_ens, _ = roc_curve(y_eval, ensemble_scores)
ax1.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={ens_auc:.3f})',
         linewidth=3, linestyle='--', color='red')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves - All Models')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 2. Precision-Recall Curves
ax2 = axes[0, 1]
for name, preds in predictions.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_eval, preds)
    ax2.plot(recall_curve, precision_curve, label=name, linewidth=2)

precision_ens, recall_ens, _ = precision_recall_curve(y_eval, ensemble_pred)
ax2.plot(recall_ens, precision_ens, label='Ensemble',
         linewidth=3, linestyle='--', color='red')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Model Comparison Bar Chart
ax3 = axes[0, 2]
metrics_to_plot = ['F1 Score', 'Precision', 'Recall', 'ROC-AUC']
x = np.arange(len(comparison_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax3.bar(x + i * width, comparison_df[metric], width, label=metric)

ax3.set_xlabel('Model')
ax3.set_ylabel('Score')
ax3.set_title('Model Performance Comparison')
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Confusion Matrix Heatmap
ax4 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
ax4.set_title(f'Confusion Matrix ({best_model_name})')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# 5. Attack Detection by Model
ax5 = axes[1, 1]
detection_rates = {
    'Isolation Forest': if_rec,
    'LOF': lof_rec,
    'Ensemble': ens_rec
}
colors = ['#3498db', '#2ecc71', '#f39c12']
bars = ax5.bar(detection_rates.keys(), detection_rates.values(), color=colors)
ax5.set_ylabel('Detection Rate (Recall)')
ax5.set_title('Attack Detection Rate by Model')
ax5.set_ylim([0, 1.0])
ax5.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}', ha='center', va='bottom')

# 6. False Positive Rate Comparison
ax6 = axes[1, 2]
fpr_comparison = {
    'Isolation Forest': confusion_matrix(y_eval, predictions['IsolationForest'])[0, 1] /
                        confusion_matrix(y_eval, predictions['IsolationForest'])[0].sum(),
    'LOF': confusion_matrix(y_eval, predictions['LOF'])[0, 1] /
           confusion_matrix(y_eval, predictions['LOF'])[0].sum(),
    'Ensemble': fpr
}
bars2 = ax6.bar(fpr_comparison.keys(), fpr_comparison.values(), color=colors)
ax6.set_ylabel('False Positive Rate')
ax6.set_title('False Positive Rate by Model (Lower is Better)')
ax6.set_ylim([0, max(fpr_comparison.values()) * 1.2])
ax6.grid(True, alpha=0.3, axis='y')

for bar in bars2:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('waf_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: waf_analysis.png")

# -----------------------------
# 11. FEATURE IMPORTANCE (SHAP)
# -----------------------------

if SHAP_AVAILABLE:
    print(f"\n🔍 GENERATING SHAP EXPLANATIONS")
    print(f"{'─' * 70}")

    try:
        sample_size = min(100, len(X_eval))
        X_eval_sample = X_eval[:sample_size]

        explainer = shap.TreeExplainer(models['IsolationForest'])
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
# 12. ATTACK PATTERN ANALYSIS
# -----------------------------

print(f"🎯 ATTACK PATTERN ANALYSIS")
print(f"{'─' * 70}")

detected_idx = np.where(best_pred == 1)[0]
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
    detected_features = X[FEATURE_COLUMNS].iloc[detected_idx]

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
# 13. ADAPTIVE THRESHOLD MECHANISM
# -----------------------------

print(f"⚙️  ADAPTIVE THRESHOLD CALIBRATION")
print(f"{'─' * 70}")

thresholds = np.linspace(0, 1, 100)
f1_scores_threshold = []

for thresh in thresholds:
    pred_thresh = (ensemble_scores >= thresh).astype(int)
    if pred_thresh.sum() > 0:
        f1_thresh = f1_score(y_eval, pred_thresh)
        f1_scores_threshold.append(f1_thresh)
    else:
        f1_scores_threshold.append(0)

optimal_threshold_idx = np.argmax(f1_scores_threshold)
optimal_threshold = thresh